#!/usr/bin/env python3
"""
Testing script to compare captions generated with RGB-only vs RGB+NDVI.
Uses a fixed list of images from a GCS bucket and outputs comparison to CSV.

Notes:
- Uses SYSTEM PROMPT V5 from `prompts_V5.py`
- Uses the question list from `prompts.py` (V5 file currently only contains the system prompt)
"""

import sys
import os
import csv
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
load_dotenv()

from google.cloud import storage
from PIL import Image

from mllm_code.prompts import questions, multi_shot_examples
from mllm_code.prompts_V5 import system_prompt as system_prompt_v5
from mllm_code.mllm_helper import LlamaPromptGenerator_mines, LlamaCaptionGenerator


# Configuration
GCS_BUCKET = os.getenv("GCS_BUCKET", "nudge-bucket")
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "rgb_vs_rgb_ndvi_comparison.csv")

# Fixed test list (basenames without extension), provided by user
DEFAULT_IMAGE_BASES: List[str] = [
    "AdamsPit_rgb_2024-09-13",
    "AvonNorthOpenPit_rgb_2024-12-29",
    "BlackwoodOpencut_rgb_2024-12-26",
    "BrumagenOpenCut_rgb_2024-12-25",
    "CadiaHillMine_rgb_2024-12-29",
    "CarbocalcioQuarry_rgb_2024-08-08",
    "Endeavour22_rgb_2024-12-27",
]

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def load_image_from_gcs(bucket, blob_name):
    """Download image from GCS and return as PIL Image."""
    blob = bucket.blob(blob_name)
    image_bytes = blob.download_as_bytes()
    return Image.open(BytesIO(image_bytes))


def _parse_base(base: str) -> Tuple[str, str]:
    """
    Parse a base name like '<mine>_rgb_<YYYY-MM-DD>' into (mine, date).
    Falls back to ('', '') if it can't parse.
    """
    try:
        parts = base.split("_")
        if len(parts) >= 3:
            return parts[0], parts[-1]
    except Exception:
        pass
    return "", ""


def find_rgb_and_ndvi_blobs(bucket, bases: List[str]) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Try to find matching RGB and NDVI blob names for each provided base.

    Expected patterns (flexible):
    - RGB: name contains the base string and an image extension
    - NDVI: same mine+date, and name contains 'ndvi' and an image extension
    """
    targets: Dict[str, Dict[str, Optional[str]]] = {
        base: {"rgb": None, "ndvi": None} for base in bases
    }

    # Precompute tokens to make matching robust to directories/prefixes.
    tokens = {}
    for base in bases:
        mine, date = _parse_base(base)
        tokens[base] = {
            "base_lower": base.lower(),
            "mine_lower": mine.lower(),
            "date": date,
        }

    # Scan blobs once; stop early when all targets are found.
    remaining = set(bases)
    for blob in bucket.list_blobs():
        name = blob.name
        lower = name.lower()
        if not lower.endswith(IMAGE_EXTENSIONS):
            continue

        # Try to match RGB and NDVI for each base (only 7 bases, so this is cheap).
        for base in list(remaining):
            info = tokens[base]
            mine_lower = info["mine_lower"]
            date = info["date"]
            if mine_lower and mine_lower not in lower:
                continue
            if date and date not in name:
                continue

            # RGB match: contains the base (e.g. ".../AdamsPit_rgb_2024-09-13.png")
            if targets[base]["rgb"] is None and info["base_lower"] in lower:
                targets[base]["rgb"] = name

            # NDVI match: mine + date + "ndvi"
            if targets[base]["ndvi"] is None and "ndvi" in lower:
                targets[base]["ndvi"] = name

            # If both found, remove from remaining set.
            if targets[base]["rgb"] and targets[base]["ndvi"]:
                remaining.discard(base)

        if not remaining:
            break

    return targets


def generate_captions_rgb_vs_rgb_ndvi(
    rgb_image: Image.Image,
    ndvi_image: Optional[Image.Image],
    blob_name_for_metadata: str,
    question: str,
    use_multishot: bool = True,
) -> Tuple[str, str]:
    """
    Generate two captions: RGB-only vs RGB+NDVI, holding everything else constant.
    
    Args:
        rgb_image: PIL Image for the RGB image
        ndvi_image: Optional PIL Image for the NDVI image (if not found, RGB+NDVI caption becomes a placeholder)
        blob_name_for_metadata: GCS blob name used for metadata extraction + prompt construction
        question: The question to ask about the image
        use_multishot: If True, include multishot examples in the prompt
        
    Returns:
        Tuple of (caption_rgb_only, caption_rgb_plus_ndvi)
    """
    model_name = 'meta/llama-4-maverick-17b-128e-instruct'
    invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
    
    # Generate prompt - pass blob name for metadata extraction
    if use_multishot:
        prompt, location, filename, country, mine_name, latitude, longitude = LlamaPromptGenerator_mines(
            blob_name_for_metadata, question, multi_shot_examples=multi_shot_examples
        )
    else:
        prompt, location, filename, country, mine_name, latitude, longitude = LlamaPromptGenerator_mines(
            blob_name_for_metadata, question, multi_shot_examples=None
        )
    
    # Generate captions - pass PIL image(s) for processing
    caption_rgb_only = LlamaCaptionGenerator(
        rgb_image, system_prompt_v5, prompt, model_name, invoke_url
    )

    if ndvi_image is None:
        caption_rgb_plus_ndvi = "NDVI_NOT_FOUND"
    else:
        caption_rgb_plus_ndvi = LlamaCaptionGenerator(
            rgb_image,
            system_prompt_v5,
            prompt,
            model_name,
            invoke_url,
            second_image_file_path_or_image=ndvi_image,
            first_image_label="RGB",
            second_image_label="NDVI",
        )
    
    return caption_rgb_only, caption_rgb_plus_ndvi


def run_rgb_vs_rgb_ndvi_comparison(image_bases: Optional[List[str]] = None, use_multishot: bool = True):
    """Run RGB-only vs RGB+NDVI comparison on a fixed list of GCS images."""
    
    image_bases = image_bases or DEFAULT_IMAGE_BASES
    print(f"Connecting to GCS bucket: {GCS_BUCKET}")
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    
    print(f"Looking up blobs for {len(image_bases)} image basenames...")
    blob_map = find_rgb_and_ndvi_blobs(bucket, image_bases)

    print("Resolved blob names:")
    for base in image_bases:
        print(f"  - {base}: rgb={blob_map[base]['rgb']} ndvi={blob_map[base]['ndvi']}")

    print(f"Processing {len(image_bases)} images")
    print(f"Questions: {len(questions)}")
    print(f"Total comparisons: {len(image_bases) * len(questions)}")
    print("=" * 60)
    
    results = []
    
    for idx, base in enumerate(image_bases, 1):
        rgb_blob = blob_map[base]["rgb"]
        ndvi_blob = blob_map[base]["ndvi"]
        print(f"\n[{idx}/{len(image_bases)}] Base: {base}")
        print(f"  RGB blob : {rgb_blob}")
        print(f"  NDVI blob: {ndvi_blob}")

        if not rgb_blob:
            # Can't do anything without the RGB image
            for question in questions:
                results.append({
                    "base": base,
                    "rgb_blob": None,
                    "ndvi_blob": ndvi_blob,
                    "question": question,
                    "rgb_only": "RGB_NOT_FOUND",
                    "rgb_plus_ndvi": "RGB_NOT_FOUND",
                })
            continue

        rgb_image = load_image_from_gcs(bucket, rgb_blob)
        ndvi_image = load_image_from_gcs(bucket, ndvi_blob) if ndvi_blob else None
        
        for q_idx, question in enumerate(questions, 1):
            print(f"  Q{q_idx}: {question[:50]}...")
            
            try:
                print("    → RGB only...")
                caption_rgb_only, caption_rgb_plus_ndvi = generate_captions_rgb_vs_rgb_ndvi(
                    rgb_image=rgb_image,
                    ndvi_image=ndvi_image,
                    blob_name_for_metadata=rgb_blob,
                    question=question,
                    use_multishot=use_multishot,
                )
                
                results.append({
                    "base": base,
                    "rgb_blob": rgb_blob,
                    "ndvi_blob": ndvi_blob,
                    "question": question,
                    "rgb_only": caption_rgb_only,
                    "rgb_plus_ndvi": caption_rgb_plus_ndvi,
                })
                print("    ✓ Done")
                
            except Exception as e:
                print(f"    ✗ Error: {e}")
                results.append({
                    "base": base,
                    "rgb_blob": rgb_blob,
                    "ndvi_blob": ndvi_blob,
                    "question": question,
                    "rgb_only": f"ERROR: {e}",
                    "rgb_plus_ndvi": f"ERROR: {e}",
                })
    
    # Write to CSV
    print(f"\n{'=' * 60}")
    print(f"Writing results to: {OUTPUT_CSV}")
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "base",
                "rgb_blob",
                "ndvi_blob",
                "question",
                "rgb_only",
                "rgb_plus_ndvi",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✅ Done! Saved {len(results)} comparisons to {OUTPUT_CSV}")


if __name__ == "__main__":
    # Default behavior: run RGB vs RGB+NDVI comparison on the fixed list.
    run_rgb_vs_rgb_ndvi_comparison()
