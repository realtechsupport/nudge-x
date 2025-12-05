#!/usr/bin/env python3
"""
Testing script to compare captions generated WITH and WITHOUT multishot examples.
Uses images from GCS bucket and outputs comparison to CSV.
"""

import sys
import os
import csv
from datetime import datetime
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()

from google.cloud import storage
from PIL import Image

from mllm_code.prompts import questions, system_prompt, multi_shot_examples
from mllm_code.config.settings import mllm_model
from mllm_code.mllm_helper import LlamaPromptGenerator_mines, LlamaCaptionGenerator


# Configuration
GCS_BUCKET = "nudge-bucket"
NUM_IMAGES = 10  # Number of images to compare
OUTPUT_CSV = "multishot_comparison.csv"


def load_image_from_gcs(bucket, blob_name):
    """Download image from GCS and return as PIL Image."""
    blob = bucket.blob(blob_name)
    image_bytes = blob.download_as_bytes()
    return Image.open(BytesIO(image_bytes))


def generate_caption(pil_image, blob_name: str, question: str, use_multishot: bool = True) -> tuple:
    """
    Generate a caption for an image with or without multishot examples.
    
    Args:
        pil_image: PIL Image object
        blob_name: GCS blob name (used for metadata extraction)
        question: The question to ask about the image
        use_multishot: If True, include multishot examples in the prompt
        
    Returns:
        Tuple of (mine_name, caption)
    """
    model_name = 'meta/llama-4-maverick-17b-128e-instruct'
    invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
    
    # Generate prompt - pass blob_name for metadata extraction
    if use_multishot:
        prompt, location, filename, country, mine_name = LlamaPromptGenerator_mines(
            blob_name, question, multi_shot_examples=multi_shot_examples
        )
    else:
        prompt, location, filename, country, mine_name = LlamaPromptGenerator_mines(
            blob_name, question, multi_shot_examples=None
        )
    
    # Generate caption - pass PIL image for processing
    caption = LlamaCaptionGenerator(pil_image, system_prompt, prompt, model_name, invoke_url)
    
    return mine_name, caption


def run_comparison():
    """Run multishot vs no-multishot comparison on GCS images."""
    
    print(f"Connecting to GCS bucket: {GCS_BUCKET}")
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    
    # Get list of image files
    blobs = bucket.list_blobs()
    image_extensions = ('.png', '.jpg', '.jpeg')
    image_files = [
        blob.name for blob in blobs 
        if blob.name.lower().endswith(image_extensions)
    ]
    
    # Limit to NUM_IMAGES
    image_files = image_files[:NUM_IMAGES]
    
    print(f"Processing {len(image_files)} images")
    print(f"Questions: {len(questions)}")
    print(f"Total comparisons: {len(image_files) * len(questions)}")
    print("=" * 60)
    
    results = []
    
    for idx, blob_name in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] Loading: {blob_name}")
        
        # Load image once from GCS
        pil_image = load_image_from_gcs(bucket, blob_name)
        
        for q_idx, question in enumerate(questions, 1):
            print(f"  Q{q_idx}: {question[:50]}...")
            
            try:
                # Generate WITH multishot
                print("    → WITH multishot...")
                mine_name, caption_with = generate_caption(
                    pil_image, blob_name, question, use_multishot=True
                )
                
                # Generate WITHOUT multishot
                print("    → WITHOUT multishot...")
                _, caption_without = generate_caption(
                    pil_image, blob_name, question, use_multishot=False
                )
                
                results.append({
                    'mine_name': mine_name,
                    'question': question,
                    'with_multishot': caption_with,
                    'without_multishot': caption_without
                })
                print("    ✓ Done")
                
            except Exception as e:
                print(f"    ✗ Error: {e}")
                results.append({
                    'mine_name': blob_name.split('_')[0],
                    'question': question,
                    'with_multishot': f"ERROR: {e}",
                    'without_multishot': f"ERROR: {e}"
                })
    
    # Write to CSV
    print(f"\n{'=' * 60}")
    print(f"Writing results to: {OUTPUT_CSV}")
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['mine_name', 'question', 'with_multishot', 'without_multishot'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✅ Done! Saved {len(results)} comparisons to {OUTPUT_CSV}")


if __name__ == "__main__":
    run_comparison()
