# Helper files to work with MLLM LLama-4
# May - Oct 2025

import os
import base64
from pathlib import Path
from io import BytesIO

import requests
import pandas as pd
from PIL import Image
from dotenv import load_dotenv

from mllm.prompts import system_prompt, multi_shot_examples

load_dotenv()

metadata_tsv = os.getenv("METADATA_TSV")
if not metadata_tsv:
    raise ValueError("METADATA_TSV environment variable is not set.")

# Read TSV metadata (tab-separated)
metadata_df = pd.read_csv(metadata_tsv, sep="\t")

# Normalize column names so code can work with both old and new schemas.
# Old (v25-v26): Mine name, Site, Country, Lat/Long
# New (v27+):    mine_name, site_location, country, gps_coordinates
rename_map = {}
if "Mine name" in metadata_df.columns:
    rename_map["Mine name"] = "mine_name"
if "Site" in metadata_df.columns:
    rename_map["Site"] = "site_location"
if "Country" in metadata_df.columns:
    rename_map["Country"] = "country"
if "Lat/Long" in metadata_df.columns:
    rename_map["Lat/Long"] = "gps_coordinates"

if rename_map:
    metadata_df = metadata_df.rename(columns=rename_map)

def compress_image(image_path_or_image, max_size=(512,512), quality=70):
    """Compress image from file path or PIL Image object."""
    if isinstance(image_path_or_image, Image.Image):
        img = image_path_or_image.convert("RGB")
    else:
        img = Image.open(image_path_or_image).convert("RGB")
    img.thumbnail(max_size)
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode()


def parse_rgb_image_name(image_path: str) -> tuple:
    """
    Parse RGB image filename to extract mine name and date.
    
    Expected format: minename_rgb_date.png
    Example: AdamsPit_rgb_2024-09-13.png
    
    Returns:
        tuple: (mine_name, date) or (None, None) if parsing fails
    """
    filename = Path(image_path).stem  # Get filename without extension
    parts = filename.split("_")
    
    if len(parts) >= 3 and parts[1].lower() == "rgb":
        mine_name = parts[0]
        date = parts[2]
        return mine_name, date
    return None, None


def find_matching_ndvi_image(rgb_image_path: str, available_images: list) -> str:
    """
    Find matching NDVI image for a given RGB image.
    
    RGB format: minename_rgb_date.png
    NDVI format: minename_ndvi_date.png
    
    Args:
        rgb_image_path: Path to the RGB image
        available_images: List of available image paths (local paths or GCS blob names)
    
    Returns:
        str: Path to matching NDVI image or None if not found
    """
    mine_name, date = parse_rgb_image_name(rgb_image_path)
    if mine_name is None or date is None:
        return None
    
    # Build expected NDVI filename pattern
    expected_ndvi = f"{mine_name}_ndvi_{date}"
    
    expected_lower = expected_ndvi.lower()
    for img_path in available_images:
        if Path(img_path).stem.lower() == expected_lower:
            return img_path
    
    return None


def find_matching_udm_image(rgb_image_path: str, available_images: list) -> str:
    """
    Find matching UDM (Urban Dwelling and Mining) binary classifier image for a given RGB image.
    
    RGB format: minename_rgb_date.png
    UDM format: minename_urban_mining_date_overlay.png
    
    Args:
        rgb_image_path: Path to the RGB image
        available_images: List of available image paths (local paths or GCS blob names)
    
    Returns:
        str: Path to matching UDM overlay image or None if not found
    """
    mine_name, date = parse_rgb_image_name(rgb_image_path)
    if mine_name is None or date is None:
        return None
    
    # Build expected UDM overlay filename pattern
    expected_udm = f"{mine_name}_urban_mining_{date}_overlay"
    
    expected_lower = expected_udm.lower()
    for img_path in available_images:
        if Path(img_path).stem.lower() == expected_lower:
            return img_path
    
    return None


def find_matching_auxiliary_images(rgb_image_path: str, available_images: list) -> dict:
    """
    Find all matching auxiliary images (NDVI, UDM) for a given RGB image.
    
    Args:
        rgb_image_path: Path to the RGB image
        available_images: List of available image paths
    
    Returns:
        dict: {
            'ndvi': path or None,
            'udm': path or None
        }
    """
    return {
        'ndvi': find_matching_ndvi_image(rgb_image_path, available_images),
        'udm': find_matching_udm_image(rgb_image_path, available_images)
    }


def get_metadata_description(mine_name: str, silent: bool = False) -> tuple:
    """Return site description and GPS coordinates by fuzzy matching mine name.
    
    Args:
        mine_name: Mine name to look up (e.g. from parse_rgb_image_name).
        silent: If True, do not print "Metadata found" / "No metadata found".

    Returns:
        tuple: (country, location, description, latitude, longitude)
    """
    mine_name = mine_name.strip().lower()
    for idx, row in metadata_df.iterrows():
        # Handle missing/NaN values robustly
        candidate = str(row.get("mine_name") or "").strip().lower()
        if not candidate or mine_name not in candidate:
            continue
        if not silent:
            print("Metadata found for this mine.")
        country = row.get("country")
        location = row.get("site_location")
        desc = row.get("metadata")
        # Normalize NaN or empty string to None
        if pd.isna(desc) or str(desc).strip() == "":
            desc = None

        # Get GPS coordinates - handle both separate and combined columns
        latitude = None
        longitude = None

        # First, try the separate Latitude/Longitude columns
        if pd.notna(row.get('Latitude')) and pd.notna(row.get('Longitude')):
            try:
                latitude = float(row['Latitude'])
                longitude = float(row['Longitude'])
            except (ValueError, TypeError):
                pass

        # If not available, try to parse the combined GPS column
        if latitude is None and "gps_coordinates" in row and pd.notna(row.get("gps_coordinates")):
            lat_long_str = str(row["gps_coordinates"]).strip().strip('"\'')
            if ',' in lat_long_str:
                parts = lat_long_str.split(',')
                if len(parts) == 2:
                    try:
                        latitude = float(parts[0].strip())
                        longitude = float(parts[1].strip())
                    except ValueError:
                        pass

        return country, location, desc, latitude, longitude
    
    if not silent:
        print("No metadata found for this mine.")
    return None, None, None, None, None


def has_metadata_for_image(image_path: str) -> bool:
    """
    Return True if this RGB image has a matching row in the metadata TSV.

    Uses parse_rgb_image_name to get mine name from filename (minename_rgb_date.png),
    then get_metadata_description to check for metadata. Images without metadata
    should be skipped by the captions pipeline. Uses silent=True to avoid printing
    for each checked image.
    """
    mine_name, _ = parse_rgb_image_name(image_path)
    if mine_name is None:
        return False
    _, _, desc, _, _ = get_metadata_description(mine_name, silent=True)
    return desc is not None


#------------------------------------------------------------------------------------------------------------
def LlamaCaptionGenerator(
    image_file_path,
    SYSTEM_PROMPT,
    prompt,
    model_name,
    invoke_url,
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 512,
    frequency_penalty: float = 0.0,
    second_image_file_path_or_image=None,
    third_image_file_path_or_image=None,
    first_image_label: str = "RGB",
    second_image_label: str = "NDVI",
    third_image_label: str = "UDM",
    quality = 70
):
    """
    Generate caption using LLAMA model.
    
    Args:
        image_file_path: File path OR PIL.Image for the first image (typically RGB)
        second_image_file_path_or_image: Optional file path OR PIL.Image for the second image (e.g. NDVI)
        third_image_file_path_or_image: Optional file path OR PIL.Image for the third image (e.g. UDM binary classifier)
        first_image_label: Label for the first image when sending multiple images
        second_image_label: Label for the second image when sending multiple images
        third_image_label: Label for the third image when sending multiple images (default: UDM)
        temperature: Controls randomness. Lower = more deterministic, Higher = more creative (0.0-1.0)
        top_p: Nucleus sampling parameter (0.0-1.0)
        max_tokens: Maximum tokens in response
        frequency_penalty: Penalizes repeated tokens (0.0-2.0, higher = less repetition)
    """
    stream = False
    # HERE: Each of the input images (1-3) are from the same unique location and collected at the same time.
    # The first input image is RGB and the subsequent ones, where made available, are bandoperations (NDBI, NDVI, etc).

    image_b64 = compress_image(image_file_path, quality = quality)
    second_image_b64 = None
    third_image_b64 = None
    
    if second_image_file_path_or_image is not None:
        second_image_b64 = compress_image(second_image_file_path_or_image, quality = quality)
    
    if third_image_file_path_or_image is not None:
        third_image_b64 = compress_image(third_image_file_path_or_image, quality = quality)

    # Build user content based on how many images are provided
    if second_image_b64 is None and third_image_b64 is None:
        # Only RGB image
        user_content = f""" {prompt} <img src="data:image/png;base64,{image_b64}" />"""
    elif third_image_b64 is None:
        # RGB + second image (e.g., NDVI)
        user_content = (
            f"""{prompt}

Image 1 ({first_image_label}):
<img src="data:image/png;base64,{image_b64}" />

Image 2 ({second_image_label}):
<img src="data:image/png;base64,{second_image_b64}" />
"""
        )
    elif second_image_b64 is None:
        # RGB + third image only (e.g., UDM without NDVI)
        user_content = (
            f"""{prompt}

Image 1 ({first_image_label}):
<img src="data:image/png;base64,{image_b64}" />

Image 2 ({third_image_label}):
<img src="data:image/png;base64,{third_image_b64}" />
"""
        )
    else:
        # All three images: RGB + NDVI + UDM
        user_content = (
            f"""{prompt}

Image 1 ({first_image_label}):
<img src="data:image/png;base64,{image_b64}" />

Image 2 ({second_image_label}):
<img src="data:image/png;base64,{second_image_b64}" />

Image 3 ({third_image_label}):
<img src="data:image/png;base64,{third_image_b64}" />
"""
        )
    headers = {
        "Authorization": f"Bearer {os.getenv('NVIDIA_API_KEY')}",
        "Accept": "text/event-stream" if stream else "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "stream": stream
    }
    try:
        response = requests.post(invoke_url, headers=headers, json=payload)
    except requests.RequestException as e:
        raise RuntimeError(f"API request failed: {e}")

    #Validating response from API
    # Check HTTP response code
    if response.status_code != 200:
        raise RuntimeError(
            f"Model API returned status {response.status_code}: {response.text}"
        )

    # Try parsing JSON safely
    try:
        response_json = response.json()
    except ValueError:
        raise RuntimeError(f"Invalid JSON response from API: {response.text}")

    # Validate structure
    if (
        "choices" not in response_json or
        not response_json["choices"] or
        "message" not in response_json["choices"][0] or
        "content" not in response_json["choices"][0]["message"]
    ):
        raise RuntimeError(f"Unexpected API response format: {response_json}")

    return response_json["choices"][0]["message"]["content"]

#------------------------------------------------------------------------------------------------------------
def LlamaPromptGenerator(image_file_path: str, question: str, multi_shot_examples: str = multi_shot_examples) -> tuple:

    info = os.path.splitext(image_file_path)[0]
    basename = info.split('/')[-1]
    parts = basename.split("_")

    # Check for band index image
    if len(parts) == 3 and parts[1].isalpha() and '-' in parts[2]:
        location, operation, date = parts
        image_type = f"{operation.upper()} result from the bands of a Sentinel-2 image"
    # Check for RGB image
    elif len(parts) == 2 and '-' in parts[1]:
        location, date = parts
        image_type = "Sentinel-2 RGB composite image"
    else:
        raise ValueError(f"Filename format not recognized: {basename}")

    site_description = get_metadata_description(location)
    site_description_text = f"Location Metadata:\n{site_description}" if site_description else ""

    prompt =  f"""
    You are analyzing a Sentinel-2 satellite image.

    Filename: {basename}
    Location: {location}
    Date: {date}
    Image Type: {image_type}
    {site_description_text}

    Answer the following, strictly using the image above:
    {question}
    """.strip()

    if multi_shot_examples:
        prompt = f"""
        {prompt.strip()}

        Below are several example inputs and their captions. 
        Use them as style references when generating your caption for the current image.

        ---
        {multi_shot_examples.strip()}
        ---

        Now generate a caption consistent with the examples above.
        """.strip()
        
    return prompt, location, basename
#------------------------------------------------------------------------------------------------------------
def LlamaPromptGenerator_mines(image_file_path: str, question: str, multi_shot_examples: str = multi_shot_examples) -> tuple:
    """
    Generate prompt for LLaMA model with mine metadata.
    
    Returns:
        tuple: (prompt, location, filename, country, mine_name, latitude, longitude)
    """
    filename = Path(image_file_path).name
    info = os.path.splitext(image_file_path)[0]
    basename = info.split('/')[-1]
    parts = basename.split("_")
    mine_name = parts[0]
    operation = parts[1]
    date = parts[2]
    image_type = f"{operation.upper()} result from the bands of a Sentinel-2 image"
    country, location, site_description, latitude, longitude = get_metadata_description(mine_name)

    # Build structured metadata section
    if site_description:
        metadata_section = f"""
    === SITE METADATA (USE THIS INFORMATION IN YOUR RESPONSE) ===
    {site_description}
    === END METADATA ===
    
    IMPORTANT: Incorporate the above metadata into your analysis. Specifically mention:
    - The minerals/resources extracted at this site
    - How the mining operations relate to the environmental conditions you observe
    """
    else:
        metadata_section = ""

    prompt = f"""
    You are analyzing a Sentinel-2 satellite image of a mine.
    
    Mine Name: {mine_name}
    Date: {date}
    Image Type: {image_type}
    Country: {country}
    Location: {location}
    {metadata_section}
    Using BOTH the image AND the metadata provided above, answer the following:
    {question}
    """.strip()

    if multi_shot_examples:
        prompt = f"""
        {prompt.strip()}

        Below are several example inputs and their captions. 
        Use them as style references when generating your caption for the current image.

        ---
        {multi_shot_examples.strip()}
        ---

        Now generate a caption consistent with the examples above. Remember to mention the specific minerals being extracted.
        """.strip()
    return prompt, location, filename, country, mine_name, latitude, longitude

#------------------------------------------------------------------------------------------------------------
def KosmosPromptGenerator(location, common_prompt, specific_prompt=""):
    site_description = get_metadata_description(location)
    prompt = f"<Provide a detailed description of the satellite image of {location}. {common_prompt} {specific_prompt}. Site context: {site_description}>"
    return prompt

#------------------------------------------------------------------------------------------------------------
def KosmosCaptionGenerator(image_path, model_name, processor, prompt, maxtokens=250):
    inputs = processor(text=prompt, images=image_path, return_tensors="pt")
    generated_ids = model_name.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=inputs["image_embeds_position_mask"],
        use_cache=True,
        max_new_tokens=maxtokens
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    processed_text = processor.post_process_generation(generated_text, cleanup_and_extract=False)

    caption, entities = processor.post_process_generation(generated_text)

    return caption, entities

