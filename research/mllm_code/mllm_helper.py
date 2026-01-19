# Helper files to work with MLLM LLama-4
# May - Oct 2025

import os
import requests, base64
import time
import sys
from pathlib import Path
from mllm_code.prompts import system_prompt, multi_shot_examples
from mllm_code.exception import mllmException
from dotenv import load_dotenv
from mllm_code.config.database_config import * 
load_dotenv()
import pandas as pd
from PIL import Image
from io import BytesIO

metadata_csv = os.getenv('METADATA_CSV')
metadata_df = pd.read_csv(metadata_csv)

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

def get_metadata_description(mine_name: str) -> tuple:
    """Return site description and GPS coordinates by fuzzy matching mine name.
    
    Returns:
        tuple: (country, location, description, latitude, longitude)
    """
    mine_name = mine_name.strip().lower()
    for idx, row in metadata_df.iterrows():
        if mine_name in row['Mine name'].lower():
            print("Meta data found for this mine\n")
            country = row['Country']
            location = row['Site']
            desc = row['metadata']
            
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
            
            # If not available, try to parse the combined Lat/Long column
            if latitude is None and 'Lat/Long' in row and pd.notna(row.get('Lat/Long')):
                lat_long_str = str(row['Lat/Long']).strip().strip('"\'')
                if ',' in lat_long_str:
                    parts = lat_long_str.split(',')
                    if len(parts) == 2:
                        try:
                            latitude = float(parts[0].strip())
                            longitude = float(parts[1].strip())
                        except ValueError:
                            pass
            
            return country, location, desc, latitude, longitude
    
    print("No meta data found for this mine\n")
    return None, None, None, None, None

#------------------------------------------------------------------------------------------------------------
def time_it(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} executed in {time.time() - start:.2f}s")
        return result
    return wrapper

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
    first_image_label: str = "RGB",
    second_image_label: str = "NDVI",
):
    """
    Generate caption using LLAMA model.
    
    Args:
        image_file_path: File path OR PIL.Image for the first image (typically RGB)
        second_image_file_path_or_image: Optional file path OR PIL.Image for the second image (e.g. NDVI)
        first_image_label: Label for the first image when sending multiple images
        second_image_label: Label for the second image when sending multiple images
        temperature: Controls randomness. Lower = more deterministic, Higher = more creative (0.0-1.0)
        top_p: Nucleus sampling parameter (0.0-1.0)
        max_tokens: Maximum tokens in response
        frequency_penalty: Penalizes repeated tokens (0.0-2.0, higher = less repetition)
    """
    stream = False
    # HERE: Each of the input images (1-3) are from the same unique location and collected at the same time.
    # The first input image is RGB and the subsequent ones, where made available, are bandoperations (NDBI, NDVI, etc).

    image_b64 = compress_image(image_file_path)
    second_image_b64 = None
    if second_image_file_path_or_image is not None:
        second_image_b64 = compress_image(second_image_file_path_or_image)

    if second_image_b64 is None:
        user_content = f""" {prompt} <img src="data:image/png;base64,{image_b64}" />"""
    else:
        # Explicitly label each image so the model knows which is which.
        user_content = (
            f"""{prompt}

Image 1 ({first_image_label}):
<img src="data:image/png;base64,{image_b64}" />

Image 2 ({second_image_label}):
<img src="data:image/png;base64,{second_image_b64}" />
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
            # Sai: enable 1, 2 or max 3 input images (for example RGB, NDBI, NDVI)
            # {"role": "user", "content": f""" {prompt} <img src="data:image/png;base64,{image_b64}" />"""}
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
        raise RuntimeError(f"❌ API request failed: {e}")

    #Validating response from API
    #  ✅ Check HTTP response code
    if response.status_code != 200:
        raise RuntimeError(
            f"❌ Model API returned status {response.status_code}: {response.text}"
        )

    # ✅ Try parsing JSON safely
    try:
        response_json = response.json()
    except ValueError:
        raise RuntimeError(f"❌ Invalid JSON response from API: {response.text}")

    # ✅ Validate structure
    if (
        "choices" not in response_json or
        not response_json["choices"] or
        "message" not in response_json["choices"][0] or
        "content" not in response_json["choices"][0]["message"]
    ):
        raise RuntimeError(f"❌ Unexpected API response format: {response_json}")

    return response_json["choices"][0]["message"]["content"]

#------------------------------------------------------------------------------------------------------------
def KosmosCaptionGenerator_N(image_file_path, prompt, invoke_url):
    stream = False
    try:
        with open(image_file_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()
    except Exception as e:
        print(f"Error processing {image_file_path}: {e}")

    headers = {
        "Authorization": f"Bearer {os.getenv('NVIDIA_API_KEY')}",
        "Accept": "text/event-stream" if stream else "application/json"
    }

    payload = {
        "messages": [
            {"role": "user", "content": f""" {prompt} <img src="data:image/png;base64,{image_b64}" />"""}
        ],
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 1.00,
        "stream": stream
    }

    response = requests.post(invoke_url, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]

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

