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

def get_metadata_description(mine_name: str) -> str:
    """Return site description by fuzzy matching mine name."""
    mine_name = mine_name.strip().lower()
    for mines, desc, country, location in zip(metadata_df['Mine name'], metadata_df['metadata'], metadata_df['Country'], metadata_df['Location']):
        if mine_name in mines.lower():
            print("Meta data found for this mine\n")
            return country, location, desc
    print("No meta data found for this mine\n")
    return None, None, None

#------------------------------------------------------------------------------------------------------------
def time_it(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} executed in {time.time() - start:.2f}s")
        return result
    return wrapper

#------------------------------------------------------------------------------------------------------------
def LlamaCaptionGenerator(image_file_path, SYSTEM_PROMPT, prompt, model_name, invoke_url):
    stream = False
    # HERE: Each of the input images (1-3) are from the same unique location and collected at the same time.
    # The first input image is RGB and the subsequent ones, where made available, are bandoperations (NDBI, NDVI, etc).

    image_b64 = compress_image(image_file_path)
    headers = {
        "Authorization": f"Bearer {os.getenv('NVIDIA_API_KEY')}",
        "Accept": "text/event-stream" if stream else "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f""" {prompt} <img src="data:image/png;base64,{image_b64}" />"""}
            # Sai: enable 1, 2 or max 3 input images (for example RGB, NDBI, NDVI)
            # {"role": "user", "content": f""" {prompt} <img src="data:image/png;base64,{image_b64}" />"""}
        ],
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 1.00,
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
    filename = Path(image_file_path).name
    info = os.path.splitext(image_file_path)[0]
    basename = info.split('/')[-1]
    parts = basename.split("_")
    mine_name = parts[0]
    operation = parts[1]
    date = parts[2]
    image_type = f"{operation.upper()} result from the bands of a Sentinel-2 image"
    country, location, site_description = get_metadata_description(mine_name)

    site_description_text = f"Location Metadata:\n{site_description}" if site_description else ""

    prompt =  f"""
    You are analyzing a Sentinel-2 satellite image of a mine.
    Mine Name: {mine_name}
    Date: {date}
    Image Type: {image_type}
    Country: {country}
    Location: {location}
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
    return prompt, location, filename, country, mine_name

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

