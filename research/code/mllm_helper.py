# Helper files to work with MLLM LLama-4
# May 2025

import os
import requests, base64
import time
from prompts.system_prompt import system_prompt
from prompts.multi_shot_examples import multi_shot_examples  # Or whatever the variable is

#------------------------------------------------------------------------------------------------------------
def time_it(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} executed in {time.time() - start:.2f}s")
        return result
    return (wrapper)

#------------------------------------------------------------------------------------------------------------
@time_it
def LlamaCaptionGenerator(image_file_path, SYSTEM_PROMPT, prompt, model_name, invoke_url):
    stream = False # Put stream to False, dont want the model to be a conversational agent for now
    #--------Opens image, base64 encodes it, converting the raw bytes into a text string---------------------
    #--------image_b64 contains the base64 string version of your image----------------
    try:
        with open(image_file_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()
    except Exception as e:
        print(f"Error processing {image_file_path}: {e}")

    #--------Defining API key------------------------------------------
    #--------API key is given followed by the string "Bearer"

    headers = {
    "Authorization": f"Bearer {os.getenv('NVIDIA_API_KEY')}",
     "Accept": "text/event-stream" if stream else "application/json"
    }

    #--------Model initialization with system prompt --------------------------------------
    # Info on sentinel-2 band operations
    # https://pro.arcgis.com/en/pro-app/3.3/help/analysis/raster-functions/band-arithmetic-function.htm
    # https://www.sciencedirect.com/science/article/pii/S0303243421000507

    payload = {
    "model": model_name,
    "messages":
        [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f""" prompt <img src="data:image/png;base64,{image_b64}" />"""}
        ],
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 1.00,
    "stream": stream
    }

    response = requests.post(invoke_url, headers=headers, json=payload)
    return (response.json()["choices"][0]["message"]["content"])
#------------------------------------------------------------------------------------------------------------

def LlamaPromptGenerator(filename: str, question: str, multi_shot_examples: str = "") -> str:
    info = os.path.splitext(filename)[0]
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
        return "Error: Filename format not recognized."


    prompt =  f"""
You are analyzing a Sentinel-2 satellite image.

Filename: {basename}
Location: {location}
Date: {date}
Image Type: {image_type}

Answer the following, strictly using the image above:
{question}
""".strip()

    if multi_shot_examples:
        return f" {prompt} \n Examples are provided below as reference. Use them to guide your analysis.\n\n{multi_shot_examples.strip()}"
    else:
        return prompt
    

    