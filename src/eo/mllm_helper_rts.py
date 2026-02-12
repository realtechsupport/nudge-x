# Helper files to work with MLLM LLama-4
# May 2025

import os
import requests, base64
import time
# for locally installed version of Kosmos-2
# from transformers import AutoProcessor, Kosmos2ForConditionalGeneration
#------------------------------------------------------------------------------------------------------------
def time_it(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} executed in {time.time() - start:.2f}s")
        return result
    return (wrapper)

#------------------------------------------------------------------------------------------------------------
def LlamaCaptionGenerator(image_file_path, SYSTEM_PROMPT, prompt, model_name, invoke_url):
    stream = False # Put stream to False, dont want the model to be a conversational agent for now
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
    "model": model_name,
    "messages":
        [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f""" {prompt} <img src="data:image/png;base64,{image_b64}" />"""}
        ],
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 1.00,
    "stream": stream
    }

    response = requests.post(invoke_url, headers=headers, json=payload)
    return (response.json()["choices"][0]["message"]["content"])

#------------------------------------------------------------------------------------------------------------
def KosmosCaptionGenerator_N (image_file_path, prompt, invoke_url):
    stream = False # Put stream to False, dont want the model to be a conversational agent for now
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
    "messages":
        [
        {"role": "user", "content": f""" {prompt} <img src="data:image/png;base64,{image_b64}" />"""}
        ],
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 1.00,
    "stream": stream
    }

    response = requests.post(invoke_url, headers=headers, json=payload)
    return (response.json()["choices"][0]["message"]["content"])
#------------------------------------------------------------------------------------------------------------
def LlamaPromptGenerator(filename: str, questions: str) -> str:
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

    #questions_block = "\n".join(f"- {q}" for q in questions)
    #questions_block = ''.join(f"- {q}" for q in questions)

    return f"""
You are analyzing a Sentinel-2 satellite image.

Filename: {basename}
Location: {location}
Date: {date}
Image Type: {image_type}

Strict analysis rules:
- Base your interpretation solely on observable environmental conditions in the image.
- Use only the location name from the filename ({location}). Do not refer to or infer any other geography.
- Do not provide generalized or assumed information not directly observable.
- If the image shows results from band operations (e.g., NDVI, FMI), interpret the specific indicators accordingly.

Answer the following, strictly using the image and metadata above:
{questions}
""".strip()

#------------------------------------------------------------------------------------------------------------
def KosmosPromptGenerator(location, common_prompt,specific_prompt=""):
    prompt = f"<Provide a detailed description of the satellite image of {location}. {common_prompt} {specific_prompt}.>"
    return (prompt)
  
 #------------------------------------------------------------------------------------------------------------
# if Kosmos is installed on your local system
def KosmosCaptionGenerator(image_path, model_name, processor, prompt, maxtokens=250):
    inputs = processor(text=prompt, images=image_path, return_tensors="pt")
    generated_ids = model_name.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=inputs["image_embeds_position_mask"],
        use_cache=True,
        max_new_tokens=maxtokens      # Increased the token count to produce big and better responses
        )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    processed_text = processor.post_process_generation(generated_text, cleanup_and_extract=False)

    caption, entities = processor.post_process_generation(generated_text)
    return(caption, entities)
    
 #------------------------------------------------------------------------------------------------------------