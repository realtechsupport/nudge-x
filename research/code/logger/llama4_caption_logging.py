import os
import json
from datetime import datetime

# Create timestamped path once per run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = "/content/drive/MyDrive/Colab/research/logs/"
os.makedirs(log_dir, exist_ok=True)
json_path = os.path.join(log_dir, f"llama4_captions_{timestamp}.json")

# Holds all caption data
caption_data = []

def log_caption(image: str, question: str, prompt: str, caption: str):
    entry = {
        "image": image,
        "question": question,
        "prompt": prompt,
        "caption": caption
    }
    caption_data.append(entry)

def save_json():
    with open(json_path, "w") as f:
        json.dump(caption_data, f, indent=2)
    print(f"✅ Captions saved to: {json_path}")
