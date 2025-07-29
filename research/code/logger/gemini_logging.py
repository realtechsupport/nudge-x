import os
import json
from datetime import datetime

# Create log path
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = "/content/drive/MyDrive/Colab/research/logs/"
os.makedirs(log_dir, exist_ok=True)
json_path = os.path.join(log_dir, f"gemini_eval_{timestamp}.json")

# Store evaluations
eval_data = []

def log_gemini_eval(caption: str, eval_result: dict):
    entry = {
        "caption": caption,
        "evaluation": eval_result
    }
    eval_data.append(entry)

def save_gemini_eval():
    with open(json_path, "w") as f:
        json.dump(eval_data, f, indent=2)
    print(f"✅ Gemini evaluations saved to: {json_path}")
