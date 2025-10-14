import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# --- Sensitive Data ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Model Configs ---
EVALUATION_MODEL_NAME = "gemini-2.5-flash"
TEMPERATURE = 0.1
TOP_P = 0.95
TOP_K = 20

# --- Evaluation Settings ---
DEFAULT_THRESHOLD = 3.5
DEFAULT_WEIGHTS = {
    "Environmental_Impact": 1,
    "Adherence_to_Constraints": 1,
    "Conciseness": 1,
    "Specificity_Terminology": 1,
    "Processes_Patterns": 1,
}

DEEPSEEK_MODEL = "deepseek-chat"