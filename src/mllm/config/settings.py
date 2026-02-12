import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()
mllm_model = "LLAMA"

# --- Sensitive Data ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Prompt Version ---
# Required via PROMPT_VERSION in .env; validate_env() enforces presence.
PROMPT_VERSION = os.getenv("PROMPT_VERSION")

# --- LLAMA Caption Generation Configs ---
LLAMA_MODEL_NAME = "meta/llama-4-maverick-17b-128e-instruct"
LLAMA_INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
LLAMA_TEMPERATURE = 0.7
LLAMA_TOP_P = 1.00
LLAMA_MAX_TOKENS = 512
LLAMA_FREQUENCY_PENALTY = 0.5  # Penalizes repeated words (0.0-2.0, higher = less repetition)

# --- Image Input Configuration ---
# Control which auxiliary images to use alongside RGB
# USE_NDVI: Include NDVI images if available
# USE_UDM: Include UDM (Urban Dwelling and Mining) binary classifier images if available
#
# Combinations:
#   USE_NDVI=False, USE_UDM=False  -> RGB only
#   USE_NDVI=True,  USE_UDM=False  -> RGB + NDVI
#   USE_NDVI=False, USE_UDM=True   -> RGB + UDM
#   USE_NDVI=True,  USE_UDM=True   -> RGB + NDVI + UDM
USE_NDVI = True
USE_UDM = True

# --- Multishot Examples ---

USE_MULTISHOT = True

# --- Evaluation Model Configs ---
EVALUATION_MODEL_NAME = "gemini-2.5-flash"
TEMPERATURE = 0.1
TOP_P = 0.95
TOP_K = 20

# --- Evaluation Settings ---
DEFAULT_THRESHOLD = 3.5
DEFAULT_WEIGHTS = {
    "Environmental_Focus": 1,
    "Adherence_to_Constraints": 1,
    "Conciseness": 1,
    "Specificity_Terminology": 1,
    "Processes_Patterns": 1,
}

DEEPSEEK_MODEL = "deepseek-chat"
