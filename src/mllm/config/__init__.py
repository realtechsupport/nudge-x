"""
Configuration package for the MLLM pipeline.
"""

import os
import sys


# Required environment variables for the pipeline to function.
# Each entry is (var_name, description).
REQUIRED_ENV_VARS = [
    ("POSTGRES_DB", "PostgreSQL database name"),
    ("POSTGRES_USER", "PostgreSQL username"),
    ("POSTGRES_PASSWORD", "PostgreSQL password"),
    ("POSTGRES_HOST", "PostgreSQL host"),
    ("POSTGRES_PORT", "PostgreSQL port"),
    ("QDRANT_HOST", "Qdrant server host"),
    ("QDRANT_PORT", "Qdrant server port"),
    ("QDRANT_MODE", "Qdrant mode (production/api/testing)"),
    ("METADATA_TSV", "Path to the mines metadata TSV file"),
    ("IMAGE_DIR", "Path to satellite images (local or gs://)"),
    ("NVIDIA_API_KEY", "NVIDIA API key for LLaMA model"),
    ("GOOGLE_API_KEY", "Google API key for Gemini evaluation"),
    ("PROMPT_VERSION", "Active prompt version (e.g. v4, v5, v6, v7)"),
]


def validate_env() -> None:
    """
    Check that all required environment variables are set.
    Prints a clear summary of any missing variables and exits if any are absent.
    
    Call this at the start of each pipeline entry point (captions, vectorization, RAG).
    """
    missing = []
    for var_name, description in REQUIRED_ENV_VARS:
        if not os.getenv(var_name):
            missing.append((var_name, description))

    if missing:
        print("\n" + "=" * 60)
        print("ERROR: Missing required environment variables.")
        print("Please set them in your .env file or shell environment.\n")
        for var_name, description in missing:
            print(f"  - {var_name:25s} ({description})")
        print("\n" + "=" * 60)
        sys.exit(1)
