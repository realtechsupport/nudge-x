import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration from environment variables
POSTGRES_DB = os.getenv("POSTGRES_DB", "captions_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "Admin1")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "einstein")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

# Image directory from environment variables
IMAGE_DIR = os.getenv("IMAGE_DIR", "images/")

# Qdrant configuration from environment variables
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))


