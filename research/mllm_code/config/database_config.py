import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration from environment variables
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")

# Image directory from environment variables
IMAGE_DIR = os.getenv("IMAGE_DIR")

IMAGE_COLLECTION_NAME =  "image_collection"
IMAGE_EMBEDDING_MODEL_NAME =  "sentence-transformers/clip-ViT-B-32"

# Qdrant configuration from environment variables
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = int(os.getenv("QDRANT_PORT"))
QDRANT_COLLECTION_NAME = 'captions_collection'

#qdrant mode
QDRANT_MODE = os.getenv("QDRANT_MODE").lower()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


#model names
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
