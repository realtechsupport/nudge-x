# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.gcs_image_handler import GCSImageHandler
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only; restrict in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

BUCKET_PATH = os.environ.get("BUCKET_PATH", "gs://nudge-bucket/")  # override with env if needed

# If your objects are private, set SIGN_URLS=True and ensure GOOGLE_APPLICATION_CREDENTIALS is set
SIGN_URLS = os.environ.get("SIGN_URLS", "true").lower() in ("1", "true", "yes")

# Initialize once
gcs_handler = GCSImageHandler(BUCKET_PATH, max_items=5, sign_urls=SIGN_URLS, url_expiry_seconds=600)

@app.get("/images")
def list_images():
    # return the precomputed listing; refresh by restarting server or create reload API if needed
    return gcs_handler.get_public_urls()
