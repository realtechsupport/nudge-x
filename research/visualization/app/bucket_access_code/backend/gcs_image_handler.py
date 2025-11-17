# backend/gcs_image_handler.py
from google.cloud import storage
from typing import List, Dict, Any
import os
import time

class GCSImageHandler:
    def __init__(self, images_folder_path: str, max_items: int = 5, sign_urls: bool = True, url_expiry_seconds: int = 300):
        """
        images_folder_path: e.g. "gs://nudge-bucket/path/to/folder/" or "gs://nudge-bucket/"
        sign_urls: if True attempt to generate v4 signed URLs (requires service account creds)
        """
        self.images_folder_path = images_folder_path
        self.max_items = max_items
        self.sign_urls = sign_urls
        self.url_expiry_seconds = url_expiry_seconds

        # parse bucket & prefix
        parts = images_folder_path.replace("gs://", "").split("/")
        self.bucket_name = parts[0]
        self.prefix = "/".join(parts[1:]) if len(parts) > 1 else ""
        self.client = storage.Client()
        self.bucket = self.client.bucket(self.bucket_name)

        # lazy list
        self.image_files = self._list_gcs_images()

    def _list_gcs_images(self) -> List[str]:
        blobs = self.bucket.list_blobs(prefix=self.prefix)
        images = [
            blob.name
            for blob in blobs
            if blob.name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
        ]
        # stable ordering
        images_sorted = sorted(images)
        return images_sorted[: self.max_items]

    def _generate_signed_url(self, blob_name: str) -> str:
        """
        Returns a v4 signed URL if service account credentials available.
        Fallback to public URL format if signing fails.
        """
        blob = self.bucket.blob(blob_name)
        try:
            # requires service-account credentials with signBlob permission
            url = blob.generate_signed_url(
                version="v4",
                expiration=self.url_expiry_seconds,
                method="GET",
            )
            return url
        except Exception as e:
            # fallback — may 403 if object is not public
            print(f"[gcs] signing failed for {blob_name}: {e}, using public url fallback")
            return f"https://storage.googleapis.com/{self.bucket_name}/{blob_name}"

    def get_public_urls(self) -> List[Dict[str, Any]]:
        """
        Return list of {"name": filename, "url": url} for first N images.
        Attempts to sign URLs if configured; falls back to public format.
        """
        results = []
        for blob_name in self.image_files:
            filename = blob_name.split("/")[-1]
            if self.sign_urls:
                url = self._generate_signed_url(blob_name)
            else:
                url = f"https://storage.googleapis.com/{self.bucket_name}/{blob_name}"
            results.append({"name": filename, "full_path": blob_name, "url": url})
        return results
