import os
import uuid
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Tuple

from dotenv import load_dotenv
from google.cloud import storage
from PIL import Image
from qdrant_client import models
from sentence_transformers import SentenceTransformer

from mllm_code.config.database_config import (
    IMAGE_COLLECTION_NAME,
    IMAGE_DIR,
    IMAGE_EMBEDDING_MODEL_NAME,
)
from mllm_code.database_pipeline.database_operations import (
    fetch_images_without_embeddings,
    mark_image_embeddings_added,
)
from mllm_code.database_pipeline.vector_db_operations import (
    create_qdrant_client,
    create_qdrant_client_api,
    create_qdrant_client_testing,
    get_or_create_collection,
)


def _parse_gcs_uri(uri: str) -> Tuple[str, str]:
    """Split a gs:// URI into (bucket, object prefix)."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {uri}")
    path = uri[5:]
    parts = path.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix


def _ensure_gcs_bucket(source: Dict[str, object], bucket_name: str):
    """Ensure a cached storage.Bucket instance for the requested bucket."""
    client = source.get("gcs_client")
    if client is None:
        client = storage.Client()
        source["gcs_client"] = client
    return client.bucket(bucket_name)


def _try_load_local_image(local_root: Path, filename: str) -> Tuple[Optional[Image.Image], Optional[str]]:
    """Try to load a local image, attempting common extensions if exact match not found."""
    IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp', '.gif', '.tif', '.tiff']
    
    candidate = (local_root / filename).resolve()
    
    # Try exact filename first
    if candidate.exists():
        try:
            with Image.open(candidate) as raw_image:
                pil_image = raw_image.convert("RGB")
            return pil_image, str(candidate)
        except Exception as exc:
            print(f"⚠️  Could not open local image '{candidate}': {exc}")
    
    # If no extension in filename, try common extensions
    if not any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
        for ext in IMAGE_EXTENSIONS:
            candidate_with_ext = (local_root / f"{filename}{ext}").resolve()
            if candidate_with_ext.exists():
                try:
                    with Image.open(candidate_with_ext) as raw_image:
                        pil_image = raw_image.convert("RGB")
                    return pil_image, str(candidate_with_ext)
                except Exception as exc:
                    print(f"⚠️  Could not open local image '{candidate_with_ext}': {exc}")
    
    return None, None


def _load_image_asset(
    filename: str,
    source: Dict[str, object],
) -> Tuple[Optional[Image.Image], Optional[str]]:
    """Load an image from local disk or GCS, returning (PIL.Image, canonical reference)."""
    local_root: Optional[Path] = source.get("local_root")  # type: ignore[assignment]
    if local_root:
        pil_image, reference = _try_load_local_image(local_root, filename)
        if pil_image is not None:
            return pil_image, reference

    gcs_bucket = source.get("gcs_bucket")
    gcs_prefix = source.get("gcs_prefix", "")

    bucket = None
    blob_name = None

    if filename.startswith("gs://"):
        bucket_name, blob_name = _parse_gcs_uri(filename)
        bucket = _ensure_gcs_bucket(source, bucket_name)
    elif gcs_bucket:
        bucket = gcs_bucket
        blob_name = "/".join(part.strip("/") for part in [gcs_prefix, filename] if part)

    if not bucket or not blob_name:
        return None, None

    IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp', '.gif', '.tif', '.tiff']
    
    # Try exact blob name first, then with extensions if no extension present
    blob_names_to_try = [blob_name]
    if not any(blob_name.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
        blob_names_to_try.extend([f"{blob_name}{ext}" for ext in IMAGE_EXTENSIONS])
    
    for try_blob_name in blob_names_to_try:
        try:
            blob = bucket.blob(try_blob_name)  # type: ignore[attr-defined]
            if not blob.exists():
                continue
            image_bytes = blob.download_as_bytes()
            with Image.open(BytesIO(image_bytes)) as raw_image:
                pil_image = raw_image.convert("RGB")
            reference = f"gs://{bucket.name}/{try_blob_name.lstrip('/')}"  # type: ignore[attr-defined]
            return pil_image, reference
        except Exception:  # noqa: BLE001
            continue
    
    print(f"⚠️  Could not find image '{blob_name}' in bucket '{bucket.name}'")  # type: ignore[attr-defined]
    return None, None


def _build_public_url(reference: Optional[str], base_url: Optional[str]) -> Optional[str]:
    """Translate a gs:// reference into a public URL when base_url is provided."""
    if not reference:
        return None
    if base_url and reference.startswith("gs://"):
        _, path = _parse_gcs_uri(reference)
        return f"{base_url.rstrip('/')}/{path}"
    return reference


def _initialize_image_source(image_dir: Optional[str]) -> Dict[str, object]:
    """Prepare helpers for retrieving images from local disk or GCS."""
    source: Dict[str, object] = {
        "local_root": None,
        "gcs_bucket": None,
        "gcs_prefix": "",
        "gcs_client": None,
    }

    if not image_dir:
        return source

    if image_dir.startswith("gs://"):
        try:
            bucket_name, prefix = _parse_gcs_uri(image_dir)
            source["gcs_prefix"] = prefix
            source["gcs_bucket"] = _ensure_gcs_bucket(source, bucket_name)
            print(f"Using GCS bucket '{bucket_name}' with prefix '{prefix}'.")
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  Could not initialise GCS bucket from IMAGE_DIR '{image_dir}': {exc}")
    else:
        local_root = Path(image_dir)
        if local_root.exists():
            source["local_root"] = local_root
            print(f"Using local image directory '{local_root}'.")
        else:
            print(f"⚠️  IMAGE_DIR '{image_dir}' does not exist; expecting fully qualified filenames.")

    return source


def _get_clip_embedding_dimension(model: SentenceTransformer) -> int:
    """
    Get the embedding dimension for a CLIP model.
    CLIP models don't support get_sentence_embedding_dimension() properly.
    """
    # First try the standard method
    vector_size = model.get_sentence_embedding_dimension()
    if vector_size is not None:
        return vector_size
    
    # For CLIP models, encode a dummy image to get the dimension
    try:
        dummy_image = Image.new('RGB', (224, 224), color='white')
        dummy_embedding = model.encode([dummy_image], convert_to_numpy=True)[0]
        vector_size = len(dummy_embedding)
        dummy_image.close()
        print(f"Detected CLIP embedding dimension: {vector_size}")
        return vector_size
    except Exception as e:
        # Default to 512 for clip-ViT-B-32
        print(f"⚠️  Could not detect embedding dimension, using default 512: {e}")
        return 512


def main():
    load_dotenv()

    qdrant_mode = os.getenv("QDRANT_MODE", "production").lower()
    image_dir = os.getenv("IMAGE_DIR", IMAGE_DIR)
    public_base_url = os.getenv("IMAGE_PUBLIC_BASE_URL") or os.getenv("GCS_BASE_URL")
    image_model_name = os.getenv("IMAGE_EMBEDDING_MODEL_NAME", IMAGE_EMBEDDING_MODEL_NAME)
    fetch_limit = int(os.getenv("IMAGE_EMBEDDING_BATCH_LIMIT", "100"))

    if qdrant_mode == "testing":
        client = create_qdrant_client_testing()
    elif qdrant_mode == "api":
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        if not qdrant_url or not qdrant_api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set for QDRANT_MODE=api.")
        client = create_qdrant_client_api(url=qdrant_url, api_key=qdrant_api_key)
    else:
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        client = create_qdrant_client(host=qdrant_host, port=qdrant_port)

    pending = fetch_images_without_embeddings(limit=fetch_limit)
    if not pending:
        print("No images pending embeddings.")
        return

    try:
        model = SentenceTransformer(image_model_name)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load image embedding model '{image_model_name}': {exc}") from exc

    # Get embedding dimension (handles CLIP models that don't support get_sentence_embedding_dimension)
    vector_size = _get_clip_embedding_dimension(model)
    get_or_create_collection(client, IMAGE_COLLECTION_NAME, vector_size)

    source = _initialize_image_source(image_dir)

    batch_ids = []
    batch_vectors = []
    batch_payloads = []
    processed_filenames = []

    for row in pending:
        filename = row.get("filename")
        if not filename:
            continue

        pil_image, reference = _load_image_asset(filename, source)
        if pil_image is None:
            print(f"⚠️  Skipping '{filename}' because it could not be loaded.")
            continue

        try:
            vector = model.encode([pil_image], convert_to_numpy=True)[0].tolist()
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  Failed to encode image '{filename}': {exc}")
            continue
        finally:
            pil_image.close()

        canonical_ref = reference or filename
        public_url = _build_public_url(canonical_ref, public_base_url)

        # Generate deterministic UUID from filename (Qdrant requires UUID or int for point IDs)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, filename))
        batch_ids.append(point_id)
        batch_vectors.append(vector)
        batch_payloads.append(
            {
                "filename": filename,
                "image_path": canonical_ref,
                "image_url": public_url,
                "location": row.get("location"),
                "caption_id": row.get("caption_id"),
            },
        )
        processed_filenames.append(filename)

    if not batch_ids:
        print("No image embeddings were generated.")
        return

    client.upsert(
        collection_name=IMAGE_COLLECTION_NAME,
        points=models.Batch(
            ids=batch_ids,
            vectors=batch_vectors,
            payloads=batch_payloads,
        ),
        wait=True,
    )

    mark_image_embeddings_added(processed_filenames)
    print(f"✅ Added {len(batch_ids)} image embeddings to '{IMAGE_COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
