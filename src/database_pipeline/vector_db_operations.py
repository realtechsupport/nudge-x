from typing import List, Dict, Tuple
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import uuid
import time
import numpy as np
import os

# HTTP timeout (seconds) for Qdrant client requests. Generous because
# `wait=True` upserts block until server-side indexing completes, which can
# exceed the qdrant-client default (~60s) on busy clusters or slow networks.
QDRANT_HTTP_TIMEOUT = int(os.getenv("QDRANT_HTTP_TIMEOUT", "300"))

# -------------------------------
# QDRANT CLIENT INITIALIZATION
# -------------------------------

def create_qdrant_client(host: str = "localhost", port: int = 6333) -> QdrantClient:
    """
    Create a Qdrant client connected to an external service.
    """
    client = QdrantClient(host=host, port=port, timeout=QDRANT_HTTP_TIMEOUT)
    try:
        client.get_collections()  # test connection
        print(f"Connected to external Qdrant at {host}:{port}.")
        return client
    except Exception as e:
        raise ConnectionError(f"Failed to connect to Qdrant at {host}:{port}: {e}")


def create_qdrant_client_testing() -> QdrantClient:
    """
    Create an in-memory Qdrant client for testing purposes.
    """
    client = QdrantClient(":memory:")
    print("Initialized in-memory Qdrant client (testing only).")
    return client


def create_qdrant_client_api(
    url: str,
    api_key: str,
    prefer_grpc: bool = False
) -> QdrantClient:
    """
    Create a Qdrant client using an HTTP/GRPC API endpoint (e.g., Qdrant Cloud).

    Args:
        url: Base URL like "https://YOUR-CLUSTER-URL.qdrant.xyz". If None, reads QDRANT_URL.
        api_key: API key/token. If None, reads QDRANT_API_KEY.
        prefer_grpc: Use gRPC if True; otherwise HTTP.

    Returns:
        Initialized QdrantClient.
    """
    if not url:
        raise ValueError("Qdrant URL is required.")
    if not api_key:
        raise ValueError("Qdrant API key is required.")

    client = QdrantClient(url=url, api_key=api_key, prefer_grpc=prefer_grpc, timeout=QDRANT_HTTP_TIMEOUT)
    try:
        client.get_collections()
        print(f"Connected to Qdrant API at {url} (gRPC={prefer_grpc}).")
        return client
    except Exception as e:
        raise ConnectionError(f"Failed to connect to Qdrant at {url}: {e}")


# -------------------------------
# EMBEDDING MODEL
# -------------------------------

def initialize_embedding_model(model_name: str) -> Tuple[SentenceTransformer, int]:
    """
    Load a sentence-transformer model and return it along with its vector size.
    """
    try:
        model = SentenceTransformer(model_name)
        vector_size = model.get_sentence_embedding_dimension()
        print(f"Sentence-transformer model '{model_name}' loaded. Vector size: {vector_size}")
        return model, vector_size
    except Exception as e:
        raise RuntimeError(f"Error loading model '{model_name}': {e}")


# -------------------------------
# COLLECTION MANAGEMENT
# -------------------------------

def get_or_create_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    distance: models.Distance = models.Distance.COSINE
) -> None:
    """
    Get an existing Qdrant collection or create it if missing.
    Preserves data if the collection already exists.
    """
    if client is None:
        raise ValueError("Qdrant client cannot be None.")

    try:
        collections = client.get_collections().collections
        if any(c.name == collection_name for c in collections):
            print(f"Collection '{collection_name}' already exists. Data preserved.")
        else:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=distance)
            )
            print(f"Collection '{collection_name}' created. Vector size: {vector_size}.")
    except Exception as e:
        raise RuntimeError(f"Error creating or getting collection '{collection_name}': {e}")


# -------------------------------
# ADD CAPTIONS WITH METADATA
# -------------------------------

def add_captions_to_vector_db(
    client: QdrantClient,
    collection_name: str,
    captions: List[Dict],
    model: SentenceTransformer,
    vector_size: int
) -> List[str]:
    """
    Vectorizes captions with rich metadata and adds them to Qdrant.
    
    Each point gets a UUID and stores the full metadata in payload.
    
    Args:
        client: Qdrant client instance
        collection_name: Qdrant collection name
        captions: List of dicts, each must have 'caption' and any metadata
        model: SentenceTransformer embedding model
        vector_size: Dimensionality of embeddings

    Returns:
        List of UUIDs for the added points
    """
    if client is None:
        raise ValueError("Qdrant client cannot be None.")
    
    # Ensure collection exists
    get_or_create_collection(client, collection_name, vector_size)
    
    points_to_upsert = []
    added_point_ids = []

    # Vectorize all captions
    chunks_text = [doc['chunk'] for doc in captions]
    print(f"Encoding {len(chunks_text)} chunks with the embedding model "
          f"(this is the slow step on CPU; progress bar below)...", flush=True)
    vectors = model.encode(
        chunks_text,
        show_progress_bar=True,
        batch_size=32,
    )
    print(f"Encoding done. Upserting {len(chunks_text)} points to Qdrant...", flush=True)

    for i, doc in enumerate(captions):
        point_id = str(uuid.uuid4())
        points_to_upsert.append(
            models.PointStruct(
                id=point_id,
                vector=vectors[i].tolist() if isinstance(vectors[i], np.ndarray) else vectors[i],
                payload=doc
            )
        )
        added_point_ids.append(point_id)

    # Upsert into Qdrant in sub-batches with retries.
    # A single 1000-point upsert with `wait=True` can blow the HTTP timeout on a
    # slow network or a busy cluster — and losing it costs the entire embedding
    # run. Splitting into smaller upserts bounds the blast radius of any one
    # failure and lets us retry without redoing the embedding step.
    upsert_batch_size = int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "100"))
    max_retries = int(os.getenv("QDRANT_UPSERT_RETRIES", "3"))
    total = len(points_to_upsert)

    for start in range(0, total, upsert_batch_size):
        batch = points_to_upsert[start:start + upsert_batch_size]
        attempt = 0
        while True:
            try:
                client.upsert(
                    collection_name=collection_name,
                    points=models.Batch(
                        ids=[p.id for p in batch],
                        vectors=[p.vector for p in batch],
                        payloads=[p.payload for p in batch],
                    ),
                    wait=True,
                )
                print(f"  upserted {start + len(batch)}/{total}", flush=True)
                break
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    print(
                        f"  giving up on upsert batch {start}-{start + len(batch)} "
                        f"after {max_retries} retries: {e}",
                        flush=True,
                    )
                    raise
                backoff = 2 ** attempt
                print(
                    f"  upsert batch {start}-{start + len(batch)} failed "
                    f"(attempt {attempt}/{max_retries}): {e!r}. "
                    f"Retrying in {backoff}s...",
                    flush=True,
                )
                time.sleep(backoff)

    print(f"Added {len(points_to_upsert)} chunks to '{collection_name}'.")
    return added_point_ids



# -------------------------------
# DELETE ALL CHUNKS BY caption_id
# -------------------------------

def delete_points_by_caption_id(
    client: QdrantClient,
    collection_name: str,
    caption_id: int
) -> None:
    """
    Deletes all points (chunks) whose payload contains caption_id == given id.
    """
    if client is None:
        raise ValueError("Qdrant client cannot be None.")

    # Ensure there is an index on the caption_id payload field so that
    # filter-based deletes work correctly on Qdrant Cloud.
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="caption_id",
            field_schema=models.PayloadSchemaType.INTEGER,
        )
    except Exception:
        # If the index already exists or the backend does not require it,
        # we can safely ignore the error and proceed with deletion.
        pass

    filt = models.Filter(
        must=[
            models.FieldCondition(
                key="caption_id",
                match=models.MatchValue(value=caption_id)
            )
        ]
    )

    client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(filter=filt)
    )
    print(f"Deleted all chunks with caption_id={caption_id} from '{collection_name}'.")
