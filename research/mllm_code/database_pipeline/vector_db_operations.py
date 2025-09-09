from typing import List, Dict, Tuple
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import uuid
import numpy as np
import os

# -------------------------------
# QDRANT CLIENT INITIALIZATION
# -------------------------------

def create_qdrant_client(host: str = "localhost", port: int = 6333) -> QdrantClient:
    """
    Create a Qdrant client connected to an external service.
    """
    client = QdrantClient(host=host, port=port)
    try:
        client.get_collections()  # test connection
        print(f"✅ Connected to external Qdrant at {host}:{port}.")
        return client
    except Exception as e:
        raise ConnectionError(f"❌ Failed to connect to Qdrant at {host}:{port}: {e}")


def create_qdrant_client_testing() -> QdrantClient:
    """
    Create an in-memory Qdrant client for testing purposes.
    """
    client = QdrantClient(":memory:")
    print("✅ Initialized in-memory Qdrant client (testing only).")
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

    client = QdrantClient(url=url, api_key=api_key, prefer_grpc=prefer_grpc)
    try:
        client.get_collections()
        print(f"✅ Connected to Qdrant API at {url} (gRPC={prefer_grpc}).")
        return client
    except Exception as e:
        raise ConnectionError(f"❌ Failed to connect to Qdrant at {url}: {e}")


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
        print(f"✅ Sentence-transformer model '{model_name}' loaded. Vector size: {vector_size}")
        return model, vector_size
    except Exception as e:
        raise RuntimeError(f"❌ Error loading model '{model_name}': {e}")


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
            print(f"✅ Collection '{collection_name}' already exists. Data preserved.")
        else:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=distance)
            )
            print(f"✅ Collection '{collection_name}' created. Vector size: {vector_size}.")
    except Exception as e:
        raise RuntimeError(f"❌ Error creating or getting collection '{collection_name}': {e}")


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
    vectors = model.encode(chunks_text)
    
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
    
    # Upsert into Qdrant
    client.upsert(
        collection_name=collection_name,
        points=models.Batch(
            ids=[p.id for p in points_to_upsert],
            vectors=[p.vector for p in points_to_upsert],
            payloads=[p.payload for p in points_to_upsert]
        ),
        wait=True
    )
    
    print(f"✅ Added {len(points_to_upsert)} chunks to '{collection_name}'.")
    return added_point_ids


# -------------------------------
# FULL CAPTIONS-ONLY WORKFLOW
# -------------------------------

def build_vector_store_from_captions(
    client: QdrantClient,
    collection_name: str,
    captions: List[Dict],
    model_name: str
):
    """
    Full workflow: initialize embedding model, ensure collection exists,
    vectorize captions, and add to Qdrant.
    """
    # Initialize model
    model, vector_size = initialize_embedding_model(model_name)

    # Vectorize and upsert captions
    add_captions_to_vector_db(client, collection_name, captions, model, vector_size)

    print(f"✅ Vector store '{collection_name}' updated with {len(captions)} captions.")


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
    print(f"🗑️ Deleted all chunks with caption_id={caption_id} from '{collection_name}'.")
