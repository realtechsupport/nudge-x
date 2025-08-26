import uuid
from typing import List, Dict
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import numpy as np

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
    captions_text = [doc['caption'] for doc in captions]
    vectors = model.encode(captions_text)
    
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
    
    print(f"✅ Added {len(points_to_upsert)} captions to '{collection_name}'.")
    return added_point_ids
