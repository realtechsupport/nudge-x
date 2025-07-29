from qdrant_client import QdrantClient, models
import uuid
import numpy as np

from config.database_config import QDRANT_HOST, QDRANT_PORT # Import Qdrant config

def get_qdrant_client():
    """Initializes and returns a Qdrant client."""
    try:
        # Connect to the Qdrant service running on localhost
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        print(f"Qdrant client initialized, connecting to {QDRANT_HOST}:{QDRANT_PORT}.")
        # Optional: Check if client can connect
        client.get_collections()
        print("Successfully connected to Qdrant.")
        return client
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        print("Please ensure Qdrant is running (e.g., via Docker) on the specified host and port.")
        return None

def get_or_create_collection(client, collection_name, vector_size):
    """
    Gets or creates a collection in Qdrant with specified vector size.
    Qdrant requires vector size upon collection creation.
    """
    if client is None:
        return None

    try:
        # Check if collection exists
        collections = client.get_collections().collections
        if any(c.name == collection_name for c in collections):
            print(f"Collection '{collection_name}' already exists.")
            return client.get_collection(collection_name=collection_name)
        else:
            # Create collection if it doesn't exist
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
            print(f"Collection '{collection_name}' created with vector size {vector_size} and Cosine distance.")
            return client.get_collection(collection_name=collection_name)
    except Exception as e:
        print(f"Error getting or creating Qdrant collection '{collection_name}': {e}")
        return None

def add_embeddings_to_vector_db(client, collection_name, image_id, image_embedding, caption_embeddings, captions_text, filename):
    """
    Adds image and caption embeddings to the Qdrant collection.
    Each embedding (image or caption) will be a separate point in Qdrant,
    linked by the original_image_id in their payload.
    Returns a list of UUIDs for the added points.
    """
    if client is None:
        print("Qdrant client not available. Cannot add embeddings.")
        return None

    points_to_upsert = []
    added_point_ids = []

    # Ensure embeddings are lists of floats for Qdrant
    if isinstance(image_embedding, np.ndarray):
        image_embedding = image_embedding.tolist()
    caption_embeddings_list = [emb.tolist() if isinstance(emb, np.ndarray) else emb for emb in caption_embeddings]

    # Add image embedding point
    image_point_id = str(uuid.uuid4())
    points_to_upsert.append(
        models.PointStruct(
            id=image_point_id,
            vector=image_embedding,
            payload={
                "type": "image",
                "original_image_id": image_id,
                "filename": filename
            }
        )
    )
    added_point_ids.append(image_point_id)

    # Add caption embedding points
    for i, caption_emb in enumerate(caption_embeddings_list):
        caption_point_id = str(uuid.uuid4())
        points_to_upsert.append(
            models.PointStruct(
                id=caption_point_id,
                vector=caption_emb,
                payload={
                    "type": "caption",
                    "original_image_id": image_id,
                    "filename": filename,
                    "caption_text": captions_text[i], # Store original caption text
                    "caption_index": i
                }
            )
        )
        added_point_ids.append(caption_point_id)

    try:
        # Upsert points in a batch
        operation_info = client.upsert(
            collection_name=collection_name,
            wait=True, # Wait for operation to complete
            points=points_to_upsert
        )
        print(f"Upsert operation to Qdrant completed: {operation_info.status.name}")
        print(f"Added {len(points_to_upsert)} embeddings to Qdrant collection '{collection_name}' for image ID {image_id}.")
        return added_point_ids # Return list of all UUIDs added for this image/captions
    except Exception as e:
        print(f"Error adding embeddings to Qdrant: {e}")
        return None

def query_vector_db(client, collection_name, query_embedding, n_results=5, filter_type=None):
    """
    Queries the Qdrant collection for similar items based on a query embedding.
    Optionally filters by payload type (e.g., 'image' or 'caption').
    """
    if client is None:
        print("Qdrant client not available. Cannot query.")
        return []

    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.tolist()

    query_filter = None
    if filter_type:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="type",
                    match=models.MatchValue(value=filter_type)
                )
            ]
        )

    try:
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=n_results,
            with_payload=True, # Retrieve metadata
            with_vectors=False # Don't retrieve the vectors themselves, just payload and score
        )
        print(f"Qdrant query returned {len(search_result)} results.")
        return search_result
    except Exception as e:
        print(f"Error querying Qdrant: {e}")
        return []

if __name__ == "__main__":
    client = get_qdrant_client()
    if client:
        COLLECTION_NAME = "test_image_caption_vectors"
        # CLIP outputs 512-dim, SBERT outputs 384-dim. For a single collection, they must match.
        # We'll use 512 as the target size and assume padding is handled externally if needed.
        VECTOR_SIZE = 512

        collection = get_or_create_collection(client, COLLECTION_NAME, VECTOR_SIZE)
        if collection:
            dummy_image_id = 123
            dummy_filename = "test_image.jpg"
            dummy_image_emb = np.random.rand(VECTOR_SIZE).astype(np.float32) # Matches VECTOR_SIZE
            dummy_caption_embs = [
                np.random.rand(VECTOR_SIZE).astype(np.float32), # Matches VECTOR_SIZE
                np.random.rand(VECTOR_SIZE).astype(np.float32)  # Matches VECTOR_SIZE
            ]
            dummy_captions_text = ["A test caption 1", "Another test caption 2"]

            added_ids = add_embeddings_to_vector_db(
                client,
                COLLECTION_NAME,
                dummy_image_id,
                dummy_image_emb,
                dummy_caption_embs,
                dummy_captions_text,
                dummy_filename
            )
            if added_ids:
                print(f"Added dummy embeddings with point IDs: {added_ids}")

                query_vec = np.random.rand(VECTOR_SIZE).astype(np.float32) # Dummy query vector
                print("\nQuerying for similar points (all types):")
                results = query_vector_db(client, COLLECTION_NAME, query_vec, n_results=3)
                for res in results:
                    print(f"  Score: {res.score:.4f}, Payload: {res.payload}")

            # client.delete_collection(collection_name=COLLECTION_NAME)
            # print(f"Collection '{COLLECTION_NAME}' deleted.")