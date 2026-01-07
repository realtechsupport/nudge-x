import sys
import tiktoken
from mllm_code.exception import mllmException
from dotenv import load_dotenv
from mllm_code.config.database_config import *
from mllm_code.database_pipeline.database_operations import *
from mllm_code.database_pipeline.vector_db_operations import (
    create_qdrant_client,
    create_qdrant_client_testing,
    create_qdrant_client_api,
    initialize_embedding_model,
    add_captions_to_vector_db,
)
import os
encoding = tiktoken.get_encoding("cl100k_base") 
# 1. Import the text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
def num_tokens(text):
     
    return len(encoding.encode(text))

def _print_points(client, collection_name: str, limit: int = 20):
    res = client.scroll(
        collection_name=collection_name,
        limit=limit,
        with_vectors=True,
        with_payload=True,
    )
    points = res[0] if isinstance(res, tuple) else res
    print(f"\n--- Listing up to {limit} points from '{collection_name}' ---")
    for idx, p in enumerate(points, start=1):
        caption = (p.payload or {}).get("caption")
        vec = p.vector if isinstance(p.vector, list) else []
        preview = vec[:5] if isinstance(vec, list) else []
        print(f"#{idx} id={p.id} dims={len(vec) if isinstance(vec, list) else 'n/a'} first5={preview} caption={(caption[:80] + '...') if caption else None}")
    print("--- End listing ---\n")


def main():
    # Load environment variables (if any)
    load_dotenv()

    # Mode: production (external Qdrant) or testing (in-memory)
    qdrant_mode = os.getenv("QDRANT_MODE", "production").lower()

    collection_name = QDRANT_COLLECTION_NAME
    model_name = EMBEDDING_MODEL_NAME

    # Connect to Qdrant
    if qdrant_mode == "testing":
        client = create_qdrant_client_testing()
    elif qdrant_mode == "api":
        # Read API connection details explicitly
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        if not qdrant_url or not qdrant_api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set for QDRANT_MODE=api")
        client = create_qdrant_client_api(url=qdrant_url, api_key=qdrant_api_key)
    else:
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        client = create_qdrant_client(host=qdrant_host, port=qdrant_port)

    # Fetch accepted captions that do not yet have embeddings
    pending = fetch_captions_without_embeddings(limit=500)
    if not pending:
        print("No accepted captions found for embedding.")
        # Optionally show current collection contents if it exists

        return

    print(f"Found {len(pending)} accepted captions without embeddings. Embedding now...")

  

    # 2. Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,  # Optimal chunk size for the model
        chunk_overlap=30,  # Maintain context between chunks
        length_function=num_tokens,
        is_separator_regex=False,
    )

    # Prepare payload docs to embed and store
    # 3. Create a list to hold the chunked documents
    chunked_docs = []
    original_caption_ids = []

    for row in pending:
        # Extract metadata fields
        mine_name = row.get("mine_name") or ""
        country = row.get("country") or ""
        location = row.get("location") or ""
        latitude = row.get("latitude")
        longitude = row.get("longitude")
        
        # Build metadata prefix for semantic search (will be embedded with caption)
        metadata_prefix = f"Mine: {mine_name}. Country: {country}. Location: {location}."
        
        # 4. Split the caption into chunks
        chunks = text_splitter.split_text(row["caption"])

        for chunk in chunks:
            # Combine metadata + chunk for embedding (enables semantic search on metadata)
            enriched_chunk = f"{metadata_prefix} {chunk}"
            
            chunked_docs.append({
                "caption_id": row["id"],
                "filename": row["filename"],
                "mine_name": mine_name,
                "country": country,
                "location": location,
                "latitude": latitude,
                "longitude": longitude,
                "chunk": enriched_chunk,  # Enriched chunk with metadata for embedding
                "caption_chunk": chunk,   # Original chunk without metadata prefix
                "full_caption": row["caption"],
                "is_accepted": row["is_accepted"],
                "is_evaluated": row["is_evaluated"],
                "created_at": str(row["created_at"]) if row.get("created_at") else None,
            })
        original_caption_ids.append(row["id"])

    # 5. Check if there are any chunks to process
    if not chunked_docs:
        print("No chunks were created from the captions.")
        return

    # Initialize model once
    model, vector_size = initialize_embedding_model(model_name)

    # Upsert into Qdrant
    # 6. Pass the chunked documents to the add function
    point_ids = add_captions_to_vector_db(client, collection_name, chunked_docs, model, vector_size)

    # Record embedding status back to DB using original caption IDs only
    mark_embeddings_added(original_caption_ids)=-09

    print(f"Embeddings created and stored for {len(point_ids)} chunks.")

    # Immediately list a sample of points (works in testing mode since same process)
    #_print_points(client, collection_name, limit=20)


if __name__ == "__main__":
    main()
