import os

import tiktoken
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

from mllm.config.database_config import (
    QDRANT_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
)
from database_pipeline.database_operations import (
    fetch_captions_without_embeddings,
    mark_embeddings_added,
    fetch_stale_embedding_caption_ids,
)
from database_pipeline.vector_db_operations import (
    create_qdrant_client,
    create_qdrant_client_testing,
    create_qdrant_client_api,
    initialize_embedding_model,
    add_captions_to_vector_db,
    delete_points_by_caption_id,
)
from mllm.config import validate_env

encoding = tiktoken.get_encoding("cl100k_base")


def num_tokens(text):
    return len(encoding.encode(text))


def main():
    load_dotenv()
    validate_env()

    qdrant_mode = os.getenv("QDRANT_MODE", "production").lower()

    collection_name = QDRANT_COLLECTION_NAME
    model_name = EMBEDDING_MODEL_NAME

    # Connect to Qdrant
    if qdrant_mode == "testing":
        client = create_qdrant_client_testing()
    elif qdrant_mode == "api":
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        if not qdrant_url or not qdrant_api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set for QDRANT_MODE=api")
        client = create_qdrant_client_api(url=qdrant_url, api_key=qdrant_api_key)
    else:
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        client = create_qdrant_client(host=qdrant_host, port=qdrant_port)

    # Remove stale embeddings for images that have newer accepted captions
    stale_ids = fetch_stale_embedding_caption_ids(limit=500)
    if stale_ids:
        print(f"Found {len(stale_ids)} stale caption embedding(s). Deleting from Qdrant...")
        for caption_id in stale_ids:
            delete_points_by_caption_id(client, collection_name, caption_id)
    else:
        print("No stale caption embeddings found.")

    # Fetch accepted captions that do not yet have embeddings
    pending = fetch_captions_without_embeddings(limit=500)
    if not pending:
        print("No accepted captions found for embedding.")
        return

    print(f"Found {len(pending)} accepted captions without embeddings. Embedding now...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=30,
        length_function=num_tokens,
        is_separator_regex=False,
    )

    chunked_docs = []
    original_caption_ids = []

    for row in pending:
        mine_name = row.get("mine_name") or ""
        country = row.get("country") or ""
        location = row.get("location") or ""
        latitude = row.get("latitude")
        longitude = row.get("longitude")

        # Build metadata prefix for semantic search (will be embedded with caption)
        metadata_prefix = f"Mine: {mine_name}. Country: {country}. Location: {location}."

        chunks = text_splitter.split_text(row["caption"])

        for chunk in chunks:
            enriched_chunk = f"{metadata_prefix} {chunk}"

            chunked_docs.append({
                "caption_id": row["id"],
                "filename": row["filename"],
                "mine_name": mine_name,
                "country": country,
                "location": location,
                "latitude": latitude,
                "longitude": longitude,
                "chunk": enriched_chunk,
                "caption_chunk": chunk,
                "full_caption": row["caption"],
                "is_accepted": row["is_accepted"],
                "is_evaluated": row["is_evaluated"],
                "created_at": str(row["created_at"]) if row.get("created_at") else None,
            })
        original_caption_ids.append(row["id"])

    if not chunked_docs:
        print("No chunks were created from the captions.")
        return

    model, vector_size = initialize_embedding_model(model_name)

    point_ids = add_captions_to_vector_db(client, collection_name, chunked_docs, model, vector_size)

    mark_embeddings_added(original_caption_ids)

    print(f"Embeddings created and stored for {len(point_ids)} chunks.")


if __name__ == "__main__":
    main()
