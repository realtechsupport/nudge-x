"""
Delete/clear the captions collection in Qdrant and reset PostgreSQL embedding tracking.
Run this before re-embedding to start fresh.

Usage:
    python -m database_pipeline.delete_captions
"""

import os
from dotenv import load_dotenv
from mllm.config.database_config import QDRANT_COLLECTION_NAME, QDRANT_URL, QDRANT_API_KEY
from database_pipeline.vector_db_operations import (
    create_qdrant_client,
    create_qdrant_client_api,
)
from database_pipeline.database_operations import connect_db

load_dotenv()


def reset_embedding_tracking():
    """Reset the caption_embeddings table in PostgreSQL so captions can be re-embedded."""
    conn = connect_db()
    if conn is None:
        print("Error: Could not connect to PostgreSQL.")
        return False
    
    try:
        cursor = conn.cursor()
        # Delete all rows from caption_embeddings to mark all captions as needing embedding
        cursor.execute("DELETE FROM caption_embeddings;")
        deleted_count = cursor.rowcount
        conn.commit()
        print(f"Reset PostgreSQL: Cleared {deleted_count} rows from caption_embeddings table.")
        return True
    except Exception as e:
        print(f"Error resetting caption_embeddings: {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()


def delete_collection():
    """Delete the captions collection based on current QDRANT_MODE."""
    qdrant_mode = os.getenv("QDRANT_MODE", "production").lower()
    
    print(f"Deleting Qdrant collection '{QDRANT_COLLECTION_NAME}' (mode: {qdrant_mode})...")
    
    if qdrant_mode == "api":
        client = create_qdrant_client_api(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    else:
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        client = create_qdrant_client(host=qdrant_host, port=qdrant_port)
    
    # Check if collection exists
    collections = client.get_collections().collections
    if any(c.name == QDRANT_COLLECTION_NAME for c in collections):
        client.delete_collection(collection_name=QDRANT_COLLECTION_NAME)
        print(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' deleted successfully.")
    else:
        print(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' does not exist.")
    
    # Also reset PostgreSQL tracking
    print("\nResetting PostgreSQL embedding tracking...")
    reset_embedding_tracking()
    
    print("\nNext step: Re-run vectorization to re-embed with metadata:")
    print("   python -m mllm.main.vectorization_pipeline")


if __name__ == "__main__":
    delete_collection()
