#from mllm_code.database_pipeline.vector_db_operations import create_qdrant_client, delete_points_by_caption_id

#client = create_qdrant_client(host="localhost", port=6333)
#delete_points_by_caption_id(client, "captions_collection", caption_id=)


from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)
client.delete_collection(collection_name="captions_collection")
# Recreate later when you upsert again