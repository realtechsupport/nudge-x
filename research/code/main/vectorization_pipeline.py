import sys
from exception import mllmException
from dotenv import load_dotenv
from database_pipeline.database_operations import fetch_captions_without_embeddings, mark_embeddings_added
from database_pipeline.vector_db_operations import (
	create_qdrant_client,
	create_qdrant_client_testing,
	initialize_embedding_model,
	add_captions_to_vector_db,
)
import os


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
	qdrant_mode = os.getenv("QDRANT_MODE", "testing").lower()

	collection_name = "captions_collection"
	model_name = "sentence-transformers/all-MiniLM-L6-v2"

	# Connect to Qdrant
	if qdrant_mode == "testing":
		client = create_qdrant_client_testing()
	else:
		qdrant_host = os.getenv("QDRANT_HOST", "localhost")
		qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
		client = create_qdrant_client(host=qdrant_host, port=qdrant_port)

	# Fetch accepted captions that do not yet have embeddings
	pending = fetch_captions_without_embeddings(limit=500)
	if not pending:
		print("No accepted captions found for embedding.")
		# Optionally show current collection contents if it exists
		try:
			_print_points(client, collection_name, limit=20)
		except Exception:
			pass
		return

	print(f"Found {len(pending)} accepted captions without embeddings. Embedding now...")

	# Prepare payload docs to embed and store
	docs = [
		{
			"caption_id": row["id"],
			"filename": row["filename"],
			"location": row["location"],
			"caption": row["caption"],
			"is_accepted": row["is_accepted"],
			"is_evaluated": row["is_evaluated"],
			"created_at": str(row["created_at"]) if row.get("created_at") else None,
		}
		for row in pending
	]

	# Initialize model once
	model, vector_size = initialize_embedding_model(model_name)

	# Upsert into Qdrant
	point_ids = add_captions_to_vector_db(client, collection_name, docs, model, vector_size)

	# Record embedding status back to DB
	caption_id_to_point_id = [
		(docs[i]["caption_id"], point_ids[i]) for i in range(len(point_ids))
	]
	mark_embeddings_added(caption_id_to_point_id)

	print(f"Embeddings created and stored for {len(point_ids)} captions.")

	# Immediately list a sample of points (works in testing mode since same process)
	_print_points(client, collection_name, limit=20)


if __name__ == "__main__":
	try:
		main()
	except Exception as e:
		raise mllmException(e, sys)
