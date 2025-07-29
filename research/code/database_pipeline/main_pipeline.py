
import os
from PIL import Image
import io
import numpy as np

# Import functions from your custom scripts
from db_operations import (
    create_tables_if_not_exists,
    save_image_and_captions,
    get_image_and_captions,
    save_embedding_metadata
)
from embedding_models import (
    load_clip_model,
    get_image_embedding_clip,
    load_sentence_transformer_model,
    get_text_embedding_sbert
)
from vector_db_operations import (
    get_qdrant_client,
    get_or_create_collection,
    add_embeddings_to_vector_db,
    query_vector_db
)
from config.database_config import IMAGE_DIR

# Define the common vector size for the Qdrant collection
# CLIP's base model output is 512. SentenceTransformer 'all-MiniLM-L6-v2' is 384.
# For a single collection in Qdrant, all vectors must have the same dimension.
# We will pad SBERT embeddings to match CLIP's 512-dimension output.
QDRANT_COLLECTION_VECTOR_SIZE = 512

def run_forward_pass(image_path, user_provided_captions):
    """
    Executes a single forward pass:
    1. Saves image and captions to traditional DB.
    2. Generates image and text embeddings.
    3. Stores embeddings in a vector DB (Qdrant) and links them in traditional DB.
    """
    print(f"\n--- Starting Forward Pass for Image: {image_path} ---")

    # Ensure tables exist
    create_tables_if_not_exists()

    # --- Step 1: Save image and captions to Traditional DB ---
    print("\n[Step 1/5] Saving image and captions to traditional DB...")
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return
    except Exception as e:
        print(f"Error reading image file: {e}")
        return

    generated_captions = user_provided_captions

    image_id = save_image_and_captions(image_path, generated_captions)
    if image_id is None:
        print("Failed to save image and captions to traditional DB. Aborting.")
        return

    # --- Step 2: Load Embedding Models ---
    print("\n[Step 2/5] Loading embedding models...")
    clip_processor, clip_model = load_clip_model()
    sbert_model = load_sentence_transformer_model()

    if not clip_processor or not clip_model or not sbert_model:
        print("Failed to load one or more embedding models. Aborting.")
        return

    # --- Step 3: Generate Embeddings ---
    print("\n[Step 3/5] Generating image and caption embeddings...")
    image_embedding = get_image_embedding_clip(image_bytes, clip_processor, clip_model)
    if image_embedding is None:
        print("Failed to generate image embedding. Aborting.")
        return
    print(f"Image embedding generated. Shape: {image_embedding.shape}")

    caption_embeddings = get_text_embedding_sbert(generated_captions, sbert_model)
    if not generated_captions: # Handle case where MLLM generates no captions
        caption_embeddings = np.array([]) # Ensure it's an empty numpy array for consistency
    elif caption_embeddings is None:
        print("Failed to generate caption embeddings. Aborting.")
        return

    if caption_embeddings.size > 0:
        print(f"Caption embeddings generated. Shape: {caption_embeddings.shape}")
    else:
        print("No captions generated or available for embedding.")


    # IMPORTANT: Pad/Truncate embeddings to match QDRANT_COLLECTION_VECTOR_SIZE
    # All vectors in a Qdrant collection must have the same dimension.
    # CLIP is 512, SBERT is 384. We'll pad SBERT embeddings to 512.

    # Pad/Truncate image embedding if its size differs (should be 512 from CLIP)
    if image_embedding.shape[0] != QDRANT_COLLECTION_VECTOR_SIZE:
        if image_embedding.shape[0] < QDRANT_COLLECTION_VECTOR_SIZE:
            image_embedding = np.pad(image_embedding, (0, QDRANT_COLLECTION_VECTOR_SIZE - image_embedding.shape[0]), 'constant')
        else: # Truncate if larger (less ideal, but for consistency)
            image_embedding = image_embedding[:QDRANT_COLLECTION_VECTOR_SIZE]
        print(f"Image embedding adjusted to {QDRANT_COLLECTION_VECTOR_SIZE} dimensions.")

    padded_caption_embeddings = []
    if caption_embeddings.size > 0:
        for emb in caption_embeddings:
            if emb.shape[0] < QDRANT_COLLECTION_VECTOR_SIZE:
                padded_emb = np.pad(emb, (0, QDRANT_COLLECTION_VECTOR_SIZE - emb.shape[0]), 'constant')
                padded_caption_embeddings.append(padded_emb)
            elif emb.shape[0] > QDRANT_COLLECTION_VECTOR_SIZE:
                padded_caption_embeddings.append(emb[:QDRANT_COLLECTION_VECTOR_SIZE])
            else:
                padded_caption_embeddings.append(emb)
        print(f"All caption embeddings adjusted to {QDRANT_COLLECTION_VECTOR_SIZE} dimensions.")
    else:
        print("No captions to pad.")


    # --- Step 4: Save Embeddings to Vector DB (Qdrant) ---
    print("\n[Step 4/5] Connecting to Qdrant and storing embeddings...")
    qdrant_client = get_qdrant_client()
    if qdrant_client is None:
        print("Qdrant client not available. Aborting.")
        return

    QDRANT_COLLECTION_NAME = "image_caption_vectors_qdrant" # Consistent collection name
    vector_collection = get_or_create_collection(qdrant_client, QDRANT_COLLECTION_NAME, QDRANT_COLLECTION_VECTOR_SIZE)
    if vector_collection is None:
        print("Failed to get or create Qdrant collection. Aborting.")
        return

    filename = os.path.basename(image_path)
    vector_db_point_ids = add_embeddings_to_vector_db(
        qdrant_client,
        QDRANT_COLLECTION_NAME,
        image_id,
        image_embedding,
        padded_caption_embeddings, # Use padded embeddings
        generated_captions,
        filename
    )
    if vector_db_point_ids is None:
        print("Failed to add embeddings to vector DB. Aborting.")
        return

    # --- Step 5: Save Embedding Metadata to Traditional DB ---
    print("\n[Step 5/5] Saving embedding metadata to traditional DB...")
    image_emb_dim = image_embedding.shape[0]
    caption_emb_dim = padded_caption_embeddings[0].shape[0] if padded_caption_embeddings else 0
    # Store the first point ID from Qdrant as the primary link for this image
    primary_vector_db_entry_id = vector_db_point_ids[0] if vector_db_point_ids else None

    metadata_id = save_embedding_metadata(image_id, image_emb_dim, caption_emb_dim, primary_vector_db_entry_id)
    if metadata_id is None:
        print("Failed to save embedding metadata to traditional DB. Process complete with partial success.")
        return

    print("\n--- Forward Pass Completed Successfully! ---")
    print(f"Image ID in traditional DB: {image_id}")
    print(f"Vector DB Point IDs added: {vector_db_point_ids}")

if __name__ == "__main__":
    # --- Create a dummy image for testing ---
    os.makedirs(IMAGE_DIR, exist_ok=True)
    sample_image_path = os.path.join(IMAGE_DIR, "example_image.jpg")
    try:
        Image.new('RGB', (224, 224), color = 'white').save(sample_image_path)
        print(f"Created dummy image at: {sample_image_path}")
    except Exception as e:
        print(f"Could not create dummy image: {e}. Please place a 'example_image.jpg' in the '{IMAGE_DIR}' directory manually.")
        sample_image_path = None

    if sample_image_path and os.path.exists(sample_image_path):
        user_input_image = sample_image_path
        mllm_generated_captions = [
            "A white square on a plain background.",
            "This is a simple geometric shape.",
            "An image demonstrating a basic color."
        ]

        # --- Run the main pipeline ---
        run_forward_pass(user_input_image, mllm_generated_captions)

        # --- Example of querying Qdrant directly after a successful run ---
        print("\n--- Testing Qdrant Query ---")
        qdrant_client_for_query = get_qdrant_client()
        if qdrant_client_for_query:
            sbert_model_for_query = load_sentence_transformer_model()
            if sbert_model_for_query:
                query_text = "a simple white object"
                query_embedding = get_text_embedding_sbert([query_text], sbert_model_for_query)[0]

                # Pad query embedding to match collection size
                if query_embedding.shape[0] < QDRANT_COLLECTION_VECTOR_SIZE:
                    query_embedding = np.pad(query_embedding, (0, QDRANT_COLLECTION_VECTOR_SIZE - query_embedding.shape[0]), 'constant')
                elif query_embedding.shape[0] > QDRANT_COLLECTION_VECTOR_SIZE:
                    query_embedding = query_embedding[:QDRANT_COLLECTION_VECTOR_SIZE]

                print(f"\nQuerying Qdrant for '{query_text}'...")
                results = query_vector_db(qdrant_client_for_query, QDRANT_COLLECTION_NAME, query_embedding, n_results=5)

                if results:
                    print("Qdrant Query Results:")
                    for res in results:
                        print(f"  Score: {res.score:.4f}, Type: {res.payload.get('type')}, Filename: {res.payload.get('filename')}, Original Image ID: {res.payload.get('original_image_id')}")
                        if res.payload.get('type') == 'caption':
                            print(f"    Caption Text: '{res.payload.get('caption_text')}'")
                else:
                    print("No results found from Qdrant query.")
            else:
                print("Could not load SBERT model for query.")
        else:
            print("Qdrant client not available for query.")
    else:
        print("\nSkipping main pipeline as no sample image is available or could be created.")

    print("\n--- Project Execution Complete ---")
    print("Remember to clean up dummy image: 'images/example_image.jpg' if created.")
    print("To stop Qdrant, go to the terminal where you ran the `docker run` command and press Ctrl+C.")