
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
import io
import os # Added for dummy image creation

def load_clip_model():
    """Loads a pre-trained CLIP model for image and text embedding."""
    try:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        print("CLIP model loaded successfully.")
        return processor, model
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        return None, None


def get_image_embedding_clip(image_path, clip_processor, clip_model):
    """Generates an embedding for an image using CLIP."""
    if not clip_processor or not clip_model:
        print("CLIP models not loaded. Cannot generate image embedding.")
        return None
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
        return image_features.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error generating image embedding with CLIP: {e}")
        return None
        
def load_sentence_transformer_model():
    """Loads a pre-trained Sentence Transformer model for text embedding."""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Sentence Transformer model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading Sentence Transformer model: {e}")
        return None

def get_text_embedding_sbert(texts, sbert_model):
    """Generates embeddings for a list of texts using SentenceTransformer."""
    if not sbert_model:
        print("Sentence Transformer model not loaded. Cannot generate text embeddings.")
        return None
    if not texts: # Handle empty list of texts
        return []
    try:
        embeddings = sbert_model.encode(texts, convert_to_numpy=True)
        return embeddings
    except Exception as e:
        print(f"Error generating text embeddings with SentenceTransformer: {e}")
        return None

if __name__ == "__main__":
    clip_processor, clip_model = load_clip_model()
    sbert_model = load_sentence_transformer_model()

    if clip_processor and clip_model:
        dummy_image_path = "images/dummy_test_image_for_embedding_check.jpg"
        os.makedirs(os.path.dirname(dummy_image_path), exist_ok=True)
        try:
            Image.new('RGB', (60, 30), color = 'blue').save(dummy_image_path)
            with open(dummy_image_path, 'rb') as f:
                dummy_image_bytes = f.read()
            img_emb = get_image_embedding_clip(dummy_image_bytes, clip_processor, clip_model)
            if img_emb is not None:
                print(f"Dummy Image Embedding Shape: {img_emb.shape}")
        except Exception as e:
            print(f"Could not create or process dummy image: {e}")
        finally:
            if os.path.exists(dummy_image_path):
                os.remove(dummy_image_path)

    if sbert_model:
        text_emb = get_text_embedding_sbert(["Hello world!", "This is a test sentence."], sbert_model)
        if text_emb is not None:
            print(f"Text Embeddings Shape: {text_emb.shape}")