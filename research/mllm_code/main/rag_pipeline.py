import os
import sys
import requests
import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from mllm_code.config.database_config import *
from mllm_code.exception import mllmException
from mllm_code.database_pipeline.vector_db_operations import create_qdrant_client, create_qdrant_client_testing, create_qdrant_client_api
from mllm_code.database_pipeline.database_operations import fetch_captions_without_embeddings, mark_embeddings_added
from mllm_code.database_pipeline.vector_db_operations import (
    create_qdrant_client,
    create_qdrant_client_testing,
    initialize_embedding_model,
    add_captions_to_vector_db,
)
from mllm_code.config.settings import DEEPSEEK_MODEL
load_dotenv()

# ---  CORE RAG LOGIC ---
# This class handles the retrieval and generation steps of the RAG pipeline
class RAGSystem:
    def __init__(
        self,
        collection_name: str,
        model_name: str,
        qdrant_host: str | None = None,
        qdrant_port: int | None = None,
        client: QdrantClient | None = None,
        image_collection_name: str | None = None,
    ):
        # Use provided client or fall back to host/port
        self.client = client if client is not None else QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        self.image_collection_name = image_collection_name
        self.model = SentenceTransformer(model_name)
        self.vector_size = self.model.get_sentence_embedding_dimension()

    def _fetch_image_metadata(self, filenames: list[str]) -> dict:
        """Retrieve image payloads from the image collection using filenames as IDs."""
        if not self.image_collection_name or not filenames:
            return {}

        unique_ids = list({name for name in filenames if name})
        if not unique_ids:
            return {}

        try:
            records = self.client.retrieve(
                collection_name=self.image_collection_name,
                ids=unique_ids,
                with_payload=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  Unable to fetch image metadata: {exc}")
            return {}

        metadata = {}
        for record in records:
            key = str(record.id)
            payload = record.payload or {}
            metadata[key] = {
                "image_path": payload.get("image_path") or payload.get("filename"),
                "image_url": payload.get("image_url"),
                "location": payload.get("location"),
            }
        return metadata

    def retrieve_context(self, query: str, top_k: int = 3):
        """Performs a similarity search to find relevant chunks plus their images."""
        query_embedding = self.model.encode(query).tolist()
        
        # Search for the most relevant vectors
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True
        )

        context_chunks = []
        filenames = []
        retrieved_items = []

        for result in search_results:
            payload = result.payload or {}
            chunk_text = payload.get("chunk", "")
            context_chunks.append(chunk_text)
            filenames.append(payload.get("filename"))
            retrieved_items.append(
                {
                    "caption_id": payload.get("caption_id"),
                    "chunk": chunk_text,
                    "filename": payload.get("filename"),
                    "location": payload.get("location"),
                }
            )

        image_metadata = self._fetch_image_metadata(filenames)

        for item in retrieved_items:
            filename = item.get("filename")
            if filename and filename in image_metadata:
                item.update(image_metadata[filename])

        combined_context = " ".join(context_chunks)
        return combined_context, retrieved_items

    def generate_response(self, query: str, context: str):
        """Calls the LLM with the query and retrieved context."""

        system_prompt = "You are a helpful assistant. Use ONLY the provided context to answer the user's question. If the information is not in the context, say 'I cannot answer this question based on the provided information.' Do not add any extra information."

        user_query_with_context = f"Context: {context}\n\nQuestion: {query}"
        

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key=" + api_key
        payload = {
            "contents": [{"parts": [{"text": user_query_with_context}]}],
            "systemInstruction": {"parts": [{"text": system_prompt}]}
        }
        
        headers = {
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status() # Raise an error for bad status codes
            
            data = response.json()
            generated_text = data.get("candidates")[0].get("content").get("parts")[0].get("text")
            return generated_text
        except requests.exceptions.RequestException as e:
            return f"An error occurred during API call: {e}"

    def generate_response_deepseek(self, query: str, context: str, model_name: str, context_items: list | None = None):
        """Calls DeepSeek Chat Completions API with the query and retrieved context.

        Args:
            query: User question
            context: Retrieved context string
        """
  

        system_prompt = """You are a retrieval-augmented assistant for DeepSeek. Answer strictly and exclusively 
        from content retrieved from the RAG vector database; if a required fact is absent, say you lack sufficient 
        information and stop, without speculation or outside knowledge. Produce exactly one concise, well-crafted 
        paragraph of neutral prose (≤250 words) in plain text only; do not use bullet points, numbering, headings, 
        tables, markdown, emojis, or extra line breaks. Begin with the direct answer, preserve essential 
        terms, numbers, and dates from the retrieved material, and never mention the retrieval system, 
        prompts, tools, internal IDs, or your reasoning; return only the final answer."""

        if context_items:
            image_lines = []
            for idx, item in enumerate(context_items, start=1):
                image_ref = item.get("image_url") or item.get("image_path") or item.get("filename") or "unknown image"
                snippet = (item.get("chunk") or "")[:150]
                image_lines.append(f"[{idx}] Related image: {image_ref}\nChunk: {snippet}")
            images_text = "\n\n".join(image_lines)
            context = f"{context}\n\nReferenced imagery:\n{images_text}"

        user_query_with_context = f"Context: {context}\n\nQuestion: {query}"

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set.")

        url = "https://api.deepseek.com/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query_with_context}
        ],
        "temperature": 0.7
        }

        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            data = response.json()
            generated_text = data.get("choices", [{}])[0].get("message", {}).get("content")
            return generated_text
        except requests.exceptions.RequestException as e:
            return f"An error occurred during API call: {e}"

    def generate_response_without_rag(self, query: str, model_name: str):
        system_prompt = "You are a helpful assistant. Answer the user's questions based on your knowledge."

        user_query = f"Question: {query}"

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set.")

        url = "https://api.deepseek.com/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            "temperature": 0.7
        }
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            data = response.json()
            generated_text = data.get("choices", [{}])[0].get("message", {}).get("content")
            return generated_text
        except requests.exceptions.RequestException as e:
            return f"An error occurred during API call: {e}"

# --- 2. MAIN EXECUTION ---
if __name__ == "__main__":
    # Determine Qdrant mode
    qdrant_mode = os.getenv("QDRANT_MODE", "production").lower()

    if qdrant_mode == "testing":
        client = create_qdrant_client_testing()
    elif qdrant_mode == "api":
        client = create_qdrant_client_api(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    else:
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        client = create_qdrant_client(host=qdrant_host, port=qdrant_port)
    
    # Initialize the RAG system
    rag = RAGSystem(
        collection_name=QDRANT_COLLECTION_NAME,
        model_name=EMBEDDING_MODEL_NAME,
        client=client,
        image_collection_name=IMAGE_COLLECTION_NAME,
    )

    while True:
        user_query = input("Ask a question about your data (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            print("Exiting RAG system.")
            break

        print(f"\n--- Processing Query: '{user_query}' ---")
        
        # Retrieval step: find relevant chunks and images
        retrieved_context, retrieved_items = rag.retrieve_context(user_query)
       
        
        # Generation step: get the LLM's answer
        llm_response = rag.generate_response_deepseek(
            user_query,
            retrieved_context,
            model_name=DEEPSEEK_MODEL,
            context_items=retrieved_items
        )
        print("\nLLM Response:", llm_response)
        print("--------------------------------------------------")
