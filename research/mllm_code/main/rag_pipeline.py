import os
import sys
import requests
import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from mllm_code.exception import mllmException
from mllm_code.database_pipeline.vector_db_operations import create_qdrant_client, create_qdrant_client_testing
from mllm_code.database_pipeline.database_operations import fetch_captions_without_embeddings, mark_embeddings_added
from mllm_code.database_pipeline.vector_db_operations import (
    create_qdrant_client,
    create_qdrant_client_testing,
    initialize_embedding_model,
    add_captions_to_vector_db,
)

load_dotenv()

# ---  CORE RAG LOGIC ---
# This class handles the retrieval and generation steps of the RAG pipeline
class RAGSystem:
    def __init__(self, collection_name: str, model_name: str, qdrant_host: str, qdrant_port: int):
        # Connect to the external Qdrant instance
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        self.model = SentenceTransformer(model_name)
        self.vector_size = self.model.get_sentence_embedding_dimension()

    def retrieve_context(self, query: str, top_k: int = 3):
        """Performs a similarity search to find relevant chunks."""
        query_embedding = self.model.encode(query).tolist()
        
        # Search for the most relevant vectors
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True
        )

        # Extract the relevant chunks from the search results
        context = [result.payload["chunk"] for result in search_results]
        return " ".join(context)

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

# --- 2. MAIN EXECUTION ---
if __name__ == "__main__":
    # Get Qdrant connection details from environment variables
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    
    # Initialize the RAG system
    rag = RAGSystem(
        collection_name="captions_collection",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port
    )

    while True:
        user_query = input("Ask a question about your data (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            print("Exiting RAG system.")
            break

        print(f"\n--- Processing Query: '{user_query}' ---")
        
        # Retrieval step: find relevant chunks
        retrieved_context = rag.retrieve_context(user_query)
        print("Retrieved Context:", retrieved_context)
        
        # Generation step: get the LLM's answer
        llm_response = rag.generate_response(user_query, retrieved_context)
        print("\nLLM Response:", llm_response)
        print("--------------------------------------------------")
