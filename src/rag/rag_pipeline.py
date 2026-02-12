import os
import requests
import json
from typing import List
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from mllm.config.database_config import (
    QDRANT_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    QDRANT_URL,
    QDRANT_API_KEY,
)
from database_pipeline.vector_db_operations import (
    create_qdrant_client,
    create_qdrant_client_testing,
    create_qdrant_client_api,
)
from mllm.config.settings import DEEPSEEK_MODEL
from mllm.config import validate_env
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
    ):
        # Use provided client or fall back to host/port
        self.client = client if client is not None else QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        self.model = SentenceTransformer(model_name)
        self.vector_size = self.model.get_sentence_embedding_dimension()

    def retrieve_context(self, query: str, top_k: int = 3):
        """Performs a similarity search to find relevant text chunks plus their metadata."""
        query_embedding = self.model.encode(query).tolist()

        # Optional filtering (useful for hierarchical document collections)
        # Example: RAG_NODE_TYPE=chunk to avoid retrieving doc/section nodes directly.
        node_type = (os.getenv("RAG_NODE_TYPE") or "").strip()
        query_filter = None
        if node_type:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="node_type",
                        match=qmodels.MatchValue(value=node_type),
                    )
                ]
            )
        
        # Search for the most relevant vectors (using query_points for qdrant-client >= 1.7)
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        ).points

        context_chunks = []
        retrieved_items = []
        parent_ids: set[str] = set()

        for result in search_results:
            payload = result.payload or {}
            # Use enriched chunk (contains metadata + caption) for context
            chunk_text = payload.get("chunk", "")
            # Use caption_chunk (without metadata prefix) for cleaner display if available
            caption_chunk = payload.get("caption_chunk", chunk_text)
            
            context_chunks.append(chunk_text)

            # Collect parent pointers when present (hierarchical documents)
            doc_point_id = payload.get("doc_point_id")
            section_id = payload.get("section_id")
            if isinstance(doc_point_id, str) and doc_point_id:
                parent_ids.add(doc_point_id)
            if isinstance(section_id, str) and section_id:
                parent_ids.add(section_id)
            
            # Include all metadata in retrieved items
            retrieved_items.append(
                {
                    "caption_id": payload.get("caption_id"),
                    "node_type": payload.get("node_type"),
                    "chunk": caption_chunk,
                    "enriched_chunk": chunk_text,
                    "filename": payload.get("filename"),
                    "mine_name": payload.get("mine_name"),
                    "country": payload.get("country"),
                    "location": payload.get("location"),
                    "latitude": payload.get("latitude"),
                    "longitude": payload.get("longitude"),
                    "doc_id": payload.get("doc_id"),
                    "doc_point_id": payload.get("doc_point_id"),
                    "section_id": payload.get("section_id"),
                    "section_title": payload.get("section_title"),
                    "section_index": payload.get("section_index"),
                }
            )

        # Optional hierarchical context expansion: prepend doc/section text for better grounding.
        # This is safe because the system prompt already forbids mentioning sources/tools/etc.
        expand_parents = (os.getenv("RAG_EXPAND_PARENTS", "false") or "false").lower() in {"1", "true", "yes"}
        if expand_parents and parent_ids:
            try:
                parent_points = self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=list(parent_ids),
                    with_payload=True,
                )
                # Order: doc → section → chunk (dedup by exact text)
                doc_texts: List[str] = []
                section_texts: List[str] = []
                for p in parent_points:
                    pp = p.payload or {}
                    text = (pp.get("chunk") or "").strip()
                    if not text:
                        continue
                    if pp.get("node_type") == "doc":
                        doc_texts.append(text)
                    elif pp.get("node_type") == "section":
                        section_texts.append(text)

                merged: List[str] = []
                seen: set[str] = set()
                for t in [*doc_texts, *section_texts, *context_chunks]:
                    tt = (t or "").strip()
                    if not tt or tt in seen:
                        continue
                    merged.append(tt)
                    seen.add(tt)
                combined_context = "\n\n".join(merged)
            except Exception:
                combined_context = " ".join(context_chunks)
        else:
            combined_context = " ".join(context_chunks)
        return combined_context, retrieved_items

    def generate_response_deepseek(self, query: str, context: str, model_name: str, context_items: list | None = None):
        """Calls DeepSeek Chat Completions API with the query and retrieved context.

        Args:
            query: User question
            context: Retrieved context string
        """
  

        system_prompt = """You are a retrieval-augmented assistant for DeepSeek. Answer strictly and exclusively
from the provided source text; if a required fact is absent, say you lack sufficient information and stop,
without speculation or outside knowledge. Produce exactly one concise, well-crafted paragraph of neutral
prose (≤250 words) in plain text only; do not use bullet points, numbering, headings, tables, markdown,
emojis, or extra line breaks. Begin with the direct answer, preserve essential terms, numbers, and dates
from the source text, and never mention or allude to the source, retrieval system, vector database,
captions, images, prompts, tools, internal IDs, or your reasoning; return only the final answer."""

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
        system_prompt = "You are a helpful assistant. Answer the user's questions based on your knowledge within 100 words."

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
    validate_env()

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
    )

    while True:
        user_query = input("Ask a question (or type 'exit' to quit): ")
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
        print(f"\nResponse:\n{llm_response}")
        print("-" * 50)
