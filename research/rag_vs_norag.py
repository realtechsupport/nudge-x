import os
import sys
import csv
from pathlib import Path
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
from mllm_code.main.rag_pipeline import *

user_questions_list = ["How do mining operations impact the environment?", 
"What negative effects does mining have on the environment ?", 
"How do mining operations in the Canadian Rockies impact the environment?", 
"How  do mining operations  in DR Congo impact the environment?", 
"How do mining operatons in Australia impact the environment? Elaborate on specific examples.",
]
qdrant_mode = os.getenv("QDRANT_MODE", "production").lower()

if qdrant_mode == "testing":
    client = create_qdrant_client_testing()
elif qdrant_mode == "api":
    client = create_qdrant_client_api(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    client = create_qdrant_client(host=qdrant_host, port=qdrant_port)
rag_system = RAGSystem(
    collection_name=QDRANT_COLLECTION_NAME,
    model_name=EMBEDDING_MODEL_NAME,
    client=client,
    image_collection_name=IMAGE_COLLECTION_NAME,
)

results = []
print("Starting the comparison...")
for user_question in user_questions_list:
    context, context_items = rag_system.retrieve_context(user_question, 3)
    response_without_rag = rag_system.generate_response_without_rag(user_question, "deepseek-chat")
    response_with_rag = rag_system.generate_response_deepseek(
        user_question,
        context,
        "deepseek-chat",
        context_items=context_items
    )

    results.append((user_question, response_without_rag, response_with_rag))
    print(f"Processed: {user_question}")

print("Comparison completed. Saving results...")
output_path = Path(__file__).resolve().parent / "data" / "rag_vs_nonrag_responses.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", encoding="utf-8", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Question", "Without VectorDB", "With VectorDB"])
    writer.writerows(results)

print(f"\nSaved comparison table to {output_path}")

