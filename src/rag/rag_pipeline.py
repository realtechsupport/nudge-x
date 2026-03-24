import os
import re
import requests
import json
from typing import Any, List
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
from mllm.config.settings import DEEPSEEK_MODEL, RAG_LLM
from mllm.config import validate_env
load_dotenv()

# Basic list of country names for simple query matching.
# Use a list rather than a set so matching order is deterministic.
COUNTRY_NAMES = [
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola",
    "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan",
    "Bangladesh", "Belarus", "Belgium", "Bolivia", "Bosnia and Herzegovina",
    "Botswana", "Brazil", "Bulgaria", "Cambodia", "Cameroon",
    "Canada", "Chile", "China", "Colombia", "Congo",
    "Costa Rica", "Croatia", "Cuba", "Cyprus", "Czech Republic",
    "Denmark", "Dominican Republic", "Ecuador", "Egypt", "El Salvador",
    "Estonia", "Ethiopia", "Finland", "France", "Georgia",
    "Germany", "Ghana", "Greece", "Guatemala", "Honduras",
    "Hungary", "Iceland", "India", "Indonesia", "Iran",
    "Iraq", "Ireland", "Israel", "Italy", "Jamaica",
    "Japan", "Jordan", "Kazakhstan", "Kenya", "Kuwait",
    "Laos", "Latvia", "Lebanon", "Libya", "Lithuania",
    "Luxembourg", "Madagascar", "Malawi", "Malaysia", "Mali",
    "Mexico", "Moldova", "Mongolia", "Montenegro", "Morocco",
    "Mozambique", "Myanmar", "Namibia", "Nepal", "Netherlands",
    "New Zealand", "Nicaragua", "Niger", "Nigeria", "North Macedonia",
    "Norway", "Oman", "Pakistan", "Panama", "Paraguay",
    "Peru", "Philippines", "Poland", "Portugal", "Qatar",
    "Romania", "Russia", "Rwanda", "Saudi Arabia", "Senegal",
    "Serbia", "Singapore", "Slovakia", "Slovenia", "South Africa",
    "South Korea", "Spain", "Sri Lanka", "Sudan", "Sweden",
    "Switzerland", "Syria", "Taiwan", "Tanzania", "Thailand",
    "Tunisia", "Turkey", "Uganda", "Ukraine", "United Arab Emirates",
    "United Kingdom", "United States", "Uruguay", "Venezuela", "Vietnam",
    "Zambia", "Zimbabwe",
]


EXHAUSTIVE_PATTERNS = [
    r"\blist all\b",
    r"\bwhich mines\b",
    r"\bwhat mines\b",
    r"\ball the\b",
    r"\bin your collection\b",
    r"\bdo you have\b",
    r"\bshow me all\b",
]

COMPARISON_PATTERNS = [
    r"\bmost recent\b",
    r"\brecently created\b",
    r"\bnewest\b",
    r"\blatest\b",
    r"\bolder\b",
    r"\bnewer\b",
    r"\bmore recent\b",
    r"\bearliest\b",
    r"\boldest\b",
]


def _extract_countries_from_query(query: str) -> list[str]:
    """Return all explicitly mentioned countries in deterministic order."""
    query_lower = query.lower()
    matches: list[str] = []

    # Prefer longer names first so, for example, "United States" is matched
    # before a shorter overlapping token in another phrase.
    for country in sorted(COUNTRY_NAMES, key=lambda name: (-len(name), name)):
        pattern = rf"\b{re.escape(country.lower())}\b"
        if re.search(pattern, query_lower):
            matches.append(country)

    return matches


def _query_type(query: str) -> str:
    q = query.lower()
    if any(re.search(pattern, q) for pattern in COMPARISON_PATTERNS):
        return "comparison"
    if any(re.search(pattern, q) for pattern in EXHAUSTIVE_PATTERNS):
        return "set"
    return "fact"

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

    def _build_query_filter(self, query: str):
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

        # If the query references one or more countries, apply a deterministic
        # country filter. Multi-country comparison questions should retrieve
        # chunks from all mentioned countries rather than only the first match.
        matched_countries = _extract_countries_from_query(query)
        if matched_countries:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="country",
                    field_schema=qmodels.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass

            country_conditions = [
                qmodels.FieldCondition(
                    key="country",
                    match=qmodels.MatchValue(value=country),
                )
                for country in matched_countries
            ]

            country_filter = (
                qmodels.Filter(should=country_conditions)
                if len(country_conditions) > 1
                else qmodels.Filter(must=country_conditions)
            )

            existing_must = list(query_filter.must) if query_filter and query_filter.must else []
            existing_should = list(query_filter.should) if query_filter and query_filter.should else []
            existing_must_not = list(query_filter.must_not) if query_filter and query_filter.must_not else []

            if query_filter is None:
                query_filter = country_filter
            elif country_filter.should:
                query_filter = qmodels.Filter(
                    must=[*existing_must, country_filter],
                    should=existing_should,
                    must_not=existing_must_not,
                )
            else:
                query_filter = qmodels.Filter(
                    must=[*existing_must, *country_conditions],
                    should=existing_should,
                    must_not=existing_must_not,
                )

        return query_filter, matched_countries

    def _format_result_item(self, payload: dict[str, Any]) -> dict[str, Any]:
        chunk_text = payload.get("chunk", "")
        caption_chunk = payload.get("caption_chunk", chunk_text)
        return {
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

    def _mine_key(self, payload: dict[str, Any]) -> str:
        mine_name = (payload.get("mine_name") or "").strip()
        if mine_name:
            return mine_name
        filename = (payload.get("filename") or "").strip()
        if filename:
            return filename
        caption_id = payload.get("caption_id")
        if caption_id is not None:
            return f"caption:{caption_id}"
        doc_id = payload.get("doc_id")
        if doc_id is not None:
            return f"doc:{doc_id}"
        return "UNKNOWN"

    def _search_points(self, query_embedding: list[float], limit: int, query_filter=None):
        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
        ).points

    def _group_ranked_results(self, search_results, per_mine_cap: int = 2):
        grouped: dict[str, list[tuple[float, dict[str, Any]]]] = {}
        for rank, result in enumerate(search_results):
            payload = result.payload or {}
            mine_key = self._mine_key(payload)
            # Smaller score is better here because rank is deterministic and query_points
            # already returned the points in descending relevance.
            grouped.setdefault(mine_key, []).append((float(rank), payload))

        ranked_groups = []
        for mine_key, rows in grouped.items():
            rows.sort(key=lambda item: item[0])
            best_rank = rows[0][0]
            limited_payloads = [payload for _, payload in rows[:per_mine_cap]]
            ranked_groups.append((best_rank, mine_key, limited_payloads))

        ranked_groups.sort(key=lambda item: item[0])
        return ranked_groups

    def _collect_parent_ids(self, payloads: list[dict[str, Any]]) -> set[str]:
        parent_ids: set[str] = set()
        for payload in payloads:
            doc_point_id = payload.get("doc_point_id")
            section_id = payload.get("section_id")
            if isinstance(doc_point_id, str) and doc_point_id:
                parent_ids.add(doc_point_id)
            if isinstance(section_id, str) and section_id:
                parent_ids.add(section_id)
        return parent_ids

    def _build_context_from_payloads(self, payloads: list[dict[str, Any]]):
        context_chunks: List[str] = []
        retrieved_items: list[dict[str, Any]] = []

        for payload in payloads:
            chunk_text = payload.get("chunk", "")
            if chunk_text:
                context_chunks.append(chunk_text)
            retrieved_items.append(self._format_result_item(payload))

        parent_ids = self._collect_parent_ids(payloads)

        expand_parents = (os.getenv("RAG_EXPAND_PARENTS", "false") or "false").lower() in {"1", "true", "yes"}
        if expand_parents and parent_ids:
            try:
                parent_points = self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=list(parent_ids),
                    with_payload=True,
                )
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

    def _retrieve_fact_context(self, query_embedding: list[float], query_filter, top_k: int):
        search_results = self._search_points(
            query_embedding=query_embedding,
            limit=top_k,
            query_filter=query_filter,
        )
        payloads = [result.payload or {} for result in search_results]
        return self._build_context_from_payloads(payloads)

    def _retrieve_entity_aware_context(self, query_embedding: list[float], query_filter, matched_countries: list[str], query_type: str, top_k: int):
        if matched_countries:
            candidate_limit = max(top_k, 1000)
            max_limit = max(candidate_limit, 1000)
            target_unique_mines = max(len(matched_countries) * 3, 6)
        elif query_type == "comparison":
            candidate_limit = max(top_k, 40)
            max_limit = 200
            target_unique_mines = 10
        else:
            candidate_limit = max(top_k, 40)
            max_limit = 160
            target_unique_mines = 12

        per_mine_cap = 2
        final_payloads: list[dict[str, Any]] = []

        while True:
            search_results = self._search_points(
                query_embedding=query_embedding,
                limit=candidate_limit,
                query_filter=query_filter,
            )
            grouped = self._group_ranked_results(search_results, per_mine_cap=per_mine_cap)
            unique_mines = len(grouped)

            max_output_mines = min(unique_mines, max(top_k, target_unique_mines))
            selected_payloads: list[dict[str, Any]] = []
            for _, _, payloads in grouped[:max_output_mines]:
                selected_payloads.extend(payloads)

            final_payloads = selected_payloads

            enough_coverage = unique_mines >= target_unique_mines
            exhausted = candidate_limit >= max_limit
            if enough_coverage or exhausted:
                break

            candidate_limit = min(candidate_limit * 2, max_limit)

        return self._build_context_from_payloads(final_payloads)

    def retrieve_context(self, query: str, top_k: int = 3):
        """Performs intent-aware retrieval, widening recall for exhaustive/comparison queries."""
        query_embedding = self.model.encode(query).tolist()
        query_filter, matched_countries = self._build_query_filter(query)
        query_type = _query_type(query)

        if query_type == "fact" and not matched_countries:
            return self._retrieve_fact_context(query_embedding, query_filter, top_k)

        return self._retrieve_entity_aware_context(
            query_embedding=query_embedding,
            query_filter=query_filter,
            matched_countries=matched_countries,
            query_type=query_type,
            top_k=top_k,
        )

    def generate_response_deepseek(self, query: str, context: str, model_name: str, context_items: list | None = None):
        """Calls DeepSeek Chat Completions API with the query and retrieved context.

        Args:
            query: User question
            context: Retrieved context string
        """


        system_prompt = """You are a retrieval-augmented assistant for DeepSeek. Answer strictly and exclusively
from the provided source text; if a required fact is absent, say you lack sufficient information and stop,
without speculation. Produce exactly one concise, well-crafted paragraph of neutral
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

    def generate_response(self, query: str, context: str, model_name: str, context_items: list | None = None):
        """
        Dispatch to the appropriate LLM implementation based on RAG_LLM.

        Currently supported: "deepseek" (default). To add another LLM, implement
        a corresponding generate_response_<name>() method and extend this
        dispatcher.
        """
        llm = (RAG_LLM or "deepseek").strip().lower()
        if llm == "deepseek":
            return self.generate_response_deepseek(query, context, model_name, context_items=context_items)
        raise ValueError(f"Unsupported RAG_LLM '{llm}'. Currently supported: deepseek")

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

    # Nudge-x RAG preamble
    print()
    print()
    print("++++++++++++++++++++++++++++++++++++++")
    print("Nudge-x RAG at your service.")
    print()

    while True:
        user_query = input("Ask a question related to the mining sites in the captions collection below ('exit' to quit):\n")
        if user_query.lower() == 'exit':
            print("Exiting RAG system.")
            break

        print(f"\n--- Processing Query: '{user_query}' ---")

        # Retrieval step: find relevant chunks and images
        retrieved_context, retrieved_items = rag.retrieve_context(user_query)
        # Generation step: get the LLM's answer via dispatcher
        llm_response = rag.generate_response(
            user_query,
            retrieved_context,
            model_name=DEEPSEEK_MODEL,
            context_items=retrieved_items,
        )
        print(f"\nResponse:\n{llm_response}")
        print("-" * 50)
