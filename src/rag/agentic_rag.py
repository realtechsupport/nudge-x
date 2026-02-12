"""
Agentic RAG v1 - Minimalistic version
Combines caption vector DB + document vector DB with self-evaluation loop.
"""
import os
import json
import requests
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from mllm.config.database_config import (
    QDRANT_COLLECTION_NAME, EMBEDDING_MODEL_NAME,
    QDRANT_URL, QDRANT_API_KEY
)
from database_pipeline.vector_db_operations import (
    create_qdrant_client, create_qdrant_client_api, create_qdrant_client_testing
)
from mllm.config.settings import DEEPSEEK_MODEL

load_dotenv()


@dataclass
class RetrievalResult:
    """Container for retrieval results from a single source."""
    source: str  # "captions" or "documents"
    chunks: List[Dict]
    combined_text: str


@dataclass
class AgentResponse:
    """Final response from the agentic RAG."""
    answer: str
    caption_sources: List[Dict]
    document_sources: List[Dict]
    iterations: int
    is_sufficient: bool


class AgenticRAG:
    """
    Minimalistic Agentic RAG that:
    1. Retrieves from caption vector DB
    2. Retrieves from document vector DB (if available)
    3. Fuses context from both sources
    4. Generates answer
    5. Self-evaluates and iterates if needed
    """
    
    def __init__(
        self,
        caption_collection: str = QDRANT_COLLECTION_NAME,
        document_collection: str = "documents",
        embedding_model: str = EMBEDDING_MODEL_NAME,
        llm_model: str = DEEPSEEK_MODEL,
        max_iterations: int = 2,
        client: Optional[QdrantClient] = None,
    ):
        self.caption_collection = caption_collection
        self.document_collection = document_collection
        self.llm_model = llm_model
        self.max_iterations = max_iterations
        
        # Initialize Qdrant client
        if client:
            self.client = client
        else:
            qdrant_mode = os.getenv("QDRANT_MODE", "production").lower()
            if qdrant_mode == "testing":
                self.client = create_qdrant_client_testing()
            elif qdrant_mode == "api":
                self.client = create_qdrant_client_api(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            else:
                self.client = create_qdrant_client()
        
        # Initialize embedding model
        self.model = SentenceTransformer(embedding_model)
        
        # Check which collections exist
        collections = [c.name for c in self.client.get_collections().collections]
        self.has_captions = caption_collection in collections
        self.has_documents = document_collection in collections
        
        print("Agentic RAG initialized:")
        print(f"   - Captions collection '{caption_collection}': {'Found' if self.has_captions else 'Not found'}")
        print(f"   - Documents collection '{document_collection}': {'Found' if self.has_documents else 'Not found'}")

        # For hierarchical document retrieval we filter by payload key "node_type".
        # Qdrant requires a payload index for filtered search on some deployments (including Qdrant Cloud).
        self._documents_node_type_filter_ready = False
        if self.has_documents:
            self._documents_node_type_filter_ready = self._ensure_payload_keyword_index(
                collection_name=self.document_collection,
                field_name="node_type",
            )

    def _ensure_payload_keyword_index(self, collection_name: str, field_name: str) -> bool:
        """
        Ensure a KEYWORD payload index exists for field_name.

        Returns True if filtering on this field should work; False if we should fall back.
        """
        try:
            schema_type = getattr(qmodels, "PayloadSchemaType", None)
            if schema_type and hasattr(schema_type, "KEYWORD"):
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=schema_type.KEYWORD,
                )
            else:
                # Older qdrant-client versions may accept "keyword" directly.
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema="keyword",
                )
            return True
        except Exception as e:
            msg = str(e).lower()
            # If it already exists, treat as ready.
            if "already exists" in msg or "exists" in msg or "conflict" in msg:
                return True
            print(f"Warning: Could not create payload index for '{field_name}' in '{collection_name}': {e}")
            print("   Falling back to client-side filtering (slower, but works).")
            return False
    
    # -------------------------
    # RETRIEVAL
    # -------------------------
    
    def _retrieve_from_collection(
        self, 
        query: str, 
        collection_name: str, 
        top_k: int = 5,
        *,
        query_filter: qmodels.Filter | None = None,
        expand_hierarchy: bool = False,
    ) -> RetrievalResult:
        """Retrieve relevant chunks from a Qdrant collection.

        If expand_hierarchy is True, and the payload contains hierarchical pointers
        (doc_point_id / section_id), we will retrieve those parent nodes and prepend
        their text to the combined context.
        """
        query_embedding = self.model.encode(query).tolist()
        
        # If filtered search isn't available (missing payload index), request more points and filter locally.
        effective_limit = top_k
        local_filter_node_type: str | None = None
        if query_filter is not None and collection_name == self.document_collection and not self._documents_node_type_filter_ready:
            effective_limit = max(top_k * 8, top_k)
            # We only use this filter for node_type="chunk" currently.
            local_filter_node_type = "chunk"
            query_filter = None

        try:
            results = self.client.query_points(
                collection_name=collection_name,
                query=query_embedding,
                limit=effective_limit,
                with_payload=True,
                query_filter=query_filter,
            ).points
        except Exception as e:
            print(f"Error retrieving from {collection_name}: {e}")
            return RetrievalResult(source=collection_name, chunks=[], combined_text="")
        
        chunks = []
        parent_ids: set[str] = set()
        for result in results:
            payload = result.payload or {}
            if local_filter_node_type and payload.get("node_type") != local_filter_node_type:
                continue
            doc_point_id = payload.get("doc_point_id")
            section_id = payload.get("section_id")
            if expand_hierarchy:
                if isinstance(doc_point_id, str) and doc_point_id:
                    parent_ids.add(doc_point_id)
                if isinstance(section_id, str) and section_id:
                    parent_ids.add(section_id)

            chunks.append({
                "text": payload.get("chunk", payload.get("text", "")),
                "score": result.score,
                **payload
            })

            if local_filter_node_type and len(chunks) >= top_k:
                break

        leaf_texts = [c["text"] for c in chunks if c.get("text")]

        if expand_hierarchy and parent_ids:
            try:
                parent_points = self.client.retrieve(
                    collection_name=collection_name,
                    ids=list(parent_ids),
                    with_payload=True,
                )
                doc_texts: List[str] = []
                section_texts: List[str] = []
                for p in parent_points:
                    pp = p.payload or {}
                    t = (pp.get("chunk") or pp.get("text") or "").strip()
                    if not t:
                        continue
                    if pp.get("node_type") == "doc":
                        doc_texts.append(t)
                    elif pp.get("node_type") == "section":
                        section_texts.append(t)

                merged: List[str] = []
                seen: set[str] = set()
                for t in [*doc_texts, *section_texts, *leaf_texts]:
                    tt = (t or "").strip()
                    if not tt or tt in seen:
                        continue
                    merged.append(tt)
                    seen.add(tt)
                combined_text = "\n\n".join(merged)
            except Exception as e:
                print(f"Error expanding hierarchy for {collection_name}: {e}")
                combined_text = "\n\n".join(leaf_texts)
        else:
            combined_text = "\n\n".join(leaf_texts)
        
        return RetrievalResult(
            source=collection_name,
            chunks=chunks,
            combined_text=combined_text
        )
    
    def retrieve(self, query: str, top_k: int = 5) -> Tuple[RetrievalResult, RetrievalResult]:
        """Retrieve from both caption and document collections."""
        caption_results = RetrievalResult(source="captions", chunks=[], combined_text="")
        document_results = RetrievalResult(source="documents", chunks=[], combined_text="")
        
        if self.has_captions:
            caption_results = self._retrieve_from_collection(
                query, self.caption_collection, top_k
            )
            print(f"Retrieved {len(caption_results.chunks)} caption chunks")
        
        if self.has_documents:
            # Hierarchical docs: retrieve chunk nodes only, then expand with parent doc/section text
            # so answers can cite concrete examples while staying grounded.
            doc_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="node_type",
                        match=qmodels.MatchValue(value="chunk"),
                    )
                ]
            )
            document_results = self._retrieve_from_collection(
                query,
                self.document_collection,
                top_k,
                query_filter=doc_filter,
                expand_hierarchy=True,
            )
            print(f"Retrieved {len(document_results.chunks)} document chunks")
        
        return caption_results, document_results
    
    # -------------------------
    # CONTEXT FUSION
    # -------------------------
    
    def _build_fused_context(
        self, 
        caption_results: RetrievalResult, 
        document_results: RetrievalResult
    ) -> str:
        """Build a fused context from both sources.

        Keep the context clean (no explicit 'evidence/source' labels) to reduce
        the chance the LLM mentions retrieval or sources in the final answer.
        """
        context_parts = []
        
        if caption_results.combined_text:
            context_parts.append(caption_results.combined_text)
        
        if document_results.combined_text:
            context_parts.append(document_results.combined_text)
        
        if not context_parts:
            return "No relevant context found."
        
        return "\n\n".join(context_parts)
    
    # -------------------------
    # GENERATION
    # -------------------------
    
    def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using DeepSeek."""
        system_prompt = """You are an expert on mining and Indigenous communities. Answer questions directly as a knowledgeable expert would.

STRICT RULES - VIOLATIONS WILL BE REJECTED:
1. NEVER start with "Based on the available evidence" or similar phrases
2. NEVER use section headers like "Environmental Impacts:", "Synthesis:", "Documentary Evidence:"
3. NEVER mention "satellite evidence", "documentary evidence", "context", or "sources"
4. NEVER use markdown formatting like **bold** headers
5. Write ONLY plain prose paragraphs
6. Answer as if you simply KNOW the information - do not explain where it came from
7. Be specific with names, locations, and facts
8. Maximum 250 words, minimum 50 words

GOOD EXAMPLE:
"Mining operations near Magela Creek have significantly impacted Indigenous water sources. The Ranger mine has recorded over 120 regulatory infringements, including contaminated water releases from tailings dams. Indigenous communities downstream depend on these waterways for fishing and cultural practices..."

BAD EXAMPLE (DO NOT DO THIS):
"Based on the available evidence, **Environmental Impacts:** The satellite imagery shows..." """

        user_prompt = f"""{context}

---
Question: {query}

Write a direct, natural answer in plain prose. No headers, no source references, no "based on evidence" phrases."""

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not set")
        
        response = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            json={
                "model": self.llm_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.7
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    # -------------------------
    # SELF-EVALUATION
    # -------------------------
    
    def _evaluate_answer(self, query: str, answer: str, context: str) -> Tuple[bool, str]:
        """
        Evaluate if the answer is sufficient.
        Returns (is_sufficient, reason)
        """
        # Simple heuristic checks first
        insufficient_phrases = [
            "i cannot answer",
            "not enough information",
            "no relevant context",
            "i don't have",
            "insufficient information",
            "not found in the context"
        ]
        
        answer_lower = answer.lower()
        for phrase in insufficient_phrases:
            if phrase in answer_lower:
                return False, f"Answer contains '{phrase}'"
        
        # Check if answer is too short
        if len(answer.split()) < 20:
            return False, "Answer too short"
        
        # Check if we have context at all
        if "No relevant context found" in context:
            return False, "No context was retrieved"
        
        # Use LLM for deeper evaluation (optional, can be disabled for speed)
        eval_prompt = f"""Evaluate if this answer adequately addresses the question.

Question: {query}

Answer: {answer}

Respond with only "SUFFICIENT" or "INSUFFICIENT: <reason>"."""

        try:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            response = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                },
                json={
                    "model": self.llm_model,
                    "messages": [{"role": "user", "content": eval_prompt}],
                    "temperature": 0.1
                }
            )
            response.raise_for_status()
            eval_result = response.json()["choices"][0]["message"]["content"].strip()
            
            if eval_result.startswith("SUFFICIENT"):
                return True, "LLM evaluation: sufficient"
            else:
                reason = eval_result.replace("INSUFFICIENT:", "").strip()
                return False, f"LLM evaluation: {reason}"
        except Exception as e:
            # If evaluation fails, assume sufficient
            return True, f"Evaluation skipped: {e}"
    
    # -------------------------
    # MAIN AGENTIC LOOP
    # -------------------------
    
    def query(self, user_query: str, top_k: int = 5) -> AgentResponse:
        """
        Main entry point: retrieve, generate, evaluate, iterate if needed.
        """
        print(f"\n{'='*60}")
        print(f"Query: {user_query}")
        print(f"{'='*60}")
        
        iteration = 0
        current_query = user_query
        final_answer = ""
        is_sufficient = False
        
        caption_results = RetrievalResult(source="captions", chunks=[], combined_text="")
        document_results = RetrievalResult(source="documents", chunks=[], combined_text="")
        
        while iteration < self.max_iterations and not is_sufficient:
            iteration += 1
            print(f"\n--- Iteration {iteration}/{self.max_iterations} ---")
            
            # Step 1: Retrieve
            caption_results, document_results = self.retrieve(current_query, top_k)
            
            # Step 2: Fuse context
            fused_context = self._build_fused_context(caption_results, document_results)
            
            # Step 3: Generate answer
            print("Generating answer...")
            final_answer = self._generate_answer(current_query, fused_context)
            
            # Step 4: Evaluate
            print("Evaluating answer...")
            is_sufficient, eval_reason = self._evaluate_answer(
                user_query, final_answer, fused_context
            )
            print(f"   Result: {'Sufficient' if is_sufficient else 'Insufficient - ' + eval_reason}")
            
            # Step 5: If not sufficient, expand query
            if not is_sufficient and iteration < self.max_iterations:
                # Simple query expansion: add context hints
                current_query = f"{user_query} (Include information about environmental impacts, community effects, and mining operations)"
                print(f"   Expanding query for next iteration...")
        
        return AgentResponse(
            answer=final_answer,
            caption_sources=[
                {
                    "mine_name": c.get("mine_name"),
                    "filename": c.get("filename"),
                    "location": c.get("location"),
                    "text_preview": (c.get("text") or "")[:100] + "..."
                }
                for c in caption_results.chunks[:3]
            ],
            document_sources=[
                {
                    # Hierarchical payload keys: title lives on doc node; chunks carry section_title and page_start.
                    "title": c.get("title") or c.get("filename") or "Document",
                    "section": c.get("section_title") or c.get("section") or "",
                    "page": c.get("page_start") or c.get("page") or "",
                    "text_preview": (c.get("text") or "")[:100] + "..."
                }
                for c in document_results.chunks[:3]
            ],
            iterations=iteration,
            is_sufficient=is_sufficient
        )
    
    def print_response(self, response: AgentResponse):
        """Pretty print the response."""
        print(f"\n{'='*60}")
        print("ANSWER:")
        print(f"{'='*60}")
        print(response.answer)
        
        print(f"\n{'='*60}")
        print("METADATA:")
        print(f"   Iterations: {response.iterations}")
        print(f"   Sufficient: {'Yes' if response.is_sufficient else 'No'}")
        
        if response.caption_sources:
            print("\nCaption Sources:")
            for src in response.caption_sources:
                print(f"   - {src.get('mine_name', 'Unknown')} ({src.get('location', '')})")
        
        if response.document_sources:
            print("\nDocument Sources:")
            for src in response.document_sources:
                title = src.get('title', 'Document')
                section = src.get('section')
                page = src.get('page')
                
                # Build display string
                display = f"   - {title}"
                if section:
                    display += f" > {section}"
                if page:
                    display += f" (p. {page})"
                print(display)


# -------------------------
# CLI INTERFACE
# -------------------------

if __name__ == "__main__":
    rag = AgenticRAG()
    
    print("\n" + "="*60)
    print("Agentic RAG v1 - Interactive Mode")
    print("="*60)
    print("Type 'exit' to quit")
    
    while True:
        user_input = input("\nYour question: ").strip()
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        response = rag.query(user_input)
        rag.print_response(response)
