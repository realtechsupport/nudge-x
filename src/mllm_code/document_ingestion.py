"""
Document Ingestion Module for Agentic RAG v1
Simple PDF → chunks → Qdrant pipeline.
"""
import os
import uuid
import re
from typing import List, Dict, Optional
from pathlib import Path
from dataclasses import dataclass

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

from mllm_code.config.database_config import (
    EMBEDDING_MODEL_NAME, QDRANT_URL, QDRANT_API_KEY
)
from mllm_code.database_pipeline.vector_db_operations import (
    create_qdrant_client, create_qdrant_client_api, create_qdrant_client_testing,
    get_or_create_collection
)

load_dotenv()

# Default collection name for documents
DOCUMENT_COLLECTION = "documents"

_DOC_ID_NAMESPACE = uuid.NAMESPACE_DNS


@dataclass
class DocumentChunk:
    """A chunk of text with metadata."""
    text: str
    doc_id: str
    doc_title: str
    page: Optional[int] = None
    section: Optional[str] = None
    chunk_index: int = 0


class DocumentIngestor:
    """
    Ingests documents (PDF or text) into Qdrant for RAG.
    """
    
    def __init__(
        self,
        collection_name: str = DOCUMENT_COLLECTION,
        embedding_model: str = EMBEDDING_MODEL_NAME,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        client: Optional[QdrantClient] = None,
    ):
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
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
        self.vector_size = self.model.get_sentence_embedding_dimension()
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Ensure collection exists
        get_or_create_collection(
            self.client, 
            self.collection_name, 
            self.vector_size
        )
        
        print("Document Ingestor initialized:")
        print(f"   - Collection: {collection_name}")
        print(f"   - Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
    
    def _extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract text from PDF with page numbers."""
        try:
            import pymupdf  # PyMuPDF
        except ImportError:
            try:
                import fitz as pymupdf  # Fallback import name
            except ImportError:
                raise ImportError("Please install PyMuPDF: pip install pymupdf")
        
        pages = []
        doc = pymupdf.open(pdf_path)
        
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():
                pages.append({
                    "text": text,
                    "page": page_num
                })
        
        doc.close()
        return pages
    
    def _extract_text_from_txt(self, txt_path: str) -> List[Dict]:
        """Extract text from a plain text file."""
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return [{"text": text, "page": 1}]

    def _split_markdown_into_sections(self, text: str, fallback_title: str) -> List[Dict]:
        """
        Split markdown into sections based on headings.

        Returns list of {section_title, text, section_index}.
        If no headings are found, returns a single section.
        """
        # Normalize newlines
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Find headings and their spans
        # Example match: "## Title"
        heading_re = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
        matches = list(heading_re.finditer(text))
        if not matches:
            return [{"section_title": fallback_title, "text": text, "section_index": 1}]

        sections: List[Dict] = []
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            title = m.group(2).strip()
            body = text[start:end].strip()
            if not body:
                # Keep empty sections out; headings-only chunks aren't useful for retrieval
                continue
            sections.append(
                {"section_title": title, "text": body, "section_index": i + 1}
            )

        if not sections:
            return [{"section_title": fallback_title, "text": text, "section_index": 1}]
        return sections

    def _truncate_for_embedding(self, text: str, max_chars: int = 6000) -> str:
        """
        Sentence-transformers will truncate anyway; this keeps the doc/section nodes
        from being overly large while preserving a representative sample.
        """
        text = (text or "").strip()
        if len(text) <= max_chars:
            return text
        head = text[: int(max_chars * 0.85)]
        tail = text[-int(max_chars * 0.15) :]
        return f"{head}\n...\n{tail}"

    def _make_point_id(self, kind: str, *parts: str) -> str:
        """
        Deterministic UUID for stable upserts.
        """
        key = ":".join([kind, *[p for p in parts if p is not None]])
        return str(uuid.uuid5(_DOC_ID_NAMESPACE, key))
    
    def _chunk_document(
        self, 
        pages: List[Dict], 
        doc_id: str, 
        doc_title: str
    ) -> List[DocumentChunk]:
        """Split document pages into chunks with metadata."""
        chunks = []
        chunk_index = 0
        
        for page_data in pages:
            page_text = page_data["text"]
            page_num = page_data.get("page", 1)
            
            # Split page into chunks
            page_chunks = self.text_splitter.split_text(page_text)
            
            for chunk_text in page_chunks:
                if chunk_text.strip():
                    chunks.append(DocumentChunk(
                        text=chunk_text,
                        doc_id=doc_id,
                        doc_title=doc_title,
                        page=page_num,
                        chunk_index=chunk_index
                    ))
                    chunk_index += 1
        
        return chunks
    
    def ingest_file(
        self, 
        file_path: str, 
        doc_title: Optional[str] = None,
        doc_id: Optional[str] = None
    ) -> int:
        """
        Ingest a single document file (PDF or TXT).
        
        Returns:
            Number of chunks added.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Generate doc ID and title if not provided
        doc_id = doc_id or str(uuid.uuid4())
        doc_title = doc_title or file_path.stem
        
        print(f"\nIngesting: {file_path.name}")
        print(f"   Title: {doc_title}")
        print(f"   Doc ID: {doc_id}")
        
        # Extract text based on file type
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            pages = self._extract_text_from_pdf(str(file_path))
        elif suffix in [".txt", ".md"]:
            pages = self._extract_text_from_txt(str(file_path))
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        print(f"   Extracted {len(pages)} pages")
        
        # Chunk the document
        chunks = self._chunk_document(pages, doc_id, doc_title)
        print(f"   Created {len(chunks)} chunks")
        
        if not chunks:
            print("   Warning: No chunks to add.")
            return 0
        
        # Vectorize chunks
        texts = [c.text for c in chunks]
        vectors = self.model.encode(texts)
        
        # Build points for Qdrant
        points = []
        for i, chunk in enumerate(chunks):
            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors[i].tolist(),
                payload={
                    "text": chunk.text,
                    "chunk": chunk.text,  # For compatibility with existing RAG
                    "doc_id": chunk.doc_id,
                    "title": chunk.doc_title,
                    "page": chunk.page,
                    "section": chunk.section,
                    "chunk_index": chunk.chunk_index,
                    "source": "document"
                }
            ))
        
        # Upsert to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True
        )
        
        print(f"   Added {len(points)} chunks to '{self.collection_name}'.")
        return len(points)

    # -------------------------------------------------------------------------
    # 3-layer hierarchical ingestion (doc → section → chunk)
    # -------------------------------------------------------------------------
    def ingest_file_hierarchical(
        self,
        file_path: str,
        doc_title: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Ingest a single document as a 3-layer hierarchy:
        - level 1: document node
        - level 2: section nodes (PDF: per-page sections, MD: per-heading sections, TXT: single section)
        - level 3: chunk nodes (split within each section)

        Returns:
            dict with counts: {doc:1, sections:N, chunks:M, total_points:...}
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        doc_id = doc_id or str(uuid.uuid4())
        doc_title = doc_title or file_path.stem

        print(f"\nHierarchical ingest: {file_path.name}")
        print(f"   Title: {doc_title}")
        print(f"   Doc ID: {doc_id}")

        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            pages = self._extract_text_from_pdf(str(file_path))
            # Sections: one per page (simple, reliable)
            sections = [
                {
                    "section_title": f"Page {p.get('page', i + 1)}",
                    "text": p["text"],
                    "section_index": int(p.get("page", i + 1)),
                    "page_start": int(p.get("page", i + 1)),
                    "page_end": int(p.get("page", i + 1)),
                }
                for i, p in enumerate(pages)
                if (p.get("text") or "").strip()
            ]
        elif suffix == ".md":
            pages = self._extract_text_from_txt(str(file_path))
            md_text = pages[0]["text"] if pages else ""
            sections = self._split_markdown_into_sections(md_text, fallback_title=doc_title)
            for s in sections:
                s["page_start"] = 1
                s["page_end"] = 1
        elif suffix in [".txt"]:
            pages = self._extract_text_from_txt(str(file_path))
            text = pages[0]["text"] if pages else ""
            sections = [{"section_title": doc_title, "text": text, "section_index": 1, "page_start": 1, "page_end": 1}]
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        if not sections:
            print("   Warning: No section text found; nothing ingested.")
            return {"doc": 0, "sections": 0, "chunks": 0, "total_points": 0}

        # Create doc node text from all section text (truncated)
        full_text = "\n\n".join([s["text"] for s in sections])
        doc_text = self._truncate_for_embedding(full_text)

        doc_point_id = self._make_point_id("doc", doc_id)

        # Prepare section and chunk nodes
        section_nodes: List[Dict] = []
        chunk_nodes: List[Dict] = []

        for s in sections:
            section_index = int(s.get("section_index", 0) or 0)
            section_id = self._make_point_id("section", doc_id, str(section_index))
            section_text = (s.get("text") or "").strip()
            if not section_text:
                continue

            section_nodes.append(
                {
                    "node_type": "section",
                    "level": 2,
                    "doc_id": doc_id,
                    "doc_point_id": doc_point_id,
                    "section_id": section_id,
                    "section_index": section_index,
                    "section_title": s.get("section_title") or f"Section {section_index}",
                    "page_start": s.get("page_start"),
                    "page_end": s.get("page_end"),
                    # Use the section text for embedding (truncated) so section-level retrieval is possible.
                    "text": self._truncate_for_embedding(section_text),
                    "chunk": self._truncate_for_embedding(section_text),  # compatibility
                    "source": "document",
                    "filename": file_path.name,
                    "path": str(file_path),
                }
            )

            # Split section into chunks
            chunks = self.text_splitter.split_text(section_text)
            for chunk_index, chunk_text in enumerate(chunks):
                chunk_text = (chunk_text or "").strip()
                if not chunk_text:
                    continue
                chunk_point_id = self._make_point_id("chunk", doc_id, str(section_index), str(chunk_index))
                chunk_nodes.append(
                    {
                        "node_type": "chunk",
                        "level": 3,
                        "doc_id": doc_id,
                        "doc_point_id": doc_point_id,
                        "section_id": section_id,
                        "section_index": section_index,
                        "section_title": s.get("section_title") or f"Section {section_index}",
                        "chunk_index": chunk_index,
                        "page_start": s.get("page_start"),
                        "page_end": s.get("page_end"),
                        "text": chunk_text,
                        "chunk": chunk_text,  # used by existing retrieval code paths
                        "source": "document",
                        "filename": file_path.name,
                        "path": str(file_path),
                    }
                )

        # Build doc node payload
        doc_node = {
            "node_type": "doc",
            "level": 1,
            "doc_id": doc_id,
            "doc_point_id": doc_point_id,
            "title": doc_title,
            "text": doc_text,
            "chunk": doc_text,  # compatibility
            "source": "document",
            "filename": file_path.name,
            "path": str(file_path),
            "sections_count": len(section_nodes),
            "chunks_count": len(chunk_nodes),
        }

        # Vectorize all nodes (doc + sections + chunks)
        payloads = [doc_node, *section_nodes, *chunk_nodes]
        texts = [p["chunk"] for p in payloads]
        vectors = self.model.encode(texts)

        point_ids = [doc_point_id] + [p["section_id"] for p in section_nodes] + [
            self._make_point_id("chunk", p["doc_id"], str(p["section_index"]), str(p["chunk_index"])) for p in chunk_nodes
        ]

        points: List[models.PointStruct] = []
        for i, payload in enumerate(payloads):
            if payload["node_type"] == "doc":
                pid = doc_point_id
            elif payload["node_type"] == "section":
                pid = payload["section_id"]
            else:
                pid = self._make_point_id("chunk", payload["doc_id"], str(payload["section_index"]), str(payload["chunk_index"]))

            points.append(
                models.PointStruct(
                    id=pid,
                    vector=vectors[i].tolist(),
                    payload=payload,
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)

        print(f"   Added {1 + len(section_nodes) + len(chunk_nodes)} points to '{self.collection_name}'.")
        return {
            "doc": 1,
            "sections": len(section_nodes),
            "chunks": len(chunk_nodes),
            "total_points": 1 + len(section_nodes) + len(chunk_nodes),
        }
    
    def ingest_directory(
        self, 
        dir_path: str, 
        extensions: List[str] = [".pdf", ".txt", ".md"]
    ) -> int:
        """
        Ingest all supported documents from a directory.
        
        Returns:
            Total number of chunks added.
        """
        dir_path = Path(dir_path)
        
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")
        
        total_chunks = 0
        files_processed = 0
        
        for ext in extensions:
            for file_path in dir_path.glob(f"*{ext}"):
                try:
                    chunks_added = self.ingest_file(str(file_path))
                    total_chunks += chunks_added
                    files_processed += 1
                except Exception as e:
                    print(f"   Error processing {file_path.name}: {e}")
        
        print("\nSummary:")
        print(f"   Files processed: {files_processed}")
        print(f"   Total chunks added: {total_chunks}")
        
        return total_chunks
    
    def ingest_text(
        self,
        text: str,
        doc_title: str,
        doc_id: Optional[str] = None
    ) -> int:
        """
        Ingest raw text directly (useful for testing).
        
        Returns:
            Number of chunks added.
        """
        doc_id = doc_id or str(uuid.uuid4())
        
        pages = [{"text": text, "page": 1}]
        chunks = self._chunk_document(pages, doc_id, doc_title)
        
        if not chunks:
            return 0
        
        texts = [c.text for c in chunks]
        vectors = self.model.encode(texts)
        
        points = []
        for i, chunk in enumerate(chunks):
            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors[i].tolist(),
                payload={
                    "text": chunk.text,
                    "chunk": chunk.text,
                    "doc_id": chunk.doc_id,
                    "title": chunk.doc_title,
                    "page": chunk.page,
                    "chunk_index": chunk.chunk_index,
                    "source": "document"
                }
            ))
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True
        )
        
        print(f"Added {len(points)} chunks from text '{doc_title}'.")
        return len(points)
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the document collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": info.status
            }
        except Exception as e:
            return {"error": str(e)}
    
    def inspect_chunks(self, limit: int = 5) -> List[Dict]:
        """
        Inspect sample chunks from the collection to verify ingestion.
        """
        try:
            # Scroll through some points
            results = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            chunks = []
            for point in results[0]:
                payload = point.payload or {}
                chunks.append({
                    "id": point.id,
                    "title": payload.get("title"),
                    "page": payload.get("page"),
                    "section": payload.get("section"),
                    "chunk_index": payload.get("chunk_index"),
                    "text_preview": (payload.get("text", "") or "")[:150] + "..."
                })
            
            return chunks
        except Exception as e:
            return [{"error": str(e)}]
    
    def print_inspection(self, limit: int = 5):
        """Print a formatted inspection of stored chunks."""
        print(f"\n{'='*60}")
        print(f"INSPECTING COLLECTION: {self.collection_name}")
        print(f"{'='*60}")
        
        stats = self.get_collection_stats()
        print("\nStats:")
        print(f"   Total chunks: {stats.get('points_count', 'N/A')}")
        print(f"   Status: {stats.get('status', 'N/A')}")
        
        chunks = self.inspect_chunks(limit)
        print(f"\nSample Chunks ({len(chunks)} shown):")
        print("-"*60)
        
        for i, chunk in enumerate(chunks, 1):
            print(f"\n[Chunk {i}]")
            print(f"   Title: {chunk.get('title')}")
            print(f"   Page: {chunk.get('page')}")
            print(f"   Section: {chunk.get('section')}")
            print(f"   Index: {chunk.get('chunk_index')}")
            print(f"   Text: {chunk.get('text_preview')}")
    
    def clear_collection(self):
        """Delete all documents from the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            get_or_create_collection(
                self.client, 
                self.collection_name, 
                self.vector_size
            )
            print(f"Cleared collection '{self.collection_name}'.")
        except Exception as e:
            print(f"Error clearing collection: {e}")


# -------------------------
# CLI INTERFACE
# -------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest documents for Agentic RAG")
    parser.add_argument("path", nargs="?", default=".", help="File or directory path to ingest")
    parser.add_argument("--title", help="Document title (for single file)")
    parser.add_argument("--hierarchical", action="store_true", help="Use 3-layer hierarchical ingestion (doc→section→chunk)")
    parser.add_argument("--clear", action="store_true", help="Clear collection before ingesting")
    parser.add_argument("--stats", action="store_true", help="Show collection stats only")
    parser.add_argument("--inspect", type=int, nargs="?", const=5, help="Inspect stored chunks (default: 5)")
    
    args = parser.parse_args()
    
    ingestor = DocumentIngestor()
    
    if args.inspect is not None:
        ingestor.print_inspection(limit=args.inspect)
    elif args.stats:
        stats = ingestor.get_collection_stats()
        print(f"\nCollection Stats: {stats}")
    elif args.clear:
        ingestor.clear_collection()
    else:
        path = Path(args.path)
        if path.is_file():
            if args.hierarchical:
                ingestor.ingest_file_hierarchical(str(path), doc_title=args.title)
            else:
                ingestor.ingest_file(str(path), doc_title=args.title)
        elif path.is_dir():
            if args.hierarchical:
                # hierarchical directory ingest: keep existing signature but call hierarchical per file
                total = 0
                for ext in [".pdf", ".txt", ".md"]:
                    for file_path in path.glob(f"*{ext}"):
                        res = ingestor.ingest_file_hierarchical(str(file_path))
                        total += int(res.get("chunks", 0))
                print(f"\nHierarchical ingest complete. Total chunk nodes added: {total}")
            else:
                ingestor.ingest_directory(str(path))
        else:
            print(f"Error: Path not found: {path}")
