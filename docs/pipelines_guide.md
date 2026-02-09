## Pipelines guide

This guide explains how to run the main pipelines, what they do, and how to
verify that they worked.

All commands assume:

- You are in the project root: `nudge-x`
- Your virtualenv is activated
- You run code from inside `src/`:

```bash
cd src
```

For environment setup and required variables, see
`[docs/setup_and_env.md](docs/setup_and_env.md)`.

---

## 1. Captions pipeline

**Entry point**:  
`[src/mllm_code/main/captions_pipeline.py](src/mllm_code/main/captions_pipeline.py)`  
Uses `[src/mllm_code/captions_generate.py](src/mllm_code/captions_generate.py)`.

### What it does

- Loads RGB images from `IMAGE_DIR` (local or `gs://`).
- Optionally finds NDVI / UDM auxiliary images for each RGB image.
- Calls the NVIDIA LLaMA model to generate multi‑question captions.
- Evaluates captions using Gemini and stores:
  - image metadata
  - caption text
  - evaluation flags (accepted / evaluated)
  - question text
  - GPS coordinates
- Writes everything into the PostgreSQL `captions` table and records a
  `run_id` in `caption_pipeline_runs`.

### How it selects images

- If `IMAGE_DIR` starts with `gs://`:
  - Lists all images under that bucket/prefix.
  - Filters to RGB images (filenames following the `*_rgb_*` naming pattern).
- If `IMAGE_DIR` is a local folder:
  - Scans the folder for `.png`, `.jpg`, `.jpeg`.
  - Filters to RGB images using the same naming pattern.

In both cases, **all RGB images under `IMAGE_DIR` are processed**.

### Command

From `src/`:

```bash
python -m mllm_code.main.captions_pipeline
```

This script:

- Loads `.env`
- Calls `validate_env()` to ensure required env vars are set
- Runs a `Captions` pipeline with a default `batch_size=2`

### How to verify

- Check the console output for:
  - a created pipeline run ID
  - per‑batch progress messages
- Inspect the database:

```bash
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB
```

Inside `psql`:

```sql
\dt
SELECT COUNT(*) FROM captions;
SELECT * FROM caption_pipeline_runs ORDER BY created_at DESC LIMIT 5;
```

You should see new rows in `captions` linked to a recent `run_id`.

---

## 2. Vectorization pipeline (captions → Qdrant)

**Entry point**:  
`[src/mllm_code/main/vectorization_pipeline.py](src/mllm_code/main/vectorization_pipeline.py)`  
Uses:

- `[src/mllm_code/database_pipeline/database_operations.py](src/mllm_code/database_pipeline/database_operations.py)`
- `[src/mllm_code/database_pipeline/vector_db_operations.py](src/mllm_code/database_pipeline/vector_db_operations.py)`

### What it does

- Fetches accepted captions from PostgreSQL that:
  - are `is_accepted = TRUE`
  - either have no embedding yet or have `embedding_added = FALSE`
    in `caption_embeddings`.
- Splits captions into chunks using `RecursiveCharacterTextSplitter` with
  a metadata prefix (mine, country, location).
- Embeds each chunk with `SentenceTransformer` and upserts into Qdrant
  collection `captions_collection`.
- Marks the corresponding captions as embedded in `caption_embeddings`.

### Command

From `src/`:

```bash
python -m mllm_code.main.vectorization_pipeline
```

### How to verify

- Check console output:
  - “Found N accepted captions without embeddings. Embedding now…”
  - “Embeddings created and stored for M chunks.”
- Optionally, connect to Qdrant and inspect `captions_collection`:
  - Using Qdrant’s UI or API, verify that the collection exists and has points.

---

## 3. RAG pipeline (captions‑only Q&A)

**Entry point**:  
`[src/mllm_code/main/rag_pipeline.py](src/mllm_code/main/rag_pipeline.py)`

### What it does

- Connects to Qdrant (`captions_collection`) and loads the same
  embedding model.
- Runs an interactive loop:
  - Takes a text query from the user.
  - Retrieves relevant caption chunks from Qdrant.
  - Calls DeepSeek to generate an answer conditioned on those chunks.

### Command

From `src/`:

```bash
python -m mllm_code.main.rag_pipeline
```

You will see a prompt:

```text
Ask a question (or type 'exit' to quit):
```

Type a question (e.g., “Where are open pit mines with nearby settlements?”)
and press Enter.

### How to verify

- Ensure Qdrant has data (run vectorization first).
- Ask a few questions and check that the responses mention specific mines,
  locations, or descriptions that match your captions.

---

## 4. Agentic RAG (captions + documents with self‑evaluation)

**Entry point**:  
`[src/mllm_code/agentic_rag.py](src/mllm_code/agentic_rag.py)`  
Works with:

- Captions collection (from vectorization pipeline)
- Documents collection (from document ingestion; see below)

### What it does

- Initializes a Qdrant client and a SentenceTransformer model.
- Checks for:
  - captions collection (`captions_collection`)
  - documents collection (`documents`)
- For each query:
  - Retrieves from captions and documents.
  - Fuses context.
  - Generates an answer with DeepSeek.
  - Optionally evaluates whether the answer is sufficient and can iterate.

### Command

From `src/`:

```bash
python -m mllm_code.agentic_rag
```

You will see an interactive prompt. Type questions and inspect the printed
answers and metadata.

### How to verify

- Ensure you have both:
  - caption embeddings in Qdrant
  - documents ingested into the `documents` collection.
- Ask a query that should touch both sources (e.g., referencing a mine
  that appears in both captions and a PDF), and inspect the printed
  caption/document sources.

---

## 5. Document ingestion

**Entry point**:  
`[src/mllm_code/document_ingestion.py](src/mllm_code/document_ingestion.py)`

### What it does

- Reads PDF / TXT / MD files, splits them into sections and chunks.
- Embeds text and stores it in the `documents` collection in Qdrant using
  a three‑level hierarchy (doc → section → chunk) when using hierarchical
  mode.

### Basic commands

From `src/`:

```bash
# Ingest a single document
python -m mllm_code.document_ingestion path/to/file.pdf --title "Doc title"

# Ingest all supported documents from a directory
python -m mllm_code.document_ingestion path/to/dir

# Hierarchical ingest (doc → section → chunk)
python -m mllm_code.document_ingestion path/to/file.pdf --hierarchical --title "Doc title"
```

### How to verify

- Use the built‑in inspect and stats options:

```bash
python -m mllm_code.document_ingestion --stats
python -m mllm_code.document_ingestion --inspect 5
```

These commands print collection stats and sample chunks from Qdrant.

---

## 6. Typical end‑to‑end workflow

For a fresh run starting from images:

1. **Set up environment**
   - Clone repo and create virtualenv (`install_instructions.txt`).
   - Configure `.env` (`docs/setup_and_env.md`).
2. **Generate captions**
   - `cd research`
   - `python -m mllm_code.main.captions_pipeline`
3. **Vectorize captions**
   - `python -m mllm_code.main.vectorization_pipeline`
4. **(Optional) Ingest documents**
   - `python -m mllm_code.document_ingestion path/to/docs --hierarchical`
5. **Run analysis**
   - RAG: `python -m mllm_code.main.rag_pipeline`
   - Agentic RAG: `python -m mllm_code.agentic_rag`
6. **Export captions for a specific run**
   - `python -m mllm_code.export_captions --list-runs`
   - `python -m mllm_code.export_captions <run_id>`

