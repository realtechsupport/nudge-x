## Environment setup and configuration

This document explains how to set up the Python environment and the required
environment variables so you can run the main pipelines.

### 1. Python environment

- **Recommended**: Python 3.10+
- Create and activate a virtual environment in the project root (`nudge-x`):

MacOS / Linux:

```bash
python3 -m venv your_venv_name
source your_venv_name/bin/activate

# 1) Install lightweight CPU-only torch first (avoids large CUDA wheels)
python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

# 2) Then install the rest of the dependencies
pip install -r requirements.txt
```

Windows:

```bash
python -m venv your_venv_name
your_venv_name\Scripts\activate

REM 1) Install lightweight CPU-only torch first (avoids large CUDA wheels)
python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

REM 2) Then install the rest of the dependencies
pip install -r requirements.txt
```

> For more detailed clone instructions, see `install_instructions.txt`.

### 2. Environment variables (`.env`)

All pipelines rely on environment variables loaded via `python-dotenv`. The
validation helper in `[src/mllm_code/config/__init__.py](src/mllm_code/config/__init__.py)`
(`validate_env()`) checks that the most important ones are set.

Create a `.env` file in the project root, using `.env.example` as a template,
and define at least the following variables:

**PostgreSQL (captions database)**  
Used by `[src/mllm_code/database_pipeline/database_operations.py](src/mllm_code/database_pipeline/database_operations.py)`.

- `POSTGRES_DB` – database name (e.g., `captions_db`)
- `POSTGRES_USER` – database user (e.g., `Admin1`)
- `POSTGRES_PASSWORD` – database password
- `POSTGRES_HOST` – host (e.g., `localhost`)
- `POSTGRES_PORT` – port (e.g., `5432`)

**Image and metadata paths**

- `IMAGE_DIR` – path to satellite images, either:
  - local folder (e.g., `/home/you/data/images`), or
  - GCS path (e.g., `gs://nudge-bucket/path/to/images`)
- `METADATA_CSV` – path to the mines metadata CSV used by `mllm_helper.py`
  (e.g., `data/metadata/Mines_Metadata_v26.csv`)

**Qdrant (vector database)**
Configured in `[src/mllm_code/config/database_config.py](src/mllm_code/config/database_config.py)`.

- `QDRANT_HOST` – host for Qdrant (e.g., `localhost` or cloud endpoint)
- `QDRANT_PORT` – port (default `6333` if unset)
- `QDRANT_MODE` – one of:
  - `production` – connect to host/port
  - `api` – use `QDRANT_URL` + `QDRANT_API_KEY`
  - `testing` – in‑memory Qdrant client
- `QDRANT_URL` – base URL for Qdrant (only for `QDRANT_MODE=api`)
- `QDRANT_API_KEY` – API key/token (only for `QDRANT_MODE=api`)

**LLM / API keys**

Used across `captions_generate.py`, `evaluation.py`, and RAG code.

- `NVIDIA_API_KEY` – for LLaMA / Kosmos calls via NVIDIA API
- `GOOGLE_API_KEY` – for Gemini caption evaluation
- `DEEPSEEK_API_KEY` – for DeepSeek in `rag_pipeline.py` and `agentic_rag.py`

**Prompt version**

- `PROMPT_VERSION` – which prompt variant to use (`v4`, `v5`, `v6`, `v7`, …).  
  The module `src/mllm_code/prompts.py` reads this and routes to the matching
  prompt file under `src/mllm_code/prompts/`. `validate_env()` enforces that
  this variable is set.

### 3. Environment validation (`validate_env()`)

Most main entry points call `validate_env()` from
`[src/mllm_code/config/__init__.py](src/mllm_code/config/__init__.py)`
before doing any work:

- `captions_pipeline.py` – captions generation
- `vectorization_pipeline.py` – caption embeddings → Qdrant
- `rag_pipeline.py` – RAG over captions

If any required variable is missing, you will see an error like:

```text
============================================================
ERROR: Missing required environment variables.
Please set them in your .env file or shell environment.

  - POSTGRES_DB              (PostgreSQL database name)
  - IMAGE_DIR                (Path to satellite images (local or gs://))
  ...
============================================================
```

Fix your `.env` file accordingly and rerun the command.

