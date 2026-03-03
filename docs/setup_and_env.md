## Environment setup and configuration

This document explains how to set up the Python environment and the required
environment variables so you can run the main pipelines.

### 1. Python environment

- **Recommended**: Python 3.10+
- Create and activate a virtual environment in the project root (`nudge-x`).

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

> For clone + sparse-checkout details, see `install_instructions.txt`.

### 2. Environment variables (`.env`)

All pipelines rely on environment variables loaded via `python-dotenv`. The
validation helper in `[src/mllm/config/__init__.py](src/mllm/config/__init__.py)`
(`validate_env()`) checks that the most important ones are set.

Create a `.env` file in the project root, using `.env.example` as a template,
and define at least the following variables.

**PostgreSQL (captions database)**  
Used by `[src/database_pipeline/database_operations.py](src/database_pipeline/database_operations.py)`.

- `POSTGRES_DB` – database name (e.g., `captions_db`)
- `POSTGRES_USER` – database user (e.g., `Admin1`)
- `POSTGRES_PASSWORD` – database password
- `POSTGRES_HOST` – host (e.g., `localhost`)
- `POSTGRES_PORT` – port (e.g., `5432`)

**Image and metadata paths**

All project data lives under `data/`; metadata TSVs are in `data/metadata/`.

- `IMAGE_DIR` – path to satellite images, either:
  - local folder (e.g., `/home/you/data/images`), or
  - GCS path (e.g., `gs://nudge-bucket/path/to/images`)
- `METADATA_TSV` – path to the mines metadata TSV used by `src/eo/mllm_helper.py`
  (e.g., `data/metadata/Mines_Metadata_v28.tsv`)

**Qdrant (vector database)**  
Configured in `[src/mllm/config/database_config.py](src/mllm/config/database_config.py)`.

- `QDRANT_HOST` – host for Qdrant (e.g., `localhost` or cloud endpoint)
- `QDRANT_PORT` – port (default `6333` if unset)
- `QDRANT_MODE` – one of:
  - `production` – connect to host/port
  - `api` – use `QDRANT_URL` + `QDRANT_API_KEY`
  - `testing` – in‑memory Qdrant client
- `QDRANT_URL` – base URL for Qdrant (only for `QDRANT_MODE=api`)
- `QDRANT_API_KEY` – API key/token (only for `QDRANT_MODE=api`)

**LLM / API keys**

Used across `mllm/captions_generate.py`, `mllm/evaluation.py`, and the RAG code.

- `NVIDIA_API_KEY` – for LLaMA / Kosmos calls via NVIDIA API
- `GOOGLE_API_KEY` – for Gemini caption evaluation
- `DEEPSEEK_API_KEY` – for DeepSeek in `rag/rag_pipeline.py` and `rag/agentic_rag.py`
- `RAG_LLM` – optional selector for which LLM implementation RAG uses (default: `deepseek`)

**Prompt version**

- `PROMPT_VERSION` – which prompt variant to use (`v4`, `v5`, `v6`, `v7`, …).  
  The module `src/mllm/prompts/__init__.py` reads this and routes to the matching
  prompt file under `src/mllm/prompts/`. `validate_env()` enforces that this variable is set.

  **Adding a new prompt version**: The router auto-discovers all `prompts_v*.py` files in
  `src/mllm/prompts/`. To add a new version:
  1. Create `src/mllm/prompts/prompts_v<N>.py`.
  2. Set `PROMPT_VERSION=v<N>` in your `.env`.
  3. No code changes to `__init__.py` are needed — the router will automatically find
     and load your new version.

  **Required variables in each prompt file**:

  Every `prompts_v<N>.py` file should define the following variables:

  - **`system_prompt`** (str, required) — The system-level instruction sent to the LLM.
    This steers the model's behaviour, tone, and output format.
  - **`questions`** (list of str, required) — One or more questions to ask about each
    image. The captions pipeline generates one caption per question per image.
    If this is missing or empty, **no captions will be generated**.
  - **`multi_shot_examples`** (str, recommended) — Example question/answer pairs that
    the LLM uses as style references. If omitted, defaults to an empty string.

  Example skeleton for a new version:

  ```python
  # prompts_v8.py
  system_prompt = """You are an expert environmental analyst..."""

  multi_shot_examples = """
  Question: ...
  Answer: ...
  """

  questions = [
      "Which environmental hazards are present in this Sentinel-2 satellite image?",
  ]
  ```

### 3. Environment validation (`validate_env()`)

Most main entry points call `validate_env()` from
`[src/mllm/config/__init__.py](src/mllm/config/__init__.py)` before doing any work:

- `mllm/main/captions_pipeline.py` – captions generation
- `mllm/main/vectorization_pipeline.py` – caption embeddings → Qdrant
- `rag/rag_pipeline.py` – RAG over captions

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
