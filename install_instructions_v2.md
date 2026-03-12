Instructions to run.
-----------------------------------------------------------------------------
1) Clone and environment
-----------------------------------------------------------------------------

Upload setup script to VM. From root: 

	sudo sh setup.sh

Activate venv (check setup.sh for path):

	source /nudge-x/venv/bin/activate

cd to nudge-x

	git sparse-checkout init
	git sparse-checkout set --no-cone /src /data /docs /.env.example /.gitignore /README.md /requirements.txt
	git checkout main

+you should see this statement after success ful checkout: 
"Your branch is up to date with 'origin/main' "

Install Python deps:

	python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
	pip install -r requirements.txt

Add your venv name to .gitignore. Copy .env.example to .env and fill values.

-----------------------------------------------------------------------------
2) Environment variables (.env)
-----------------------------------------------------------------------------

Required (see .env.example for full list):

	POSTGRES_DB=captions_db
	POSTGRES_USER=db_user
	POSTGRES_PASSWORD=db_password
	POSTGRES_HOST=localhost
	POSTGRES_PORT=5432
	IMAGE_DIR=                    # local path or gs://bucket/prefix
	METADATA_TSV=                 # e.g. data/metadata/Mines_Metadata_v28.tsv
	PROMPT_VERSION=v7             # must match a prompts_v*.py file
	NVIDIA_API_KEY=
	GOOGLE_API_KEY=
	DEEPSEEK_API_KEY=

Qdrant (local): QDRANT_HOST, QDRANT_PORT, QDRANT_MODE=production
Qdrant (cloud): QDRANT_MODE=api, QDRANT_URL, QDRANT_API_KEY
RAG LLM: RAG_LLM=deepseek (optional)

-----------------------------------------------------------------------------
3) PostgreSQL
-----------------------------------------------------------------------------

	sudo apt update
	sudo apt install -y postgresql postgresql-contrib
	sudo systemctl start postgresql
	sudo systemctl enable postgresql

Create DB and user:

	sudo -u postgres psql

In psql (use lowercase username; Postgres lowercases unquoted names):

	CREATE USER db_user WITH PASSWORD 'db_password';
	CREATE DATABASE captions_db OWNER db_user;
	\c captions_db
	GRANT ALL ON SCHEMA public TO db_user;
	ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO db_user;
	CREATE EXTENSION IF NOT EXISTS pgcrypto;
	\q

Verify:

	psql -h localhost -U db_user -d captions_db -W

-----------------------------------------------------------------------------
4) Pipelines (run from src)
-----------------------------------------------------------------------------

	cd src

Generate captions:

	python3 -m mllm.main.captions_pipeline

Vectorize accepted captions to Qdrant:

	python3 -m mllm.main.vectorization_pipeline

RAG (captions only):

	python3 -m rag.rag_pipeline

Agentic RAG (captions + documents). Run only if you have ingested documents (see Ingest documents below):

	python3 -m rag.agentic_rag

Ingest documents:

	python3 -m rag.document_ingestion path/to/file_or_dir
	python3 -m rag.document_ingestion path/to/dir

Reset caption embeddings (then re-run vectorization):

	python3 -m database_pipeline.delete_captions

-----------------------------------------------------------------------------
5) Export captions
-----------------------------------------------------------------------------

	List runs:    python3 -m mllm.export_captions --list-runs
	Export run:   python3 -m mllm.export_captions <run_id>

Output: data/frontend_captions/captions_<short_id>_<date>.tsv

-----------------------------------------------------------------------------
6) Database backup and restore
-----------------------------------------------------------------------------

Create dump (on VM with data):

	DUMP_NAME="captions_db_$(date +%F_%H-%M-%S).dump"
	pg_dump -h localhost -U "$POSTGRES_USER" -d "$POSTGRES_DB" -F c -f "$DUMP_NAME"

File in current directory (e.g. ~/). Download: SSH toolbar → Download file. From local:

	gcloud compute scp VM_NAME:"$DUMP_NAME" ./
	scp -i ~/.ssh/YOUR_KEY VM_USER@VM_IP:~/captions_db_*.dump ./

Upload to VM: SSH toolbar → Upload file, or:

	scp -i ~/.ssh/YOUR_KEY ./captions_db_*.dump VM_USER@VM_IP:~/

Restore (on destination VM; empty DB first if it already has tables):

	sudo -u postgres psql -c "DROP DATABASE IF EXISTS captions_db;"
	sudo -u postgres psql -c "CREATE DATABASE captions_db OWNER db_user;"
	sudo -u postgres psql -d captions_db -c "CREATE EXTENSION IF NOT EXISTS pgcrypto;"

	pg_restore -h localhost -U db_user -d captions_db --no-owner --no-privileges ~/captions_db_2026-03-05_04-02-31.dump

Use actual dump path on that VM. Find: ls -la ~ or find ~ -maxdepth 2 -name "*.dump"

-----------------------------------------------------------------------------
7) DB quick reference
-----------------------------------------------------------------------------

	Start:    sudo systemctl start postgresql
	Connect:  psql -h localhost -U db_user -d captions_db -W
	Tables:   \dt
	Count:    SELECT COUNT(*) FROM captions;

-----------------------------------------------------------------------------
8) Prompt versions
-----------------------------------------------------------------------------

Each prompts_v*.py must define: system_prompt, multi_shot_examples, questions.
Set PROMPT_VERSION in .env (e.g. v7). New version: add prompts_vN.py, set PROMPT_VERSION=vN.
