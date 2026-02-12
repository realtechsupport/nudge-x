## Export and maintenance utilities

This document explains how to:

- Export captions for a specific pipeline run to TSV.
- Reset caption embeddings and the Qdrant captions collection when you want
  to rebuild embeddings from scratch.

---

## 1. Export captions by `run_id`

**Script**:  
`[src/mllm/export_captions.py](src/mllm/export_captions.py)`

### What it does

- Lists or exports captions that belong to a particular pipeline run
  (`run_id` from `caption_pipeline_runs`).
- Only exports **accepted** captions (`is_accepted = TRUE`) from the
  `captions` table.
- Writes them to a TSV file with frontend‑style columns:
  - `filename`
  - `mine_name`
  - `location`
  - `country`
  - `GPS_coordinates` (either `lat,long` or `Unknown`)
  - `caption`

### List available runs

From `src/`:

```bash
python -m mllm.export_captions --list-runs
```

This prints recent `run_id` values with metadata from `caption_pipeline_runs`,
for example:

```text
Recent pipeline runs:
  - 123e4567-e89b-12d3-a456-426614174000 | 2026-02-09 ... | prompt=v7 | model=llama-4 | shots=5
  - ...
```

### Export accepted captions for a run

```bash
python -m mllm.export_captions <run_id>
```

By default, the script writes to:

```text
data/frontend_captions/captions_<short_id>_<YYYY-MM-DD>.tsv
```

where `<short_id>` is the first part of the UUID (before the first `-`).

You will see a confirmation like:

```text
Wrote 120 captions to data/frontend_captions/captions_123e4567_2026-02-09.tsv
```

### Custom output path

You can override the default filename and directory:

```bash
python -m mllm.export_captions <run_id> --output /path/to/output.tsv
```

The script creates the directory if it does not exist.

---

## 2. Reset embeddings and Qdrant captions collection

**Script**:  
`[src/database_pipeline/delete_captions.py](src/database_pipeline/delete_captions.py)`

### What it does

When you want to rebuild all caption embeddings from scratch, this script:

1. Deletes the Qdrant captions collection (`QDRANT_COLLECTION_NAME`), if it exists.
2. Clears the `caption_embeddings` table in PostgreSQL so all captions appear
   as “not embedded” again.

It does **not** delete the main `captions` table; your captions remain intact.

### Command

From `src/`:

```bash
python -m database_pipeline.delete_captions
```

The script:

- Chooses how to connect to Qdrant based on `QDRANT_MODE` (`production` vs `api`).
- Checks if the collection exists and deletes it if present.
- Calls `reset_embedding_tracking()` to truncate `caption_embeddings`.

You will see output similar to:

```text
Deleting Qdrant collection 'captions_collection' (mode: production)...
Qdrant collection 'captions_collection' deleted successfully.

Resetting PostgreSQL embedding tracking...
Reset PostgreSQL: Cleared 120 rows from caption_embeddings table.

Next step: Re-run vectorization to re-embed with metadata:
   python -m mllm.main.vectorization_pipeline
```

### When to use this script

Typical scenarios:

- You have changed how captions are chunked or embedded and want to rebuild
  the vector store from the same captions.
- You want to clean up a broken or partial embedding run.

After running this script, the usual sequence is:

1. Ensure captions exist in PostgreSQL (`captions` table).
2. Run:

   ```bash
   cd src
   python -m mllm.main.vectorization_pipeline
   ```

3. Optionally verify Qdrant (`captions_collection`) now has fresh points.

---

## 3. Backing up your PostgreSQL database

Your captions database currently runs on a VM-local PostgreSQL instance.  
If that VM or its disk is deleted, the database is lost **unless you have
backups**. The simplest approach is to regularly export the DB using `pg_dump`
and copy the dump file off the VM.

> These commands are examples; adjust usernames/paths as needed.

### 3.1 Create a backup on the VM

Run this on the VM:

```bash
mkdir -p backups

pg_dump -h localhost -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  > backups/captions_db_$(date +%F).sql
```

This produces a file like `backups/captions_db_2026-02-09.sql` containing all
schema and data.

### 3.2 Copy the backup off the VM

From your laptop:

```bash
scp your_user@YOUR_VM_IP:/home/your_user/nudge-x/backups/captions_db_2026-02-09.sql .
```

Or from the VM to a private GCS bucket:

```bash
gsutil cp backups/captions_db_2026-02-09.sql gs://your-private-backup-bucket/
```

Now the backup survives even if the VM is deleted.

### 3.3 Restore the backup elsewhere

On any machine with PostgreSQL installed:

1. Create an empty database:

   ```bash
   createdb -h localhost -U your_local_user captions_db
   ```

2. Load the backup:

   ```bash
   psql -h localhost -U your_local_user -d captions_db \
     -f captions_db_2026-02-09.sql
   ```

3. Point your `.env` at the restored DB:

   ```env
   POSTGRES_DB=captions_db
   POSTGRES_USER=your_local_user
   POSTGRES_PASSWORD=your_local_password
   POSTGRES_HOST=localhost
   POSTGRES_PORT=5432
   ```

All pipelines and export scripts will then use this restored database.

