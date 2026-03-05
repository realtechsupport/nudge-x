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

The captions database runs on a VM-local PostgreSQL instance. If that VM or its disk is deleted, the database is lost unless you have backups. This section describes how to create a dump and copy it to your local machine. For restoring a dump on another VM (including creating the database and user), see [database_setup.md](database_setup.md) sections 6.3–6.5.

**Recommended format:** Use the custom dump format (`.dump`) so the backup can be restored on another machine with a different database user via `pg_restore --no-owner --no-privileges`.

### 3.1 Create a backup on the VM

On the VM where PostgreSQL and `captions_db` run:

```bash
DUMP_NAME="captions_db_$(date +%F_%H-%M-%S).dump"
pg_dump -h localhost -U "$POSTGRES_USER" -d "$POSTGRES_DB" -F c -f "$DUMP_NAME"
```

- This creates a file like `captions_db_2026-03-05_03-58-34.dump` in the current directory.
- `-F c` = custom format (compressed; use with `pg_restore` when restoring).
- Enter the database password when prompted.

### 3.2 Copy the backup to your local machine (two methods)

#### Option A – Download via the browser SSH window

1. In Google Cloud Console, open the VM using the **SSH** button (web terminal in the browser).
2. In the SSH window toolbar (three dots or gear icon), choose **Download file**.
3. When prompted for the path, enter the path to the dump file on the VM (the path you used in step 3.1).
4. The file is downloaded to your local machine.

#### Option B – Use `scp` or `gcloud compute scp` from your local machine

From a local terminal:

- With `gcloud`:
  ```bash
  gcloud compute scp VM_NAME:PATH_TO_DUMP_FILE PATH_TO_LOCAL_DESTINATION
  ```
- With `scp`:
  ```bash
  scp -i ~/.ssh/YOUR_KEY VM_USER@VM_IP:PATH_TO_DUMP_FILE PATH_TO_LOCAL_DESTINATION
  ```
  Omit `-i ~/.ssh/YOUR_KEY` if your default SSH key is used for this VM. If the VM was set up with a dedicated key, use `-i` with that key path; you will be prompted for the key’s passphrase unless you add it to `ssh-agent` (`ssh-add ~/.ssh/YOUR_KEY`).

Replace `PATH_TO_DUMP_FILE` with the path to the dump on the VM (e.g. `captions_db_2026-03-05_03-58-34.dump` if you used the timestamped name in step 3.1), `PATH_TO_LOCAL_DESTINATION` with the path on your local machine, and use the VM’s name, SSH user, and IP as below. Use the **same VM username** you use when logging in via SSH (e.g. `vm-access`), not necessarily your local machine username.

#### Copy from local machine to VM

To upload a dump from your local machine to a VM (e.g. before restoring there), you can use the browser or the command line.

**Option A — Upload via the browser SSH window**

1. In Google Cloud Console, open the VM with the **SSH** button.
2. In the SSH window toolbar (three dots or gear icon), choose **Upload file**.
3. Select the `.dump` file from your local machine. The browser may not ask for a destination path; the file is then saved in the **current working directory** of the SSH session (often your home directory when you first open the session).
4. To find the file on the VM, run: `ls -la ~` or `find ~ -maxdepth 2 -name "*.dump"`.

**Option B — Use `scp` or `gcloud compute scp` from your local terminal**

- With `gcloud`: `gcloud compute scp PATH_TO_LOCAL_DUMP VM_USER@VM_NAME:PATH_ON_VM`
- With `scp`: `scp -i ~/.ssh/YOUR_KEY PATH_TO_LOCAL_DUMP VM_USER@VM_IP:PATH_ON_VM`

Example: `scp -i ~/.ssh/id_ed25519_vm /Users/yourname/Desktop/captions_db_2026-03-05_04-02-31.dump vm-access@35.208.244.178:~/`

Replace `PATH_TO_LOCAL_DUMP` with the path to the `.dump` file on your machine, `PATH_ON_VM` with the path on the VM (e.g. `~/`), and use the same VM user, key, and IP/name as in the download section above.

**Finding VM name, user, and IP**

- **Google Cloud Console:** Compute Engine → VM instances. The table shows **External IP** and the instance **Name** (use as `VM_NAME` for `gcloud`). The SSH user is the username that works when you SSH into the VM (e.g. from the SSH button or from `ssh -i ~/.ssh/your_key user@VM_IP`).
- **From your local machine (with gcloud):**  
  `gcloud compute instances list`  
  shows all VMs and external IPs. For one VM:  
  `gcloud compute instances describe VM_NAME --format='get(networkInterfaces[0].accessConfigs[0].natIP)'`
- **From inside the VM (e.g. in the browser SSH session):**  
  `curl -s ifconfig.me`  
  prints the VM’s public (external) IP. Use this as `VM_IP` for `scp`.

Once the dump is on your local machine, it is safe even if the VM is deleted.

### 3.3 Restore the backup

To restore the dump on the same or another machine, create an empty database and a database user, then run `pg_restore`. For step-by-step instructions, see [database_setup.md](database_setup.md) sections 6.3–6.5.

