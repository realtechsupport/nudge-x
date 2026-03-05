# PostgreSQL Database Setup

This guide explains how to set up the PostgreSQL database for the nudge-x captions pipeline.

The approach: each person installs PostgreSQL on their own VM, creates `captions_db`, and runs the app locally. If someone already has data (pipeline runs, captions, embeddings), they can export a dump file and share it so others can import the same data into their own database.

---

## 1. Install PostgreSQL on your VM (Ubuntu)

```bash
sudo apt update
sudo apt install -y postgresql postgresql-contrib
```

Start PostgreSQL and enable it on boot:

```bash
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

Verify it is running:

```bash
sudo systemctl status postgresql
```

You should see `active (running)`.

---

## 2. Create the database and user

Switch to the `postgres` system user and open the PostgreSQL shell :(use below command)

```bash
sudo -u postgres psql
```

You will see postgres=# 

Run the following SQL commands. Replace `db_user` and `db_password` with the username and password you will use in your `.env` file (PostgreSQL lowercases unquoted names; use lowercase for the username to avoid confusion):

```sql
-- Create the application user
CREATE USER db_user WITH PASSWORD 'db_password';

-- Create the database owned by that user
CREATE DATABASE captions_db OWNER db_user;

-- Grant full access
GRANT ALL PRIVILEGES ON DATABASE captions_db TO db_user;

-- Switch into the new database
\c captions_db

-- Grant schema-level access (required for table creation)
GRANT ALL ON SCHEMA public TO db_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO db_user;

-- Enable pgcrypto so gen_random_uuid() works (used by the app for run IDs)
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Exit psql
\q
```

---

## 3. Verify the connection

Test that you can connect from the command line with the database user you just created:

```bash
psql -h localhost -U db_user -d captions_db -W
```

Enter the password when prompted. If you see the `captions_db=>` prompt, the connection works. Type `\q` to exit.

If you get "peer authentication failed", edit `pg_hba.conf` to allow password auth for local TCP connections:

```bash
sudo nano /etc/postgresql/*/main/pg_hba.conf
```

Find the line for IPv4 local connections and ensure it says:

```
host    all    all    127.0.0.1/32    scram-sha-256
```

Save, then restart PostgreSQL:

```bash
sudo systemctl restart postgresql
```

Try the `psql` connection again.

---

## 4. Configure `.env`

In the project root (`nudge-x`), create or edit `.env` and set the database variables to match what you created in step 2:

```env
POSTGRES_DB=captions_db
POSTGRES_USER=db_user
POSTGRES_PASSWORD=db_password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
```

Plus all other required variables (Qdrant, API keys, `IMAGE_DIR`, `METADATA_TSV`, `PROMPT_VERSION`, etc.) — see `.env.example` for the full list and [setup_and_env.md](setup_and_env.md) for descriptions.

---

## 5. Run the app (tables are created automatically)

The app creates all tables on first run. From the project root:

```bash
cd src
python -m mllm.main.captions_pipeline
```

On the first run, `create_table_if_not_exists()` will create:

- `caption_pipeline_runs` — pipeline run metadata (run ID, prompt version, model, etc.)
- `captions` — generated captions with image metadata
- `caption_embeddings` — tracks which captions have been vectorized
- `frontend_captions_view` — a read-only view of accepted captions

You can verify in psql:

```bash
psql -h localhost -U your_user -d captions_db -W
```

```sql
\dt
```

You should see the three tables listed. Type `\q` to exit.

---

## 6. Sharing data: export a dump and import on another VM

If you have data (pipeline runs, captions) and want to provide it to another person or VM, export a dump, transfer it, then restore it on the destination VM.

### 6.1 On the source VM (the one with data): create a dump

```bash
DUMP_NAME="captions_db_$(date +%F_%H-%M-%S).dump"
pg_dump -h localhost -U DB_USER -d captions_db -F c -f "$DUMP_NAME"
```

- Replace `DB_USER` with the database user from your `.env`.
- This creates a file like `captions_db_2026-03-05_03-58-34.dump` in the current directory.
- `-F c` = custom format (compressed; use with `pg_restore` when restoring).
- Enter the database password when prompted.

### 6.2 Transfer the dump to the destination VM or to your local machine

Two common methods:

**Option A — Download via the browser SSH window (VM to local machine)**

1. In Google Cloud Console, open the source VM with the **SSH** button.
2. In the SSH window toolbar (three dots or gear icon), choose **Download file**.
3. When prompted, enter the path to the dump file on the VM (the path you used in step 6.1).
4. The file is downloaded to your local machine. You can then upload or copy it to the destination VM if needed.

**Option B — Use `scp` or `gcloud compute scp` from your local machine**

From a local terminal:

- With `gcloud`: `gcloud compute scp VM_NAME:PATH_TO_DUMP_FILE PATH_TO_LOCAL_OR_DESTINATION`
- With `scp`: `scp -i ~/.ssh/YOUR_KEY VM_USER@VM_IP:PATH_TO_DUMP_FILE PATH_TO_LOCAL_OR_DESTINATION`  
  Omit `-i ~/.ssh/YOUR_KEY` if your default SSH key is used for this VM. If the VM was set up with a dedicated key, use `-i` with that key path; you will be prompted for the key’s passphrase unless you add it to `ssh-agent` (`ssh-add ~/.ssh/YOUR_KEY`).

Replace `PATH_TO_DUMP_FILE` with the path to the dump on the VM (e.g. `captions_db_2026-03-05_03-58-34.dump` if you used the timestamped name in step 6.1), `PATH_TO_LOCAL_OR_DESTINATION` with the destination path, and use the VM’s name, SSH user, and IP as below. Use the **same VM username** you use when logging in via SSH (e.g. `vm-access`), not necessarily your local machine username.

**Copy from local machine to VM**

When the dump is on your local machine and you need it on a VM (e.g. to run `pg_restore` there), you can upload it directly in the browser or use the command line.

**Option A — Upload via the browser SSH window**

1. In Google Cloud Console, open the **destination VM** with the **SSH** button.
2. In the SSH window toolbar (three dots or gear icon), choose **Upload file**.
3. Select the `.dump` file from your local machine. The browser may not ask for a destination path; the file is then saved in the **current working directory** of the SSH session (often your home directory when you first open the session).
4. To find the file on the VM, run: `ls -la ~` or `find ~ -maxdepth 2 -name "*.dump"`. Use the path you see (e.g. `~/captions_db_2026-03-05_04-02-31.dump`) in the `pg_restore` command in step 6.4.

**Option B — Use `scp` or `gcloud compute scp` from your local terminal**

- With `gcloud`: `gcloud compute scp PATH_TO_LOCAL_DUMP VM_USER@VM_NAME:PATH_ON_VM`
- With `scp`: `scp -i ~/.ssh/YOUR_KEY PATH_TO_LOCAL_DUMP VM_USER@VM_IP:PATH_ON_VM`

Example (dump from Desktop to the VM user’s home directory):  
`scp -i ~/.ssh/id_ed25519_vm /Users/yourname/Desktop/captions_db_2026-03-05_04-02-31.dump vm-access@35.208.244.178:~/`

Replace `PATH_TO_LOCAL_DUMP` with the path to the `.dump` file on your machine, `PATH_ON_VM` with the path where it should go on the VM (e.g. `~/` or `~/captions_db_2026-03-05_04-02-31.dump`), and use the same VM user, key, and IP/name as above.

**Finding VM name, user, and IP**

- **Google Cloud Console:** Compute Engine → VM instances. The table shows **External IP** and the instance **Name** (use as `VM_NAME` for `gcloud`). The SSH user is the username that works when you SSH into the VM (e.g. from the SSH button or from `ssh -i ~/.ssh/your_key user@VM_IP`).
- **From your local machine (with gcloud):**  
  `gcloud compute instances list`  
  shows all VMs and external IPs. For one VM:  
  `gcloud compute instances describe VM_NAME --format='get(networkInterfaces[0].accessConfigs[0].natIP)'`
- **From inside the VM (e.g. in the browser SSH session):**  
  `curl -s ifconfig.me`  
  prints the VM’s public (external) IP. Use this as `VM_IP` for `scp`.

For more detail on creating and downloading backups, see [export_and_maintenance.md](export_and_maintenance.md) section 3.

### 6.3 On the destination VM: install Postgres and create the database

Follow steps 1–4 above on the destination VM (install Postgres, create a database user and `captions_db`, set `.env`). The destination can use different credentials from the source; only the dump file is shared.

Ensure `pgcrypto` is enabled (step 2 includes this).

### 6.4 On the destination VM: restore the dump

```bash
pg_restore -h localhost -U DESTINATION_DB_USER -d captions_db --no-owner --no-privileges PATH_TO_DUMP_FILE
```

- Replace `DESTINATION_DB_USER` with the database user created on the destination VM.
- Replace `PATH_TO_DUMP_FILE` with the **path to the dump file on this machine** (the VM where you are running `pg_restore`). For example, if the file is in your home directory: `~/captions_db_2026-03-05_04-02-31.dump`; if it’s in the project folder: `/home/your_user/nudge-x/captions_db_2026-03-05_04-02-31.dump`.
- `--no-owner` and `--no-privileges` avoid errors when the original roles do not exist on the destination.
- Enter the destination database password when prompted.

Warnings such as `role "..." does not exist` are expected and can be ignored.

### 6.5 Verify the data was imported

```bash
psql -h localhost -U DESTINATION_DB_USER -d captions_db -W
```

```sql
\dt
SELECT COUNT(*) FROM caption_pipeline_runs;
SELECT COUNT(*) FROM captions;
SELECT COUNT(*) FROM caption_embeddings;
\q
```

The counts should match the source. The destination VM can then run the pipelines and export scripts against this database.

---

## 7. Keeping databases in sync (manual)

Each VM has its own independent database. To share updated data with another VM:

1. Create a fresh dump on the source VM (step 6.1).
2. Transfer the dump (step 6.2).
3. On the destination VM, drop and recreate the database, then restore:

```bash
# Drop and recreate (WARNING: deletes all existing data in captions_db on this VM)
sudo -u postgres psql -c "DROP DATABASE IF EXISTS captions_db;"
sudo -u postgres psql -c "CREATE DATABASE captions_db OWNER DESTINATION_DB_USER;"
sudo -u postgres psql -d captions_db -c "CREATE EXTENSION IF NOT EXISTS pgcrypto;"

# Restore the dump
pg_restore -h localhost -U DESTINATION_DB_USER -d captions_db --no-owner --no-privileges PATH_TO_DUMP_FILE
```

Replace `DESTINATION_DB_USER` with the database user on this VM. Replace `PATH_TO_DUMP_FILE` with the path to the `.dump` file on this machine (e.g. `~/captions_db_2026-03-05_04-02-31.dump`). This is a manual process. For automatic sync or multiple users writing to the same database, use a shared cloud database (e.g. Neon, Supabase) and point `.env` at it.

---

## 8. Backing up your database

For creating backups and downloading the dump to your local machine (including the two transfer methods: browser Download file and scp), see [export_and_maintenance.md](export_and_maintenance.md) section 3. To restore a backup, follow step 6.4 above.

---

## Quick reference

| Task           | Command                                                                                       |
| -------------- | --------------------------------------------------------------------------------------------- |
| Start Postgres | `sudo systemctl start postgresql`                                                             |
| Stop Postgres  | `sudo systemctl stop postgresql`                                                              |
| Connect to DB  | `psql -h localhost -U DB_USER -d captions_db -W`                                              |
| List tables    | `\dt` (inside psql)                                                                           |
| Count captions | `SELECT COUNT(*) FROM captions;`                                                               |
| Create dump    | `DUMP_NAME="captions_db_$(date +%F_%H-%M-%S).dump"; pg_dump -h localhost -U DB_USER -d captions_db -F c -f "$DUMP_NAME"` |
| Restore dump   | `pg_restore -h localhost -U DESTINATION_DB_USER -d captions_db --no-owner --no-privileges PATH_TO_DUMP_FILE` |
| Postgres version | `psql --version`                                                                            |


For environment variable details, see [setup_and_env.md](setup_and_env.md).
