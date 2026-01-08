import psycopg2
import csv
import os
import argparse
from datetime import datetime

# Database connection info
DB_NAME = "captions_db"
DB_USER = "Admin1"
DB_PASSWORD = "einstein"
DB_HOST = "localhost"
DB_PORT = 5432

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Export captions for a specific pipeline run")
parser.add_argument("--run_id", required=True, help="UUID of the pipeline run to export")
args = parser.parse_args()

# Make dated filename with run_id
today = datetime.today().strftime("%Y-%m-%d")
OUTPUT_DIR = "data"
# Use short run_id (first 8 chars) for filename
run_id_short = args.run_id[:8]
OUTPUT_FILE = f"{OUTPUT_DIR}/frontend_captions_{run_id_short}_{today}.tsv"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    print("Connecting to database...")
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cur = conn.cursor()

    print(f"Running SELECT query for run_id: {args.run_id}")
    cur.execute("""
        SELECT id, filename, mine_name, location AS site_location, country,
               CASE WHEN latitude IS NOT NULL AND longitude IS NOT NULL 
                    THEN latitude || ',' || longitude ELSE 'Unknown' END AS GPS_coordinates,
               caption
        FROM captions
        WHERE run_id = %s AND is_accepted = TRUE
        ORDER BY created_at DESC
    """, (args.run_id,))

    rows = cur.fetchall()
    headers = [desc[0] for desc in cur.description]

    print(f"Writing {len(rows)} rows to {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(headers)
        writer.writerows(rows)

    print("Done! File saved as:", OUTPUT_FILE)

except Exception as e:
    print("Error:", e)

finally:
    if 'cur' in locals():
        cur.close()
    if 'conn' in locals():
        conn.close()
