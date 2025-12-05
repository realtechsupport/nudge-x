import psycopg2
import csv
import os
from datetime import datetime

# Database connection info
DB_NAME = "captions_db"
DB_USER = "Admin1"
DB_PASSWORD = "einstein"
DB_HOST = "localhost"
DB_PORT = 5432

# Make dated filename
today = datetime.today().strftime("%Y-%m-%d")
OUTPUT_DIR = "data"
OUTPUT_FILE = f"{OUTPUT_DIR}/frontend_captions_{today}.tsv"

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

    print("Running SELECT query...")
    cur.execute("SELECT * FROM frontend_captions")

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
