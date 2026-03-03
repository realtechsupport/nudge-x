"""
Export accepted captions for a specific pipeline run to TSV.

Usage:
    python -m mllm.export_captions --list-runs
    python -m mllm.export_captions <run_id>
    python -m mllm.export_captions <run_id> --output /path/to/output.tsv
"""

import argparse
import csv
import os
import re
import sys
from datetime import date
from pathlib import Path
from typing import Iterable, Tuple

from dotenv import load_dotenv

from database_pipeline.database_operations import connect_db


def list_runs(limit: int = 20) -> None:
    """Print recent pipeline runs."""
    conn = connect_db()
    if conn is None:
        print("Error: Could not connect to PostgreSQL.")
        sys.exit(1)

    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT run_id, created_at, prompt_version, model_name, num_shots
            FROM caption_pipeline_runs
            ORDER BY created_at DESC
            LIMIT %s;
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        if not rows:
            print("No pipeline runs found.")
            return

        print("Recent pipeline runs:")
        for run_id, created_at, prompt_version, model_name, num_shots in rows:
            print(
                f"  - {run_id} | {created_at} | prompt={prompt_version} | "
                f"model={model_name} | shots={num_shots}"
            )
    finally:
        cursor.close()
        conn.close()


def run_exists(run_id: str) -> bool:
    conn = connect_db()
    if conn is None:
        print("Error: Could not connect to PostgreSQL.")
        sys.exit(1)

    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM caption_pipeline_runs WHERE run_id = %s;",
            (run_id,),
        )
        return cursor.fetchone() is not None
    finally:
        cursor.close()
        conn.close()


def get_run_prompt_version(run_id: str) -> str | None:
    """Fetch the prompt_version stored for a given pipeline run."""
    conn = connect_db()
    if conn is None:
        return None

    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT prompt_version
            FROM caption_pipeline_runs
            WHERE run_id = %s;
            """,
            (run_id,),
        )
        row = cursor.fetchone()
        return row[0] if row else None
    finally:
        cursor.close()
        conn.close()


def fetch_captions(run_id: str) -> Iterable[Tuple]:
    """Fetch accepted captions for a run_id."""
    conn = connect_db()
    if conn is None:
        print("Error: Could not connect to PostgreSQL.")
        sys.exit(1)

    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT filename, mine_name, location, country, latitude, longitude, caption
            FROM captions
            WHERE run_id = %s
              AND is_accepted = TRUE
            ORDER BY created_at DESC;
            """,
            (run_id,),
        )
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()


def format_gps(latitude, longitude) -> str:
    if latitude is None or longitude is None:
        return "Unknown"
    return f"{latitude},{longitude}"


def write_tsv(rows: Iterable[Tuple], output_path: str) -> int:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            ["filename", "mine_name", "location", "country", "GPS_coordinates", "caption"]
        )
        count = 0
        for filename, mine_name, location, country, latitude, longitude, caption in rows:
            writer.writerow(
                [
                    filename,
                    mine_name or "",
                    location or "",
                    country or "",
                    format_gps(latitude, longitude),
                    caption or "",
                ]
            )
            count += 1
        return count


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Export accepted captions to TSV by pipeline run_id."
    )
    parser.add_argument("run_id", nargs="?", help="Pipeline run_id (UUID)")
    parser.add_argument("--output", help="Output TSV path (optional)")
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List recent pipeline runs and exit",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max runs to show with --list-runs (default: 20)",
    )

    args = parser.parse_args()

    if args.list_runs:
        list_runs(limit=args.limit)
        return

    if not args.run_id:
        print("Error: run_id is required unless --list-runs is used.")
        parser.print_help()
        sys.exit(2)

    run_id = args.run_id.strip()
    if not run_exists(run_id):
        print(f"Error: run_id '{run_id}' does not exist.")
        sys.exit(1)

    # Determine prompt version tag from DB
    raw_prompt_version = get_run_prompt_version(run_id) or ""
    prompt_version_norm = raw_prompt_version.strip().lower()
    if prompt_version_norm.startswith("v"):
        prompt_version_norm = prompt_version_norm[1:]
    prompt_tag = f"promptv{prompt_version_norm}" if prompt_version_norm else "prompt"

    # Determine metadata version tag from METADATA_TSV env var (e.g., Mines_Metadata_v28.tsv -> metadataV28)
    metadata_path = os.getenv("METADATA_TSV", "")
    meta_tag = "metadata"
    match = re.search(r"[Vv](\d+)", metadata_path)
    if match:
        meta_tag = f"metadataV{match.group(1)}"

    rows = fetch_captions(run_id)
    output_path = args.output
    if not output_path:
        short_id = run_id.split("-")[0]
        today = date.today().isoformat()
        # Resolve project root (two levels up from this file: src/mllm/...)
        project_root = Path(__file__).resolve().parents[2]
        default_dir = project_root / "data" / "frontend_captions"
        output_path = str(
            default_dir
            / f"captions_{prompt_tag}_{meta_tag}_{short_id}_{today}.tsv"
        )

    count = write_tsv(rows, output_path)
    print(f"Wrote {count} captions to {output_path}")


if __name__ == "__main__":
    main()
