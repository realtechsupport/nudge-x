"""One-shot backfill: derive `country_code` from `country` for existing Qdrant points.

Run after deploying the index-time normalization in vectorization_pipeline.py
to update points that were written before `country_code` existed.

Usage:
    python -m database_pipeline.backfill_country_code
    python -m database_pipeline.backfill_country_code --dry-run

Honors the same QDRANT_MODE env var as the rest of the pipeline.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from typing import Any

from dotenv import load_dotenv

from mllm.config.database_config import (
    QDRANT_COLLECTION_NAME,
    QDRANT_URL,
    QDRANT_API_KEY,
)
from database_pipeline.vector_db_operations import (
    create_qdrant_client,
    create_qdrant_client_testing,
    create_qdrant_client_api,
)
from rag.country_normalizer import to_iso2, iso2_to_name


SCROLL_BATCH = 256


def _build_client():
    qdrant_mode = os.getenv("QDRANT_MODE", "production").lower()
    if qdrant_mode == "testing":
        return create_qdrant_client_testing()
    if qdrant_mode == "api":
        return create_qdrant_client_api(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    return create_qdrant_client(host=qdrant_host, port=qdrant_port)


def _iter_points(client, collection_name: str):
    next_offset: Any = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=SCROLL_BATCH,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            yield p
        if next_offset is None:
            break


def backfill(collection_name: str, dry_run: bool = False) -> dict[str, int]:
    client = _build_client()

    seen = 0
    needs_update = 0
    skipped_already_set = 0
    unresolved: Counter[str] = Counter()
    pending_updates: list[tuple[Any, dict[str, Any]]] = []

    for point in _iter_points(client, collection_name):
        seen += 1
        payload = point.payload or {}

        existing_code = payload.get("country_code")
        country_raw = payload.get("country")

        if existing_code:
            skipped_already_set += 1
            continue

        if not country_raw:
            continue

        code = to_iso2(country_raw)
        if not code:
            unresolved[country_raw] += 1
            continue

        canonical_name = iso2_to_name(code) or country_raw
        new_payload = {"country_code": code, "country": canonical_name}
        pending_updates.append((point.id, new_payload))
        needs_update += 1

    print(f"Scanned {seen} points.")
    print(f"  already had country_code: {skipped_already_set}")
    print(f"  to update:                {needs_update}")
    if unresolved:
        print(f"  unresolved country values ({sum(unresolved.values())}):")
        for value, count in unresolved.most_common(20):
            print(f"    {count:>6} x {value!r}")

    if dry_run:
        print("Dry run — no writes performed.")
        return {
            "scanned": seen,
            "updated": 0,
            "skipped": skipped_already_set,
            "unresolved": sum(unresolved.values()),
        }

    for point_id, new_payload in pending_updates:
        client.set_payload(
            collection_name=collection_name,
            payload=new_payload,
            points=[point_id],
            wait=False,
        )

    print(f"Wrote country_code on {needs_update} points.")
    return {
        "scanned": seen,
        "updated": needs_update,
        "skipped": skipped_already_set,
        "unresolved": sum(unresolved.values()),
    }


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default=QDRANT_COLLECTION_NAME)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    backfill(collection_name=args.collection, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
