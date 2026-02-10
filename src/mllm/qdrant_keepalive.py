#!/usr/bin/env python3
"""
Ping Qdrant to keep it active.

Intended for Qdrant Cloud: requires QDRANT_URL and (usually) QDRANT_API_KEY.
"""

import os
import sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient


def main() -> int:
    try:
        load_dotenv()
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")

        if not url:
            raise RuntimeError("QDRANT_URL is not set.")

        client = QdrantClient(url=url, api_key=api_key)
        collections = client.get_collections()
        print(f"Qdrant ping OK - {len(collections.collections)} collections")
        return 0
    except Exception as e:
        print(f"Qdrant keepalive ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

