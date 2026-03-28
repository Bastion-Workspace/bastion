"""
Backfill document_chunks table for full-text search.

Reprocesses completed documents so their chunks are stored in PostgreSQL
(document_chunks). Safe to run multiple times; existing chunks are replaced
per document.

Usage (from backend container or with PYTHONPATH=backend):
    python scripts/backfill_document_chunks.py
    python scripts/backfill_document_chunks.py --batch 50 --delay 1.0
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    from services.document_chunk_backfill import run_backfill

    parser = argparse.ArgumentParser(description="Backfill document_chunks for full-text search")
    parser.add_argument("--batch", type=int, default=100, help="Batch size for progress logging")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay in seconds between documents")
    parser.add_argument("--limit", type=int, default=5000, help="Max documents to process")
    args = parser.parse_args()

    async def _run():
        logger.info("Starting backfill (limit=%s, delay=%s)...", args.limit, args.delay)
        result = await run_backfill(batch_size=args.batch, delay=args.delay, limit=args.limit)
        logger.info("Backfill complete: %s", result)
        return result

    asyncio.run(_run())


if __name__ == "__main__":
    main()
