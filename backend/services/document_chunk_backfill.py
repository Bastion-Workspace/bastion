"""
Backfill document_chunks table for full-text search.

Reprocesses completed documents so their chunks are stored in PostgreSQL.
Uses the same eligibility rules and index-freshness columns as document-service
(idempotent default reprocess). Image sidecar / watcher flows are separate.

Used by scripts/backfill_document_chunks.py and document_tasks.backfill_document_chunks_task.
"""

import asyncio
import logging
from typing import Any, Dict, List

from bastion_indexing.policy import (
    APP_CHUNK_INDEX_SCHEMA_VERSION,
    sql_eligible_doc_types_tuple,
)

logger = logging.getLogger(__name__)


async def fetch_documents_to_backfill(limit: int = 5000) -> List[Dict[str, Any]]:
    """Fetch rows needing primary chunk index repair or initial chunk persistence."""
    from services.database_manager.database_helpers import fetch_all

    rls_context = {"user_id": "", "user_role": "admin"}
    eligible = list(sql_eligible_doc_types_tuple())
    rows = await fetch_all(
        """
        SELECT document_id, user_id, collection_type, filename, folder_id, doc_type
        FROM document_metadata
        WHERE processing_status = 'completed'
          AND (exempt_from_vectorization IS FALSE OR exempt_from_vectorization IS NULL)
          AND doc_type = ANY($2::text[])
          AND NOT (LOWER(doc_type) = 'zip' AND COALESCE(is_zip_container, false))
          AND (
            chunk_indexed_at IS NULL
            OR chunk_indexed_file_hash IS DISTINCT FROM file_hash
            OR COALESCE(chunk_index_schema_version, 0) < $3
            OR NOT EXISTS (
                SELECT 1 FROM document_chunks dc WHERE dc.document_id = document_metadata.document_id
            )
          )
        ORDER BY upload_date DESC
        LIMIT $1
        """,
        limit,
        eligible,
        APP_CHUNK_INDEX_SCHEMA_VERSION,
        rls_context=rls_context,
    )
    return [dict(r) for r in rows] if rows else []


async def reconcile_legacy_chunk_index_rows() -> None:
    """
    Optional one-shot: stamp chunk_indexed_* for completed eligible documents that
    already have document_chunks rows but never received freshness columns (pre-migration).
    """
    from services.database_manager.database_helpers import execute

    eligible = list(sql_eligible_doc_types_tuple())
    await execute(
        """
        UPDATE document_metadata dm
        SET chunk_indexed_at = CURRENT_TIMESTAMP,
            chunk_indexed_file_hash = COALESCE(dm.file_hash, ''),
            chunk_index_schema_version = $1,
            updated_at = CURRENT_TIMESTAMP
        WHERE dm.processing_status = 'completed'
          AND (dm.exempt_from_vectorization IS FALSE OR dm.exempt_from_vectorization IS NULL)
          AND dm.doc_type = ANY($2::text[])
          AND NOT (LOWER(dm.doc_type) = 'zip' AND COALESCE(dm.is_zip_container, false))
          AND dm.chunk_indexed_at IS NULL
          AND EXISTS (SELECT 1 FROM document_chunks dc WHERE dc.document_id = dm.document_id)
        """,
        APP_CHUNK_INDEX_SCHEMA_VERSION,
        eligible,
        rls_context={"user_id": "", "user_role": "admin"},
    )


async def backfill_one(doc: Dict[str, Any]) -> bool:
    """Reprocess one document so its chunks are written to document_chunks (via document-service)."""
    from clients.document_service_client import get_document_service_client

    doc_id = doc["document_id"]
    user_id = doc.get("user_id") or ""

    try:
        dsc = get_document_service_client()
        await dsc.initialize(required=True)
        await dsc.reprocess_via_document_service(
            doc_id, user_id or None, force_reprocess=False
        )
        return True
    except Exception as e:
        logger.warning("Reprocess failed for %s: %s", doc_id, e)
        return False


async def run_backfill(
    batch_size: int = 100,
    delay: float = 0.5,
    limit: int = 5000,
) -> Dict[str, Any]:
    """Backfill document_chunks by reprocessing documents. Returns counts."""
    docs = await fetch_documents_to_backfill(limit=limit)
    total = len(docs)
    if total == 0:
        return {"success": True, "total": 0, "ok": 0, "failed": 0}

    ok = 0
    fail = 0
    for i, doc in enumerate(docs):
        if await backfill_one(doc):
            ok += 1
        else:
            fail += 1
        if delay > 0:
            await asyncio.sleep(delay)
        if (i + 1) % batch_size == 0:
            logger.info("Backfill progress: %s/%s (ok=%s, fail=%s)", i + 1, total, ok, fail)

    return {"success": True, "total": total, "ok": ok, "failed": fail}
