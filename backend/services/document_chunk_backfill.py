"""
Backfill document_chunks table for full-text search.

Reprocesses completed documents so their chunks are stored in PostgreSQL.
Used by scripts/backfill_document_chunks.py and document_tasks.backfill_document_chunks_task.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any

from config import settings

logger = logging.getLogger(__name__)


async def fetch_documents_to_backfill(limit: int = 5000) -> List[Dict[str, Any]]:
    """Fetch document_id, user_id, collection_type, filename, folder_id for completed non-exempt documents."""
    from services.database_manager.database_helpers import fetch_all

    rls_context = {"user_id": "", "user_role": "admin"}
    rows = await fetch_all(
        """
        SELECT document_id, user_id, collection_type, filename, folder_id, doc_type
        FROM document_metadata
        WHERE processing_status = 'completed'
          AND (exempt_from_vectorization IS FALSE OR exempt_from_vectorization IS NULL)
        ORDER BY upload_date DESC
        LIMIT $1
        """,
        limit,
        rls_context=rls_context,
    )
    return [dict(r) for r in rows] if rows else []


async def backfill_one(doc: Dict[str, Any]) -> bool:
    """Reprocess one document so its chunks are written to document_chunks."""
    from services.service_container import get_service_container

    container = await get_service_container()
    document_service = container.document_service
    folder_service = container.folder_service

    doc_id = doc["document_id"]
    user_id = doc.get("user_id") or ""
    collection_type = doc.get("collection_type") or "user"
    filename = doc.get("filename") or ""
    folder_id = doc.get("folder_id")
    doc_type = doc.get("doc_type") or "txt"

    file_path = None
    try:
        folder_path = await folder_service.get_document_file_path(
            filename=filename,
            folder_id=folder_id,
            user_id=user_id,
            collection_type=collection_type,
        )
        if folder_path and Path(folder_path).exists():
            file_path = Path(folder_path)
        if not file_path or not file_path.exists():
            filename_with_id = f"{doc_id}_{filename}"
            folder_path = await folder_service.get_document_file_path(
                filename=filename_with_id,
                folder_id=folder_id,
                user_id=user_id,
                collection_type=collection_type,
            )
            if folder_path and Path(folder_path).exists():
                file_path = Path(folder_path)
    except Exception as e:
        logger.debug("Folder path resolution failed for %s: %s", doc_id, e)

    if not file_path or not file_path.exists():
        upload_dir = Path(settings.UPLOAD_DIR)
        for potential_file in upload_dir.rglob(f"{doc_id}_*"):
            if potential_file.is_file():
                file_path = potential_file
                break
        if not file_path and filename:
            for potential_file in upload_dir.rglob(filename):
                if potential_file.is_file():
                    file_path = potential_file
                    break

    if not file_path or not file_path.exists():
        logger.warning("File not found for document %s", doc_id)
        return False

    try:
        await document_service._process_document_async(
            doc_id, file_path, doc_type, user_id or None
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
