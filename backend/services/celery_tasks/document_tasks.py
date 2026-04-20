"""
Document Celery Tasks
Background reprocessing after document content save so the save response returns immediately.
"""

import logging
from typing import Dict, Any, List, Optional

from services.celery_app import celery_app
from services.celery_tasks.async_runner import run_async

logger = logging.getLogger(__name__)


async def _async_reprocess_document_after_save(doc_id: str, user_id: str) -> Dict[str, Any]:
    """
    Resolve document file path and run full reprocess (re-embed + entity extraction).
    Called from Celery after content has been written to disk by update_document_content.

    Always reads user_id, team_id, and collection_type from the DB row so that
    Qdrant collection routing is correct regardless of what user_id was in the task args.
    """
    from services.service_container import get_service_container

    container = await get_service_container()
    document_service = container.document_service

    doc_info = await document_service.get_document(doc_id)
    if not doc_info:
        logger.warning(f"Document not found for reprocess: {doc_id}")
        return {"success": False, "error": "Document not found", "document_id": doc_id}

    # The task caller's user_id is only used as a fallback when the row has no owner.
    path_user_id = getattr(doc_info, "user_id", None) or user_id

    from clients.document_service_client import DocumentServiceClient

    client = DocumentServiceClient()
    try:
        await client.initialize(required=True)
        await client.reprocess_via_document_service(
            doc_id, path_user_id, force_reprocess=True
        )
    finally:
        await client.close()
    return {"success": True, "document_id": doc_id}


@celery_app.task(bind=True, name="services.celery_tasks.document_tasks.reprocess_document_after_save")
def reprocess_document_after_save_task(self, doc_id: str, user_id: str) -> Dict[str, Any]:
    """
    Celery task: run vector re-embedding and entity extraction after document save.
    Save API returns immediately after writing to disk; this task runs in the background.
    """
    try:
        logger.info(f"Document reprocess task started: {doc_id}")
        result = run_async(_async_reprocess_document_after_save(doc_id, user_id))
        logger.info(f"Document reprocess task completed: {doc_id}")
        return result
    except Exception as e:
        logger.error(f"Document reprocess task failed: {doc_id} - {e}")
        return {
            "success": False,
            "error": str(e),
            "document_id": doc_id,
            "message": "Background re-indexing failed",
        }


@celery_app.task(bind=True, name="services.celery_tasks.document_tasks.bulk_reindex_batch")
def bulk_reindex_batch_task(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Celery task: process a batch of documents for bulk re-embedding.

    Each item in batch is a dict with keys: document_id, user_id, team_id, collection_type.
    These values come directly from document_metadata rows so collection routing is always
    authoritative. Processes documents sequentially within the batch and reports PROGRESS
    state after each document so Flower/callers can monitor throughput.

    Routed to the dedicated 'reindex' queue to keep bulk operations isolated from
    user-facing agent and chat tasks on the 'default' queue.
    """
    total = len(batch)
    results: Dict[str, Any] = {"success": 0, "failed": 0, "skipped": 0, "errors": []}

    for i, doc in enumerate(batch):
        self.update_state(
            state="PROGRESS",
            meta={"current": i, "total": total, "success": results["success"], "failed": results["failed"]},
        )
        doc_id = doc.get("document_id", "")
        if not doc_id:
            results["skipped"] += 1
            continue
        try:
            result = run_async(_async_reprocess_document_after_save(doc_id, doc.get("user_id", "")))
            if result.get("success"):
                results["success"] += 1
            else:
                results["failed"] += 1
                results["errors"].append({"document_id": doc_id, "error": result.get("error", "unknown")})
        except Exception as e:
            logger.error(f"Bulk reindex failed for document {doc_id}: {e}")
            results["failed"] += 1
            results["errors"].append({"document_id": doc_id, "error": str(e)})

    logger.info(
        "Bulk reindex batch complete: %d success, %d failed, %d skipped out of %d",
        results["success"], results["failed"], results["skipped"], total,
    )
    return results


async def _async_backfill_document_chunks(limit: int = 5000, batch_size: int = 50, delay: float = 0.3) -> Dict[str, Any]:
    """Backfill document_chunks table by reprocessing completed documents."""
    from services.document_chunk_backfill import run_backfill

    return await run_backfill(limit=limit, batch_size=batch_size, delay=delay)


@celery_app.task(bind=True, name="services.celery_tasks.document_tasks.backfill_document_chunks")
def backfill_document_chunks_task(self, limit: int = 5000, batch_size: int = 50, delay: float = 0.3) -> Dict[str, Any]:
    """
    Celery task: backfill document_chunks for full-text search by reprocessing completed documents.
    Safe to run multiple times.
    """
    try:
        logger.info("Backfill document_chunks task started (limit=%s)", limit)
        result = run_async(_async_backfill_document_chunks(limit=limit, batch_size=batch_size, delay=delay))
        logger.info("Backfill document_chunks task completed: %s", result)
        return result
    except Exception as e:
        logger.error("Backfill document_chunks task failed: %s", e)
        return {"success": False, "error": str(e)}
