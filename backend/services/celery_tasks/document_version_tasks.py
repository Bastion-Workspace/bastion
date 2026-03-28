"""
Document version retention pruning.
Runs periodically via Celery Beat to prune old versions per retention policy.
"""

import logging
from datetime import datetime
from typing import Dict, Any

from services.celery_app import celery_app, TaskStatus
from services.celery_tasks.async_runner import run_async

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="services.celery_tasks.document_version_tasks.prune_document_versions_task")
def prune_document_versions_task(
    self,
    retention_days: int = 90,
    keep_every_nth: int = 10,
    max_versions_per_document: int = 200,
) -> Dict[str, Any]:
    """Prune old document versions across all documents. Keeps first version, current, recent, and every Nth older."""
    try:
        logger.info("Starting document version pruning (retention_days=%s, keep_every_nth=%s)", retention_days, keep_every_nth)

        async def _run():
            from repositories import document_version_repository as version_repo
            from services.document_version_service import prune_old_versions

            doc_ids = await version_repo.get_document_ids_with_versions()
            total_pruned = 0
            total_kept = 0
            for doc_id in doc_ids:
                result = await prune_old_versions(
                    doc_id,
                    retention_days=retention_days,
                    keep_every_nth=keep_every_nth,
                    max_versions=max_versions_per_document,
                    user_id=None,
                    collection_type="user",
                )
                total_pruned += result.get("pruned", 0)
                total_kept += result.get("kept", 0)
            return total_pruned, total_kept

        total_pruned, total_kept = run_async(_run())

        logger.info("Document version pruning: pruned=%s, kept=%s", total_pruned, total_kept)
        return {
            "success": True,
            "task_id": self.request.id,
            "timestamp": datetime.now().isoformat(),
            "pruned_count": total_pruned,
            "kept_count": total_kept,
            "message": f"Pruned {total_pruned} old version(s), kept {total_kept}",
        }
    except Exception as e:
        logger.exception("Document version pruning failed: %s", e)
        self.update_state(
            state=TaskStatus.FAILURE,
            meta={
                "error": str(e),
                "message": "Document version pruning failed",
                "timestamp": datetime.now().isoformat(),
            },
        )
        return {
            "success": False,
            "error": str(e),
            "message": "Document version pruning failed",
        }
