"""
Expired document edit proposal cleanup.
Runs periodically via Celery Beat to delete proposals past expires_at.
"""

import logging
from datetime import datetime
from typing import Dict, Any

from services.celery_app import celery_app, TaskStatus
from services.celery_tasks.async_runner import run_async

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="services.celery_tasks.proposal_cleanup_tasks.cleanup_expired_proposals")
def cleanup_expired_proposals(self) -> Dict[str, Any]:
    """Delete document_edit_proposals where expires_at < NOW(). Uses DB function to bypass RLS."""
    try:
        logger.info("Starting cleanup of expired document edit proposals")

        async def _run():
            from services.database_manager.database_helpers import fetch_value
            return await fetch_value("SELECT cleanup_expired_document_edit_proposals()")

        deleted = run_async(_run())

        deleted = int(deleted) if deleted is not None else 0
        logger.info("Cleanup expired proposals: deleted %s row(s)", deleted)
        return {
            "success": True,
            "task_id": self.request.id,
            "timestamp": datetime.now().isoformat(),
            "deleted_count": deleted,
            "message": f"Cleaned up {deleted} expired proposal(s)",
        }
    except Exception as e:
        logger.exception("Cleanup expired proposals failed: %s", e)
        self.update_state(
            state=TaskStatus.FAILURE,
            meta={
                "error": str(e),
                "message": "Expired proposal cleanup failed",
                "timestamp": datetime.now().isoformat(),
            },
        )
        return {
            "success": False,
            "error": str(e),
            "message": "Expired proposal cleanup failed",
        }
