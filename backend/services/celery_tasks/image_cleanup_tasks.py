"""
Periodic cleanup of orphaned generated image files under web_sources/images.
"""

import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

from services.celery_app import celery_app, TaskStatus
from services.celery_tasks.async_runner import run_async

logger = logging.getLogger(__name__)

_GEN_PATTERN = re.compile(
    r"^gen_[a-fA-F0-9]+\.(?:png|jpg|jpeg|webp)$",
    re.IGNORECASE,
)


@celery_app.task(bind=True, name="services.celery_tasks.image_cleanup_tasks.cleanup_orphaned_generated_images_task")
def cleanup_orphaned_generated_images_task(self, days: int = 7) -> Dict[str, Any]:
    """
    Remove gen_* image files in web_sources/images older than ``days`` that have no
    document_metadata row (not promoted to the library).
    """

    try:
        logger.info(
            "Generated image cleanup: starting (older than %s days, no document record)",
            days,
        )

        async def _run() -> int:
            from config import settings
            from services.database_manager.database_helpers import fetch_all

            base = Path(settings.UPLOAD_DIR) / "web_sources" / "images"
            if not base.is_dir():
                return 0

            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            deleted = 0

            for entry in base.iterdir():
                if entry.is_dir():
                    continue
                if not entry.is_file() or not _GEN_PATTERN.match(entry.name):
                    continue
                try:
                    mtime = datetime.fromtimestamp(entry.stat().st_mtime, tz=timezone.utc)
                except OSError:
                    continue
                if mtime >= cutoff:
                    continue

                fname_lower = entry.name.lower()
                rows = await fetch_all(
                    """
                    SELECT document_id FROM document_metadata
                    WHERE doc_type = 'image' AND LOWER(filename) = $1
                    LIMIT 1
                    """,
                    fname_lower,
                )
                if rows:
                    continue

                try:
                    entry.unlink()
                    deleted += 1
                    logger.info("Removed orphaned generated image: %s", entry)
                    sidecar = entry.parent / f"{entry.stem}.metadata.json"
                    if sidecar.is_file():
                        try:
                            sidecar.unlink()
                        except OSError as se:
                            logger.warning("Could not remove sidecar %s: %s", sidecar, se)
                except OSError as e:
                    logger.warning("Failed to delete %s: %s", entry, e)

            return deleted

        deleted_count = run_async(_run())

        logger.info("Generated image cleanup: removed %s file(s)", deleted_count)
        return {
            "success": True,
            "task_id": self.request.id,
            "timestamp": datetime.now().isoformat(),
            "deleted_count": deleted_count,
            "days": days,
            "message": f"Removed {deleted_count} orphaned generated image file(s)",
        }

    except Exception as e:
        logger.error("Generated image cleanup task failed: %s", e)
        self.update_state(
            state=TaskStatus.FAILURE,
            meta={
                "error": str(e),
                "message": "Generated image cleanup failed",
                "timestamp": datetime.now().isoformat(),
            },
        )
        return {
            "success": False,
            "error": str(e),
            "message": "Background generated image cleanup failed",
        }
