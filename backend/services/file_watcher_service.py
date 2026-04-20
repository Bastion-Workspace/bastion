"""
File watcher stub for the backend.

The document library lives on document-service (./uploads is not mounted here).
Live watching and sidecar ingestion run in document-service (see document-service/ds_services/file_watcher_service.py).
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class _DisabledFileWatcher:
    """Placeholder when uploads are processed in document-service (no local watcher)."""

    running = False

    async def start(self) -> None:
        logger.info("File watcher disabled: document library is owned by document-service")

    async def stop(self) -> None:
        return None

    async def rescan_uploads(self, dry_run: bool = False) -> Dict[str, Any]:
        return {
            "success": False,
            "error": "File watcher disabled when document-service owns the upload tree",
            "new_count": 0,
            "scanned_count": 0,
        }


_file_watcher_instance = None


async def get_file_watcher():
    """Return the global file watcher (disabled stub — real watcher runs in document-service)."""
    global _file_watcher_instance

    if _file_watcher_instance is None:
        _file_watcher_instance = _DisabledFileWatcher()

    return _file_watcher_instance
