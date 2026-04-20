"""
File recovery: delegate scan/recover to document-service (library lives on DS volume).
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class FileRecoveryService:
    """Recover orphaned files via document-service ScanAndRecoverFiles."""

    async def scan_and_recover_user_files(
        self,
        user_id: str,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        try:
            from clients.document_service_client import get_document_service_client

            dsc = get_document_service_client()
            await dsc.initialize(required=True)
            ok, data, err = await dsc.scan_and_recover_json(
                user_id,
                {"user_id": user_id, "dry_run": dry_run},
            )
            if not ok:
                return {"success": False, "error": err or "Scan and recover failed"}
            if isinstance(data, dict):
                return data
            return {"success": True, "result": data}
        except Exception as e:
            logger.error("File recovery failed: %s", e)
            return {"success": False, "error": str(e)}


_file_recovery_service: Optional[FileRecoveryService] = None


async def get_file_recovery_service() -> FileRecoveryService:
    global _file_recovery_service
    if _file_recovery_service is None:
        _file_recovery_service = FileRecoveryService()
    return _file_recovery_service
