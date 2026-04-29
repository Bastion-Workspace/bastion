"""Upload-relative file paths and file recovery scan."""

import json
from pathlib import Path

from protos import document_service_pb2


class FileIoHandlersMixin:
    async def ReadUploadRelativeFile(
        self,
        request: document_service_pb2.JsonToolRequest,
        context,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_read_upload_relative_file(request)

    async def WriteUploadRelativeFile(
        self,
        request: document_service_pb2.JsonToolRequest,
        context,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_write_upload_relative_file(request)

    async def ListUploadRelativeDir(
        self,
        request: document_service_pb2.JsonToolRequest,
        context,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_list_upload_relative_dir(request)

    async def ScanAndRecoverFiles(
        self,
        request: document_service_pb2.JsonToolRequest,
        context,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_scan_recover(request)

    def _safe_path_under_upload(self, rel_path: str) -> Path:
        from ds_config import settings

        root = Path(settings.UPLOAD_DIR).resolve()
        p = Path(rel_path or "")
        if p.is_absolute():
            raise ValueError("absolute path not allowed")
        full = (root / p).resolve()
        try:
            full.relative_to(root)
        except ValueError as e:
            raise ValueError("path escapes upload root") from e
        return full

    async def _json_read_upload_relative_file(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            fp = self._safe_path_under_upload(p["rel_path"])
            if not fp.is_file():
                return document_service_pb2.JsonToolResponse(success=False, error="not found")
            text = fp.read_text(encoding="utf-8")
            return document_service_pb2.JsonToolResponse(
                success=True, result_json=json.dumps({"content": text})
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_write_upload_relative_file(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            fp = self._safe_path_under_upload(p["rel_path"])
            if p.get("delete"):
                if not fp.is_file():
                    return document_service_pb2.JsonToolResponse(success=False, error="not found")
                fp.unlink()
                return document_service_pb2.JsonToolResponse(
                    success=True, result_json=json.dumps({"ok": True, "deleted": True})
                )
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(p.get("content", ""), encoding="utf-8")
            return document_service_pb2.JsonToolResponse(
                success=True, result_json=json.dumps({"ok": True})
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_list_upload_relative_dir(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            dp = self._safe_path_under_upload(p.get("rel_path", ""))
            if not dp.is_dir():
                return document_service_pb2.JsonToolResponse(success=False, error="not a directory")
            names = sorted([x.name for x in dp.iterdir()])
            return document_service_pb2.JsonToolResponse(
                success=True, result_json=json.dumps({"entries": names})
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_scan_recover(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            from ds_services.file_recovery_service import FileRecoveryService

            svc = FileRecoveryService()
            result = await svc.scan_and_recover_user_files(
                p.get("user_id") or request.user_id,
                dry_run=p.get("dry_run", False),
            )
            return document_service_pb2.JsonToolResponse(
                success=True, result_json=json.dumps(result)
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))
