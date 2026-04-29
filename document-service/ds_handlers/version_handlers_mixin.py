"""Document version list, content, diff, rollback."""

import json
from uuid import UUID

from protos import document_service_pb2


class VersionHandlersMixin:
    async def GetDocumentVersions(
        self,
        request: document_service_pb2.JsonToolRequest,
        context,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_get_document_versions(request)

    async def GetVersionContent(
        self,
        request: document_service_pb2.JsonToolRequest,
        context,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_get_version_content(request)

    async def DiffVersions(
        self,
        request: document_service_pb2.JsonToolRequest,
        context,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_diff_versions(request)

    async def RollbackToVersion(
        self,
        request: document_service_pb2.JsonToolRequest,
        context,
    ) -> document_service_pb2.JsonToolResponse:
        return await self._json_rollback_to_version(request)

    async def _json_get_document_versions(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            from ds_services import document_version_service as dvs

            versions = await dvs.list_versions(
                p["document_id"],
                skip=int(p.get("skip", 0)),
                limit=int(p.get("limit", 100)),
                user_id=p.get("user_id") or request.user_id,
                collection_type=p.get("collection_type", "user"),
            )
            return document_service_pb2.JsonToolResponse(
                success=True, result_json=json.dumps({"versions": versions})
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_get_version_content(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            from ds_services import document_version_service as dvs

            content = await dvs.get_version_content(
                UUID(p["version_id"]),
                user_id=p.get("user_id") or request.user_id,
                collection_type=p.get("collection_type", "user"),
            )
            return document_service_pb2.JsonToolResponse(
                success=True, result_json=json.dumps({"content": content})
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_diff_versions(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            from ds_services import document_version_service as dvs

            result = await dvs.diff_versions(
                p["document_id"],
                UUID(p["from_version"]),
                UUID(p["to_version"]),
                user_id=p.get("user_id") or request.user_id,
                collection_type=p.get("collection_type", "user"),
            )
            return document_service_pb2.JsonToolResponse(
                success=result is not None, result_json=json.dumps(result or {})
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))

    async def _json_rollback_to_version(self, request: document_service_pb2.JsonToolRequest):
        try:
            p = json.loads(request.payload_json or "{}")
            from ds_services import document_version_service as dvs

            result = await dvs.rollback_to_version(
                p["document_id"], UUID(p["version_id"]), p.get("user_id") or request.user_id
            )
            return document_service_pb2.JsonToolResponse(
                success=bool(result.get("success")),
                result_json=json.dumps(result),
            )
        except Exception as e:
            return document_service_pb2.JsonToolResponse(success=False, error=str(e))
