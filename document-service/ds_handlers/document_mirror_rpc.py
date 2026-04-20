"""
gRPC DocumentMirror: JSON dispatch to ParallelDocumentService for backend facade parity.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict

from protos import document_service_pb2

logger = logging.getLogger(__name__)


def _jsonable_model(obj: Any) -> Any:
    """Serialize Pydantic v2/v1 models and primitives for JSON."""
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, dict):
        return {k: _jsonable_model(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable_model(x) for x in obj]
    return obj


async def handle_document_mirror_request(
    request: document_service_pb2.JsonToolRequest,
) -> document_service_pb2.JsonToolResponse:
    """
    Payload: {"action": "<name>", ...action-specific fields}
    user_id on the protobuf request is used when payload omits user_id.
    """
    try:
        payload = json.loads(request.payload_json or "{}")
    except json.JSONDecodeError as e:
        return document_service_pb2.JsonToolResponse(success=False, error=f"invalid json: {e}")

    action = (payload.get("action") or "").strip()
    if not action:
        return document_service_pb2.JsonToolResponse(success=False, error="missing action")

    uid = (payload.get("user_id") or request.user_id or "").strip()

    try:
        from shims.services.service_container import get_service_container
        from ds_models.api_models import (
            BulkCategorizeRequest,
            DocumentFilterRequest,
            DocumentUpdateRequest,
        )

        c = await get_service_container()
        ds = c.document_service

        result: Dict[str, Any] = {}

        if action == "get_document":
            doc = await ds.get_document(payload["document_id"])
            result = {"document": _jsonable_model(doc)}

        elif action == "check_document_exists":
            result = {"exists": await ds.check_document_exists(payload["doc_id"])}

        elif action == "list_documents":
            docs = await ds.list_documents(
                int(payload.get("skip", 0)),
                int(payload.get("limit", 100)),
            )
            result = {"documents": _jsonable_model(docs)}

        elif action == "filter_documents":
            fr = DocumentFilterRequest.model_validate(payload["filter_request"])
            resp = await ds.filter_documents(fr)
            result = {"response": _jsonable_model(resp)}

        elif action == "get_document_status":
            st = await ds.get_document_status(payload["doc_id"])
            result = {"status": _jsonable_model(st)}

        elif action == "update_document_metadata":
            ur = DocumentUpdateRequest.model_validate(payload["update_request"])
            ok = await ds.update_document_metadata(payload["document_id"], ur)
            result = {"success": bool(ok)}

        elif action == "bulk_categorize_documents":
            br = BulkCategorizeRequest.model_validate(payload["bulk_request"])
            resp = await ds.bulk_categorize_documents(br)
            result = {"response": _jsonable_model(resp)}

        elif action == "delete_document_database_only":
            ok = await ds.delete_document_database_only(payload["document_id"])
            result = {"success": bool(ok)}

        elif action == "cleanup_orphaned_embeddings":
            n = await ds.cleanup_orphaned_embeddings()
            result = {"cleaned_count": int(n)}

        elif action == "get_duplicate_documents":
            dup = await ds.get_duplicate_documents()
            out: Dict[str, Any] = {}
            for k, v in dup.items():
                if isinstance(v, list):
                    out[k] = _jsonable_model(v)
                else:
                    out[k] = v
            result = {"duplicates": out}

        elif action == "get_documents_stats":
            stats = await ds.get_documents_stats()
            result = {"stats": dict(stats) if stats else {}}

        elif action == "get_document_categories_overview":
            ov = await ds.get_document_categories_overview()
            result = {"overview": _jsonable_model(ov)}

        elif action == "import_from_url":
            up = await ds.import_from_url(
                payload["url"],
                payload.get("content_type") or "html",
            )
            result = {"upload": _jsonable_model(up)}

        elif action == "get_documents_with_hierarchy":
            data = await ds.get_documents_with_hierarchy(
                int(payload.get("limit", 100)),
                int(payload.get("offset", 0)),
            )
            result = {"data": _jsonable_model(data)}

        elif action == "get_zip_hierarchy":
            data = await ds.get_zip_hierarchy(payload["document_id"])
            result = {"data": _jsonable_model(data)}

        elif action == "delete_zip_with_children":
            data = await ds.delete_zip_with_children(
                payload["parent_document_id"],
                bool(payload.get("delete_children", True)),
            )
            result = {"data": _jsonable_model(data)}

        elif action == "process_url_async":
            asyncio.create_task(
                ds._process_url_async(
                    payload["document_id"],
                    payload["url"],
                    payload.get("content_type") or "html",
                    payload.get("user_id") or uid or None,
                )
            )
            result = {"queued": True}

        elif action == "resume_incomplete_processing":
            if hasattr(ds, "_resume_incomplete_processing"):
                await ds._resume_incomplete_processing()
            elif hasattr(ds, "_resume_incomplete_processing_sequential"):
                await ds._resume_incomplete_processing_sequential()
            result = {"resumed": True}

        elif action == "remove_team_upload_directory":
            from ds_services.folder_service import FolderService

            fs = FolderService()
            await fs.initialize()
            tid = str(payload.get("team_id") or "").strip()
            if not tid:
                return document_service_pb2.JsonToolResponse(
                    success=False, error="missing team_id"
                )
            ok = await fs.remove_team_upload_directory(tid)
            result = {"success": bool(ok)}

        else:
            return document_service_pb2.JsonToolResponse(
                success=False,
                error=f"unknown action: {action}",
            )

        return document_service_pb2.JsonToolResponse(
            success=True,
            result_json=json.dumps(result, default=str),
        )
    except Exception as e:
        logger.exception("DocumentMirror action=%s failed", action)
        return document_service_pb2.JsonToolResponse(success=False, error=str(e))
