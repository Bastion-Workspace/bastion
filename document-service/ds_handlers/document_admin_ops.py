"""
Shared document admin operations (ParallelDocumentService + FolderService).
Used by DocumentMirror JSON dispatch and typed DocumentAdmin gRPC RPCs.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def jsonable_model(obj: Any) -> Any:
    """Serialize Pydantic v2/v1 models and primitives for JSON."""
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, dict):
        return {k: jsonable_model(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [jsonable_model(x) for x in obj]
    return obj


async def _ds():
    from shims.services.service_container import get_service_container

    c = await get_service_container()
    return c.document_service


async def op_get_document(payload: Dict[str, Any], uid: str) -> Dict[str, Any]:
    ds = await _ds()
    doc = await ds.get_document(payload["document_id"])
    return {"document": jsonable_model(doc)}


async def op_check_document_exists(payload: Dict[str, Any], uid: str) -> Dict[str, Any]:
    ds = await _ds()
    return {"exists": await ds.check_document_exists(payload["doc_id"])}


async def op_list_documents(payload: Dict[str, Any], uid: str) -> Dict[str, Any]:
    ds = await _ds()
    docs = await ds.list_documents(
        int(payload.get("skip", 0)),
        int(payload.get("limit", 100)),
    )
    return {"documents": jsonable_model(docs)}


async def op_filter_documents(payload: Dict[str, Any], uid: str) -> Dict[str, Any]:
    from ds_models.api_models import DocumentFilterRequest

    ds = await _ds()
    fr = DocumentFilterRequest.model_validate(payload["filter_request"])
    resp = await ds.filter_documents(fr)
    return {"response": jsonable_model(resp)}


async def op_get_document_status(payload: Dict[str, Any], uid: str) -> Dict[str, Any]:
    ds = await _ds()
    st = await ds.get_document_status(payload["doc_id"])
    return {"status": jsonable_model(st)}


async def op_update_document_metadata(payload: Dict[str, Any], uid: str) -> Dict[str, Any]:
    from ds_models.api_models import DocumentUpdateRequest

    ds = await _ds()
    ur = DocumentUpdateRequest.model_validate(payload["update_request"])
    ok = await ds.update_document_metadata(payload["document_id"], ur)
    return {"success": bool(ok)}


async def op_bulk_categorize_documents(payload: Dict[str, Any], uid: str) -> Dict[str, Any]:
    from ds_models.api_models import BulkCategorizeRequest

    ds = await _ds()
    br = BulkCategorizeRequest.model_validate(payload["bulk_request"])
    resp = await ds.bulk_categorize_documents(br)
    return {"response": jsonable_model(resp)}


async def op_delete_document_database_only(payload: Dict[str, Any], uid: str) -> Dict[str, Any]:
    ds = await _ds()
    ok = await ds.delete_document_database_only(payload["document_id"])
    return {"success": bool(ok)}


async def op_cleanup_orphaned_embeddings(payload: Dict[str, Any], uid: str) -> Dict[str, Any]:
    ds = await _ds()
    n = await ds.cleanup_orphaned_embeddings()
    return {"cleaned_count": int(n)}


async def op_get_duplicate_documents(payload: Dict[str, Any], uid: str) -> Dict[str, Any]:
    ds = await _ds()
    dup = await ds.get_duplicate_documents()
    out: Dict[str, Any] = {}
    for k, v in dup.items():
        if isinstance(v, list):
            out[k] = jsonable_model(v)
        else:
            out[k] = v
    return {"duplicates": out}


async def op_get_documents_stats(payload: Dict[str, Any], uid: str) -> Dict[str, Any]:
    ds = await _ds()
    stats = await ds.get_documents_stats()
    return {"stats": dict(stats) if stats else {}}


async def op_get_document_categories_overview(payload: Dict[str, Any], uid: str) -> Dict[str, Any]:
    ds = await _ds()
    ov = await ds.get_document_categories_overview()
    return {"overview": jsonable_model(ov)}


async def op_import_from_url(payload: Dict[str, Any], uid: str) -> Dict[str, Any]:
    ds = await _ds()
    up = await ds.import_from_url(
        payload["url"],
        payload.get("content_type") or "html",
    )
    return {"upload": jsonable_model(up)}


async def op_get_documents_with_hierarchy(payload: Dict[str, Any], uid: str) -> Dict[str, Any]:
    ds = await _ds()
    data = await ds.get_documents_with_hierarchy(
        int(payload.get("limit", 100)),
        int(payload.get("offset", 0)),
    )
    return {"data": jsonable_model(data)}


async def op_get_zip_hierarchy(payload: Dict[str, Any], uid: str) -> Dict[str, Any]:
    ds = await _ds()
    data = await ds.get_zip_hierarchy(payload["document_id"])
    return {"data": jsonable_model(data)}


async def op_delete_zip_with_children(payload: Dict[str, Any], uid: str) -> Dict[str, Any]:
    ds = await _ds()
    data = await ds.delete_zip_with_children(
        payload["parent_document_id"],
        bool(payload.get("delete_children", True)),
    )
    return {"data": jsonable_model(data)}


async def op_process_url_async(payload: Dict[str, Any], uid: str) -> Dict[str, Any]:
    ds = await _ds()
    asyncio.create_task(
        ds._process_url_async(
            payload["document_id"],
            payload["url"],
            payload.get("content_type") or "html",
            payload.get("user_id") or uid or None,
        )
    )
    return {"queued": True}


async def op_resume_incomplete_processing(payload: Dict[str, Any], uid: str) -> Dict[str, Any]:
    ds = await _ds()
    if hasattr(ds, "_resume_incomplete_processing"):
        await ds._resume_incomplete_processing()
    elif hasattr(ds, "_resume_incomplete_processing_sequential"):
        await ds._resume_incomplete_processing_sequential()
    return {"resumed": True}


async def op_remove_team_upload_directory(
    payload: Dict[str, Any], uid: str, folder_service: Any = None
) -> Dict[str, Any]:
    fs = folder_service
    if fs is None:
        from ds_services.folder_service import FolderService

        fs = FolderService()
        await fs.initialize()
    tid = str(payload.get("team_id") or "").strip()
    if not tid:
        raise ValueError("missing team_id")
    ok = await fs.remove_team_upload_directory(tid)
    return {"success": bool(ok)}


ACTION_OPS = {
    "get_document": op_get_document,
    "check_document_exists": op_check_document_exists,
    "list_documents": op_list_documents,
    "filter_documents": op_filter_documents,
    "get_document_status": op_get_document_status,
    "update_document_metadata": op_update_document_metadata,
    "bulk_categorize_documents": op_bulk_categorize_documents,
    "delete_document_database_only": op_delete_document_database_only,
    "cleanup_orphaned_embeddings": op_cleanup_orphaned_embeddings,
    "get_duplicate_documents": op_get_duplicate_documents,
    "get_documents_stats": op_get_documents_stats,
    "get_document_categories_overview": op_get_document_categories_overview,
    "import_from_url": op_import_from_url,
    "get_documents_with_hierarchy": op_get_documents_with_hierarchy,
    "get_zip_hierarchy": op_get_zip_hierarchy,
    "delete_zip_with_children": op_delete_zip_with_children,
    "process_url_async": op_process_url_async,
    "resume_incomplete_processing": op_resume_incomplete_processing,
    "remove_team_upload_directory": op_remove_team_upload_directory,
}
