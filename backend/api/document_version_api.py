"""
Document Version API - List, content, diff, and rollback (document-service gRPC).
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from api.document_api import check_document_access
from clients.document_service_client import get_document_service_client
from models.api_models import AuthenticatedUserResponse
from utils.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["document-versions"])


@router.get("/api/documents/{doc_id}/versions")
async def get_document_versions(
    doc_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """List versions for a document (newest first)."""
    doc_info = await check_document_access(doc_id, current_user, "read")
    collection_type = getattr(doc_info, "collection_type", "user")
    user_id = getattr(doc_info, "user_id", None) or current_user.user_id

    dsc = get_document_service_client()
    await dsc.initialize(required=True)
    ok, data, err = await dsc.get_document_versions_json(
        current_user.user_id,
        {
            "document_id": doc_id,
            "skip": skip,
            "limit": limit,
            "user_id": user_id,
            "collection_type": collection_type,
        },
    )
    if not ok or not data:
        raise HTTPException(status_code=500, detail=err or "Failed to list versions")
    return {"document_id": doc_id, "versions": data.get("versions", [])}


@router.get("/api/documents/{doc_id}/versions/diff")
async def get_versions_diff(
    doc_id: str,
    from_version: UUID = Query(..., alias="from"),
    to_version: UUID = Query(..., alias="to"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Get unified diff between two versions."""
    doc_info = await check_document_access(doc_id, current_user, "read")
    collection_type = getattr(doc_info, "collection_type", "user")
    user_id = getattr(doc_info, "user_id", None) or current_user.user_id

    dsc = get_document_service_client()
    await dsc.initialize(required=True)
    ok, data, err = await dsc.diff_versions_json(
        current_user.user_id,
        {
            "document_id": doc_id,
            "from_version": str(from_version),
            "to_version": str(to_version),
            "user_id": user_id,
            "collection_type": collection_type,
        },
    )
    if not ok:
        raise HTTPException(status_code=500, detail=err or "Diff failed")
    if not data:
        raise HTTPException(status_code=404, detail="One or both versions not found")
    return data


@router.get("/api/documents/{doc_id}/versions/{version_id}/content")
async def get_version_content_endpoint(
    doc_id: str,
    version_id: UUID,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Get full content of a specific version."""
    doc_info = await check_document_access(doc_id, current_user, "read")
    collection_type = getattr(doc_info, "collection_type", "user")
    user_id = getattr(doc_info, "user_id", None) or current_user.user_id

    dsc = get_document_service_client()
    await dsc.initialize(required=True)
    ok, data, err = await dsc.get_version_content_json(
        current_user.user_id,
        {
            "version_id": str(version_id),
            "user_id": user_id,
            "collection_type": collection_type,
        },
    )
    if not ok or not data:
        raise HTTPException(status_code=500, detail=err or "Failed to load version content")
    content = data.get("content")
    if content is None:
        raise HTTPException(status_code=404, detail="Version not found or content missing")
    return {"document_id": doc_id, "version_id": str(version_id), "content": content}


@router.post("/api/documents/{doc_id}/versions/{version_id}/rollback")
async def rollback_document_version(
    doc_id: str,
    version_id: UUID,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Rollback document to a prior version."""
    await check_document_access(doc_id, current_user, "write")

    dsc = get_document_service_client()
    await dsc.initialize(required=True)
    ok, data, err = await dsc.rollback_to_version_json(
        current_user.user_id,
        {
            "document_id": doc_id,
            "version_id": str(version_id),
            "user_id": current_user.user_id,
        },
    )
    if not ok or not data:
        raise HTTPException(status_code=500, detail=err or "Rollback failed")
    if not data.get("success"):
        raise HTTPException(
            status_code=400 if data.get("error") != "Document not found" else 404,
            detail=data.get("message", data.get("error", "Rollback failed")),
        )
    return data
