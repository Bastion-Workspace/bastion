"""
Document Version API - List, content, diff, and rollback for document versions.
"""

import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, Query

from api.document_api import check_document_access
from utils.auth_middleware import get_current_user
from models.api_models import AuthenticatedUserResponse
from services.document_version_service import (
    list_versions,
    get_version_content,
    diff_versions,
    rollback_to_version,
)

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
    versions = await list_versions(doc_id, skip=skip, limit=limit, user_id=user_id, collection_type=collection_type)
    return {"document_id": doc_id, "versions": versions}


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
    result = await diff_versions(
        doc_id, from_version, to_version, user_id=user_id, collection_type=collection_type
    )
    if result is None:
        raise HTTPException(status_code=404, detail="One or both versions not found")
    return result


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
    content = await get_version_content(version_id, user_id=user_id, collection_type=collection_type)
    if content is None:
        raise HTTPException(status_code=404, detail="Version not found or content missing")
    return {"document_id": doc_id, "version_id": str(version_id), "content": content}


@router.post("/api/documents/{doc_id}/versions/{version_id}/rollback")
async def rollback_document_version(
    doc_id: str,
    version_id: UUID,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Rollback document to a prior version. Current state is saved as a new version first."""
    await check_document_access(doc_id, current_user, "write")
    result = await rollback_to_version(doc_id, version_id, current_user.user_id)
    if not result.get("success"):
        raise HTTPException(
            status_code=400 if result.get("error") != "Document not found" else 404,
            detail=result.get("message", result.get("error", "Rollback failed")),
        )
    return result
