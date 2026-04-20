"""
API endpoints for per-document file encryption (document-service gRPC).
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api.document_api import check_document_access
from clients.document_service_client import get_document_service_client
from config import settings
from models.api_models import AuthenticatedUserResponse
from utils.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["documents"])


class EncryptBody(BaseModel):
    password: str = Field(..., min_length=1)
    confirm_password: str = Field(..., min_length=1)


class PasswordBody(BaseModel):
    password: str = Field(..., min_length=1)


class ChangePasswordBody(BaseModel):
    old_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=1)


class HeartbeatBody(BaseModel):
    session_token: str = Field(..., min_length=1)


def _no_store_headers() -> dict:
    return {
        "Cache-Control": "no-store, no-cache, must-revalidate",
        "Pragma": "no-cache",
    }


def _metadata_subset(doc) -> dict:
    return {
        "document_id": doc.document_id,
        "title": doc.title,
        "filename": doc.filename,
        "author": doc.author,
        "description": doc.description,
        "category": doc.category.value if doc.category else None,
        "tags": doc.tags,
        "user_id": getattr(doc, "user_id", None),
        "collection_type": getattr(doc, "collection_type", None),
        "folder_id": getattr(doc, "folder_id", None),
        "is_encrypted": getattr(doc, "is_encrypted", False),
    }


def _ds_encrypt_error_status(err: str | None) -> int:
    if not err:
        return 500
    low = err.lower()
    if "too many" in low or "rate" in low:
        return 429
    if "configuration" in low or "redis" in low:
        return 503
    return 400 if "password" in low or "invalid" in low or "not encrypted" in low else 500


@router.post("/api/documents/{doc_id}/encrypt")
async def encrypt_document_endpoint(
    doc_id: str,
    body: EncryptBody,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    await check_document_access(doc_id, current_user, "write")
    dsc = get_document_service_client()
    await dsc.initialize(required=True)
    ok, _data, err = await dsc.encrypt_document_json(
        current_user.user_id,
        {
            "document_id": doc_id,
            "password": body.password,
            "confirm_password": body.confirm_password,
            "user_id": current_user.user_id,
        },
    )
    if not ok:
        raise HTTPException(status_code=_ds_encrypt_error_status(err), detail=err or "Encryption failed")
    return {"status": "success", "message": "Document encrypted"}


@router.post("/api/documents/{doc_id}/decrypt-session")
async def decrypt_session_endpoint(
    doc_id: str,
    body: PasswordBody,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    doc = await check_document_access(doc_id, current_user, "read")
    dsc = get_document_service_client()
    await dsc.initialize(required=True)
    ok, data, err = await dsc.create_decrypt_session_json(
        current_user.user_id,
        {
            "document_id": doc_id,
            "password": body.password,
            "user_id": current_user.user_id,
        },
    )
    if not ok or not data:
        status = _ds_encrypt_error_status(err)
        raise HTTPException(status_code=status, detail=err or "Unlock failed")

    plaintext = data.get("content")
    session_token = data.get("session_token")
    if plaintext is None or session_token is None:
        raise HTTPException(status_code=500, detail="Invalid unlock response")

    payload = {
        "content": plaintext,
        "session_token": session_token,
        "ttl_seconds": settings.FILE_ENCRYPTION_SESSION_TTL_SECONDS,
        "metadata": _metadata_subset(doc),
    }
    return JSONResponse(content=payload, headers=_no_store_headers())


@router.post("/api/documents/{doc_id}/encryption-heartbeat")
async def encryption_heartbeat_endpoint(
    doc_id: str,
    body: HeartbeatBody,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    await check_document_access(doc_id, current_user, "read")
    dsc = get_document_service_client()
    await dsc.initialize(required=True)
    ok, data, err = await dsc.encryption_heartbeat_json(
        current_user.user_id,
        {
            "document_id": doc_id,
            "session_token": body.session_token,
            "user_id": current_user.user_id,
        },
    )
    if not ok:
        raise HTTPException(status_code=500, detail=err or "Heartbeat failed")
    remaining = data.get("ttl_seconds") if data else None
    if remaining is None:
        raise HTTPException(status_code=423, detail="Encryption session expired or invalid")
    return {"remaining_ttl_seconds": remaining}


@router.post("/api/documents/{doc_id}/encryption-lock")
async def encryption_lock_endpoint(
    doc_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    await check_document_access(doc_id, current_user, "read")
    dsc = get_document_service_client()
    await dsc.initialize(required=True)
    ok, _data, err = await dsc.encryption_lock_json(
        current_user.user_id,
        {"document_id": doc_id, "user_id": current_user.user_id},
    )
    if not ok:
        raise HTTPException(status_code=500, detail=err or "Lock failed")
    return {"status": "success"}


@router.post("/api/documents/{doc_id}/change-encryption-password")
async def change_encryption_password_endpoint(
    doc_id: str,
    body: ChangePasswordBody,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    await check_document_access(doc_id, current_user, "write")
    dsc = get_document_service_client()
    await dsc.initialize(required=True)
    ok, _data, err = await dsc.change_encryption_password_json(
        current_user.user_id,
        {
            "document_id": doc_id,
            "old_password": body.old_password,
            "new_password": body.new_password,
            "user_id": current_user.user_id,
        },
    )
    if not ok:
        raise HTTPException(
            status_code=400 if err and ("password" in err.lower() or "invalid" in err.lower()) else 500,
            detail=err or "Failed to change password",
        )
    return {"status": "success", "message": "Password changed; document is locked"}


@router.post("/api/documents/{doc_id}/remove-encryption")
async def remove_encryption_endpoint(
    doc_id: str,
    body: PasswordBody,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    await check_document_access(doc_id, current_user, "write")
    dsc = get_document_service_client()
    await dsc.initialize(required=True)
    ok, _data, err = await dsc.remove_encryption_json(
        current_user.user_id,
        {
            "document_id": doc_id,
            "password": body.password,
            "user_id": current_user.user_id,
        },
    )
    if not ok:
        raise HTTPException(
            status_code=400 if err and "password" in err.lower() else 500,
            detail=err or "Failed to remove encryption",
        )
    return {"status": "success", "message": "Encryption removed; re-indexing queued"}
