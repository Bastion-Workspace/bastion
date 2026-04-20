"""REST API for document sharing, shared-with-me, and edit locks."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException

from models.api_models import AuthenticatedUserResponse
from models.document_sharing_models import (
    AcquireLockResponse,
    CreateShareRequest,
    DocumentLockResponse,
    DocumentSharingContextResponse,
    ShareableUsersListResponse,
    ShareableUserResponse,
    SharedItemResponse,
    ShareInfoResponse,
    ShareListResponse,
    SharerGroupResponse,
    SharedWithMeResponse,
    UpdateShareRequest,
)
from services.document_sharing_service import document_sharing_service
from services.database_manager.database_helpers import fetch_one
from api.document_api import check_document_access
from utils.auth_middleware import get_current_user
from utils.websocket_manager import get_websocket_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["document-sharing"])


def _lock_to_response(doc_id: str, row: Optional[Dict[str, Any]]) -> DocumentLockResponse:
    if not row:
        return DocumentLockResponse(document_id=doc_id, active=False)
    now = datetime.now(timezone.utc)
    exp = row.get("expires_at")
    active = bool(exp and exp > now)
    return DocumentLockResponse(
        document_id=doc_id,
        locked_by_user_id=row.get("locked_by_user_id"),
        locked_by_username=row.get("locked_by_username"),
        acquired_at=row.get("acquired_at"),
        expires_at=exp,
        active=active,
    )


def _row_to_share_info(row: dict) -> ShareInfoResponse:
    return ShareInfoResponse(
        share_id=row["share_id"],
        document_id=row.get("document_id"),
        folder_id=row.get("folder_id"),
        shared_by_user_id=row["shared_by_user_id"],
        shared_with_user_id=row["shared_with_user_id"],
        shared_with_username=row.get("shared_with_username"),
        share_type=row["share_type"],
        created_at=row.get("created_at"),
        expires_at=row.get("expires_at"),
    )


async def _notify_share_recipients(shared_with_user_id: str, shared_by_user_id: str, payload: dict):
    try:
        ws = get_websocket_manager()
        message = {"type": "document_share_update", **payload}
        await ws.broadcast_to_users([shared_with_user_id, shared_by_user_id], message)
    except Exception as e:
        logger.warning("WebSocket share notification failed: %s", e)


@router.get("/api/users/shareable", response_model=ShareableUsersListResponse)
async def list_shareable_users(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    rows = await document_sharing_service.list_shareable_users(current_user.user_id)
    return ShareableUsersListResponse(
        users=[
            ShareableUserResponse(
                user_id=r["user_id"],
                username=r["username"],
                avatar_url=r.get("avatar_url"),
            )
            for r in rows
        ]
    )


@router.get("/api/shared-with-me", response_model=SharedWithMeResponse)
async def shared_with_me(current_user: AuthenticatedUserResponse = Depends(get_current_user)):
    raw = await document_sharing_service.get_shared_with_me(current_user.user_id)
    groups: dict = {}
    for row in raw:
        sid = row["shared_by_user_id"]
        if sid not in groups:
            groups[sid] = {"sharer_user_id": sid, "sharer_username": row.get("sharer_username") or sid, "items": []}
        if row.get("document_id"):
            groups[sid]["items"].append(
                SharedItemResponse(
                    item_type="document",
                    document_id=row["document_id"],
                    title=row.get("title"),
                    filename=row.get("filename"),
                    share_type=row["share_type"],
                    share_id=row["share_id"],
                )
            )
        else:
            groups[sid]["items"].append(
                SharedItemResponse(
                    item_type="folder",
                    folder_id=row.get("folder_id"),
                    name=row.get("folder_name"),
                    parent_folder_id=row.get("folder_parent_id"),
                    share_type=row["share_type"],
                    share_id=row["share_id"],
                )
            )
    return SharedWithMeResponse(
        groups=[
            SharerGroupResponse(
                sharer_user_id=g["sharer_user_id"],
                sharer_username=g["sharer_username"],
                items=g["items"],
            )
            for g in groups.values()
        ]
    )


@router.get("/api/documents/{doc_id}/shares", response_model=ShareListResponse)
async def list_document_shares(
    doc_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    try:
        rows = await document_sharing_service.list_shares_for_document(
            doc_id, current_user.user_id, current_user.role
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return ShareListResponse(shares=[_row_to_share_info(r) for r in rows])


@router.post("/api/documents/{doc_id}/shares", response_model=ShareInfoResponse)
async def create_document_share(
    doc_id: str,
    body: CreateShareRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    try:
        row = await document_sharing_service.create_document_share(
            doc_id,
            current_user.user_id,
            body.shared_with_user_id,
            body.share_type,
            current_user.role,
            body.expires_at,
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    await _notify_share_recipients(
        body.shared_with_user_id,
        current_user.user_id,
        {"document_id": doc_id, "action": "created"},
    )
    return _row_to_share_info(row)


@router.get("/api/folders/{folder_id}/shares", response_model=ShareListResponse)
async def list_folder_shares(
    folder_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    try:
        rows = await document_sharing_service.list_shares_for_folder(
            folder_id, current_user.user_id, current_user.role
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return ShareListResponse(shares=[_row_to_share_info(r) for r in rows])


@router.post("/api/folders/{folder_id}/shares", response_model=ShareInfoResponse)
async def create_folder_share(
    folder_id: str,
    body: CreateShareRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    try:
        row = await document_sharing_service.create_folder_share(
            folder_id,
            current_user.user_id,
            body.shared_with_user_id,
            body.share_type,
            current_user.role,
            body.expires_at,
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    await _notify_share_recipients(
        body.shared_with_user_id,
        current_user.user_id,
        {"folder_id": folder_id, "action": "created"},
    )
    return _row_to_share_info(row)


@router.put("/api/shares/{share_id}", response_model=ShareInfoResponse)
async def update_share(
    share_id: str,
    body: UpdateShareRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    try:
        row = await document_sharing_service.update_share(
            share_id, body.share_type, current_user.user_id, current_user.role
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    await _notify_share_recipients(
        row["shared_with_user_id"],
        row["shared_by_user_id"],
        {"share_id": share_id, "action": "updated"},
    )
    return _row_to_share_info(row)


@router.delete("/api/shares/{share_id}")
async def delete_share(
    share_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    row = await fetch_one(
        "SELECT shared_by_user_id, shared_with_user_id FROM document_shares WHERE share_id = $1",
        share_id,
        rls_context={"user_id": current_user.user_id, "user_role": current_user.role},
    )
    if not row:
        raise HTTPException(status_code=404, detail="Share not found")
    try:
        await document_sharing_service.revoke_share(share_id, current_user.user_id, current_user.role)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    await _notify_share_recipients(
        row["shared_with_user_id"],
        row["shared_by_user_id"],
        {"share_id": share_id, "action": "revoked"},
    )
    return {"ok": True}


@router.get("/api/documents/{doc_id}/sharing-context", response_model=DocumentSharingContextResponse)
async def get_sharing_context(
    doc_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    await check_document_access(doc_id, current_user, "read")
    ctx = await document_sharing_service.get_sharing_context_for_user(
        doc_id, current_user.user_id, current_user.role
    )
    return DocumentSharingContextResponse(**ctx)


@router.get("/api/documents/{doc_id}/lock", response_model=DocumentLockResponse)
async def get_document_lock(
    doc_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    await check_document_access(doc_id, current_user, "read")
    row = await document_sharing_service.get_lock_row(doc_id, current_user.user_id, current_user.role)
    return _lock_to_response(doc_id, row)


@router.post("/api/documents/{doc_id}/lock", response_model=AcquireLockResponse)
async def acquire_document_lock(
    doc_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    ctx = await document_sharing_service.get_sharing_context_for_user(
        doc_id, current_user.user_id, current_user.role
    )
    if not ctx.get("can_write"):
        raise HTTPException(status_code=403, detail="Write access required to lock")
    await check_document_access(doc_id, current_user, "write")
    result = await document_sharing_service.acquire_lock(doc_id, current_user.user_id, current_user.role)
    lock_row = result.get("lock")
    try:
        await get_websocket_manager().send_document_status_update(doc_id, "lock_changed")
    except Exception:
        pass
    lr = _lock_to_response(doc_id, lock_row)
    if not result["success"]:
        return AcquireLockResponse(success=False, message=result.get("message", "Lock failed"), lock=lr)
    return AcquireLockResponse(success=True, message=result.get("message", ""), lock=lr)


@router.delete("/api/documents/{doc_id}/lock")
async def release_document_lock(
    doc_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    try:
        await document_sharing_service.release_lock(doc_id, current_user.user_id, current_user.role)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    try:
        await get_websocket_manager().send_document_status_update(doc_id, "lock_changed")
    except Exception:
        pass
    return {"ok": True}


@router.post("/api/documents/{doc_id}/lock/heartbeat", response_model=DocumentLockResponse)
async def heartbeat_document_lock(
    doc_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    try:
        row = await document_sharing_service.heartbeat_lock(
            doc_id, current_user.user_id, current_user.role
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return _lock_to_response(doc_id, row)