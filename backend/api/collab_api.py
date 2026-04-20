"""Collaborative editing: Yjs WebSocket and manual flush."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from api.document_api import check_document_access
from models.api_models import AuthenticatedUserResponse
from services.collab_room_manager import get_collab_room_manager
from services.document_sharing_service import document_sharing_service
from utils.auth_middleware import decode_jwt_token, get_current_user
from utils.collab_websocket_adapter import FastAPIWebsocketAdapter

logger = logging.getLogger(__name__)

router = APIRouter(tags=["collaboration"])


def _ws_user_from_payload(payload: dict) -> AuthenticatedUserResponse:
    uid = payload.get("user_id") or payload.get("sub")
    if not uid:
        raise ValueError("Missing user_id in token")
    return AuthenticatedUserResponse(
        user_id=str(uid),
        username=str(payload.get("username", uid)),
        email=str(payload.get("email", "")),
        display_name=payload.get("display_name"),
        role=str(payload.get("role", "user")),
        preferences={},
        federation_discoverable=payload.get("federation_discoverable"),
    )


@router.websocket("/api/ws/collab/{document_id}")
async def collaborative_document_ws(websocket: WebSocket, document_id: str):
    token: Optional[str] = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=4001, reason="Missing token")
        return
    try:
        payload = decode_jwt_token(token)
        user = _ws_user_from_payload(payload)
    except ValueError:
        await websocket.close(code=4003, reason="Invalid token")
        return

    try:
        ctx = await document_sharing_service.get_sharing_context_for_user(
            document_id, user.user_id, user.role
        )
        if not ctx.get("collab_eligible"):
            await websocket.close(code=4003, reason="Collaboration not enabled for this document")
            return
        await check_document_access(document_id, user, "write")
    except HTTPException as e:
        await websocket.close(code=4003, reason=str(e.detail))
        return
    except Exception as e:
        logger.warning("collab ws auth failed: %s", e)
        await websocket.close(code=4000, reason="Authorization failed")
        return

    await websocket.accept()
    path = f"/api/ws/collab/{document_id}"
    adapter = FastAPIWebsocketAdapter(websocket, path)
    manager = get_collab_room_manager()
    try:
        await manager.serve_connection(document_id, adapter)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning("collab ws error for %s: %s", document_id, e)
    finally:
        if websocket.application_state == WebSocketState.CONNECTED:
            try:
                await websocket.close()
            except Exception:
                pass


@router.post("/api/documents/{document_id}/collab-flush")
async def collaborative_flush(
    document_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    await check_document_access(document_id, current_user, "write")
    ctx = await document_sharing_service.get_sharing_context_for_user(
        document_id, current_user.user_id, current_user.role
    )
    if not ctx.get("collab_eligible"):
        raise HTTPException(status_code=400, detail="Document is not in collaborative mode")
    try:
        await get_collab_room_manager().flush_now(document_id)
    except Exception as e:
        logger.error("collab flush failed: %s", e)
        raise HTTPException(status_code=500, detail="Flush failed") from e
    return {"ok": True, "document_id": document_id}
