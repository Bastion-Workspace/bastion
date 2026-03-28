"""
Device proxy API - WebSocket endpoint for Bastion Local Proxy daemon connections.
"""

import asyncio
import json
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from services.device_token_service import resolve_token, update_last_connected
from utils.websocket_manager import get_websocket_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Device Proxy"])

AUTH_TIMEOUT_SECONDS = 10


@router.websocket("/api/ws/device")
async def device_websocket(websocket: WebSocket):
    """WebSocket endpoint for local proxy daemons. Authenticate via first message: {"type": "auth", "token": "..."}."""
    await websocket.accept()
    client_ip = getattr(websocket.client, "host", None) or "unknown"

    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=AUTH_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.warning("Device auth failed: ip=%s, reason=no auth message within %ss", client_ip, AUTH_TIMEOUT_SECONDS)
        await websocket.close(code=4001)
        return

    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Device auth failed: ip=%s, reason=invalid json", client_ip)
        await websocket.close(code=4001)
        return

    if not isinstance(msg, dict):
        logger.warning("Device auth failed: ip=%s, reason=first message is not a JSON object", client_ip)
        await websocket.close(code=4001)
        return

    if msg.get("type") != "auth":
        logger.warning("Device auth failed: ip=%s, reason=first message type is not auth", client_ip)
        await websocket.close(code=4001)
        return

    token = msg.get("token") if isinstance(msg.get("token"), str) else None
    if not token:
        logger.warning("Device auth failed: ip=%s, reason=missing token", client_ip)
        await websocket.close(code=4001)
        return

    row = await resolve_token(token)
    if not row:
        logger.warning("Device auth failed: ip=%s, reason=invalid or revoked token", client_ip)
        await websocket.close(code=4003)
        return

    user_id = row["user_id"]
    device_id = row.get("device_name") or "default"
    token_id = str(row["id"])
    ws_manager = get_websocket_manager()
    ws_manager._register_connection(websocket, session_id=user_id)
    ws_manager.register_device(user_id, device_id, websocket, [])
    try:
        await update_last_connected(token_id, ip=client_ip)
    except Exception as e:
        logger.debug("Could not update last_connected_at: %s", e)
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(msg, dict):
                logger.debug("Device message ignored: expected JSON object, got %s", type(msg).__name__)
                continue
            msg_type = msg.get("type")
            if msg_type == "register":
                capabilities = msg.get("capabilities") or []
                dev_id = msg.get("device_id") or device_id
                ws_manager.unregister_device(user_id, device_id)
                ws_manager.register_device(user_id, dev_id, websocket, capabilities)
                device_id = dev_id
            elif msg_type == "result":
                request_id = msg.get("request_id")
                if request_id:
                    result = {
                        "success": True,
                        "result_json": json.dumps(msg.get("result", {})),
                        "formatted": msg.get("formatted", ""),
                    }
                    ws_manager.resolve_device_invocation(request_id, result, user_id)
            elif msg_type == "error":
                request_id = msg.get("request_id")
                if request_id:
                    result = {
                        "success": False,
                        "error": msg.get("error", "Unknown error"),
                        "formatted": msg.get("error", "Unknown error"),
                    }
                    ws_manager.resolve_device_invocation(request_id, result, user_id)
            elif msg_type == "heartbeat":
                pass
    except WebSocketDisconnect:
        pass
    finally:
        ws_manager.unregister_device(user_id, device_id)
        ws_manager.disconnect(websocket, session_id=user_id)
