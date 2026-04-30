"""
REST API for mobile push tokens, notification preferences, and cross-surface ACK.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from models.api_models import AuthenticatedUserResponse
from services.database_manager.database_helpers import execute, fetch_one
from services.push_notification_service import register_push_token, revoke_push_token
from utils.auth_middleware import get_current_user
from utils.websocket_manager import get_websocket_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["notifications"])


class PushTokenBody(BaseModel):
    token: str = Field(..., min_length=10, max_length=512)
    platform: str = Field(default="android", max_length=20)
    device_id: str = Field(..., min_length=4, max_length=255)
    app_version: Optional[str] = Field(default=None, max_length=64)


class NotificationPreferencesBody(BaseModel):
    """Partial update merged into users.preferences.notification_preferences."""

    cascade_order: Optional[list] = None
    email_address: Optional[str] = None
    push_enabled: Optional[bool] = None
    telegram_enabled: Optional[bool] = None
    email_enabled: Optional[bool] = None
    quiet_hours: Optional[dict] = None
    per_event_overrides: Optional[dict] = None


class AckBody(BaseModel):
    notification_id: str = Field(..., min_length=8, max_length=64)


@router.post("/api/notifications/push-token")
async def post_push_token(
    body: PushTokenBody,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    try:
        await register_push_token(
            current_user.user_id,
            body.token,
            body.platform,
            body.device_id,
            body.app_version,
        )
        return {"success": True}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("push-token register failed: %s", e)
        raise HTTPException(status_code=500, detail="Registration failed")


@router.delete("/api/notifications/push-token/{device_id:path}")
async def delete_push_token(
    device_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    did = (device_id or "").strip()
    if not did:
        raise HTTPException(status_code=400, detail="device_id required")
    await revoke_push_token(current_user.user_id, did)
    return {"success": True}


@router.get("/api/notifications/preferences")
async def get_notification_preferences(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    row = await fetch_one(
        "SELECT preferences FROM users WHERE user_id = $1",
        current_user.user_id,
    )
    prefs: Dict[str, Any] = {}
    if row and row.get("preferences"):
        raw = row["preferences"]
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = {}
        if isinstance(raw, dict):
            np = raw.get("notification_preferences")
            if isinstance(np, dict):
                prefs = np
    return {"notification_preferences": prefs}


@router.put("/api/notifications/preferences")
async def put_notification_preferences(
    body: NotificationPreferencesBody,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    row = await fetch_one(
        "SELECT preferences FROM users WHERE user_id = $1",
        current_user.user_id,
    )
    all_prefs: Dict[str, Any] = {}
    if row and row.get("preferences"):
        raw = row["preferences"]
        if isinstance(raw, str):
            try:
                all_prefs = json.loads(raw)
            except json.JSONDecodeError:
                all_prefs = {}
        elif isinstance(raw, dict):
            all_prefs = dict(raw)

    np = all_prefs.get("notification_preferences")
    if not isinstance(np, dict):
        np = {}
    incoming = body.model_dump(exclude_unset=True)
    for k, v in incoming.items():
        if v is not None:
            np[k] = v
    all_prefs["notification_preferences"] = np

    await execute(
        "UPDATE users SET preferences = $1::jsonb WHERE user_id = $2",
        json.dumps(all_prefs),
        current_user.user_id,
    )
    return {"success": True, "notification_preferences": np}


@router.post("/api/notifications/ack")
async def post_notification_ack(
    body: AckBody,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Fan out dismiss to other sessions (same as WebSocket notification_ack)."""
    nid = (body.notification_id or "").strip()
    if not nid:
        raise HTTPException(status_code=400, detail="notification_id required")
    ws = get_websocket_manager()
    if ws:
        await ws.send_notification_ack_to_user_sessions(
            current_user.user_id,
            nid,
            exclude_websocket=None,
        )
    return {"success": True}
