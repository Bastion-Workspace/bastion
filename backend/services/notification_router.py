"""
Central notification routing: surface-aware in-app delivery and external cascade.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional

from services.database_manager.database_helpers import execute, fetch_one
from services.email_service import EmailService
from utils.websocket_manager import get_websocket_manager

logger = logging.getLogger(__name__)

CRITICAL_SUBTYPES = frozenset(
    {"approval_required", "budget_exceeded", "shell_command_approval"}
)

DEFAULT_CASCADE = ["push", "telegram", "discord", "slack", "teams", "email"]


async def _log_notification(
    user_id: str,
    notification_id: str,
    event_type: str,
    conversation_id: Optional[str],
    channel: str,
    status: str,
) -> None:
    try:
        nid = uuid.UUID(str(notification_id))
        await execute(
            """
            INSERT INTO notification_log (user_id, notification_id, event_type, conversation_id, channel, status)
            VALUES ($1, $2::uuid, $3, $4, $5, $6)
            """,
            user_id,
            str(nid),
            (event_type or "")[:100],
            (conversation_id or "")[:255] or None,
            channel[:50],
            status[:20],
        )
    except Exception as e:
        logger.debug("notification_log insert skipped: %s", e)


async def _load_notification_preferences(user_id: str) -> Dict[str, Any]:
    try:
        row = await fetch_one(
            "SELECT preferences FROM users WHERE user_id = $1",
            user_id,
        )
        if not row:
            return {}
        prefs = row.get("preferences") or {}
        if isinstance(prefs, str):
            prefs = json.loads(prefs)
        if not isinstance(prefs, dict):
            return {}
        np = prefs.get("notification_preferences") or {}
        return np if isinstance(np, dict) else {}
    except Exception as e:
        logger.debug("load notification preferences failed: %s", e)
        return {}


def _event_subtype(payload: Dict[str, Any]) -> Optional[str]:
    st = payload.get("subtype")
    if st:
        return str(st).strip()
    return None


def _resolve_cascade_order(prefs: Dict[str, Any], event_subtype: Optional[str]) -> List[str]:
    np = prefs or {}
    order: List[str] = []
    overrides = np.get("per_event_overrides") or {}
    if isinstance(overrides, dict) and event_subtype:
        ev = overrides.get(event_subtype)
        if isinstance(ev, dict) and ev.get("cascade_order"):
            order = list(ev["cascade_order"])
    if not order:
        order = list(np.get("cascade_order") or DEFAULT_CASCADE)
    order = [str(x).strip().lower() for x in order if str(x).strip()]
    if event_subtype in CRITICAL_SUBTYPES:
        for ch in DEFAULT_CASCADE:
            if ch not in order:
                order.append(ch)
    seen = set()
    out: List[str] = []
    for ch in order:
        if ch not in seen:
            seen.add(ch)
            out.append(ch)
    return out


def _quiet_hours_block(prefs: Dict[str, Any]) -> bool:
    qh = prefs.get("quiet_hours") or {}
    if not isinstance(qh, dict) or not qh.get("enabled"):
        return False
    # Optional future: parse tz and window; default off when enabled without full config
    return False


class ChannelContext:
    def __init__(self, user_id: str, payload: Dict[str, Any], prefs: Dict[str, Any]):
        self.user_id = user_id
        self.payload = payload
        self.prefs = prefs


ChannelSendFn = Callable[[ChannelContext], Awaitable[bool]]


async def _channel_push(ctx: ChannelContext) -> bool:
    if not ctx.prefs.get("push_enabled", True):
        return False
    if _quiet_hours_block(ctx.prefs):
        return False
    from services.push_notification_service import send_push_for_user

    title = str(ctx.payload.get("title") or "Bastion")[:200]
    body = str(ctx.payload.get("preview") or ctx.payload.get("message") or "")[:500]
    data = {
        "notification_type": str(ctx.payload.get("subtype") or ctx.payload.get("type") or "agent")[:80],
        "conversation_id": str(ctx.payload.get("conversation_id") or ""),
        "notification_id": str(ctx.payload.get("notification_id") or ""),
    }
    return await send_push_for_user(ctx.user_id, title, body, data)


async def _channel_telegram(ctx: ChannelContext) -> bool:
    if ctx.prefs.get("telegram_enabled") is False:
        return False
    try:
        from clients.connections_service_client import get_connections_service_client

        client = await get_connections_service_client()
        text = f"**{ctx.payload.get('title', 'Bastion')}**\n\n{ctx.payload.get('preview') or ctx.payload.get('message') or ''}"[
            :4000
        ]
        result = await client.send_outbound_message(
            user_id=ctx.user_id,
            provider="telegram",
            connection_id="",
            message=text,
            format="markdown",
            recipient_chat_id="",
        )
        return bool(result.get("success"))
    except Exception as e:
        logger.debug("telegram cascade failed: %s", e)
        return False


async def _channel_discord(ctx: ChannelContext) -> bool:
    try:
        from clients.connections_service_client import get_connections_service_client

        client = await get_connections_service_client()
        text = f"**{ctx.payload.get('title', 'Bastion')}**\n\n{ctx.payload.get('preview') or ctx.payload.get('message') or ''}"[
            :4000
        ]
        result = await client.send_outbound_message(
            user_id=ctx.user_id,
            provider="discord",
            connection_id="",
            message=text,
            format="markdown",
            recipient_chat_id="",
        )
        return bool(result.get("success"))
    except Exception as e:
        logger.debug("discord cascade failed: %s", e)
        return False


async def _channel_slack(ctx: ChannelContext) -> bool:
    try:
        from clients.connections_service_client import get_connections_service_client

        client = await get_connections_service_client()
        text = f"*{ctx.payload.get('title', 'Bastion')}*\n{ctx.payload.get('preview') or ctx.payload.get('message') or ''}"[
            :4000
        ]
        result = await client.send_outbound_message(
            user_id=ctx.user_id,
            provider="slack",
            connection_id="",
            message=text,
            format="markdown",
            recipient_chat_id="",
        )
        return bool(result.get("success"))
    except Exception as e:
        logger.debug("slack cascade failed: %s", e)
        return False


async def _channel_teams(ctx: ChannelContext) -> bool:
    if ctx.prefs.get("teams_enabled") is False:
        return False
    try:
        from clients.connections_service_client import get_connections_service_client

        client = await get_connections_service_client()
        text = f"**{ctx.payload.get('title', 'Bastion')}**\n\n{ctx.payload.get('preview') or ctx.payload.get('message') or ''}"[
            :4000
        ]
        result = await client.send_outbound_message(
            user_id=ctx.user_id,
            provider="teams",
            connection_id="",
            message=text,
            format="markdown",
            recipient_chat_id="",
        )
        return bool(result.get("success"))
    except Exception as e:
        logger.debug("teams cascade failed: %s", e)
        return False


async def _channel_email(ctx: ChannelContext) -> bool:
    if ctx.prefs.get("email_enabled") is False:
        return False
    to = (ctx.prefs.get("email_address") or "").strip()
    if not to:
        return False
    try:
        svc = EmailService()
        subj = str(ctx.payload.get("title") or "Bastion notification")[:200]
        body = str(ctx.payload.get("preview") or ctx.payload.get("message") or "")[:8000]
        ok = await svc.send_email(to_email=to, subject=subj, body_text=body)
        return bool(ok)
    except Exception as e:
        logger.debug("email cascade failed: %s", e)
        return False


CHANNEL_HANDLERS: Dict[str, ChannelSendFn] = {
    "push": _channel_push,
    "telegram": _channel_telegram,
    "discord": _channel_discord,
    "slack": _channel_slack,
    "teams": _channel_teams,
    "email": _channel_email,
}


async def route_notification(
    user_id: str,
    event_type: str,
    payload: Dict[str, Any],
    originating_surface_id: Optional[str] = None,
) -> None:
    """
    Route agent_notification-style payloads to WebSocket sessions and optional external cascade.

    - Assigns stable notification_id for cross-surface ACK.
    - Suppresses in-app when user is focused on the same conversation (chat_completion only).
    - Delivers in-app to session WebSockets, optionally excluding originating_surface_id.
    - Runs external cascade when no focused surfaces remain for delivery (see logic below).
    """
    if not user_id or not payload:
        return
    uid = str(user_id).strip()
    prefs = await _load_notification_preferences(uid)
    subtype = _event_subtype(payload)
    conv_id = (payload.get("conversation_id") or "").strip() or None

    notification_id = str(uuid.uuid4())
    out_payload = {**payload, "notification_id": notification_id}
    if out_payload.get("type") is None:
        out_payload["type"] = "agent_notification"

    ws = get_websocket_manager()
    if not ws:
        return

    exclude_surfaces: List[str] = []
    if originating_surface_id:
        exclude_surfaces.append(str(originating_surface_id).strip())

    suppress_in_app = False
    if subtype == "chat_completion" and conv_id:
        if ws.is_conversation_active_on_any_surface(uid, conv_id):
            suppress_in_app = True

    if suppress_in_app:
        await _log_notification(
            uid, notification_id, event_type, conv_id, "suppressed", "suppressed"
        )
        return

    active = ws.get_active_surface_ids(uid)
    has_focused = bool(active)

    try:
        await ws.send_to_session(
            out_payload,
            uid,
            exclude_surface_ids=exclude_surfaces if exclude_surfaces else None,
        )
        await _log_notification(uid, notification_id, event_type, conv_id, "in_app", "sent")
    except Exception as e:
        logger.debug("in-app notification send failed: %s", e)
        await _log_notification(uid, notification_id, event_type, conv_id, "in_app", "failed")

    if has_focused and subtype not in CRITICAL_SUBTYPES:
        return

    ctx = ChannelContext(uid, out_payload, prefs)
    cascade = _resolve_cascade_order(prefs, subtype)
    for ch in cascade:
        handler = CHANNEL_HANDLERS.get(ch)
        if not handler:
            continue
        try:
            ok = await handler(ctx)
            if ok:
                await _log_notification(uid, notification_id, event_type, conv_id, ch, "sent")
                return
            await _log_notification(uid, notification_id, event_type, conv_id, ch, "failed")
        except Exception as e:
            logger.debug("cascade channel %s error: %s", ch, e)
            await _log_notification(uid, notification_id, event_type, conv_id, ch, "failed")


def register_notification_channel(name: str, handler: ChannelSendFn) -> None:
    """Register or replace an external cascade channel (extensibility)."""
    CHANNEL_HANDLERS[str(name).strip().lower()] = handler
