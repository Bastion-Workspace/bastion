"""
WebSocket bell notification when an assistant message is persisted after streaming.
Clients suppress when the same conversation is already selected in that tab (sessionStorage).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

NOTIFICATION_SUBTYPE = "chat_completion"
PREVIEW_MAX_LEN = 200
TITLE_MAX_LEN = 120


async def resolve_display_agent_name(
    user_id: str,
    metadata_received: Dict[str, Any],
    agent_name_used: Optional[str],
) -> str:
    """
    Human-readable agent label for bell notifications.
    Prefer stream metadata (profile display name), then DB name by agent_profile_id, then raw agent id.
    """
    meta = metadata_received or {}
    display = (meta.get("agent_display_name") or meta.get("agent_profile_name") or "").strip()
    if display:
        return display[:200]

    pid = (meta.get("agent_profile_id") or "").strip()
    if pid and user_id:
        try:
            import uuid as _uuid

            from services.database_manager.database_helpers import fetch_one

            rls = {"user_id": user_id, "user_role": "user"}
            row = await fetch_one(
                "SELECT name FROM agent_profiles WHERE id = $1::uuid AND user_id = $2",
                _uuid.UUID(pid),
                user_id,
                rls_context=rls,
            )
            if row and (row.get("name") or "").strip():
                return str(row["name"]).strip()[:200]
        except (ValueError, TypeError):
            pass
        except Exception as e:
            logger.debug("resolve_display_agent_name profile lookup skipped: %s", e)

    raw = (agent_name_used or meta.get("delegated_agent") or "").strip()
    if raw.lower() in ("custom_agent", "orchestrator", "system", "unknown", ""):
        return "Assistant"
    return raw[:200]


def _preview_from_body(text: str, max_len: int = PREVIEW_MAX_LEN) -> str:
    if not text or not str(text).strip():
        return ""
    s = re.sub(r"\s+", " ", str(text).strip())
    return s[:max_len] + ("…" if len(s) > max_len else "")


async def notify_chat_reply_ready(
    user_id: str,
    conversation_id: str,
    *,
    response_text: str,
    agent_name: Optional[str] = None,
    conversation_title: Optional[str] = None,
    originating_surface_id: Optional[str] = None,
) -> None:
    """
    Send agent_notification (subtype chat_completion) via NotificationRouter.
    Failures are logged and ignored so chat persistence is never blocked.
    """
    if not user_id or not conversation_id or not str(conversation_id).strip():
        return
    try:
        preview = _preview_from_body(response_text)
        if not preview:
            preview = "Reply ready"

        agent = (agent_name or "").strip() or "Assistant"
        title_raw = (conversation_title or "").strip() or "Reply ready"
        title = title_raw[:TITLE_MAX_LEN] + ("…" if len(title_raw) > TITLE_MAX_LEN else "")

        now = datetime.now(timezone.utc)
        payload = {
            "type": "agent_notification",
            "subtype": NOTIFICATION_SUBTYPE,
            "conversation_id": str(conversation_id).strip(),
            "agent_name": agent,
            "title": title,
            "preview": preview,
            "timestamp": now.isoformat(),
        }
        from services.notification_router import route_notification

        await route_notification(
            user_id,
            NOTIFICATION_SUBTYPE,
            payload,
            originating_surface_id=originating_surface_id,
        )
    except Exception as e:
        logger.debug("chat completion notification skipped: %s", e)
