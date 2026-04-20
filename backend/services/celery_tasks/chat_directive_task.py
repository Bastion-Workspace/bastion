"""
Chat directive timeline injection.

Used to persist @team chat messages into the team timeline (agent_messages) so that
subsequent heartbeats can incorporate them, and to optionally wake an idle team by
enqueuing an ad-hoc heartbeat.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from services.celery_app import celery_app
from services.celery_tasks.async_runner import run_async

logger = logging.getLogger(__name__)


def _parse_heartbeat_config(raw: Any) -> Dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            import json

            parsed = json.loads(raw) if raw else {}
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _coerce_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            # DB timestamps are typically ISO without Z; handle both.
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return None
    return None


def _heartbeat_enabled(cfg: Dict[str, Any]) -> bool:
    en = cfg.get("enabled")
    if isinstance(en, bool):
        return en
    if isinstance(en, str):
        return en.lower() in ("true", "1", "yes")
    return False


def _interval_seconds(cfg: Dict[str, Any]) -> Optional[int]:
    try:
        interval_sec = cfg.get("interval_seconds")
        if interval_sec is None and "interval" in cfg:
            interval_sec = int(cfg.get("interval", 0)) * 60
        if interval_sec and int(interval_sec) > 0:
            return int(interval_sec)
    except Exception:
        return None
    return None


@celery_app.task(
    name="services.celery_tasks.chat_directive_task.post_chat_directive_to_timeline",
    soft_time_limit=120,
)
def post_chat_directive_to_timeline(
    line_id: str,
    user_id: str,
    user_message: str,
    leader_response: str,
    conversation_id: str = "",
    ceo_profile_id: str = "",
) -> Dict[str, Any]:
    """
    Persist the @team chat message to the team timeline and optionally wake the team
    via an ad-hoc heartbeat.
    """
    from services import agent_message_service
    from services.database_manager.database_helpers import fetch_one

    now = datetime.now(timezone.utc)
    conv_id = (conversation_id or "").strip()
    meta = {
        "source": "chat",
        "conversation_id": conv_id,
        "created_at": now.isoformat(),
    }

    directive_msg = None
    try:
        directive_msg = run_async(
            agent_message_service.create_message(
                line_id=line_id,
                from_agent_id=None,
                to_agent_id=None,
                message_type="user_directive",
                content=(user_message or "").strip(),
                metadata=meta,
                parent_message_id=None,
                user_id=user_id,
            )
        )
    except Exception as e:
        logger.warning("Failed to create user_directive timeline message: %s", e)

    parent_id = (directive_msg or {}).get("id")

    try:
        from_id = (ceo_profile_id or "").strip() or None
        run_async(
            agent_message_service.create_message(
                line_id=line_id,
                from_agent_id=from_id,
                to_agent_id=None,
                message_type="response",
                content=(leader_response or "").strip()[:5000],
                metadata={**meta, "in_reply_to_conversation_id": conv_id},
                parent_message_id=parent_id,
                user_id=user_id,
            )
        )
    except Exception as e:
        logger.warning("Failed to create leader response timeline message: %s", e)

    # Wake-up logic: if the team is idle (no scheduled beat soon), enqueue a manual heartbeat.
    try:
        row = run_async(
            fetch_one(
                "SELECT status, heartbeat_config, next_beat_at FROM agent_lines WHERE id = $1 AND user_id = $2",
                line_id,
                user_id,
            )
        )
        if not row:
            return {"ok": False, "woke": False, "message": "Line not found"}

        status = (row.get("status") or "").lower()
        if status != "active":
            return {"ok": True, "woke": False, "message": f"Line status is {status} (no wake)"}

        cfg = _parse_heartbeat_config(row.get("heartbeat_config"))
        enabled = _heartbeat_enabled(cfg)
        next_at = _coerce_dt(row.get("next_beat_at"))

        wake = False
        if not enabled:
            wake = True
        else:
            interval_sec = _interval_seconds(cfg)
            if not next_at:
                wake = True
            elif interval_sec:
                # If the next scheduled beat is more than ~2 intervals away, wake.
                delta = (next_at - now).total_seconds()
                if delta > float(interval_sec) * 2.0:
                    wake = True

        if wake:
            from services.celery_tasks.team_heartbeat_tasks import execute_team_heartbeat

            execute_team_heartbeat.apply_async(
                kwargs={"line_id": line_id, "user_id": user_id, "from_manual_trigger": True},
                countdown=30,
            )
            return {"ok": True, "woke": True, "message": "Ad-hoc heartbeat enqueued"}

        return {"ok": True, "woke": False, "message": "Scheduled heartbeat is due soon"}
    except Exception as e:
        logger.warning("Directive wake-up check failed: %s", e)
        return {"ok": True, "woke": False, "message": f"Wake-up check failed: {e}"}

