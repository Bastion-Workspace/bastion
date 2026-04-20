"""
Team Heartbeat Utilities - helpers and notification functions for agent team heartbeats.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _heartbeat_enabled(config: Optional[Dict[str, Any]]) -> bool:
    if not config or not isinstance(config, dict):
        return False
    en = config.get("enabled")
    if isinstance(en, bool):
        return en
    if isinstance(en, str):
        return en.lower() in ("true", "1", "yes")
    return False


def heartbeat_requires_open_goals(config: Optional[Dict[str, Any]]) -> bool:
    """When True (default), autonomous heartbeats only enqueue if there is an open goal under 100% progress."""
    if not config or not isinstance(config, dict):
        return True
    v = config.get("require_open_goals")
    if v is False:
        return False
    if isinstance(v, str) and v.strip().lower() in ("false", "0", "no"):
        return False
    return True


def _compute_next_beat_at(heartbeat_config: Optional[Dict[str, Any]], from_time: Optional[datetime] = None) -> Optional[datetime]:
    """Compute next beat UTC from heartbeat_config (schedule_type + interval or timezone-aware cron)."""
    from services.line_heartbeat_schedule import compute_next_beat_at

    return compute_next_beat_at(heartbeat_config, from_time)


async def _fetch_teams_due_heartbeat() -> List[Dict[str, Any]]:
    """Teams with heartbeat enabled, next_beat_at due, and (by default) at least one open goal under 100%.

    If heartbeat_config.require_open_goals is false, lines are eligible even with no open goals (briefing mode).

    When require_open_goals is true (default), scheduling stops when every tracked goal is done/cancelled
    or at 100% progress; use **Run Heartbeat** manually for a final wrap-up if needed.
    """
    from services.database_manager.database_helpers import fetch_all

    rows = await fetch_all(
        """
        SELECT t.id, t.user_id, t.name, t.heartbeat_config, t.next_beat_at, t.last_beat_at
        FROM agent_lines t
        WHERE t.status = 'active'
          AND (t.heartbeat_config->>'enabled')::text IN ('true', '1')
          AND t.next_beat_at IS NOT NULL
          AND t.next_beat_at <= NOW()
          AND (
            lower(coalesce(trim(t.heartbeat_config->>'require_open_goals'), 'true')) IN ('false', '0', 'no')
            OR EXISTS (
              SELECT 1 FROM agent_line_goals g
              WHERE g.line_id = t.id
                AND g.status IN ('active', 'blocked')
                AND COALESCE(g.progress_pct, 0) < 100
            )
          )
        ORDER BY t.next_beat_at ASC NULLS FIRST
        """
    )
    result = []
    for r in rows:
        cfg = r.get("heartbeat_config")
        if isinstance(cfg, str):
            try:
                import json
                cfg = json.loads(cfg) if cfg else {}
            except Exception:
                cfg = {}
        if not _heartbeat_enabled(cfg):
            continue
        result.append({
            "id": str(r["id"]),
            "user_id": r["user_id"],
            "name": r.get("name", ""),
            "heartbeat_config": cfg if isinstance(cfg, dict) else {},
        })
    return result


async def _fetch_teams_with_pending_worker_tasks() -> List[Dict[str, Any]]:
    """Teams that have non-root agents with at least one 'assigned' task and none 'in_progress'.
    Avoids re-dispatching agents who are already working."""
    from services.database_manager.database_helpers import fetch_all

    rows = await fetch_all(
        """
        SELECT DISTINCT t.id AS line_id, t.user_id
        FROM agent_lines t
        JOIN agent_line_memberships m ON m.line_id = t.id AND m.reports_to IS NOT NULL
        JOIN agent_tasks tk ON tk.line_id = t.id
            AND tk.assigned_agent_id = m.agent_profile_id
            AND tk.status = 'assigned'
        WHERE t.status = 'active'
        AND NOT EXISTS (
            SELECT 1 FROM agent_tasks tk2
            WHERE tk2.line_id = t.id
              AND tk2.assigned_agent_id = m.agent_profile_id
              AND tk2.status = 'in_progress'
        )
        """
    )
    return [{"id": str(r["line_id"]), "user_id": r["user_id"]} for r in rows]


async def _agent_has_reports(line_id: str, agent_profile_id: str) -> bool:
    """True if this agent has at least one direct report on the team."""
    from services.database_manager.database_helpers import fetch_one

    row = await fetch_one(
        """
        SELECT EXISTS (
            SELECT 1 FROM agent_line_memberships sub
            WHERE sub.line_id = $1 AND sub.reports_to = (
                SELECT m.id FROM agent_line_memberships m
                WHERE m.line_id = $1 AND m.agent_profile_id = $2
            )
        ) AS has_reports
        """,
        line_id,
        agent_profile_id,
    )
    return bool(row and row.get("has_reports"))


async def _send_team_notification(
    user_id: str,
    line_id: str,
    team_name: str,
    subtype: str,
    message: str = "",
    error_details: Optional[str] = None,
) -> None:
    """Send team event to user via WebSocket (internal notify endpoint)."""
    import os
    base = os.getenv("BACKEND_URL", "http://backend:8000")
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{base}/api/agent-factory/internal/notify-execution-event",
                json={
                    "user_id": user_id,
                    "subtype": subtype,
                    "line_id": line_id,
                    "team_name": team_name,
                    "message": (message or "")[:500],
                    "error_details": (error_details or "")[:500] if error_details else None,
                },
                timeout=5.0,
            )
    except Exception as e:
        logger.warning("Team notification send failed: %s", e)


async def _send_team_execution_status(line_id: str, status: str, agent_id=None) -> None:
    """Send execution_status to team timeline subscribers via internal HTTP endpoint."""
    import os
    import httpx
    base = os.getenv("BACKEND_URL", "http://backend:8000")
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{base}/api/agent-factory/internal/notify-line-timeline",
                json={
                    "line_id": line_id,
                    "payload": {"type": "execution_status", "status": status, "agent_id": agent_id},
                },
                timeout=3.0,
            )
    except Exception as e:
        logger.debug("Send execution_status failed: %s", e)
