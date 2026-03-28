"""
Agent Message service - inter-agent messages and team timeline.
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MESSAGE_TYPES = (
    "task_assignment",
    "status_update",
    "request",
    "response",
    "delegation",
    "escalation",
    "report",
    "system",
)


def _row_to_message(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not row:
        return {}
    out = {
        "id": str(row["id"]),
        "line_id": str(row["line_id"]),
        "from_agent_id": str(row["from_agent_id"]) if row.get("from_agent_id") else None,
        "to_agent_id": str(row["to_agent_id"]) if row.get("to_agent_id") else None,
        "message_type": row.get("message_type", "report"),
        "content": row.get("content") or "",
        "metadata": row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
        "parent_message_id": str(row["parent_message_id"]) if row.get("parent_message_id") else None,
        "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
    }
    if "from_agent_name" in row:
        out["from_agent_name"] = row.get("from_agent_name")
    if "from_agent_handle" in row:
        out["from_agent_handle"] = row.get("from_agent_handle")
    if "to_agent_name" in row:
        out["to_agent_name"] = row.get("to_agent_name")
    if "to_agent_handle" in row:
        out["to_agent_handle"] = row.get("to_agent_handle")
    if "from_agent_color" in row and row.get("from_agent_color"):
        out["from_agent_color"] = row.get("from_agent_color")
    if "to_agent_color" in row and row.get("to_agent_color"):
        out["to_agent_color"] = row.get("to_agent_color")
    return out


async def create_message(
    line_id: str,
    from_agent_id: Optional[str],
    to_agent_id: Optional[str],
    message_type: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    parent_message_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create an inter-agent message and emit WebSocket timeline update."""
    from services.database_manager.database_helpers import fetch_one, execute

    if message_type not in MESSAGE_TYPES:
        message_type = "report"
    meta = metadata if isinstance(metadata, dict) else {}
    meta_json = json.dumps(meta)

    row_returning = await fetch_one(
        """
        INSERT INTO agent_messages (line_id, from_agent_id, to_agent_id, message_type, content, metadata, parent_message_id)
        VALUES ($1, $2::uuid, $3::uuid, $4, $5, $6::jsonb, $7::uuid)
        RETURNING id
        """,
        line_id,
        from_agent_id,
        to_agent_id,
        message_type,
        content or "",
        meta_json,
        parent_message_id,
    )
    if not row_returning:
        raise RuntimeError("Failed to insert agent message")
    new_id = row_returning["id"]
    row = await fetch_one(
        """
        SELECT m.*, fa.name AS from_agent_name, fa.handle AS from_agent_handle,
               ta.name AS to_agent_name, ta.handle AS to_agent_handle
        FROM agent_messages m
        LEFT JOIN agent_profiles fa ON fa.id = m.from_agent_id
        LEFT JOIN agent_profiles ta ON ta.id = m.to_agent_id
        WHERE m.id = $1
        """,
        new_id,
    )
    msg = _row_to_message(row)

    try:
        import os
        import httpx
        base = os.getenv("BACKEND_URL", "http://backend:8000")
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{base}/api/agent-factory/internal/notify-line-timeline",
                json={"line_id": line_id, "payload": {"type": "team_timeline_update", "message": msg}},
                timeout=3.0,
            )
    except Exception as e:
        logger.debug("WebSocket team timeline emit skipped: %s", e)

    return msg


async def get_line_timeline(
    line_id: str,
    user_id: str,
    limit: int = 50,
    offset: int = 0,
    message_type_filter: Optional[str] = None,
    agent_filter: Optional[str] = None,
    since: Optional[str] = None,
) -> Dict[str, Any]:
    """Paginated timeline for a team. Returns { items, total }."""
    from services.database_manager.database_helpers import fetch_one, fetch_all

    team = await fetch_one("SELECT id FROM agent_lines WHERE id = $1 AND user_id = $2", line_id, user_id)
    if not team:
        return {"items": [], "total": 0}

    where = ["m.line_id = $1"]
    args = [line_id]
    pos = 2
    if message_type_filter:
        where.append(f"m.message_type = ${pos}")
        args.append(message_type_filter)
        pos += 1
    if agent_filter:
        where.append(f"(m.from_agent_id::text = ${pos} OR m.to_agent_id::text = ${pos})")
        args.append(agent_filter)
        pos += 1
    if since:
        where.append(f"m.created_at >= ${pos}::timestamptz")
        args.append(since)
        pos += 1
    where_sql = " AND ".join(where)

    total_row = await fetch_one(
        f"SELECT COUNT(*)::int AS c FROM agent_messages m WHERE {where_sql}",
        *args,
    )
    total = total_row.get("c", 0) if total_row else 0

    args.extend([limit, offset])
    rows = await fetch_all(
        f"""
        SELECT m.*, fa.name AS from_agent_name, fa.handle AS from_agent_handle,
               ta.name AS to_agent_name, ta.handle AS to_agent_handle,
               fm.color AS from_agent_color, tm.color AS to_agent_color
        FROM agent_messages m
        LEFT JOIN agent_profiles fa ON fa.id = m.from_agent_id
        LEFT JOIN agent_profiles ta ON ta.id = m.to_agent_id
        LEFT JOIN agent_line_memberships fm ON fm.line_id = m.line_id AND fm.agent_profile_id = m.from_agent_id
        LEFT JOIN agent_line_memberships tm ON tm.line_id = m.line_id AND tm.agent_profile_id = m.to_agent_id
        WHERE {where_sql}
        ORDER BY m.created_at DESC
        LIMIT ${pos} OFFSET ${pos + 1}
        """,
        *args,
    )
    return {"items": [_row_to_message(r) for r in rows], "total": total}


async def get_agent_messages(
    agent_profile_id: str,
    line_id: str,
    user_id: str,
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    """Messages for one agent (from or to) in a team."""
    from services.database_manager.database_helpers import fetch_all, fetch_one

    team = await fetch_one("SELECT id FROM agent_lines WHERE id = $1 AND user_id = $2", line_id, user_id)
    if not team:
        return {"items": [], "total": 0}

    total_row = await fetch_one(
        """
        SELECT COUNT(*)::int AS c FROM agent_messages
        WHERE line_id = $1 AND (from_agent_id = $2 OR to_agent_id = $2)
        """,
        line_id,
        agent_profile_id,
    )
    total = total_row.get("c", 0) if total_row else 0

    rows = await fetch_all(
        """
        SELECT m.*, fa.name AS from_agent_name, fa.handle AS from_agent_handle,
               ta.name AS to_agent_name, ta.handle AS to_agent_handle,
               fm.color AS from_agent_color, tm.color AS to_agent_color
        FROM agent_messages m
        LEFT JOIN agent_profiles fa ON fa.id = m.from_agent_id
        LEFT JOIN agent_profiles ta ON ta.id = m.to_agent_id
        LEFT JOIN agent_line_memberships fm ON fm.line_id = m.line_id AND fm.agent_profile_id = m.from_agent_id
        LEFT JOIN agent_line_memberships tm ON tm.line_id = m.line_id AND tm.agent_profile_id = m.to_agent_id
        WHERE m.line_id = $1 AND (m.from_agent_id = $2 OR m.to_agent_id = $2)
        ORDER BY m.created_at DESC
        LIMIT $3 OFFSET $4
        """,
        line_id,
        agent_profile_id,
        limit,
        offset,
    )
    return {"items": [_row_to_message(r) for r in rows], "total": total}


async def get_thread(parent_message_id: str, user_id: str) -> List[Dict[str, Any]]:
    """Return threaded conversation: root message + all replies."""
    from services.database_manager.database_helpers import fetch_one, fetch_all

    root = await fetch_one(
        """
        SELECT m.*, fa.name AS from_agent_name, fa.handle AS from_agent_handle,
               ta.name AS to_agent_name, ta.handle AS to_agent_handle,
               fm.color AS from_agent_color, tm.color AS to_agent_color
        FROM agent_messages m
        JOIN agent_lines t ON t.id = m.line_id AND t.user_id = $2
        LEFT JOIN agent_profiles fa ON fa.id = m.from_agent_id
        LEFT JOIN agent_profiles ta ON ta.id = m.to_agent_id
        LEFT JOIN agent_line_memberships fm ON fm.line_id = m.line_id AND fm.agent_profile_id = m.from_agent_id
        LEFT JOIN agent_line_memberships tm ON tm.line_id = m.line_id AND tm.agent_profile_id = m.to_agent_id
        WHERE m.id = $1
        """,
        parent_message_id,
        user_id,
    )
    if not root:
        return []
    replies = await fetch_all(
        """
        SELECT m.*, fa.name AS from_agent_name, fa.handle AS from_agent_handle,
               ta.name AS to_agent_name, ta.handle AS to_agent_handle,
               fm.color AS from_agent_color, tm.color AS to_agent_color
        FROM agent_messages m
        LEFT JOIN agent_profiles fa ON fa.id = m.from_agent_id
        LEFT JOIN agent_profiles ta ON ta.id = m.to_agent_id
        LEFT JOIN agent_line_memberships fm ON fm.line_id = m.line_id AND fm.agent_profile_id = m.from_agent_id
        LEFT JOIN agent_line_memberships tm ON tm.line_id = m.line_id AND tm.agent_profile_id = m.to_agent_id
        WHERE m.parent_message_id = $1
        ORDER BY m.created_at ASC
        """,
        parent_message_id,
    )
    return [_row_to_message(root)] + [_row_to_message(r) for r in replies]


async def clear_line_timeline(line_id: str, user_id: str) -> int:
    """Delete all timeline messages for the team. Returns number of rows deleted. Verifies team ownership."""
    from services.database_manager.database_helpers import execute, fetch_one

    team = await fetch_one("SELECT id FROM agent_lines WHERE id = $1 AND user_id = $2", line_id, user_id)
    if not team:
        raise ValueError("Team not found")
    row = await fetch_one(
        "WITH d AS (DELETE FROM agent_messages WHERE line_id = $1 RETURNING id) SELECT COUNT(*)::int AS n FROM d",
        line_id,
    )
    return (row.get("n") or 0) if row else 0


async def get_line_timeline_summary(line_id: str, user_id: str) -> Dict[str, Any]:
    """Stats for timeline: message count today, active threads, last activity."""
    from datetime import datetime, timezone
    from services.database_manager.database_helpers import fetch_one

    team = await fetch_one("SELECT id FROM agent_lines WHERE id = $1 AND user_id = $2", line_id, user_id)
    if not team:
        return {"message_count_today": 0, "active_threads": 0, "last_activity_at": None}

    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    count_today = await fetch_one(
        "SELECT COUNT(*)::int AS c FROM agent_messages WHERE line_id = $1 AND created_at >= $2",
        line_id,
        today_start,
    )
    threads = await fetch_one(
        "SELECT COUNT(DISTINCT COALESCE(parent_message_id, id))::int AS c FROM agent_messages WHERE line_id = $1",
        line_id,
    )
    last = await fetch_one(
        "SELECT created_at FROM agent_messages WHERE line_id = $1 ORDER BY created_at DESC LIMIT 1",
        line_id,
    )
    return {
        "message_count_today": count_today.get("c", 0) if count_today else 0,
        "active_threads": threads.get("c", 0) if threads else 0,
        "last_activity_at": last["created_at"].isoformat() if last and last.get("created_at") else None,
    }
