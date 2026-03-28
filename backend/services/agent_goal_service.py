"""
Agent Goal service - goal hierarchy for teams.
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _is_uuid(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and _UUID_RE.match(s.strip()))


def _row_to_goal(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not row:
        return {}
    return {
        "id": str(row["id"]),
        "line_id": str(row["line_id"]),
        "parent_goal_id": str(row["parent_goal_id"]) if row.get("parent_goal_id") else None,
        "title": row.get("title", ""),
        "description": row.get("description"),
        "status": row.get("status", "active"),
        "assigned_agent_id": str(row["assigned_agent_id"]) if row.get("assigned_agent_id") else None,
        "priority": row.get("priority", 0),
        "progress_pct": row.get("progress_pct", 0),
        "due_date": row.get("due_date").isoformat() if row.get("due_date") else None,
        "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
        "updated_at": row.get("updated_at").isoformat() if row.get("updated_at") else None,
    }


async def create_goal(
    line_id: str,
    user_id: str,
    title: str,
    description: Optional[str] = None,
    parent_goal_id: Optional[str] = None,
    assigned_agent_id: Optional[str] = None,
    status: str = "active",
    priority: int = 0,
    progress_pct: int = 0,
    due_date: Optional[str] = None,
) -> Dict[str, Any]:
    from services.database_manager.database_helpers import fetch_one, execute

    team = await fetch_one("SELECT id FROM agent_lines WHERE id = $1 AND user_id = $2", line_id, user_id)
    if not team:
        raise ValueError("Team not found")
    await execute(
        """
        INSERT INTO agent_line_goals (line_id, parent_goal_id, title, description, status, assigned_agent_id, priority, progress_pct, due_date)
        VALUES ($1, $2::uuid, $3, $4, $5, $6::uuid, $7, $8, $9::date)
        """,
        line_id,
        parent_goal_id,
        title,
        description or "",
        status,
        assigned_agent_id,
        priority,
        progress_pct,
        due_date,
    )
    row = await fetch_one(
        "SELECT * FROM agent_line_goals WHERE line_id = $1 AND title = $2 ORDER BY created_at DESC LIMIT 1",
        line_id,
        title,
    )
    result = _row_to_goal(row)
    try:
        from services.agent_line_notify import notify_line_event
        await notify_line_event(line_id, "goal_updated", {"goal": result})
    except Exception as e:
        logger.debug("WebSocket goal_updated emit skipped: %s", e)
    return result


async def update_goal(
    goal_id: str,
    user_id: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
    status: Optional[str] = None,
    assigned_agent_id: Optional[str] = None,
    priority: Optional[int] = None,
    progress_pct: Optional[int] = None,
    due_date: Optional[str] = None,
) -> Dict[str, Any]:
    if not _is_uuid(goal_id):
        raise ValueError(
            "goal_id must be the goal's UUID from list_team_goals or get_team_status_board, not the goal title or description."
        )
    from services.database_manager.database_helpers import fetch_one, execute

    row = await fetch_one(
        "SELECT g.* FROM agent_line_goals g JOIN agent_lines t ON t.id = g.line_id WHERE g.id = $1 AND t.user_id = $2",
        goal_id,
        user_id,
    )
    if not row:
        raise ValueError("Goal not found")
    updates = ["updated_at = NOW()"]
    args = []
    pos = 1
    for name, val in [
        ("title", title),
        ("description", description),
        ("status", status),
        ("assigned_agent_id", assigned_agent_id),
        ("priority", priority),
        ("progress_pct", progress_pct),
        ("due_date", due_date),
    ]:
        if val is not None:
            if name == "assigned_agent_id":
                updates.append(f"{name} = ${pos}::uuid")
            elif name == "due_date":
                updates.append(f"{name} = ${pos}::date")
            else:
                updates.append(f"{name} = ${pos}")
            args.append(val)
            pos += 1
    if len(args) == 0:
        return _row_to_goal(row)
    args.append(goal_id)
    await execute(f"UPDATE agent_line_goals SET {', '.join(updates)} WHERE id = ${pos}", *args)
    updated_row = await fetch_one("SELECT * FROM agent_line_goals WHERE id = $1", goal_id)
    result = _row_to_goal(updated_row)
    try:
        from services.agent_line_notify import notify_line_event
        await notify_line_event(str(row["line_id"]), "goal_updated", {"goal": result})
    except Exception as e:
        logger.debug("WebSocket goal_updated emit skipped: %s", e)
    return result


async def delete_goal(goal_id: str, user_id: str) -> None:
    if not _is_uuid(goal_id):
        raise ValueError(
            "goal_id must be the goal's UUID from list_team_goals or get_team_status_board, not the goal title or description."
        )
    from services.database_manager.database_helpers import execute, fetch_one

    row = await fetch_one(
        "SELECT g.id, g.line_id FROM agent_line_goals g JOIN agent_lines t ON t.id = g.line_id WHERE g.id = $1 AND t.user_id = $2",
        goal_id,
        user_id,
    )
    if not row:
        raise ValueError("Goal not found")
    line_id = str(row["line_id"])
    await execute("DELETE FROM agent_line_goals WHERE id = $1", goal_id)
    try:
        from services.agent_line_notify import notify_line_event
        await notify_line_event(line_id, "goal_updated", {"goal_id": goal_id, "deleted": True})
    except Exception as e:
        logger.debug("WebSocket goal_updated emit skipped: %s", e)


def _build_goal_tree(goals: List[Dict], parent_id: Optional[str] = None) -> List[Dict]:
    nodes = []
    for g in goals:
        if (g.get("parent_goal_id") or None) == parent_id:
            children = _build_goal_tree(goals, g["id"])
            g["children"] = children
            nodes.append(g)
    return nodes


async def get_goal_tree(line_id: str, user_id: str) -> List[Dict[str, Any]]:
    if not _is_uuid(line_id):
        raise ValueError(
            "line_id must be the team's UUID from get_team_status_board or pipeline context, not the team name."
        )
    from services.database_manager.database_helpers import fetch_all, fetch_one

    team = await fetch_one("SELECT id FROM agent_lines WHERE id = $1 AND user_id = $2", line_id, user_id)
    if not team:
        return []
    rows = await fetch_all("SELECT * FROM agent_line_goals WHERE line_id = $1 ORDER BY priority DESC, created_at", line_id)
    goals = [_row_to_goal(r) for r in rows]
    return _build_goal_tree(goals, None)


async def get_goal_ancestry(goal_id: str, user_id: str) -> List[Dict[str, Any]]:
    """From leaf goal up to root (for context injection)."""
    if not _is_uuid(goal_id):
        raise ValueError(
            "goal_id must be the goal's UUID from list_team_goals or get_team_status_board, not the goal title or description."
        )
    from services.database_manager.database_helpers import fetch_one, fetch_all

    ancestry = []
    current_id = goal_id
    while current_id:
        row = await fetch_one(
            "SELECT g.* FROM agent_line_goals g JOIN agent_lines t ON t.id = g.line_id WHERE g.id = $1 AND t.user_id = $2",
            current_id,
            user_id,
        )
        if not row:
            break
        ancestry.append(_row_to_goal(row))
        current_id = str(row["parent_goal_id"]) if row.get("parent_goal_id") else None
    return list(reversed(ancestry))


async def update_progress(goal_id: str, user_id: str, progress_pct: int) -> Dict[str, Any]:
    """Set goal progress; at 100% also mark completed so heartbeats stop (see team heartbeat query)."""
    pct = max(0, min(100, progress_pct))
    if pct >= 100:
        return await update_goal(goal_id, user_id, progress_pct=pct, status="completed")
    return await update_goal(goal_id, user_id, progress_pct=pct)


async def update_goal_progress_from_tasks(goal_id: str, user_id: str) -> None:
    """Recompute goal progress from sibling tasks (done/total) and update the goal. Call when a task is marked done."""
    from services.database_manager.database_helpers import fetch_one, fetch_all, execute

    goal_row = await fetch_one(
        "SELECT g.id, g.status, g.line_id FROM agent_line_goals g JOIN agent_lines t ON t.id = g.line_id WHERE g.id = $1 AND t.user_id = $2",
        goal_id,
        user_id,
    )
    if not goal_row:
        return
    rows = await fetch_all(
        "SELECT status FROM agent_tasks WHERE goal_id = $1",
        goal_id,
    )
    total = len([r for r in rows if (r.get("status") or "") not in ("cancelled",)])
    if total == 0:
        return
    done = len([r for r in rows if (r.get("status") or "") == "done"])
    progress_pct = min(100, int(100 * done / total))
    new_status = "completed" if progress_pct >= 100 else None
    updates = ["progress_pct = $1", "updated_at = NOW()"]
    args = [progress_pct]
    if new_status:
        updates.append("status = $2")
        args.append(new_status)
    args.append(goal_id)
    await execute(
        f"UPDATE agent_line_goals SET {', '.join(updates)} WHERE id = ${len(args)}",
        *args,
    )
    updated_row = await fetch_one("SELECT * FROM agent_line_goals WHERE id = $1", goal_id)
    if updated_row:
        try:
            from services.agent_line_notify import notify_line_event
            await notify_line_event(str(goal_row["line_id"]), "goal_updated", {"goal": _row_to_goal(updated_row)})
        except Exception as e:
            logger.debug("WebSocket goal_updated emit skipped: %s", e)

    await _propagate_to_parent_goals(goal_id, user_id)


async def _propagate_to_parent_goals(goal_id: str, user_id: str) -> None:
    """Walk up the goal hierarchy recalculating parent progress from children."""
    from services.database_manager.database_helpers import fetch_one, fetch_all, execute

    parent_row = await fetch_one(
        "SELECT g.id, g.parent_goal_id, g.line_id FROM agent_line_goals g "
        "JOIN agent_lines t ON t.id = g.line_id WHERE g.id = $1 AND t.user_id = $2",
        goal_id,
        user_id,
    )
    if not parent_row or not parent_row.get("parent_goal_id"):
        return
    parent_id = str(parent_row["parent_goal_id"])
    line_id = str(parent_row["line_id"])
    children = await fetch_all(
        "SELECT progress_pct, status FROM agent_line_goals WHERE parent_goal_id = $1",
        parent_id,
    )
    if not children:
        return
    children = [c for c in children if (c.get("status") or "") != "cancelled"]
    if not children:
        return
    avg_pct = min(100, int(sum(c.get("progress_pct", 0) for c in children) / len(children)))
    new_status = "completed" if avg_pct >= 100 else None
    updates = ["progress_pct = $1", "updated_at = NOW()"]
    args = [avg_pct]
    if new_status:
        updates.append("status = $2")
        args.append(new_status)
    args.append(parent_id)
    await execute(
        f"UPDATE agent_line_goals SET {', '.join(updates)} WHERE id = ${len(args)}",
        *args,
    )
    updated_parent = await fetch_one("SELECT * FROM agent_line_goals WHERE id = $1", parent_id)
    if updated_parent:
        try:
            from services.agent_line_notify import notify_line_event
            await notify_line_event(line_id, "goal_updated", {"goal": _row_to_goal(updated_parent)})
        except Exception as e:
            logger.debug("WebSocket goal_updated emit skipped: %s", e)
    await _propagate_to_parent_goals(parent_id, user_id)


async def get_goals_for_agent(agent_profile_id: str, line_id: str, user_id: str) -> List[Dict[str, Any]]:
    from services.database_manager.database_helpers import fetch_all, fetch_one

    team = await fetch_one("SELECT id FROM agent_lines WHERE id = $1 AND user_id = $2", line_id, user_id)
    if not team:
        return []
    rows = await fetch_all(
        "SELECT * FROM agent_line_goals WHERE line_id = $1 AND assigned_agent_id = $2 ORDER BY priority DESC",
        line_id,
        agent_profile_id,
    )
    return [_row_to_goal(r) for r in rows]


async def reset_line_goals_progress(line_id: str, user_id: str) -> int:
    """Reset all goals for the team to progress_pct=0 and status=active (except cancelled). Returns count updated."""
    from services.database_manager.database_helpers import fetch_one

    team = await fetch_one(
        "SELECT id FROM agent_lines WHERE id = $1 AND user_id = $2", line_id, user_id
    )
    if not team:
        return 0
    row = await fetch_one(
        "WITH u AS ("
        "  UPDATE agent_line_goals SET progress_pct = 0, status = 'active', updated_at = NOW() "
        "  WHERE line_id = $1 AND status != 'cancelled' RETURNING id"
        ") SELECT COUNT(*)::int AS n FROM u",
        line_id,
    )
    return (row.get("n") or 0) if row else 0


async def clear_line_agent_memory(line_id: str, user_id: str) -> int:
    """Clear agent_memory for all agents in the team (e.g. last_heartbeat_summary). Returns count deleted."""
    from services.database_manager.database_helpers import fetch_one

    row = await fetch_one(
        "WITH d AS ("
        "  DELETE FROM agent_memory WHERE user_id = $1 AND agent_profile_id IN ("
        "    SELECT agent_profile_id FROM agent_line_memberships WHERE line_id = $2"
        "  ) RETURNING id"
        ") SELECT COUNT(*)::int AS n FROM d",
        user_id,
        line_id,
    )
    return (row.get("n") or 0) if row else 0
