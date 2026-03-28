"""
Agent Task service - task/ticket system for teams.
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

TASK_STATUSES = ("backlog", "assigned", "in_progress", "review", "done", "cancelled")
TRANSITIONS = {
    "backlog": ("assigned", "cancelled"),
    "assigned": ("in_progress", "review", "backlog", "cancelled"),
    "in_progress": ("review", "assigned", "cancelled"),
    "review": ("done", "in_progress", "cancelled"),
    "done": (),
    "cancelled": (),
}


def _row_to_task(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not row:
        return {}
    return {
        "id": str(row["id"]),
        "line_id": str(row["line_id"]),
        "title": row.get("title", ""),
        "description": row.get("description"),
        "status": row.get("status", "backlog"),
        "assigned_agent_id": str(row["assigned_agent_id"]) if row.get("assigned_agent_id") else None,
        "created_by_agent_id": str(row["created_by_agent_id"]) if row.get("created_by_agent_id") else None,
        "goal_id": str(row["goal_id"]) if row.get("goal_id") else None,
        "priority": row.get("priority", 0),
        "thread_id": str(row["thread_id"]) if row.get("thread_id") else None,
        "execution_id": str(row["execution_id"]) if row.get("execution_id") else None,
        "metadata": row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
        "due_date": row.get("due_date").isoformat() if row.get("due_date") else None,
        "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
        "updated_at": row.get("updated_at").isoformat() if row.get("updated_at") else None,
    }


async def create_task(
    line_id: str,
    user_id: str,
    title: str,
    description: Optional[str] = None,
    assigned_agent_id: Optional[str] = None,
    goal_id: Optional[str] = None,
    priority: int = 0,
    created_by_agent_id: Optional[str] = None,
    due_date: Optional[str] = None,
) -> Dict[str, Any]:
    from services.database_manager.database_helpers import fetch_one, execute
    from services import agent_line_service

    if not _is_uuid(line_id):
        raise ValueError(
            "line_id must be a valid UUID. In team context use the line_id from get_team_status_board."
        )
    if goal_id is not None and goal_id and not _is_uuid(goal_id):
        raise ValueError(
            "goal_id must be the goal's UUID from list_team_goals or get_team_status_board, not the goal title or description."
        )
    team = await fetch_one("SELECT id FROM agent_lines WHERE id = $1 AND user_id = $2", line_id, user_id)
    if not team:
        raise ValueError("Team not found")

    resolved_assigned_id = assigned_agent_id
    if assigned_agent_id and not _is_uuid(assigned_agent_id):
        team_with_members = await agent_line_service.get_line(line_id, user_id)
        members = team_with_members.get("members") or []
        needle = (assigned_agent_id or "").strip().lower()
        for m in members:
            pid = m.get("agent_profile_id")
            if not pid:
                continue
            name = (m.get("agent_name") or "").strip().lower()
            handle = (m.get("agent_handle") or "").strip().lower()
            if needle == name or needle == handle or needle in (name, handle):
                resolved_assigned_id = str(pid)
                break
        if not _is_uuid(resolved_assigned_id):
            raise ValueError(
                f"No team member matching '{assigned_agent_id}'. "
                "Use agent_profile_id from get_team_status_board (e.g. create_task_for_agent(..., assigned_agent_id=<uuid>))."
            )

    status = "assigned" if resolved_assigned_id else "backlog"
    await execute(
        """
        INSERT INTO agent_tasks (line_id, title, description, status, assigned_agent_id, created_by_agent_id, goal_id, priority, due_date)
        VALUES ($1, $2, $3, $4, $5::uuid, $6::uuid, $7::uuid, $8, $9::date)
        """,
        line_id,
        title,
        description or "",
        status,
        resolved_assigned_id,
        created_by_agent_id,
        goal_id,
        priority,
        due_date,
    )
    row = await fetch_one(
        "SELECT * FROM agent_tasks WHERE line_id = $1 AND title = $2 ORDER BY created_at DESC LIMIT 1",
        line_id,
        title,
    )
    result = _row_to_task(row)
    if resolved_assigned_id and status == "assigned":
        try:
            from services.celery_tasks.team_heartbeat_tasks import dispatch_single_worker
            dispatch_single_worker.apply_async(
                args=[line_id, user_id, resolved_assigned_id],
                countdown=2,
            )
        except Exception as e:
            logger.warning("Failed to enqueue dispatch_single_worker after create_task: %s", e)
    try:
        from services.agent_line_notify import notify_line_event
        await notify_line_event(line_id, "task_updated", {"task": result})
    except Exception as e:
        logger.debug("WebSocket task_updated emit skipped: %s", e)
    return result


async def update_task(
    task_id: str,
    user_id: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
    status: Optional[str] = None,
    assigned_agent_id: Optional[str] = None,
    goal_id: Optional[str] = None,
    priority: Optional[int] = None,
    thread_id: Optional[str] = None,
    execution_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    due_date: Optional[str] = None,
) -> Dict[str, Any]:
    if not _is_uuid(task_id):
        raise ValueError(
            "task_id must be the task's UUID from check_my_tasks or get_team_status_board, not the task title."
        )
    if goal_id is not None and goal_id and not _is_uuid(goal_id):
        raise ValueError(
            "goal_id must be the goal's UUID from list_team_goals or get_team_status_board, not the goal title or description."
        )
    from services.database_manager.database_helpers import fetch_one, execute

    row = await fetch_one(
        "SELECT t.* FROM agent_tasks t JOIN agent_lines tm ON tm.id = t.line_id WHERE t.id = $1 AND tm.user_id = $2",
        task_id,
        user_id,
    )
    if not row:
        raise ValueError("Task not found")
    updates = ["updated_at = NOW()"]
    args = []
    pos = 1
    for name, val in [
        ("title", title),
        ("description", description),
        ("status", status),
        ("assigned_agent_id", assigned_agent_id),
        ("goal_id", goal_id),
        ("priority", priority),
        ("thread_id", thread_id),
        ("execution_id", execution_id),
        ("metadata", metadata),
        ("due_date", due_date),
    ]:
        if val is not None:
            if name in ("assigned_agent_id", "goal_id", "thread_id", "execution_id"):
                updates.append(f"{name} = ${pos}::uuid")
            elif name == "metadata":
                updates.append(f"{name} = ${pos}::jsonb")
            elif name == "due_date":
                updates.append(f"{name} = ${pos}::date")
            else:
                updates.append(f"{name} = ${pos}")
            args.append(val)
            pos += 1
    if len(args) == 0:
        return _row_to_task(row)
    args.append(task_id)
    await execute(f"UPDATE agent_tasks SET {', '.join(updates)} WHERE id = ${pos}", *args)
    updated_row = await fetch_one("SELECT * FROM agent_tasks WHERE id = $1", task_id)
    result = _row_to_task(updated_row)
    try:
        from services.agent_line_notify import notify_line_event
        await notify_line_event(str(row["line_id"]), "task_updated", {"task": result})
    except Exception as e:
        logger.debug("WebSocket task_updated emit skipped: %s", e)
    return result


async def assign_task(task_id: str, agent_profile_id: str, user_id: str) -> Dict[str, Any]:
    from services import agent_message_service
    from services.database_manager.database_helpers import fetch_one

    task = await get_task(task_id, user_id)
    if not task:
        raise ValueError("Task not found")
    line_id = task["line_id"]
    updated = await update_task(
        task_id, user_id, assigned_agent_id=agent_profile_id, status="assigned"
    )
    try:
        await agent_message_service.create_message(
            line_id=line_id,
            from_agent_id=task.get("created_by_agent_id"),
            to_agent_id=agent_profile_id,
            message_type="task_assignment",
            content=f"Task assigned: {task.get('title', '')}",
            metadata={"task_id": task_id},
            user_id=user_id,
        )
    except Exception as e:
        logger.warning("Failed to create task_assignment message: %s", e)
    return updated


async def get_agent_work_queue(
    agent_profile_id: str, line_id: str, user_id: str
) -> List[Dict[str, Any]]:
    from services.database_manager.database_helpers import fetch_all, fetch_one

    team = await fetch_one("SELECT id FROM agent_lines WHERE id = $1 AND user_id = $2", line_id, user_id)
    if not team:
        return []
    rows = await fetch_all(
        "SELECT * FROM agent_tasks WHERE line_id = $1 AND assigned_agent_id = $2 AND status NOT IN ('done', 'cancelled') ORDER BY priority DESC, created_at",
        line_id,
        agent_profile_id,
    )
    return [_row_to_task(r) for r in rows]


def _can_transition(current: str, new: str) -> bool:
    allowed = TRANSITIONS.get(current, ())
    return new in allowed


async def transition_task(task_id: str, user_id: str, new_status: str) -> Dict[str, Any]:
    if new_status not in TASK_STATUSES:
        raise ValueError(f"Invalid status: {new_status}")
    task = await get_task(task_id, user_id)
    if not task:
        raise ValueError("Task not found")
    current = task.get("status", "backlog")
    if current == new_status:
        return task
    if not _can_transition(current, new_status):
        if current in ("done", "cancelled"):
            raise ValueError(
                f"Task is already {current}; status cannot be changed. "
                "Completed or cancelled tasks are terminal."
            )
        allowed = TRANSITIONS.get(current, ())
        allowed_txt = ", ".join(allowed) if allowed else "none"
        raise ValueError(
            f"Cannot transition from '{current}' to '{new_status}'. "
            f"Allowed next statuses: {allowed_txt}."
        )
    updated = await update_task(task_id, user_id, status=new_status)
    if new_status == "done" and updated.get("goal_id"):
        try:
            from services import agent_goal_service
            await agent_goal_service.update_goal_progress_from_tasks(updated["goal_id"], user_id)
        except Exception as e:
            logger.warning("Auto-update goal progress after task done failed: %s", e)
    return updated


async def list_line_tasks(
    line_id: str,
    user_id: str,
    status_filter: Optional[str] = None,
    agent_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    from services.database_manager.database_helpers import fetch_all, fetch_one

    team = await fetch_one("SELECT id FROM agent_lines WHERE id = $1 AND user_id = $2", line_id, user_id)
    if not team:
        return []
    q = "SELECT * FROM agent_tasks WHERE line_id = $1"
    args = [line_id]
    pos = 2
    if status_filter:
        q += f" AND status = ${pos}"
        args.append(status_filter)
        pos += 1
    if agent_filter:
        q += f" AND assigned_agent_id = ${pos}"
        args.append(agent_filter)
        pos += 1
    q += " ORDER BY priority DESC, created_at"
    rows = await fetch_all(q, *args)
    return [_row_to_task(r) for r in rows]


async def get_task(task_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    if not _is_uuid(task_id):
        raise ValueError(
            "task_id must be the task's UUID from check_my_tasks or get_team_status_board, not the task title."
        )
    from services.database_manager.database_helpers import fetch_one

    row = await fetch_one(
        "SELECT t.* FROM agent_tasks t JOIN agent_lines tm ON tm.id = t.line_id WHERE t.id = $1 AND tm.user_id = $2",
        task_id,
        user_id,
    )
    if not row:
        return None
    task = _row_to_task(row)
    thread_id = task.get("thread_id")
    if thread_id:
        from services import agent_message_service
        try:
            thread = await agent_message_service.get_thread(thread_id, user_id)
            task["thread"] = thread
        except Exception:
            task["thread"] = []
    return task


async def delete_task(task_id: str, user_id: str) -> None:
    from services.database_manager.database_helpers import execute, fetch_one

    row = await fetch_one(
        "SELECT t.id, t.line_id FROM agent_tasks t JOIN agent_lines tm ON tm.id = t.line_id WHERE t.id = $1 AND tm.user_id = $2",
        task_id,
        user_id,
    )
    if not row:
        raise ValueError("Task not found")
    line_id = str(row["line_id"])
    await execute("DELETE FROM agent_tasks WHERE id = $1", task_id)
    try:
        from services.agent_line_notify import notify_line_event
        await notify_line_event(line_id, "task_updated", {"task_id": task_id, "deleted": True})
    except Exception as e:
        logger.debug("WebSocket task_updated emit skipped: %s", e)


async def delete_all_line_tasks(line_id: str, user_id: str) -> int:
    """Delete all tasks for the team. Returns number of rows deleted. Verifies team ownership. Goals are not modified."""
    from services.database_manager.database_helpers import fetch_one

    team = await fetch_one("SELECT id FROM agent_lines WHERE id = $1 AND user_id = $2", line_id, user_id)
    if not team:
        raise ValueError("Team not found")
    row = await fetch_one(
        "WITH d AS (DELETE FROM agent_tasks WHERE line_id = $1 RETURNING id) SELECT COUNT(*)::int AS n FROM d",
        line_id,
    )
    return (row.get("n") or 0) if row else 0
