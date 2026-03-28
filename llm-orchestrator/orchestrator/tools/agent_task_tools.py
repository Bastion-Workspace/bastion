"""
Agent Task Tools - task/ticket system for teams (create, work queue, status, escalate).
"""

import logging
import re
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action
from orchestrator.utils.line_context import line_id_from_metadata

logger = logging.getLogger(__name__)

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _is_uuid(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and _UUID_RE.match(s.strip()))


class CreateTaskForAgentInputs(BaseModel):
    """Inputs for create_task_for_agent. Team is always the current team (from context)."""
    title: str = Field(description="Task title")
    assigned_agent_id: str = Field(
        description="Agent to assign to: name, @handle, or agent_profile_id UUID from get_team_status_board"
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional detailed description of what the task requires"
    )
    goal_id: Optional[str] = Field(
        default=None,
        description="Optional: goal UUID from list_team_goals / briefing to link this task to that goal. Omit for a standalone task. Do not pass task_id—new tasks do not exist yet; the tool returns task_id.",
    )
    priority: int = Field(
        default=0,
        description="Priority level; 0 = normal, higher = more urgent"
    )


class CreateTaskForAgentOutputs(BaseModel):
    """Outputs from create_task_for_agent."""
    formatted: str = Field(description="Human-readable result")
    task_id: str = Field(default="", description="Created task ID")
    success: bool = Field(description="Whether creation succeeded")


async def create_task_for_agent_tool(
    title: str,
    assigned_agent_id: str,
    description: Optional[str] = None,
    goal_id: Optional[str] = None,
    priority: int = 0,
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a task and assign it to an agent on your current team. This is the ONLY way to assign work so that the worker agent is actually invoked. Writing to the workspace or sending a message does NOT create a task or trigger the worker; only create_task_for_agent does. If this call returns success=false or an error, the assignment is NOT complete—fix the error and retry. Team is taken from context (you must be running in a team)."""
    meta = _pipeline_metadata or {}
    resolved_team = line_id_from_metadata(meta)
    created_by = meta.get("agent_profile_id")
    if not resolved_team:
        return {"formatted": "Not in an agent line context; create_task_for_agent is only available when running as a line member.", "task_id": "", "success": False}
    if goal_id and not _is_uuid(goal_id):
        return {
            "formatted": "goal_id must be the goal's UUID (id from list_team_goals), not the goal title or description.",
            "task_id": "",
            "success": False,
        }
    try:
        client = await get_backend_tool_client()
        result = await client.create_agent_task(
            team_id=resolved_team,
            user_id=user_id,
            title=title,
            description=description,
            assigned_agent_id=assigned_agent_id,
            goal_id=goal_id,
            priority=priority,
            created_by_agent_id=created_by,
        )
        ok = result.get("success", False)
        tid = result.get("task_id", "")
        return {
            "formatted": f"Task created and assigned: {title} (task_id: {tid})" if ok and tid else (f"Task created and assigned: {title}" if ok else (result.get("error") or "Failed.")),
            "task_id": tid,
            "success": ok,
        }
    except Exception as e:
        logger.warning("create_task_for_agent failed: %s", e)
        return {"formatted": str(e), "task_id": "", "success": False}


class CheckMyTasksInputs(BaseModel):
    """Inputs for check_my_tasks. line_id and agent_profile_id are injected from pipeline context."""

    pass


class CheckMyTasksOutputs(BaseModel):
    """Outputs from check_my_tasks."""
    formatted: str = Field(description="Human-readable work queue")
    success: bool = Field(description="Whether fetch succeeded")


async def check_my_tasks_tool(
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Get the current agent's work queue (tasks assigned to me)."""
    meta = _pipeline_metadata or {}
    resolved_team = line_id_from_metadata(meta)
    agent_id = meta.get("agent_profile_id")
    if not resolved_team or not agent_id:
        return {"formatted": "line_id and agent_profile_id (from context) are required.", "success": False}
    try:
        client = await get_backend_tool_client()
        result = await client.get_agent_work_queue(
            team_id=resolved_team, agent_profile_id=agent_id, user_id=user_id
        )
        if not result.get("success"):
            return {"formatted": result.get("error", "Failed to load tasks"), "success": False}
        tasks = result.get("tasks", [])
        if not tasks:
            return {"formatted": "No tasks assigned to you.", "success": True}
        lines = [f"You have {len(tasks)} task(s):"]
        for t in tasks:
            line = f"- [{t.get('status', '')}] {t.get('title', '')} (id: {t.get('id', '')})"
            if t.get("goal_id"):
                line += f" [goal_id: {t['goal_id']}]"
            priority = t.get("priority", 0)
            if priority != 0:
                line += f" priority={priority}"
            due_date = t.get("due_date")
            if due_date:
                line += f", due {due_date}"
            lines.append(line)
            if t.get("description"):
                desc = (t["description"] or "").strip()[:300]
                if desc:
                    lines.append(f"  Description: {desc}")
        return {"formatted": "\n".join(lines), "success": True}
    except Exception as e:
        logger.warning("check_my_tasks failed: %s", e)
        return {"formatted": str(e), "success": False}


class UpdateTaskStatusInputs(BaseModel):
    """Inputs for update_task_status."""
    task_id: str = Field(
        description="Task UUID (the 'id' field from check_my_tasks or get_team_status_board). Do NOT use the task title."
    )
    new_status: str = Field(
        description=(
            "Target status: backlog, assigned, in_progress, review, done, or cancelled. "
            "Rules: backlog→assigned|cancelled; assigned→in_progress|review|backlog|cancelled; "
            "in_progress→review|assigned|cancelled; review→done|in_progress|cancelled. "
            "done and cancelled are terminal—do not call this tool to move a completed or cancelled task to another status."
        )
    )


class UpdateTaskStatusOutputs(BaseModel):
    """Outputs from update_task_status."""
    formatted: str = Field(description="Human-readable result")
    success: bool = Field(description="Whether update succeeded")


async def update_task_status_tool(
    task_id: str,
    new_status: str,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Update a task's status. task_id must be the task's UUID (id from check_my_tasks), not the task title.

    Only valid state-machine transitions are accepted; done and cancelled tasks cannot be changed.
    If the tool returns success=false, read the message and do not retry the same transition.
    """
    if not _is_uuid(task_id):
        return {
            "formatted": "task_id must be the task's UUID (the 'id' from check_my_tasks or get_team_status_board), not the task title. Call check_my_tasks to get task ids.",
            "success": False,
        }
    try:
        client = await get_backend_tool_client()
        result = await client.update_task_status(task_id=task_id, user_id=user_id, new_status=new_status)
        ok = result.get("success", False)
        err = (result.get("error") or "").strip()
        if ok:
            msg = f"Task status set to {new_status}."
        else:
            msg = err or "Update failed."
        return {
            "formatted": msg,
            "success": ok,
        }
    except Exception as e:
        logger.warning("update_task_status failed: %s", e)
        return {"formatted": str(e), "success": False}


class EscalateTaskInputs(BaseModel):
    """Inputs for escalate_task."""
    task_id: str = Field(
        description="Task UUID (id from check_my_tasks or get_team_status_board), not the task title"
    )
    assigned_agent_id: str = Field(
        description="Agent to escalate to: name, @handle, or agent_profile_id UUID from get_team_status_board (e.g. manager)"
    )


class EscalateTaskOutputs(BaseModel):
    """Outputs from escalate_task."""
    formatted: str = Field(description="Human-readable result")
    success: bool = Field(description="Whether escalation succeeded")


async def escalate_task_tool(
    task_id: str,
    assigned_agent_id: str,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Escalate a task by reassigning to another agent. task_id must be the task's UUID (id from check_my_tasks), not the task title."""
    if not _is_uuid(task_id):
        return {
            "formatted": "task_id must be the task's UUID (the 'id' from check_my_tasks or get_team_status_board), not the task title.",
            "success": False,
        }
    try:
        client = await get_backend_tool_client()
        result = await client.assign_task_to_agent(
            task_id=task_id, agent_profile_id=assigned_agent_id, user_id=user_id
        )
        ok = result.get("success", False)
        return {
            "formatted": f"Task escalated to agent {assigned_agent_id}." if ok else (result.get("error") or "Failed."),
            "success": ok,
        }
    except Exception as e:
        logger.warning("escalate_task failed: %s", e)
        return {"formatted": str(e), "success": False}


register_action(
    name="create_task_for_agent",
    category="tasks",
    description="Create a task and assign it to an agent. REQUIRED to trigger worker execution—write_to_workspace and send_to_agent do NOT schedule workers; only this tool does. If it fails, retry until success.",
    inputs_model=CreateTaskForAgentInputs,
    params_model=None,
    outputs_model=CreateTaskForAgentOutputs,
    tool_function=create_task_for_agent_tool,
)
register_action(
    name="check_my_tasks",
    category="tasks",
    description="Get the current agent's work queue (tasks assigned to me)",
    inputs_model=CheckMyTasksInputs,
    params_model=None,
    outputs_model=CheckMyTasksOutputs,
    tool_function=check_my_tasks_tool,
)
register_action(
    name="update_task_status",
    category="tasks",
    description=(
        "Move a task along its workflow. Respect the status machine: "
        "done and cancelled are final—never try to set review/in_progress on an already-done task. "
        "On failure, use the error text; do not repeat an invalid transition."
    ),
    inputs_model=UpdateTaskStatusInputs,
    params_model=None,
    outputs_model=UpdateTaskStatusOutputs,
    tool_function=update_task_status_tool,
)
register_action(
    name="escalate_task",
    category="tasks",
    description="Escalate a task by reassigning to another agent (e.g. manager)",
    inputs_model=EscalateTaskInputs,
    params_model=None,
    outputs_model=EscalateTaskOutputs,
    tool_function=escalate_task_tool,
)
