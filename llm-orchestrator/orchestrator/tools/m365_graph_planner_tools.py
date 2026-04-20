"""Microsoft Planner tools via Graph (connection-scoped)."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.tools.m365_invoke_common import invoke_m365_graph
from orchestrator.utils.action_io_registry import register_action


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _format_planner_plans(plans: List[Dict[str, Any]]) -> str:
    if not plans:
        return "Microsoft Planner: no plans found."
    lines = [
        "Microsoft Planner plans. Use the id values below for get_planner_tasks.",
        f"count: {len(plans)}",
        "",
    ]
    for i, p in enumerate(plans, 1):
        lines.append(f'{i}. "{p.get("title", "")}"')
        lines.append(f"   id: {p.get('id', '')}")
        if p.get("owner"):
            lines.append(f"   owner: {p['owner']}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _format_planner_tasks(tasks: List[Dict[str, Any]], plan_id: str) -> str:
    if not tasks:
        return f"Microsoft Planner: no tasks in plan {plan_id}."
    lines = [
        "Microsoft Planner tasks. Use only these titles and ids in your reply.",
        f"plan_id: {plan_id}",
        f"count: {len(tasks)}",
        "",
    ]
    for i, t in enumerate(tasks, 1):
        title = t.get("title") or "(no title)"
        lines.append(f'{i}. Title: "{title}"')
        lines.append(f"   task_id: {t.get('id', '')}")
        pct = t.get("percent_complete")
        if pct is not None:
            lines.append(f"   percent_complete: {pct}")
        due = t.get("due_datetime", "")
        if due:
            lines.append(f"   due: {due}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _format_planner_mutation(operation: str, success: bool, task_id: str, error: str, title: str = "") -> str:
    if not success:
        return f"Planner {operation} failed: {error or 'unknown error'}"
    parts = [f"Planner {operation} succeeded."]
    if title:
        parts.append(f'  title: "{title}"')
    if task_id:
        parts.append(f"  task_id: {task_id}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Input / Output models
# ---------------------------------------------------------------------------

class ConnParams(BaseModel):
    connection_id: Optional[int] = Field(default=None, description="Microsoft 365 connection id")


class PlannerPlansOutputs(BaseModel):
    plans: List[Dict[str, Any]] = Field(default_factory=list)
    formatted: str = Field(description="Human-readable summary")


class GetPlannerTasksInputs(ConnParams):
    plan_id: str = Field(
        description="Plan id from list_planner_plans (the long id string, not the plan title)"
    )


class PlannerTasksOutputs(BaseModel):
    tasks: List[Dict[str, Any]] = Field(default_factory=list)
    formatted: str = Field(description="Human-readable summary")


class CreatePlannerTaskInputs(ConnParams):
    plan_id: str = Field(
        description="Plan id from list_planner_plans (the long id string, not the plan title)"
    )
    title: str = Field(description="Task title")
    bucket_id: str = Field(
        default="",
        description="Optional bucket id within the plan"
    )


class PlannerMutationOutputs(BaseModel):
    task_id: str = Field(default="")
    success: bool = Field(default=True)
    formatted: str = Field(description="Human-readable summary")


class UpdatePlannerTaskInputs(ConnParams):
    task_id: str = Field(
        description="Task id from get_planner_tasks (the long id string, not the task title)"
    )
    title: Optional[str] = Field(default=None, description="New title (omit to leave unchanged)")
    percent_complete: Optional[int] = Field(
        default=None, ge=0, le=100,
        description="Completion percentage 0-100 (omit to leave unchanged)"
    )
    due_datetime: Optional[str] = Field(
        default=None,
        description="ISO 8601 due date/time (omit to leave unchanged)"
    )


class OkOutputs(BaseModel):
    success: bool = Field(default=True)
    formatted: str = Field(description="Human-readable summary")


class DeletePlannerTaskInputs(ConnParams):
    task_id: str = Field(
        description="Task id from get_planner_tasks (the long id string, not the task title)"
    )
    etag: str = Field(default="", description="Task etag for concurrency (optional)")


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

async def list_planner_plans_tool(
    user_id: str = "system",
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    out = await invoke_m365_graph("list_planner_plans", user_id, connection_id, {})
    plans = out.get("plans") or []
    plans = plans if isinstance(plans, list) else []
    err = out.get("error")
    body = _format_planner_plans(plans)
    return {
        "plans": plans,
        "formatted": (f"Error: {err}\n\n" if err else "") + body,
    }


async def get_planner_tasks_tool(
    user_id: str = "system",
    plan_id: str = "",
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    out = await invoke_m365_graph(
        "get_planner_tasks", user_id, connection_id, {"plan_id": plan_id}
    )
    tasks = out.get("tasks") or []
    tasks = tasks if isinstance(tasks, list) else []
    err = out.get("error")
    body = _format_planner_tasks(tasks, plan_id)
    return {
        "tasks": tasks,
        "formatted": (f"Error: {err}\n\n" if err else "") + body,
    }


async def create_planner_task_tool(
    user_id: str = "system",
    plan_id: str = "",
    title: str = "Task",
    bucket_id: str = "",
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    out = await invoke_m365_graph(
        "create_planner_task",
        user_id,
        connection_id,
        {"plan_id": plan_id, "title": title, "bucket_id": bucket_id},
    )
    ok = out.get("success", False)
    tid = out.get("task_id") or ""
    err = out.get("error") or ""
    return {
        "task_id": tid,
        "success": ok,
        "formatted": _format_planner_mutation("create_task", ok, tid, err, title=title),
    }


async def update_planner_task_tool(
    user_id: str = "system",
    task_id: str = "",
    title: Optional[str] = None,
    percent_complete: Optional[int] = None,
    due_datetime: Optional[str] = None,
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {"task_id": task_id}
    if title is not None:
        params["title"] = title
    if percent_complete is not None:
        params["percent_complete"] = percent_complete
    if due_datetime is not None:
        params["due_datetime"] = due_datetime
    out = await invoke_m365_graph("update_planner_task", user_id, connection_id, params)
    ok = out.get("success", False)
    err = out.get("error") or ""
    return {
        "success": ok,
        "formatted": _format_planner_mutation("update_task", ok, task_id, err),
    }


async def delete_planner_task_tool(
    user_id: str = "system",
    task_id: str = "",
    etag: str = "",
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    out = await invoke_m365_graph(
        "delete_planner_task",
        user_id,
        connection_id,
        {"task_id": task_id, "etag": etag},
    )
    ok = out.get("success", False)
    err = out.get("error") or ""
    return {
        "success": ok,
        "formatted": _format_planner_mutation("delete_task", ok, task_id, err),
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

register_action(
    name="list_planner_plans",
    category="productivity",
    description="List Microsoft Planner plans for the user",
    inputs_model=ConnParams,
    outputs_model=PlannerPlansOutputs,
    tool_function=list_planner_plans_tool,
)
register_action(
    name="get_planner_tasks",
    category="productivity",
    description="List tasks in a Planner plan",
    inputs_model=GetPlannerTasksInputs,
    outputs_model=PlannerTasksOutputs,
    tool_function=get_planner_tasks_tool,
)
register_action(
    name="create_planner_task",
    category="productivity",
    description="Create a task in a Planner plan",
    inputs_model=CreatePlannerTaskInputs,
    outputs_model=PlannerMutationOutputs,
    tool_function=create_planner_task_tool,
)
register_action(
    name="update_planner_task",
    category="productivity",
    description="Update a Planner task",
    inputs_model=UpdatePlannerTaskInputs,
    outputs_model=OkOutputs,
    tool_function=update_planner_task_tool,
)
register_action(
    name="delete_planner_task",
    category="productivity",
    description="Delete a Planner task",
    inputs_model=DeletePlannerTaskInputs,
    outputs_model=OkOutputs,
    tool_function=delete_planner_task_tool,
)
