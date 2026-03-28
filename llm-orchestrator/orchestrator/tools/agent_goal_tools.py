"""
Agent Goal Tools - goal hierarchy for teams (list, report progress, delegate goal to tasks).
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action
from orchestrator.utils.line_context import line_id_from_metadata

logger = logging.getLogger(__name__)


class ListTeamGoalsInputs(BaseModel):
    """Inputs for list_team_goals. line_id is injected from pipeline context."""

    pass


class ListTeamGoalsOutputs(BaseModel):
    """Outputs from list_team_goals."""
    formatted: str = Field(description="Human-readable goal tree")
    tree_json: str = Field(default="[]", description="JSON array of tree nodes")


async def list_team_goals_tool(
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """List the goal tree for the current line. line_id is taken from pipeline context."""
    resolved_team = line_id_from_metadata(_pipeline_metadata or {})
    if not resolved_team:
        return {"formatted": "line_id is required.", "tree_json": "[]"}
    if not _is_uuid(resolved_team):
        return {
            "formatted": "line_id must be the agent line UUID (from get_team_status_board or pipeline context), not the line name. Call get_team_status_board to see the line id in context.",
            "tree_json": "[]",
        }
    try:
        client = await get_backend_tool_client()
        result = await client.get_team_goals_tree(team_id=resolved_team, user_id=user_id)
        if not result.get("success"):
            return {"formatted": result.get("error", "Failed to load goals"), "tree_json": "[]"}
        tree = result.get("tree", [])
        lines = []

        def fmt(node: Dict, indent: int = 0) -> None:
            prefix = "  " * indent
            title = node.get("title", "")
            status = node.get("status", "")
            pct = node.get("progress_pct", 0)
            goal_id = node.get("id", "")
            assigned_agent_id = node.get("assigned_agent_id")
            parts = [f"{prefix}- {title} [{status}] {pct}%"]
            if goal_id:
                parts.append(f"(goal_id: {goal_id})")
            if assigned_agent_id:
                parts.append(f"assigned to {assigned_agent_id}")
            lines.append(" ".join(parts))
            for child in node.get("children", []):
                fmt(child, indent + 1)

        for root in tree:
            fmt(root)
        return {
            "formatted": "\n".join(lines) if lines else "No goals yet.",
            "tree_json": json.dumps(tree),
        }
    except Exception as e:
        logger.warning("list_team_goals failed: %s", e)
        return {"formatted": str(e), "tree_json": "[]"}


class ReportGoalProgressInputs(BaseModel):
    """Inputs for report_goal_progress."""
    goal_id: str = Field(
        description="Goal UUID (the 'id' field from list_team_goals or get_team_status_board). Do NOT use the goal title or description."
    )
    progress_pct: int = Field(ge=0, le=100, description="Progress percentage 0-100")


class ReportGoalProgressOutputs(BaseModel):
    """Outputs from report_goal_progress."""
    formatted: str = Field(description="Human-readable result")
    success: bool = Field(description="Whether update succeeded")


def _is_uuid(s: Optional[str]) -> bool:
    if not s or not isinstance(s, str):
        return False
    return bool(re.match(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$", s.strip()))


async def report_goal_progress_tool(
    goal_id: str,
    progress_pct: int,
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Report progress on an assigned goal. goal_id must be the goal's UUID (id from list_team_goals or get_team_status_board), not the goal title or description."""
    if not _is_uuid(goal_id):
        return {
            "formatted": "goal_id must be the goal's UUID (the 'id' field from list_team_goals or get_team_status_board), not the goal title or description. Call list_team_goals to get each goal's id.",
            "success": False,
        }
    try:
        client = await get_backend_tool_client()
        result = await client.update_goal_progress(
            goal_id=goal_id, user_id=user_id, progress_pct=progress_pct
        )
        ok = result.get("success", False)
        return {
            "formatted": f"Goal progress set to {progress_pct}%." if ok else (result.get("error") or "Update failed."),
            "success": ok,
        }
    except Exception as e:
        logger.warning("report_goal_progress failed: %s", e)
        return {"formatted": str(e), "success": False}


class DelegateGoalToTasksInputs(BaseModel):
    """Inputs for delegate_goal_to_tasks. line_id is injected from pipeline context."""

    goal_id: str = Field(
        description="Goal UUID (id from list_team_goals or get_team_status_board), not the goal title"
    )


class DelegateGoalToTasksOutputs(BaseModel):
    """Outputs from delegate_goal_to_tasks."""
    formatted: str = Field(description="Human-readable summary of created tasks")
    tasks_created: int = Field(description="Number of tasks created")
    task_ids: List[str] = Field(default_factory=list, description="Created task IDs")


def _parse_delegate_json(text: str) -> Optional[List[Dict[str, Any]]]:
    """Extract tasks array from LLM response (JSON or markdown code block)."""
    if not text or not text.strip():
        return None
    raw = text.strip()
    if raw.startswith("```"):
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if m:
            raw = m.group(1).strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "tasks" in data:
            return data.get("tasks")
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    return None


async def delegate_goal_to_tasks_tool(
    goal_id: str,
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Decompose a goal into 2-5 concrete tasks via LLM, create each task linked to the goal,
    and assign to suggested agents. Requires OPENROUTER_API_KEY. line_id is taken from pipeline context.
    """
    resolved_team = line_id_from_metadata(_pipeline_metadata or {})
    if not resolved_team or not goal_id:
        return {"formatted": "goal_id and line_id are required.", "tasks_created": 0, "task_ids": []}
    if not _is_uuid(goal_id):
        return {
            "formatted": "goal_id must be the goal's UUID (the 'id' from list_team_goals or get_team_status_board), not the goal title or description.",
            "tasks_created": 0,
            "task_ids": [],
        }
    try:
        client = await get_backend_tool_client()
        ancestry = await client.get_goal_ancestry(goal_id=goal_id, user_id=user_id)
        if not ancestry.get("success") or not ancestry.get("goals"):
            return {"formatted": ancestry.get("error") or "Goal not found.", "tasks_created": 0, "task_ids": []}
        goal = ancestry["goals"][-1]
        title = goal.get("title", "")
        description = goal.get("description", "") or ""

        from orchestrator.utils.llm_credentials_from_metadata import get_openrouter_credentials
        api_key, base_url = get_openrouter_credentials(_pipeline_metadata)
        if not api_key:
            return {"formatted": "Delegate goal to tasks requires OpenRouter API key (configure in Settings > AI Models).", "tasks_created": 0, "task_ids": []}

        from orchestrator.utils.openrouter_client import get_openrouter_client
        openrouter = get_openrouter_client(api_key=api_key, base_url=base_url)
        model = os.getenv("FAST_MODEL", "anthropic/claude-3-haiku")
        try:
            from config.settings import settings
            model = getattr(settings, "FAST_MODEL", model)
        except Exception:
            pass

        prompt = (
            f"Given this goal: {title}\n"
            f"{description}\n\n"
            "Decompose it into 2-5 concrete tasks. For each task provide: title, short description, and optionally assignee_handle (e.g. @ceo or @researcher). "
            "Output ONLY valid JSON with no other text: {\"tasks\": [{\"title\": \"...\", \"description\": \"...\", \"assignee_handle\": \"@handle or null\"}]}"
        )
        response = await openrouter.chat.completions.create(
            messages=[
                {"role": "system", "content": "You output only valid JSON. No markdown, no explanation."},
                {"role": "user", "content": prompt[:8000]},
            ],
            model=model,
            temperature=0.2,
            max_tokens=1024,
        )
        content = (response.choices[0].message.content or "").strip()
        tasks_spec = _parse_delegate_json(content)
        if not tasks_spec or not isinstance(tasks_spec, list):
            return {"formatted": "Could not parse task list from LLM.", "tasks_created": 0, "task_ids": []}

        created_ids: List[str] = []
        created_by = (_pipeline_metadata or {}).get("agent_profile_id")
        for i, t in enumerate(tasks_spec[:10]):
            if not isinstance(t, dict):
                continue
            t_title = (t.get("title") or "").strip()
            if not t_title:
                continue
            t_desc = (t.get("description") or "").strip()
            assignee_handle = (t.get("assignee_handle") or "").strip() or None
            assigned_agent_id: Optional[str] = None
            if assignee_handle:
                try:
                    resolved = await client.resolve_agent_handle(handle=assignee_handle, user_id=user_id)
                    if resolved and resolved.get("found"):
                        assigned_agent_id = resolved.get("agent_profile_id")
                except Exception as e:
                    logger.debug("delegate_goal_to_tasks: resolve %s failed: %s", assignee_handle, e)
            result = await client.create_agent_task(
                team_id=resolved_team,
                user_id=user_id,
                title=t_title,
                description=t_desc or None,
                assigned_agent_id=assigned_agent_id,
                goal_id=goal_id,
                priority=0,
                created_by_agent_id=created_by,
            )
            if result.get("success") and result.get("task_id"):
                created_ids.append(result["task_id"])

        summary = f"Created {len(created_ids)} task(s) for goal: {title}."
        if created_ids:
            summary += " Task IDs: " + ", ".join(created_ids[:5]) + ("..." if len(created_ids) > 5 else "")
        return {"formatted": summary, "tasks_created": len(created_ids), "task_ids": created_ids}
    except Exception as e:
        logger.warning("delegate_goal_to_tasks failed: %s", e)
        return {"formatted": str(e), "tasks_created": 0, "task_ids": []}


register_action(
    name="list_team_goals",
    category="goals",
    description="List the goal tree for a team",
    inputs_model=ListTeamGoalsInputs,
    params_model=None,
    outputs_model=ListTeamGoalsOutputs,
    tool_function=list_team_goals_tool,
)
register_action(
    name="report_goal_progress",
    category="goals",
    description="Report progress (0-100%) on an assigned goal",
    inputs_model=ReportGoalProgressInputs,
    params_model=None,
    outputs_model=ReportGoalProgressOutputs,
    tool_function=report_goal_progress_tool,
)
register_action(
    name="delegate_goal_to_tasks",
    category="goals",
    description="Decompose a goal into 2-5 tasks via LLM, create them linked to the goal, and assign to suggested agents. Do NOT also call create_task_for_agent for the same work — this tool handles creation and assignment.",
    inputs_model=DelegateGoalToTasksInputs,
    params_model=None,
    outputs_model=DelegateGoalToTasksOutputs,
    tool_function=delegate_goal_to_tasks_tool,
)
