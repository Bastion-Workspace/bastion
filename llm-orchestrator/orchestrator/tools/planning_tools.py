"""
Self-managed planning tools for multi-step LLM workflows.

In-memory plan store keyed by user_id. Lets the LLM create, track, and adapt
a task plan during long-horizon work (research, Agent Factory LLM steps, etc.).
"""

import json
import logging
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from orchestrator.utils.action_io_registry import register_action
from orchestrator.utils.tool_type_models import PlanStep

logger = logging.getLogger(__name__)

_plan_store: Dict[str, Dict[str, Any]] = {}

STATUS_PENDING = "pending"
STATUS_IN_PROGRESS = "in_progress"
STATUS_COMPLETE = "complete"
STATUS_SKIPPED = "skipped"


def _format_plan(goal: str, steps: List[Dict[str, Any]]) -> str:
    """Build human-readable checklist for LLM consumption."""
    lines = [f"Goal: {goal}", ""]
    completed = 0
    for s in steps:
        status = s.get("status", STATUS_PENDING)
        if status == STATUS_COMPLETE or status == STATUS_SKIPPED:
            completed += 1
        marker = "[x]" if status == STATUS_COMPLETE else "[~]" if status == STATUS_SKIPPED else "[ ]"
        title = s.get("title", "?")
        result = s.get("result_summary", "")
        line = f"  {marker} {s.get('step_id', '?')}: {title}"
        if result:
            line += f" — {result[:80]}{'...' if len(result) > 80 else ''}"
        lines.append(line)
    lines.append("")
    lines.append(f"Progress: {completed}/{len(steps)} steps done")
    return "\n".join(lines)


def _steps_to_plan_steps(steps: List[Dict[str, Any]]) -> List[PlanStep]:
    """Convert internal step dicts to PlanStep models."""
    return [
        PlanStep(
            step_id=s.get("step_id", ""),
            title=s.get("title", ""),
            status=s.get("status", STATUS_PENDING),
            result_summary=s.get("result_summary", ""),
        )
        for s in steps
    ]


def _steps_to_output(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert internal step dicts to output-schema dicts for JSON-safe return."""
    return [s.model_dump() for s in _steps_to_plan_steps(steps)]


# --- create_plan ---


class CreatePlanInputs(BaseModel):
    goal: str = Field(description="What the overall plan aims to achieve")
    steps: str = Field(
        description='JSON array of step title strings, e.g. ["Search for X", "Analyze Y"]'
    )


class CreatePlanOutputs(BaseModel):
    created: bool = Field(description="Whether a new plan was created")
    goal: str = Field(description="Plan goal")
    steps: List[PlanStep] = Field(description="All plan steps")
    total_steps: int = Field(description="Number of steps")
    completed_steps: int = Field(description="Number of completed/skipped steps")
    formatted: str = Field(description="Human-readable plan summary")


def create_plan_tool(
    goal: str,
    steps: str,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Create a new plan, replacing any existing one for this user. Steps is a JSON array of title strings."""
    step_titles: List[str] = []
    if steps.strip():
        try:
            parsed = json.loads(steps)
            if isinstance(parsed, list):
                step_titles = [str(t) for t in parsed]
        except (json.JSONDecodeError, TypeError):
            logger.warning("create_plan_tool: invalid steps JSON, using empty list")
    plan_steps: List[Dict[str, Any]] = [
        {
            "step_id": f"step_{i + 1}",
            "title": t,
            "status": STATUS_PENDING,
            "result_summary": "",
        }
        for i, t in enumerate(step_titles)
    ]
    _plan_store[user_id] = {"goal": goal, "steps": plan_steps}
    formatted = _format_plan(goal, plan_steps)
    return {
        "created": True,
        "goal": goal,
        "steps": _steps_to_output(plan_steps),
        "total_steps": len(plan_steps),
        "completed_steps": 0,
        "formatted": formatted,
    }


# --- get_plan ---


class GetPlanInputs(BaseModel):
    pass


class GetPlanOutputs(BaseModel):
    has_plan: bool = Field(description="Whether a plan exists for this user")
    goal: str = Field(default="", description="Plan goal if any")
    steps: List[PlanStep] = Field(default_factory=list, description="All plan steps")
    total_steps: int = Field(default=0, description="Number of steps")
    completed_steps: int = Field(default=0, description="Number of completed/skipped steps")
    formatted: str = Field(description="Human-readable plan summary or no-plan message")


def get_plan_tool(user_id: str = "system") -> Dict[str, Any]:
    """Return the current plan state (goal, all steps with statuses). If no plan exists, says so."""
    plan = _plan_store.get(user_id)
    if not plan:
        msg = "No plan yet. Use create_plan_tool to create one."
        return {
            "has_plan": False,
            "goal": "",
            "steps": [],
            "total_steps": 0,
            "completed_steps": 0,
            "formatted": msg,
        }
    steps = plan.get("steps", [])
    goal = plan.get("goal", "")
    completed = sum(
        1 for s in steps if s.get("status") in (STATUS_COMPLETE, STATUS_SKIPPED)
    )
    return {
        "has_plan": True,
        "goal": goal,
        "steps": _steps_to_output(steps),
        "total_steps": len(steps),
        "completed_steps": completed,
        "formatted": _format_plan(goal, steps),
    }


# --- update_plan_step ---


class UpdatePlanStepInputs(BaseModel):
    step_id: str = Field(description="Which step to update (e.g. step_1)")
    status: str = Field(
        description="New status: pending, in_progress, complete, or skipped"
    )
    result_summary: str = Field(
        default="",
        description="Optional summary of what was accomplished",
    )


class UpdatePlanStepOutputs(BaseModel):
    updated_step_id: str = Field(description="Step that was updated")
    goal: str = Field(description="Plan goal")
    steps: List[PlanStep] = Field(description="All plan steps")
    total_steps: int = Field(description="Number of steps")
    completed_steps: int = Field(description="Number of completed/skipped steps")
    formatted: str = Field(description="Human-readable plan summary")


def update_plan_step_tool(
    step_id: str,
    status: str,
    result_summary: str = "",
    user_id: str = "system",
) -> Dict[str, Any]:
    """Update a step's status and optional result. When marking complete, next pending step becomes in_progress."""
    plan = _plan_store.get(user_id)
    if not plan:
        msg = "No plan yet. Use create_plan_tool first."
        return {
            "updated_step_id": step_id,
            "goal": "",
            "steps": [],
            "total_steps": 0,
            "completed_steps": 0,
            "formatted": msg,
        }
    steps = plan.get("steps", [])
    for i, s in enumerate(steps):
        if s.get("step_id") == step_id:
            s["status"] = status
            s["result_summary"] = result_summary or s.get("result_summary", "")
            if status == STATUS_COMPLETE and i + 1 < len(steps):
                next_s = steps[i + 1]
                if next_s.get("status") == STATUS_PENDING:
                    next_s["status"] = STATUS_IN_PROGRESS
            break
    goal = plan.get("goal", "")
    completed = sum(
        1 for s in steps if s.get("status") in (STATUS_COMPLETE, STATUS_SKIPPED)
    )
    return {
        "updated_step_id": step_id,
        "goal": goal,
        "steps": _steps_to_output(steps),
        "total_steps": len(steps),
        "completed_steps": completed,
        "formatted": _format_plan(goal, steps),
    }


# --- add_plan_step ---


class AddPlanStepInputs(BaseModel):
    title: str = Field(description="What the new step aims to accomplish")
    after_step_id: str = Field(
        default="",
        description="Insert after this step (empty = append to end)",
    )


class AddPlanStepOutputs(BaseModel):
    new_step_id: str = Field(description="Assigned step ID for the new step")
    goal: str = Field(description="Plan goal")
    steps: List[PlanStep] = Field(description="All plan steps")
    total_steps: int = Field(description="Number of steps")
    completed_steps: int = Field(description="Number of completed/skipped steps")
    formatted: str = Field(description="Human-readable plan summary")


def add_plan_step_tool(
    title: str,
    after_step_id: str = "",
    user_id: str = "system",
) -> Dict[str, Any]:
    """Add a new step to the plan. After_step_id empty = append; otherwise insert after that step."""
    plan = _plan_store.get(user_id)
    if not plan:
        msg = "No plan yet. Use create_plan_tool first."
        return {
            "new_step_id": "step_1",
            "goal": "",
            "steps": [],
            "total_steps": 0,
            "completed_steps": 0,
            "formatted": msg,
        }
    steps = plan.get("steps", [])
    max_num = 0
    for s in steps:
        sid = s.get("step_id", "")
        if sid.startswith("step_") and sid[5:].isdigit():
            max_num = max(max_num, int(sid[5:]))
    new_id = f"step_{max_num + 1}"
    new_step = {
        "step_id": new_id,
        "title": title,
        "status": STATUS_PENDING,
        "result_summary": "",
    }
    if not after_step_id:
        steps.append(new_step)
    else:
        inserted = False
        for i, s in enumerate(steps):
            if s.get("step_id") == after_step_id:
                steps.insert(i + 1, new_step)
                inserted = True
                break
        if not inserted:
            steps.append(new_step)
    goal = plan.get("goal", "")
    completed = sum(
        1 for s in steps if s.get("status") in (STATUS_COMPLETE, STATUS_SKIPPED)
    )
    return {
        "new_step_id": new_id,
        "goal": goal,
        "steps": _steps_to_output(steps),
        "total_steps": len(steps),
        "completed_steps": completed,
        "formatted": _format_plan(goal, steps),
    }


# --- Registry ---

register_action(
    name="create_plan",
    category="planning",
    description="Create a new multi-step plan (replaces any existing plan). Steps: JSON array of step title strings.",
    inputs_model=CreatePlanInputs,
    outputs_model=CreatePlanOutputs,
    tool_function=create_plan_tool,
)
register_action(
    name="get_plan",
    category="planning",
    description="Get the current plan: goal and all steps with status. Use to check progress.",
    inputs_model=GetPlanInputs,
    outputs_model=GetPlanOutputs,
    tool_function=get_plan_tool,
)
register_action(
    name="update_plan_step",
    category="planning",
    description="Update a step's status (pending, in_progress, complete, skipped) and optional result summary. Marks next step in_progress when completing.",
    inputs_model=UpdatePlanStepInputs,
    outputs_model=UpdatePlanStepOutputs,
    tool_function=update_plan_step_tool,
)
register_action(
    name="add_plan_step",
    category="planning",
    description="Add a new step to the plan. Optionally insert after a specific step_id, or append if after_step_id is empty.",
    inputs_model=AddPlanStepInputs,
    outputs_model=AddPlanStepOutputs,
    tool_function=add_plan_step_tool,
)
