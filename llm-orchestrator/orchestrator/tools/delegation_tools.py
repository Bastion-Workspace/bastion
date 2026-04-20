"""
Subagent delegation for Agent Factory: supervisor steps delegate to other agents with a shared scratchpad.
"""

import json
import logging
import re
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from orchestrator.tools.agent_invocation_tools import invoke_agent_tool
from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)

_SCRATCHPAD_JSON_MAX = 120_000


def _safe_scratchpad_json(scratchpad: Any) -> str:
    try:
        text = json.dumps(scratchpad, ensure_ascii=False, default=str)
        if len(text) > _SCRATCHPAD_JSON_MAX:
            return text[:_SCRATCHPAD_JSON_MAX] + "\n... (truncated)"
        return text
    except (TypeError, ValueError):
        return str(scratchpad)[:_SCRATCHPAD_JSON_MAX]


def subagent_scratchpad_key(subagent: Dict[str, Any], index: int) -> str:
    """Stable key for shared_scratchpad entries."""
    role = (subagent.get("role") or "").strip()
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", role)[:48].strip("_") or "agent"
    pid = str(subagent.get("agent_profile_id") or "").replace("-", "")[:8]
    return f"sub_{index}_{slug}_{pid}"


def _compose_delegation_input(
    task: str,
    context_json: str,
    output_hint: str,
    scratchpad: Dict[str, Any],
) -> str:
    parts: list[str] = []
    if scratchpad:
        parts.append(
            "## Shared scratchpad (context from this supervisor step)\n"
            + _safe_scratchpad_json(scratchpad)
        )
    cj = (context_json or "").strip()
    if cj:
        parts.append("## Additional context (JSON)\n" + cj)
    oh = (output_hint or "").strip()
    if oh:
        parts.append("## Desired output\n" + oh)
    parts.append("## Task\n" + (task or "").strip() or "(no task text)")
    return "\n\n".join(parts)


async def run_subagent_delegation(
    *,
    task: str,
    context_json: str,
    output_hint: str,
    agent_profile_id: str,
    playbook_id: Optional[str],
    scratchpad_key: str,
    agent_display_name: str,
    user_id: str,
    _pipeline_metadata: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Run a subagent via invoke_agent, merge results into metadata.shared_scratchpad[scratchpad_key].
    """
    meta = _pipeline_metadata or {}
    scratchpad: Dict[str, Any] = meta.setdefault("shared_scratchpad", {})
    composed = _compose_delegation_input(task, context_json, output_hint, scratchpad)

    result = await invoke_agent_tool(
        input_content=composed,
        agent_handle="",
        agent_profile_id=agent_profile_id,
        playbook_id=playbook_id,
        user_id=user_id,
        _pipeline_metadata=meta,
    )

    scratchpad[scratchpad_key] = {
        "agent_display_name": agent_display_name,
        "agent_profile_id": agent_profile_id,
        "agent_response": result.get("agent_response", ""),
        "typed_outputs": result.get("typed_outputs") or {},
        "status": result.get("status", "complete"),
    }

    extra = (
        f"\n\n## Scratchpad key `{scratchpad_key}` updated with this subagent's outputs."
        if result.get("status") == "complete"
        else ""
    )
    formatted = (result.get("formatted") or "") + extra
    out = dict(result)
    out["formatted"] = formatted
    out["scratchpad_key"] = scratchpad_key
    return out


class DelegateToInputs(BaseModel):
    task: str = Field(description="Specific instruction for the subagent")
    context_json: str = Field(
        default="",
        description="Optional JSON string with extra structured context for the subagent",
    )
    output_hint: str = Field(
        default="",
        description="Optional hint describing the shape or focus of the response you need",
    )


class DelegateToParams(BaseModel):
    agent_profile_id: str = Field(
        default="",
        description="Target agent profile UUID (required at runtime unless pre-bound by step subagents)",
    )
    playbook_id: Optional[str] = Field(default=None, description="Optional playbook override")
    scratchpad_key: str = Field(default="subagent", description="Key used to store this run in the shared scratchpad")
    agent_display_name: str = Field(default="", description="Human-readable name for logging and scratchpad")


class DelegateToOutputs(BaseModel):
    formatted: str = Field(description="Human-readable summary of the delegation result")
    agent_response: str = Field(default="", description="Subagent response text")
    agent_name: str = Field(default="", description="Invoked agent display name")
    status: str = Field(description="complete, error, or rejected")
    typed_outputs: Dict[str, Any] = Field(default_factory=dict)
    scratchpad_key: str = Field(default="", description="Scratchpad slot updated")


async def delegate_to_tool(
    task: str,
    context_json: str = "",
    output_hint: str = "",
    agent_profile_id: str = "",
    playbook_id: Optional[str] = None,
    scratchpad_key: str = "subagent",
    agent_display_name: str = "",
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Delegate a task to another agent with shared scratchpad context.
    When used from playbook subagents, agent_profile_id and scratchpad_key are pre-bound by the executor.
    """
    if not (agent_profile_id or "").strip():
        msg = "delegate_to requires agent_profile_id (add subagents on the LLM Agent / Deep Agent step, or bind parameters)."
        return {
            "formatted": msg,
            "agent_response": "",
            "agent_name": "",
            "status": "error",
            "typed_outputs": {},
            "scratchpad_key": "",
        }
    name = (agent_display_name or "").strip() or agent_profile_id[:8]
    return await run_subagent_delegation(
        task=task,
        context_json=context_json or "",
        output_hint=output_hint or "",
        agent_profile_id=agent_profile_id.strip(),
        playbook_id=playbook_id,
        scratchpad_key=(scratchpad_key or "subagent").strip() or "subagent",
        agent_display_name=name,
        user_id=user_id,
        _pipeline_metadata=_pipeline_metadata,
    )


register_action(
    name="delegate_to",
    category="agent",
    description=(
        "Delegate a task to another Agent Factory agent with shared scratchpad context. "
        "Prefer configuring subagents on the step instead of calling this with raw profile IDs."
    ),
    inputs_model=DelegateToInputs,
    params_model=DelegateToParams,
    outputs_model=DelegateToOutputs,
    tool_function=delegate_to_tool,
)
