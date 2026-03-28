"""
Agent Invocation Tools - Agent-to-agent and playbook invocation for Agent Factory.

invoke_agent_tool: Run another agent's playbook with input content; returns envelope
with both input_submitted and agent_response for downstream steps and output routing.
invoke_playbook_tool: Run a playbook directly (no agent profile) with input as {query}.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)

MAX_CHAIN_DEPTH = 5


def _definition_steps(playbook: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return playbook definition steps, tolerating definition as dict or JSON string."""
    definition = playbook.get("definition") or {}
    if isinstance(definition, str):
        try:
            definition = json.loads(definition) if definition else {}
        except (json.JSONDecodeError, TypeError):
            definition = {}
    if not isinstance(definition, dict):
        return []
    return definition.get("steps", [])


class InvokeAgentInputs(BaseModel):
    """Required inputs for invoke_agent."""
    agent_handle: str = Field(
        default="",
        description="Optional @handle of the agent to invoke (e.g. validator). Omit if agent_profile_id is set in params.",
    )
    input_content: str = Field(description="Content to send to the agent (wire from e.g. {step_1.formatted})")


class InvokeAgentParams(BaseModel):
    """Optional configuration for invoke_agent."""
    timeout_seconds: int = Field(default=300, description="Max seconds for the invoked agent to run")
    include_input_in_output: bool = Field(
        default=True,
        description="Include input_submitted in formatted output for downstream steps/notifications",
    )
    agent_profile_id: Optional[str] = Field(
        default=None,
        description="Optional profile ID to invoke directly; skips handle resolution. Use when the playbook designer knows the agent ID.",
    )
    playbook_id: Optional[str] = Field(
        default=None,
        description="Optional playbook ID to run instead of the agent's default playbook.",
    )


class InvokeAgentOutputs(BaseModel):
    """Outputs from invoke_agent (envelope: input + agent response)."""
    formatted: str = Field(description="Human-readable summary: input + agent response")
    input_submitted: str = Field(description="Content that was sent to the invoked agent")
    agent_response: str = Field(description="Formatted response from the invoked agent")
    agent_name: str = Field(description="Display name of the invoked agent")
    agent_handle: str = Field(description="Handle of the invoked agent")
    status: str = Field(description="Task status from invoked agent: complete, error, rejected")
    typed_outputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Typed output fields from the invoked agent for downstream wiring",
    )


async def invoke_agent_tool(
    input_content: str,
    agent_handle: str = "",
    timeout_seconds: int = 300,
    include_input_in_output: bool = True,
    agent_profile_id: Optional[str] = None,
    playbook_id: Optional[str] = None,
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Invoke another Agent Factory agent by handle or by profile ID; run its playbook with the given input.
    Returns an envelope with both input_submitted and agent_response.
    Respects chain depth limit and cycle detection via _pipeline_metadata.
    Provide agent_profile_id (in params) to skip handle resolution when the playbook designer knows the agent ID.
    """
    profile_id = None
    agent_name = agent_handle or agent_profile_id or ""

    if agent_profile_id:
        profile_id = agent_profile_id
        agent_name = agent_name or agent_profile_id
    elif agent_handle and agent_handle.strip():
        try:
            client = await get_backend_tool_client()
            resolved = await client.resolve_agent_handle(handle=agent_handle.strip(), user_id=user_id)
        except Exception as e:
            logger.exception("invoke_agent: resolve_agent_handle failed: %s", e)
            err_msg = f"Could not resolve agent handle @{agent_handle}: {e}"
            return {
                "formatted": err_msg,
                "input_submitted": input_content,
                "agent_response": "",
                "agent_name": "",
                "agent_handle": agent_handle,
                "status": "error",
                "typed_outputs": {},
            }
        if not resolved or not resolved.get("found"):
            err_msg = f"Agent @{agent_handle} not found or inactive."
            return {
                "formatted": err_msg,
                "input_submitted": input_content,
                "agent_response": "",
                "agent_name": "",
                "agent_handle": agent_handle,
                "status": "error",
                "typed_outputs": {},
            }
        profile_id = resolved.get("agent_profile_id")
        agent_name = resolved.get("agent_name") or agent_handle
    else:
        err_msg = "Either agent_handle or agent_profile_id (param) must be provided."
        return {
            "formatted": err_msg,
            "input_submitted": input_content,
            "agent_response": "",
            "agent_name": "",
            "agent_handle": agent_handle or "",
            "status": "error",
            "typed_outputs": {},
        }

    meta = _pipeline_metadata or {}
    depth = meta.get("_agent_chain_depth", 0)
    path = list(meta.get("_agent_chain_path", []))
    display_handle = agent_handle or agent_name or str(profile_id)

    if depth >= MAX_CHAIN_DEPTH:
        err_msg = f"Agent chain depth limit ({MAX_CHAIN_DEPTH}) reached; cannot invoke @{display_handle}."
        return {
            "formatted": err_msg,
            "input_submitted": input_content,
            "agent_response": "",
            "agent_name": agent_name,
            "agent_handle": display_handle,
            "status": "error",
            "typed_outputs": {},
        }

    if profile_id in path:
        err_msg = "Circular agent chain detected; cannot invoke an agent already in the chain."
        return {
            "formatted": err_msg,
            "input_submitted": input_content,
            "agent_response": "",
            "agent_name": agent_name,
            "agent_handle": display_handle,
            "status": "error",
            "typed_outputs": {},
        }

    child_metadata: Dict[str, Any] = {
        "user_id": user_id,
        "agent_profile_id": profile_id,
        "trigger_input": input_content,
        "_agent_chain_depth": depth + 1,
        "_agent_chain_path": path + [profile_id],
    }
    if playbook_id:
        child_metadata["playbook_id"] = playbook_id
    lid = (meta.get("line_id") or meta.get("team_id")) if meta else None
    if lid:
        child_metadata["line_id"] = str(lid).strip()
        child_metadata["team_id"] = child_metadata["line_id"]  # legacy alias for nested runs

    conv_id = (meta.get("conversation_id") or "").strip()
    if conv_id:
        child_metadata["conversation_id"] = conv_id

    try:
        from orchestrator.agents.custom_agent_runner import CustomAgentRunner
        runner = CustomAgentRunner()
        result = await runner.process(query=input_content, metadata=child_metadata)
    except Exception as e:
        logger.exception("invoke_agent: CustomAgentRunner.process failed: %s", e)
        err_msg = f"Invoked agent @{display_handle} failed: {e}"
        return {
            "formatted": err_msg,
            "input_submitted": input_content,
            "agent_response": "",
            "agent_name": agent_name,
            "agent_handle": display_handle,
            "status": "error",
            "typed_outputs": {},
        }

    agent_formatted = result.get("formatted", "") or result.get("response", "")
    task_status = result.get("task_status", "complete")
    typed = result.get("typed_outputs", {})

    if include_input_in_output:
        formatted = f"## Input submitted to @{display_handle}\n\n{input_content}\n\n## Response from {agent_name}\n\n{agent_formatted}"
    else:
        formatted = agent_formatted

    return {
        "formatted": formatted,
        "input_submitted": input_content,
        "agent_response": agent_formatted,
        "agent_name": agent_name,
        "agent_handle": display_handle,
        "status": task_status,
        "typed_outputs": typed,
    }


class InvokePlaybookInputs(BaseModel):
    """Required inputs for invoke_playbook."""
    input_content: str = Field(description="Content to send to the playbook as {query}")


class InvokePlaybookParams(BaseModel):
    """Optional configuration for invoke_playbook."""
    playbook_id: Optional[str] = Field(
        default=None,
        description="Playbook ID to run (pre-wired when used as agent tool).",
    )


class InvokePlaybookOutputs(BaseModel):
    """Outputs from invoke_playbook."""
    formatted: str = Field(description="Human-readable output from the playbook")
    status: str = Field(description="complete or error")
    typed_outputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Typed output fields from playbook steps for downstream wiring",
    )


async def invoke_playbook_tool(
    input_content: str,
    playbook_id: str,
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run a playbook directly (no agent profile, no persona) with input_content as {query}.
    Use for utility workflows that do not need profile context.
    """
    try:
        client = await get_backend_tool_client()
        playbook = await client.get_playbook(user_id, playbook_id)
        if not playbook:
            return {
                "formatted": f"Playbook not found or access denied: {playbook_id}",
                "status": "error",
                "typed_outputs": {},
            }
        steps = _definition_steps(playbook)
        if not steps:
            return {
                "formatted": f"Playbook '{playbook.get('name', '')}' has no steps defined.",
                "status": "error",
                "typed_outputs": {},
            }
    except Exception as e:
        logger.exception("invoke_playbook: get_playbook failed: %s", e)
        return {
            "formatted": f"Failed to load playbook: {e}",
            "status": "error",
            "typed_outputs": {},
        }

    from orchestrator.checkpointer import get_async_postgres_saver
    from orchestrator.engines.playbook_graph_builder import build_playbook_graph

    checkpointer = await get_async_postgres_saver()
    graph = build_playbook_graph(steps, checkpointer=checkpointer)
    metadata = dict(_pipeline_metadata or {})
    metadata["user_id"] = user_id
    initial_state: Dict[str, Any] = {
        "playbook_state": {},
        "inputs": {"query": input_content},
        "user_id": user_id,
        "metadata": metadata,
        "execution_trace": [],
    }
    config = {"configurable": {"thread_id": f"playbook_tool_{user_id}_{playbook_id}"}}

    try:
        result = await graph.ainvoke(initial_state, config=config)
    except Exception as e:
        logger.exception("invoke_playbook: graph.ainvoke failed: %s", e)
        return {
            "formatted": f"Playbook execution failed: {e}",
            "status": "error",
            "typed_outputs": {},
        }

    playbook_state = result.get("playbook_state") or {}
    parts: List[str] = []
    typed_outputs: Dict[str, Any] = {}
    for key, value in playbook_state.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict) and value.get("formatted"):
            parts.append(value["formatted"])
            typed_outputs[key] = {k: v for k, v in value.items() if k != "formatted"}
        elif isinstance(value, str):
            parts.append(value)
    formatted = "\n\n".join(parts) if parts else "Playbook completed (no formatted output)."

    return {
        "formatted": formatted,
        "status": "complete",
        "typed_outputs": typed_outputs,
    }


register_action(
    name="invoke_agent",
    category="agent",
    description="Invoke another agent by @handle; run its playbook with input. Returns input and response.",
    inputs_model=InvokeAgentInputs,
    params_model=InvokeAgentParams,
    outputs_model=InvokeAgentOutputs,
    tool_function=invoke_agent_tool,
)

register_action(
    name="invoke_playbook",
    category="agent",
    description="Run a playbook directly with input as query (no agent profile or persona).",
    inputs_model=InvokePlaybookInputs,
    params_model=InvokePlaybookParams,
    outputs_model=InvokePlaybookOutputs,
    tool_function=invoke_playbook_tool,
)
