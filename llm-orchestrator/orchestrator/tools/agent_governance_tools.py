"""
Agent Governance Tools - propose_hire, propose_strategy_change (create approval queue entries).
"""

import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action
from orchestrator.utils.line_context import line_id_from_metadata

logger = logging.getLogger(__name__)


class ProposeHireInputs(BaseModel):
    """Inputs for propose_hire. line_id is auto-injected from pipeline context."""
    proposed_name: str = Field(description="Name for the new agent")
    proposed_role: str = Field(default="worker", description="Role in the team (e.g. ceo, manager, worker)")
    proposed_handle: Optional[str] = Field(default=None, description="Optional @handle for the agent")
    reason: Optional[str] = Field(default=None, description="Reason for the hire request")


class ProposeHireOutputs(BaseModel):
    """Outputs from propose_hire."""
    formatted: str = Field(description="Human-readable result")
    approval_id: str = Field(default="", description="Approval queue ID if created")
    success: bool = Field(description="Whether the approval request was created")


async def propose_hire_tool(
    proposed_name: str,
    proposed_role: str = "worker",
    proposed_handle: Optional[str] = None,
    reason: Optional[str] = None,
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Propose adding a new agent to the line. Creates an approval request; on user approval, profile can be created and added. line_id is auto-injected from pipeline context."""
    meta = _pipeline_metadata or {}
    resolved_user = user_id or meta.get("user_id", "system")
    from_agent_id = meta.get("agent_profile_id", "")
    line_uuid = line_id_from_metadata(meta)
    if not resolved_user or not line_uuid:
        return {"formatted": "user_id and line_id (from context) are required.", "approval_id": "", "success": False}
    try:
        client = await get_backend_tool_client()
        prompt = f"Proposed hire: {proposed_name} as {proposed_role}" + (f" (@{proposed_handle})" if proposed_handle else "")
        if reason:
            prompt += f"\nReason: {reason}"
        preview = {
            "governance_type": "hire_agent",
            "line_id": line_uuid,
            "team_id": line_uuid,  # Backward compat for _handle_approved_hire
            "proposed_name": proposed_name,
            "proposed_role": proposed_role,
            "proposed_handle": proposed_handle,
            "reason": reason,
        }
        approval_id = await client.park_approval(
            user_id=resolved_user,
            agent_profile_id=from_agent_id,
            execution_id=None,
            step_name="propose_hire",
            prompt=prompt,
            preview_data=preview,
            governance_type="hire_agent",
        )
        if approval_id:
            return {
                "formatted": f"Hire proposal submitted for approval (id: {approval_id}).",
                "approval_id": approval_id,
                "success": True,
            }
        return {"formatted": "Failed to create approval request.", "approval_id": "", "success": False}
    except Exception as e:
        logger.warning("propose_hire failed: %s", e)
        return {"formatted": str(e), "approval_id": "", "success": False}


class ProposeStrategyChangeInputs(BaseModel):
    """Inputs for propose_strategy_change."""
    description: str = Field(description="Description of the proposed strategy or playbook change")
    scope: Optional[str] = Field(default=None, description="Scope: playbook, team_goals, or both")


class ProposeStrategyChangeOutputs(BaseModel):
    """Outputs from propose_strategy_change."""
    formatted: str = Field(description="Human-readable result")
    approval_id: str = Field(default="", description="Approval queue ID if created")
    success: bool = Field(description="Whether the approval request was created")


async def propose_strategy_change_tool(
    description: str,
    scope: Optional[str] = None,
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Propose a strategy or playbook change. Creates an approval request for the user to review."""
    meta = _pipeline_metadata or {}
    resolved_user = user_id or meta.get("user_id", "system")
    from_agent_id = meta.get("agent_profile_id", "")
    line_uuid = line_id_from_metadata(meta) or ""
    if not resolved_user:
        return {"formatted": "user_id is required.", "approval_id": "", "success": False}
    try:
        client = await get_backend_tool_client()
        prompt = f"Proposed strategy change: {description}"
        if scope:
            prompt += f"\nScope: {scope}"
        preview = {
            "governance_type": "strategy_change",
            "description": description,
            "scope": scope or "playbook",
            "line_id": line_uuid,
            "team_id": line_uuid,
        }
        approval_id = await client.park_approval(
            user_id=resolved_user,
            agent_profile_id=from_agent_id,
            execution_id=None,
            step_name="propose_strategy_change",
            prompt=prompt,
            preview_data=preview,
            governance_type="strategy_change",
        )
        if approval_id:
            return {
                "formatted": f"Strategy change proposal submitted for approval (id: {approval_id}).",
                "approval_id": approval_id,
                "success": True,
            }
        return {"formatted": "Failed to create approval request.", "approval_id": "", "success": False}
    except Exception as e:
        logger.warning("propose_strategy_change failed: %s", e)
        return {"formatted": str(e), "approval_id": "", "success": False}


register_action(
    name="propose_hire",
    category="governance",
    description="Propose adding a new agent to the team (creates approval request)",
    inputs_model=ProposeHireInputs,
    params_model=None,
    outputs_model=ProposeHireOutputs,
    tool_function=propose_hire_tool,
)
register_action(
    name="propose_strategy_change",
    category="governance",
    description="Propose a strategy or playbook change (creates approval request)",
    inputs_model=ProposeStrategyChangeInputs,
    params_model=None,
    outputs_model=ProposeStrategyChangeOutputs,
    tool_function=propose_strategy_change_tool,
)
