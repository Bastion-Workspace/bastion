"""
Account Tools - List email/calendar/contacts accounts available to the agent.
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action
from orchestrator.utils.tool_type_models import AccountRef

logger = logging.getLogger(__name__)


class ListAccountsInputs(BaseModel):
    """Inputs for list_accounts_tool."""

    service_type: str = Field(
        default="all",
        description="Filter by type: email, calendar, contacts, or all",
    )


class ListAccountsOutputs(BaseModel):
    """Outputs for list_accounts_tool."""

    accounts: List[AccountRef] = Field(
        description="Available accounts bound to this agent",
    )
    count: int = Field(description="Number of accounts")
    formatted: str = Field(description="Human-readable list for LLM/chat")


async def list_accounts_tool(
    user_id: str = "system",
    service_type: str = "all",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """List email/calendar/contacts accounts. When run in a pipeline, only returns accounts bound to the agent profile."""
    try:
        agent_profile_id = (_pipeline_metadata or {}).get("agent_profile_id", "")
        client = await get_backend_tool_client()
        accounts = await client.list_user_accounts(
            user_id=user_id,
            agent_profile_id=agent_profile_id,
            service_type=service_type,
        )
        refs = [AccountRef(**a) for a in accounts]
        lines = [
            f"{i + 1}. [{r.type}] {r.label} (provider: {r.provider}, connection_id: {r.connection_id})"
            for i, r in enumerate(refs)
        ]
        formatted = "\n".join(lines) if lines else "No accounts configured for this agent."
        return {
            "accounts": [r.model_dump() for r in refs],
            "count": len(refs),
            "formatted": formatted,
        }
    except Exception as e:
        logger.error("list_accounts_tool error: %s", e)
        return {
            "accounts": [],
            "count": 0,
            "formatted": f"Error: {e}",
        }


register_action(
    name="list_accounts",
    category="accounts",
    description="List email/calendar/contact accounts available to this agent. Call this first to discover connection IDs.",
    inputs_model=ListAccountsInputs,
    outputs_model=ListAccountsOutputs,
    tool_function=list_accounts_tool,
)
