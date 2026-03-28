"""
Agent monitoring tools - query agent execution (run) history via backend gRPC.
"""
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)


class AgentRunRecord(BaseModel):
    """One agent run from the execution log."""
    execution_id: str = Field(description="UUID of the execution")
    agent_name: str = Field(description="Name of the agent profile")
    query: str = Field(description="User query that triggered the run")
    status: str = Field(description="completed, failed, or running")
    started_at: str = Field(description="ISO timestamp when the run started")
    duration_ms: Optional[int] = Field(default=None, description="Duration in milliseconds")
    connectors_called: List[str] = Field(default_factory=list, description="Connectors used")
    entities_discovered: int = Field(default=0, description="Entities discovered")
    error_details: Optional[str] = Field(default=None, description="Error message if failed")
    steps_completed: int = Field(default=0, description="Steps completed")
    steps_total: int = Field(default=0, description="Total steps")


class GetAgentRunHistoryInputs(BaseModel):
    """Required inputs for get_agent_run_history."""
    agent_profile_id: Optional[str] = Field(
        default=None,
        description="Agent profile ID to inspect; omit to query the current agent's own history or all agents",
    )


class GetAgentRunHistoryParams(BaseModel):
    """Optional parameters for get_agent_run_history."""
    limit: int = Field(default=10, ge=1, le=50, description="Max number of runs to return")
    status: Optional[str] = Field(
        default=None,
        description="Filter by status: completed, failed, or running",
    )
    start_date: Optional[str] = Field(default=None, description="Start of range (YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, description="End of range (YYYY-MM-DD)")


class GetAgentRunHistoryOutputs(BaseModel):
    """Structured output from get_agent_run_history."""
    success: bool = Field(description="Whether the query succeeded")
    runs: List[AgentRunRecord] = Field(description="List of run records")
    total: int = Field(description="Number of runs returned")
    agent_name: str = Field(description="Agent name when querying a single profile")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


def _format_run(r: Dict[str, Any], index: int) -> str:
    status_icon = "OK" if r.get("status") == "completed" else "FAIL" if r.get("status") == "failed" else "..."
    started = r.get("started_at", "")[:19].replace("T", " ")
    duration = r.get("duration_ms")
    duration_str = f", {duration}ms" if duration is not None else ""
    query_preview = (r.get("query") or "")[:60]
    if len(r.get("query") or "") > 60:
        query_preview += "..."
    line = f"{index}. [{status_icon}] {started} — \"{query_preview}\"{duration_str}"
    if r.get("error_details"):
        line += f" — Error: {r['error_details'][:80]}"
    return line


async def get_agent_run_history_tool(
    user_id: str = "system",
    agent_profile_id: Optional[str] = None,
    limit: int = 10,
    status: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Query agent run history for the user. Omit agent_profile_id for this agent's history
    or to see all agents; pass agent_profile_id to inspect a specific agent.
    """
    try:
        client = await get_backend_tool_client()
        result = await client.get_agent_run_history(
            user_id=user_id,
            agent_profile_id=agent_profile_id,
            limit=limit,
            status=status,
            start_date=start_date,
            end_date=end_date,
        )
        if not result.get("success"):
            err = result.get("error", "Failed to get run history.")
            return {
                "success": False,
                "runs": [],
                "total": 0,
                "agent_name": "",
                "formatted": err,
            }
        runs_raw = result.get("runs", [])
        agent_name = result.get("agent_name", "")
        total = result.get("total", 0)
        runs = [
            AgentRunRecord(
                execution_id=r.get("execution_id", ""),
                agent_name=r.get("agent_name", ""),
                query=r.get("query", ""),
                status=r.get("status", ""),
                started_at=r.get("started_at", ""),
                duration_ms=r.get("duration_ms"),
                connectors_called=r.get("connectors_called") or [],
                entities_discovered=r.get("entities_discovered", 0),
                error_details=r.get("error_details"),
                steps_completed=r.get("steps_completed", 0),
                steps_total=r.get("steps_total", 0),
            )
            for r in runs_raw
        ]
        title = f"Agent: {agent_name}" if agent_name else "Agent run history"
        parts = [f"{title} — last {total} run(s):"]
        for i, r in enumerate(runs_raw, 1):
            parts.append(_format_run(r, i))
        formatted = "\n".join(parts)
        return {
            "success": True,
            "runs": [r.model_dump() for r in runs],
            "total": total,
            "agent_name": agent_name,
            "formatted": formatted,
        }
    except Exception as e:
        logger.error("get_agent_run_history_tool error: %s", e)
        err = str(e)
        return {
            "success": False,
            "runs": [],
            "total": 0,
            "agent_name": "",
            "formatted": f"Error: {err}",
        }


register_action(
    name="get_agent_run_history",
    category="agent",
    description=(
        "Query agent execution (run) history. Omit agent_profile_id for this agent's history; "
        "pass agent_profile_id to inspect another agent. Filter by status (completed/failed/running) or date range."
    ),
    inputs_model=GetAgentRunHistoryInputs,
    params_model=GetAgentRunHistoryParams,
    outputs_model=GetAgentRunHistoryOutputs,
    tool_function=get_agent_run_history_tool,
)
