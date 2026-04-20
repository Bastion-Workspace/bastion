"""
Agent monitoring tools - query agent execution (run) history via backend gRPC.
"""
import json
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


# ----- get_execution_trace (per-run step trace) -----


class ExecutionStepDetail(BaseModel):
    """One playbook step from an execution trace."""

    step_index: int = Field(description="Step order index")
    step_name: str = Field(description="Step name in playbook")
    step_type: str = Field(description="Step type: tool, llm_agent, llm_task, etc.")
    action_name: str = Field(default="", description="Tool action name when step_type is tool")
    status: str = Field(description="completed, failed, or skipped")
    started_at: str = Field(default="", description="ISO start time")
    completed_at: str = Field(default="", description="ISO end time")
    duration_ms: Optional[int] = Field(default=None, description="Duration in milliseconds")
    inputs_json: str = Field(default="", description="Truncated JSON of resolved inputs (empty if include_io false)")
    outputs_json: str = Field(default="", description="Truncated JSON of step outputs (empty if include_io false)")
    error_details: str = Field(default="", description="Error text when status is failed")
    tool_call_trace_json: str = Field(
        default="",
        description="JSON array of tool invocations within llm_agent steps (empty if include_tool_calls false)",
    )
    input_tokens: int = Field(default=0, description="Input tokens attributed to this step when available")
    output_tokens: int = Field(default=0, description="Output tokens attributed to this step when available")


class GetExecutionTraceInputs(BaseModel):
    """Required inputs for get_execution_trace."""

    execution_id: str = Field(description="UUID of the execution row (from get_agent_run_history)")


class GetExecutionTraceParams(BaseModel):
    """Optional parameters for get_execution_trace."""

    include_io: bool = Field(
        default=True,
        description="Include per-step inputs_json and outputs_json (truncated server-side)",
    )
    include_tool_calls: bool = Field(
        default=False,
        description="Include tool_call_trace_json per step (larger payload; use for deep debugging)",
    )


class GetExecutionTraceOutputs(BaseModel):
    """Structured output from get_execution_trace."""

    success: bool = Field(description="Whether the query succeeded")
    execution_id: str = Field(description="Execution UUID")
    agent_name: str = Field(description="Agent profile display name")
    query: str = Field(description="User query that started the run")
    status: str = Field(description="Run status: completed, failed, or running")
    started_at: str = Field(default="", description="Run started at (ISO)")
    completed_at: str = Field(default="", description="Run completed at (ISO)")
    duration_ms: Optional[int] = Field(default=None, description="Total run duration ms")
    tokens_input: Optional[int] = Field(default=None, description="Total input tokens for the run")
    tokens_output: Optional[int] = Field(default=None, description="Total output tokens for the run")
    cost_usd: Optional[str] = Field(default=None, description="Estimated cost when available")
    model_used: str = Field(default="", description="Model identifier when logged")
    error_details: str = Field(default="", description="Top-level run error when failed")
    steps: List[ExecutionStepDetail] = Field(description="Ordered step traces")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


def _status_icon(status: str) -> str:
    s = (status or "").lower()
    if s == "completed":
        return "OK"
    if s == "failed":
        return "FAIL"
    if s == "skipped":
        return "SKIP"
    return "..."


def _format_tool_calls_line(tool_call_trace_json: str, max_items: int = 8) -> str:
    if not tool_call_trace_json or not tool_call_trace_json.strip():
        return ""
    try:
        arr = json.loads(tool_call_trace_json)
    except json.JSONDecodeError:
        return f"     Tool trace (raw): {tool_call_trace_json[:200]}..."
    if not isinstance(arr, list) or not arr:
        return ""
    parts: List[str] = []
    for item in arr[:max_items]:
        if isinstance(item, dict):
            name = item.get("tool_name") or item.get("name") or "tool"
            dur = item.get("duration_ms")
            st = item.get("status") or ""
            frag = f"{name}"
            if dur is not None:
                frag += f" ({dur}ms)"
            if st and st != "completed":
                frag += f" [{st}]"
            args = item.get("args")
            if isinstance(args, dict) and args:
                keys = list(args.keys())[:3]
                frag += f" args:{keys}"
            parts.append(frag)
        else:
            parts.append(str(item)[:120])
    more = f" (+{len(arr) - max_items} more)" if len(arr) > max_items else ""
    return "     Tool calls: " + "; ".join(parts) + more


async def get_execution_trace_tool(
    user_id: str = "system",
    execution_id: str = "",
    include_io: bool = True,
    include_tool_calls: bool = False,
) -> Dict[str, Any]:
    """Load one execution and its step-level trace for the user."""
    eid = (execution_id or "").strip()
    if not eid:
        return {
            "success": False,
            "execution_id": "",
            "agent_name": "",
            "query": "",
            "status": "",
            "started_at": "",
            "completed_at": "",
            "duration_ms": None,
            "tokens_input": None,
            "tokens_output": None,
            "cost_usd": None,
            "model_used": "",
            "error_details": "",
            "steps": [],
            "formatted": "Error: execution_id is required.",
        }
    try:
        client = await get_backend_tool_client()
        result = await client.get_execution_trace(
            user_id=user_id,
            execution_id=eid,
            include_io=include_io,
            include_tool_calls=include_tool_calls,
        )
        if not result.get("success"):
            err = result.get("error", "Failed to load execution trace.")
            return {
                "success": False,
                "execution_id": eid,
                "agent_name": "",
                "query": "",
                "status": "",
                "started_at": "",
                "completed_at": "",
                "duration_ms": None,
                "tokens_input": None,
                "tokens_output": None,
                "cost_usd": None,
                "model_used": "",
                "error_details": "",
                "steps": [],
                "formatted": err,
            }
        steps_raw = result.get("steps") or []
        steps = [
            ExecutionStepDetail(
                step_index=int(s.get("step_index", 0)),
                step_name=s.get("step_name") or "",
                step_type=s.get("step_type") or "",
                action_name=s.get("action_name") or "",
                status=s.get("status") or "",
                started_at=s.get("started_at") or "",
                completed_at=s.get("completed_at") or "",
                duration_ms=s.get("duration_ms"),
                inputs_json=s.get("inputs_json") or "",
                outputs_json=s.get("outputs_json") or "",
                error_details=s.get("error_details") or "",
                tool_call_trace_json=s.get("tool_call_trace_json") or "",
                input_tokens=int(s.get("input_tokens") or 0),
                output_tokens=int(s.get("output_tokens") or 0),
            )
            for s in steps_raw
        ]
        done = sum(1 for s in steps if (s.status or "").lower() == "completed")
        total = len(steps)
        dur = result.get("duration_ms")
        dur_str = f" ({dur}ms)" if dur is not None else ""
        lines = [
            f"Execution {result.get('execution_id', eid)} — Agent: {result.get('agent_name', '')} — "
            f"{result.get('status', '')}{dur_str}",
        ]
        if result.get("model_used"):
            lines.append(f"Model: {result.get('model_used')}")
        if result.get("tokens_input") is not None or result.get("tokens_output") is not None:
            lines.append(
                f"Tokens (run): in={result.get('tokens_input')} out={result.get('tokens_output')}"
                + (f" cost={result.get('cost_usd')}" if result.get("cost_usd") else "")
            )
        if result.get("error_details"):
            lines.append(f"Run error: {result['error_details'][:500]}")
        lines.append(f"Steps ({done}/{total} completed):")
        for i, s in enumerate(steps, 1):
            ms = s.duration_ms
            ms_str = f" — {ms}ms" if ms is not None else ""
            tok_str = ""
            if s.input_tokens or s.output_tokens:
                tok_str = f", {s.input_tokens} in / {s.output_tokens} out tokens"
            act = f" ({s.action_name})" if s.action_name else ""
            err_frag = f" — Error: {s.error_details[:120]}" if s.error_details else ""
            lines.append(
                f"  {i}. [{_status_icon(s.status)}] {s.step_name} ({s.step_type}){act}{ms_str}{tok_str}{err_frag}"
            )
            if include_tool_calls:
                tline = _format_tool_calls_line(s.tool_call_trace_json)
                if tline:
                    lines.append(tline)
        formatted = "\n".join(lines)
        return {
            "success": True,
            "execution_id": result.get("execution_id") or eid,
            "agent_name": result.get("agent_name") or "",
            "query": result.get("query") or "",
            "status": result.get("status") or "",
            "started_at": result.get("started_at") or "",
            "completed_at": result.get("completed_at") or "",
            "duration_ms": result.get("duration_ms"),
            "tokens_input": result.get("tokens_input"),
            "tokens_output": result.get("tokens_output"),
            "cost_usd": result.get("cost_usd"),
            "model_used": result.get("model_used") or "",
            "error_details": result.get("error_details") or "",
            "steps": [x.model_dump() for x in steps],
            "formatted": formatted,
        }
    except Exception as e:
        logger.error("get_execution_trace_tool error: %s", e)
        return {
            "success": False,
            "execution_id": eid,
            "agent_name": "",
            "query": "",
            "status": "",
            "started_at": "",
            "completed_at": "",
            "duration_ms": None,
            "tokens_input": None,
            "tokens_output": None,
            "cost_usd": None,
            "model_used": "",
            "error_details": "",
            "steps": [],
            "formatted": f"Error: {e}",
        }


register_action(
    name="get_execution_trace",
    category="agent",
    description=(
        "Load one agent execution with per-step trace (timing, status, optional I/O and tool-call trace). "
        "Use execution_id from get_agent_run_history. Set include_tool_calls true for llm_agent tool details."
    ),
    inputs_model=GetExecutionTraceInputs,
    params_model=GetExecutionTraceParams,
    outputs_model=GetExecutionTraceOutputs,
    tool_function=get_execution_trace_tool,
)
