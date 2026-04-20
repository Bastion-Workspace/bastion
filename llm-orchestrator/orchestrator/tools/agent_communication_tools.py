"""
Agent Communication Tools - inter-agent messaging and team timeline.

send_to_agent: Send a message to another agent; optionally wait for response.
read_team_timeline, read_my_messages, get_team_status_board.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.tools.agent_invocation_tools import invoke_agent_tool
from orchestrator.utils.action_io_registry import register_action
from orchestrator.utils.line_context import line_id_from_metadata

logger = logging.getLogger(__name__)

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _is_uuid(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and _UUID_RE.match(s.strip()))

MESSAGE_TYPES = (
    "task_assignment",
    "status_update",
    "request",
    "response",
    "delegation",
    "escalation",
    "report",
    "system",
)


class SendToAgentInputs(BaseModel):
    """Required inputs for send_to_agent."""
    target_agent: str = Field(
        description="Recipient: @handle (e.g. @ceo), agent profile UUID from the roster, or exact display name as shown on get_team_status_board"
    )
    message: str = Field(description="Content to send to the target agent")


class SendToAgentParams(BaseModel):
    """Optional configuration for send_to_agent."""
    data: Optional[Dict[str, Any]] = Field(default=None, description="Optional structured data payload")
    wait_for_response: bool = Field(default=False, description="If true, invoke target agent and return its response")
    timeout_minutes: int = Field(default=5, description="Max minutes to wait when wait_for_response is true")
    message_type: str = Field(default="request", description="One of: task_assignment, status_update, request, response, delegation, escalation, report, system")
    team_id: Optional[str] = Field(default=None, description="Agent line UUID for timeline (same as line_id in context); optional override")
    agent_profile_id: Optional[str] = Field(default=None, description="Target agent profile ID (optional if target_agent handle is set)")


class SendToAgentOutputs(BaseModel):
    """Outputs from send_to_agent."""
    formatted: str = Field(description="Human-readable summary")
    message_id: str = Field(description="ID of the created message when persisted")
    conversation_id: str = Field(default="", description="Thread or conversation identifier if applicable")
    response: str = Field(default="", description="Target agent response when wait_for_response is true")
    response_status: str = Field(default="", description="complete, error, or empty when not waiting")


async def send_to_agent_tool(
    target_agent: str,
    message: str,
    data: Optional[Dict[str, Any]] = None,
    wait_for_response: bool = False,
    timeout_minutes: int = 5,
    message_type: str = "request",
    team_id: Optional[str] = None,
    agent_profile_id: Optional[str] = None,
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Send a message to another agent. Optionally persist to team timeline and/or wait for the target's response.
    """
    metadata = _pipeline_metadata or {}
    from_agent_id = metadata.get("agent_profile_id")
    resolved_team_id = team_id or line_id_from_metadata(metadata)
    if resolved_team_id and not _is_uuid(resolved_team_id):
        return {
            "formatted": "line_id must be the agent line UUID, not the line name. It is automatically provided from context.",
            "message_id": "",
            "conversation_id": "",
            "response": "",
            "response_status": "error",
        }
    to_profile_id = agent_profile_id

    if not to_profile_id and target_agent.strip():
        try:
            client = await get_backend_tool_client()
            resolved = await client.resolve_agent_handle(handle=target_agent.strip(), user_id=user_id)
            if resolved and resolved.get("found"):
                to_profile_id = resolved.get("agent_profile_id")
        except Exception as e:
            logger.warning("send_to_agent: resolve handle failed: %s", e)

    if not to_profile_id:
        return {
            "formatted": (
                f"Could not resolve target agent {target_agent!r}. "
                "Use @handle, the agent's profile UUID, or the exact display name from get_team_status_board."
            ),
            "message_id": "",
            "conversation_id": "",
            "response": "",
            "response_status": "error",
        }

    message_type = message_type if message_type in MESSAGE_TYPES else "request"
    payload_meta = dict(data) if isinstance(data, dict) else {}
    if not wait_for_response and resolved_team_id and to_profile_id:
        payload_meta["trigger_dispatch"] = True

    if resolved_team_id and from_agent_id:
        try:
            client = await get_backend_tool_client()
            result = await client.create_agent_message(
                user_id=user_id,
                team_id=resolved_team_id,
                from_agent_id=from_agent_id,
                to_agent_id=to_profile_id,
                message_type=message_type,
                content=message,
                metadata=payload_meta,
                parent_message_id=None,
            )
            message_id = result.get("message_id", "") if result.get("success") else ""
        except Exception as e:
            logger.warning("send_to_agent: create_agent_message failed: %s", e)
            message_id = ""
    else:
        message_id = ""

    if not wait_for_response:
        return {
            "formatted": f"Message sent to @{target_agent}" + (f" (message_id={message_id})" if message_id else ""),
            "message_id": message_id,
            "conversation_id": "",
            "response": "",
            "response_status": "",
        }

    timeout_seconds = min(300, timeout_minutes * 60)
    try:
        inv = await invoke_agent_tool(
            input_content=message,
            agent_handle=target_agent if not agent_profile_id else "",
            agent_profile_id=to_profile_id,
            timeout_seconds=timeout_seconds,
            user_id=user_id,
            _pipeline_metadata=_pipeline_metadata,
        )
        resp_text = inv.get("agent_response", "") or inv.get("formatted", "")
        status = inv.get("status", "complete")
        if resolved_team_id and to_profile_id and resp_text:
            try:
                resp_client = await get_backend_tool_client()
                await resp_client.create_agent_message(
                    user_id=user_id,
                    team_id=resolved_team_id,
                    from_agent_id=to_profile_id,
                    to_agent_id=from_agent_id,
                    message_type="response",
                    content=resp_text,
                    metadata={"in_reply_to": message_id} if message_id else {},
                    parent_message_id=message_id or None,
                )
            except Exception as timeline_err:
                logger.warning("send_to_agent: post response to timeline failed: %s", timeline_err)

        conv_append = (metadata.get("conversation_id") or "").strip()
        if conv_append and resp_text and resolved_team_id:
            try:
                append_client = await get_backend_tool_client()
                await append_client.append_line_agent_chat_message(
                    user_id=user_id,
                    conversation_id=conv_append,
                    content=resp_text,
                    agent_profile_id=to_profile_id or "",
                    agent_display_name=(inv.get("agent_name") or "").strip(),
                    line_id=resolved_team_id,
                    line_agent_handle=(inv.get("agent_handle") or "").strip(),
                    delegated_by_agent_id=from_agent_id or "",
                )
            except Exception as chat_err:
                logger.warning("send_to_agent: append line chat message failed: %s", chat_err)

        return {
            "formatted": f"Sent to @{target_agent}; response: {resp_text[:200]}..." if len(resp_text) > 200 else f"Sent to @{target_agent}; response: {resp_text}",
            "message_id": message_id,
            "conversation_id": "",
            "response": resp_text,
            "response_status": status,
        }
    except Exception as e:
        logger.exception("send_to_agent wait_for_response invoke failed: %s", e)
        return {
            "formatted": f"Message sent to @{target_agent} but response failed: {e}",
            "message_id": message_id,
            "conversation_id": "",
            "response": "",
            "response_status": "error",
        }


class ReadTeamTimelineInputs(BaseModel):
    """No required inputs; line_id from pipeline metadata."""


class ReadTeamTimelineParams(BaseModel):
    """Optional params for read_team_timeline."""
    limit: int = Field(default=20, description="Max number of messages to return (1-100)")
    since_hours: int = Field(default=24, description="Only messages from the last N hours; 0 = no filter")


class ReadTeamTimelineOutputs(BaseModel):
    """Outputs from read_team_timeline."""
    formatted: str = Field(description="Human-readable summary of recent messages")
    items: List[Dict[str, Any]] = Field(default_factory=list, description="List of message objects")
    total: int = Field(default=0, description="Total count")


async def read_team_timeline_tool(
    limit: int = 20,
    since_hours: int = 24,
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Read recent team timeline messages (what other agents have said). Use this to catch up on team communication.
    Requires agent line context (line_id in metadata); only available when running in a line.
    """
    metadata = _pipeline_metadata or {}
    line_id = line_id_from_metadata(metadata)
    if not line_id:
        return {
            "formatted": "No agent line context; cannot read timeline. This tool is only available when running in a line.",
            "items": [],
            "total": 0,
        }
    if not _is_uuid(line_id):
        return {
            "formatted": "line_id must be the agent line UUID, not the line name. It is automatically provided from context.",
            "items": [],
            "total": 0,
        }
    limit = max(1, min(100, limit))
    try:
        client = await get_backend_tool_client()
        result = await client.read_team_timeline(
            team_id=line_id,
            user_id=user_id,
            limit=limit,
            since_hours=since_hours,
        )
        if not result.get("success"):
            return {
                "formatted": result.get("error") or "Failed to read timeline.",
                "items": [],
                "total": 0,
            }
        items = result.get("items") or []
        total = result.get("total") or 0
        lines = []
        for m in items:
            from_name = m.get("from_agent_name") or m.get("from_agent_handle") or "Agent"
            to_name = m.get("to_agent_name") or m.get("to_agent_handle") or "team"
            content = (m.get("content") or "")[:150]
            if len(m.get("content") or "") > 150:
                content += "..."
            created = m.get("created_at") or ""
            lines.append(f"- {from_name} -> {to_name}: \"{content}\" ({created})")
        formatted = (
            f"Recent team messages ({len(items)} in this response, {total} total matching filter):\n" + "\n".join(lines)
            if lines
            else f"No messages in the requested window (total={total})."
        )
        return {"formatted": formatted, "items": items, "total": total}
    except Exception as e:
        logger.warning("read_team_timeline failed: %s", e)
        return {"formatted": str(e), "items": [], "total": 0}


class ReadMyMessagesInputs(BaseModel):
    """No required inputs; agent_profile_id and line_id from pipeline metadata."""


class ReadMyMessagesParams(BaseModel):
    """Optional params for read_my_messages."""
    limit: int = Field(default=50, description="Max number of messages to return (1-100)")


class ReadMyMessagesOutputs(BaseModel):
    """Outputs from read_my_messages."""
    formatted: str = Field(description="Human-readable summary of messages to/from this agent")
    items: List[Dict[str, Any]] = Field(default_factory=list, description="List of message objects")
    total: int = Field(default=0, description="Total count")


async def read_my_messages_tool(
    limit: int = 50,
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Read messages sent to or from the current agent on the team timeline. Use this to check your inbox.
    Requires team context; only available when running in a team.
    """
    metadata = _pipeline_metadata or {}
    line_id = line_id_from_metadata(metadata)
    agent_profile_id = metadata.get("agent_profile_id")
    if not line_id or not agent_profile_id:
        return {
            "formatted": "No agent line or agent context; cannot read messages.",
            "items": [],
            "total": 0,
        }
    if not _is_uuid(line_id):
        return {
            "formatted": "line_id must be the agent line UUID, not the line name. It is automatically provided from context.",
            "items": [],
            "total": 0,
        }
    limit = max(1, min(100, limit))
    try:
        client = await get_backend_tool_client()
        result = await client.read_agent_messages(
            team_id=line_id,
            agent_profile_id=agent_profile_id,
            user_id=user_id,
            limit=limit,
        )
        if not result.get("success"):
            return {
                "formatted": result.get("error") or "Failed to read messages.",
                "items": [],
                "total": 0,
            }
        items = result.get("items") or []
        total = result.get("total") or 0
        lines = []
        for m in items:
            from_name = m.get("from_agent_name") or m.get("from_agent_handle") or "Agent"
            to_name = m.get("to_agent_name") or m.get("to_agent_handle") or "team"
            content = (m.get("content") or "")[:150]
            if len(m.get("content") or "") > 150:
                content += "..."
            created = m.get("created_at") or ""
            direction = "to me" if m.get("to_agent_id") == agent_profile_id else "from me"
            lines.append(f"- [{direction}] {from_name} -> {to_name}: \"{content}\" ({created})")
        formatted = (
            f"Messages to/from you ({len(items)} in this response, {total} total matching filter):\n" + "\n".join(lines)
            if lines
            else f"No messages (total={total})."
        )
        return {"formatted": formatted, "items": items, "total": total}
    except Exception as e:
        logger.warning("read_my_messages failed: %s", e)
        return {"formatted": str(e), "items": [], "total": 0}


class GetTeamStatusBoardInputs(BaseModel):
    """No required inputs; line_id from pipeline metadata."""


class GetTeamStatusBoardOutputs(BaseModel):
    """Outputs from get_team_status_board."""
    formatted: str = Field(description="Human-readable team overview")
    board: Dict[str, Any] = Field(default_factory=dict, description="Structured board (team_name, members with tasks, goals, last_activity_at)")


async def get_team_status_board_tool(
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Get a comprehensive team status overview: all members with their current tasks, goal assignments, and last activity.
    Use this to understand who is doing what. Only available when running in a team.
    """
    metadata = _pipeline_metadata or {}
    line_id = line_id_from_metadata(metadata)
    if not line_id:
        return {
            "formatted": "No agent line context; cannot get status board.",
            "board": {},
        }
    if not _is_uuid(line_id):
        return {
            "formatted": "line_id must be the agent line UUID, not the line name. It is automatically provided from context.",
            "board": {},
        }
    try:
        client = await get_backend_tool_client()
        result = await client.get_team_status_board(team_id=line_id, user_id=user_id)
        if not result.get("success"):
            return {
                "formatted": result.get("error") or "Failed to get status board.",
                "board": {},
            }
        board = result.get("board") or {}
        team_name = board.get("team_name", "Team")
        members = board.get("members") or []
        current_agent_id = (metadata.get("agent_profile_id") or "").strip()
        lines = [f"Line: {team_name} (line_id: {line_id})", ""]
        for m in members:
            name = m.get("agent_name") or "Unknown"
            handle = m.get("agent_handle") or ""
            role = m.get("role") or "worker"
            task_count = m.get("task_count") or 0
            goal_count = m.get("goal_count") or 0
            last_at = m.get("last_activity_at") or "no activity"
            aid = str(m.get("agent_profile_id") or "")
            is_you = aid == current_agent_id
            you_tag = " (you)" if is_you else ""
            line = f"- {name}{you_tag} ({role}): {task_count} tasks, {goal_count} goals, last active {last_at}"
            direct_reports = m.get("direct_reports") or []
            if direct_reports:
                report_names = [r.get("agent_name") or r.get("agent_handle") or r.get("agent_profile_id") for r in direct_reports]
                line += f"; direct reports: {', '.join(report_names)}"
            manager_name = m.get("reports_to_agent_name")
            if manager_name:
                line += f"; reports to: {manager_name}"
            lines.append(line)
            # Show task/goal details with UUIDs (limit to first 3 each to avoid token bloat)
            tasks = m.get("tasks", [])[:3]
            goals = m.get("goals", [])[:3]
            if tasks:
                task_parts = []
                for t in tasks:
                    title = (t.get("title") or "")[:50]
                    tid = t.get("id", "")
                    status = t.get("status", "")
                    task_parts.append(f'"{title}" (task_id: {tid}, {status})')
                lines.append(f"  Tasks: {', '.join(task_parts)}" + ("..." if len(m.get("tasks", [])) > 3 else ""))
            if goals:
                goal_parts = []
                for g in goals:
                    title = (g.get("title") or "")[:50]
                    gid = g.get("id", "")
                    pct = g.get("progress_pct", 0)
                    goal_parts.append(f'"{title}" (goal_id: {gid}, {pct}%)')
                lines.append(f"  Goals: {', '.join(goal_parts)}" + ("..." if len(m.get("goals", [])) > 3 else ""))
        if current_agent_id:
            me = next((m for m in members if str(m.get("agent_profile_id") or "") == current_agent_id), None)
            if me:
                lines.append("")
                lines.append("Your context:")
                direct_reports = me.get("direct_reports") or []
                if direct_reports:
                    parts = []
                    for r in direct_reports:
                        name = r.get("agent_name") or r.get("agent_handle") or "Unknown"
                        pid = r.get("agent_profile_id")
                        parts.append(f"{name} (agent_profile_id: {pid})" if pid else name)
                    lines.append("  Your direct reports (to assign work so they run: create_task_for_agent; workspace/messages alone do not schedule workers): " + ", ".join(parts))
                else:
                    lines.append("  Your direct reports: (none)")
                manager_name = me.get("reports_to_agent_name")
                if manager_name:
                    lines.append(f"  Your manager: {manager_name}")
                peers = me.get("peers") or []
                if peers:
                    lines.append(f"  Your peers (same manager): {', '.join((p.get('agent_name') or p.get('agent_handle') or p.get('agent_profile_id') for p in peers))}")
        formatted = "\n".join(lines)
        return {"formatted": formatted, "board": board}
    except Exception as e:
        logger.warning("get_team_status_board failed: %s", e)
        return {"formatted": str(e), "board": {}}


register_action(
    name="send_to_agent",
    category="agent_communication",
    description="Send a message to another agent; optionally wait for response. To assign work so the worker runs on the next cycle, use create_task_for_agent instead; this only sends a message (or runs them now if wait_for_response=true).",
    inputs_model=SendToAgentInputs,
    params_model=SendToAgentParams,
    outputs_model=SendToAgentOutputs,
    tool_function=send_to_agent_tool,
)
register_action(
    name="read_team_timeline",
    category="agent_communication",
    description="Read recent team timeline messages (what other agents have said). Defaults to last 24h; pass since_hours=0 for all messages. Use when running in a team.",
    inputs_model=ReadTeamTimelineInputs,
    params_model=ReadTeamTimelineParams,
    outputs_model=ReadTeamTimelineOutputs,
    tool_function=read_team_timeline_tool,
)
register_action(
    name="read_my_messages",
    category="agent_communication",
    description="Read messages sent to or from the current agent on the team timeline. Use when running in a team.",
    inputs_model=ReadMyMessagesInputs,
    params_model=ReadMyMessagesParams,
    outputs_model=ReadMyMessagesOutputs,
    tool_function=read_my_messages_tool,
)
register_action(
    name="get_team_status_board",
    category="agent_communication",
    description="Get team overview: all members with their tasks, goals, and last activity. Use when running in a team.",
    inputs_model=GetTeamStatusBoardInputs,
    params_model=None,
    outputs_model=GetTeamStatusBoardOutputs,
    tool_function=get_team_status_board_tool,
)
