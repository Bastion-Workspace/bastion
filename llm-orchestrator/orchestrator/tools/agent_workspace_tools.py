"""
Agent Workspace Tools - team shared workspace read/write (Blackboard pattern).
"""

import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action
from orchestrator.utils.line_context import line_id_from_metadata

logger = logging.getLogger(__name__)

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _is_uuid(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and _UUID_RE.match(s.strip()))


class WriteToWorkspaceInputs(BaseModel):
    """Required inputs for write_to_workspace."""
    key: str = Field(description="Workspace key (e.g. campaign_brief, competitor_analysis)")
    value: str = Field(description="Content to store")


class WriteToWorkspaceOutputs(BaseModel):
    """Outputs from write_to_workspace."""
    formatted: str = Field(description="Human-readable result")
    success: bool = Field(default=False)
    key: str = Field(default="")
    updated_at: str = Field(default="")


async def write_to_workspace_tool(
    key: str,
    value: str,
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Write a named entry to the team's shared workspace. Other agents can read it with read_workspace.
    Use for artifacts like campaign_brief, competitor_analysis, approved_taglines. Only available in a team.
    Does NOT create tasks or trigger worker dispatch—use create_task_for_agent to assign work.
    """
    metadata = _pipeline_metadata or {}
    line_id = line_id_from_metadata(metadata)
    agent_profile_id = metadata.get("agent_profile_id")
    if not line_id:
        return {"formatted": "No agent line context; cannot write to workspace.", "success": False, "key": "", "updated_at": ""}
    if not _is_uuid(line_id):
        return {"formatted": "line_id must be the agent line UUID, not the line name. It is automatically provided from context.", "success": False, "key": "", "updated_at": ""}
    key = (key or "").strip()
    if not key:
        return {"formatted": "key is required.", "success": False, "key": "", "updated_at": ""}
    try:
        client = await get_backend_tool_client()
        result = await client.set_workspace_entry(
            team_id=line_id,
            key=key,
            value=value or "",
            user_id=user_id,
            updated_by_agent_id=agent_profile_id,
        )
        if not result.get("success"):
            return {
                "formatted": result.get("error") or "Failed to write.",
                "success": False,
                "key": key,
                "updated_at": "",
            }
        return {
            "formatted": f"Wrote to workspace key \"{key}\".",
            "success": True,
            "key": key,
            "updated_at": result.get("updated_at") or "",
        }
    except Exception as e:
        logger.warning("write_to_workspace failed: %s", e)
        return {"formatted": str(e), "success": False, "key": key, "updated_at": ""}


class ReadWorkspaceInputs(BaseModel):
    """Optional key; if omitted, list all keys."""


class ReadWorkspaceParams(BaseModel):
    """Optional params for read_workspace."""
    key: Optional[str] = Field(default=None, description="Workspace key to read; if empty, list all keys")


class ReadWorkspaceOutputs(BaseModel):
    """Outputs from read_workspace."""
    formatted: str = Field(description="Human-readable content or list of keys")
    entry: Optional[Dict[str, Any]] = Field(default=None, description="Single entry when key was provided")
    entries: List[Dict[str, Any]] = Field(default_factory=list, description="List of key/metadata when listing")


async def read_workspace_tool(
    key: Optional[str] = None,
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Read one workspace entry by key, or list all workspace keys (with who updated and when).
    Use to get shared artifacts from other agents. Only available in a team.
    """
    metadata = _pipeline_metadata or {}
    line_id = line_id_from_metadata(metadata)
    if not line_id:
        return {"formatted": "No agent line context; cannot read workspace.", "entry": None, "entries": []}
    if not _is_uuid(line_id):
        return {"formatted": "line_id must be the agent line UUID, not the line name. It is automatically provided from context.", "entry": None, "entries": []}
    key = (key or "").strip() if key else None
    try:
        client = await get_backend_tool_client()
        result = await client.read_workspace(team_id=line_id, user_id=user_id, key=key)
        if not result.get("success"):
            return {
                "formatted": result.get("error") or "Failed to read workspace.",
                "entry": None,
                "entries": [],
            }
        if result.get("single") and result.get("entry"):
            ent = result["entry"]
            val = (ent.get("value") or "")[:2000]
            if len(ent.get("value") or "") > 2000:
                val += "..."
            formatted = f"Workspace key \"{ent.get('key')}\" (updated by {ent.get('updated_by_agent_name') or 'unknown'}, {ent.get('updated_at') or 'unknown'}):\n{val}"
            return {"formatted": formatted, "entry": ent, "entries": []}
        entries = result.get("entries") or []
        lines = [f"- {e.get('key')} (updated by {e.get('updated_by_agent_name') or 'unknown'}, {e.get('updated_at') or 'unknown'})" for e in entries]
        formatted = "Workspace keys:\n" + "\n".join(lines) if lines else "No workspace entries yet."
        return {"formatted": formatted, "entry": None, "entries": entries}
    except Exception as e:
        logger.warning("read_workspace failed: %s", e)
        return {"formatted": str(e), "entry": None, "entries": []}


register_action(
    name="write_to_workspace",
    category="agent_communication",
    description="Write a named entry to the team shared workspace. Other agents can read it. Does NOT create tasks or trigger workers; use create_task_for_agent to assign work.",
    inputs_model=WriteToWorkspaceInputs,
    params_model=None,
    outputs_model=WriteToWorkspaceOutputs,
    tool_function=write_to_workspace_tool,
)
register_action(
    name="read_workspace",
    category="agent_communication",
    description="Read one workspace entry by key or list all keys. Use when running in a team.",
    inputs_model=ReadWorkspaceInputs,
    params_model=ReadWorkspaceParams,
    outputs_model=ReadWorkspaceOutputs,
    tool_function=read_workspace_tool,
)
