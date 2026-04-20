"""
Agent Memory Tools - Persistent key-value and log memory per agent (Agent Factory).

get_agent_memory, set_agent_memory, list_agent_memories, delete_agent_memory, append_agent_memory.
Used by playbooks to store and recall state across runs.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)


def _user_and_profile(metadata: Optional[Dict[str, Any]]) -> tuple[str, str]:
    """Resolve user_id and agent_profile_id from pipeline metadata."""
    meta = metadata or {}
    user_id = meta.get("user_id", "system")
    profile_id = meta.get("agent_profile_id", "") or meta.get("profile_id", "")
    return user_id, profile_id


class GetAgentMemoryInputs(BaseModel):
    """Inputs for get_agent_memory."""
    memory_key: str = Field(description="Key to read (e.g. last_price, report_summary)")


class GetAgentMemoryOutputs(BaseModel):
    """Outputs from get_agent_memory."""
    value: Optional[Dict[str, Any]] = Field(default=None, description="Stored value (parsed JSON)")
    found: bool = Field(description="True if the key existed")
    formatted: str = Field(description="Human-readable summary")


async def get_agent_memory_tool(
    memory_key: str,
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Read a value from this agent's persistent memory."""
    uid, profile_id = _user_and_profile(_pipeline_metadata)
    if not profile_id:
        return {
            "value": None,
            "found": False,
            "formatted": "Agent memory requires agent_profile_id (run in an agent playbook).",
        }
    client = await get_backend_tool_client()
    value = await client.get_agent_memory(user_id=uid, agent_profile_id=profile_id, memory_key=memory_key)
    if value is None:
        return {
            "value": None,
            "found": False,
            "formatted": f"No value stored for key '{memory_key}'.",
        }
    return {
        "value": value,
        "found": True,
        "formatted": f"Key '{memory_key}': {json.dumps(value)[:200]}{'...' if len(json.dumps(value)) > 200 else ''}",
    }


class SetAgentMemoryInputs(BaseModel):
    """Inputs for set_agent_memory."""
    memory_key: str = Field(description="Key to write")
    memory_value: Dict[str, Any] = Field(description="Value to store (JSON object)")


class SetAgentMemoryParams(BaseModel):
    """Params for set_agent_memory."""
    memory_type: str = Field(default="kv", description="kv, log, or counter")
    expires_at: Optional[str] = Field(default=None, description="Optional ISO 8601 expiry")


class SetAgentMemoryOutputs(BaseModel):
    """Outputs from set_agent_memory."""
    success: bool = Field(description="True if written")
    formatted: str = Field(description="Human-readable summary")


async def set_agent_memory_tool(
    memory_key: str,
    memory_value: Dict[str, Any],
    memory_type: str = "kv",
    expires_at: Optional[str] = None,
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write a value to this agent's persistent memory."""
    uid, profile_id = _user_and_profile(_pipeline_metadata)
    if not profile_id:
        return {
            "success": False,
            "formatted": "Agent memory requires agent_profile_id (run in an agent playbook).",
        }
    client = await get_backend_tool_client()
    ok = await client.set_agent_memory(
        user_id=uid,
        agent_profile_id=profile_id,
        memory_key=memory_key,
        memory_value=memory_value,
        memory_type=memory_type,
        expires_at=expires_at,
    )
    return {
        "success": ok,
        "formatted": f"Stored '{memory_key}'." if ok else "Failed to store.",
    }


class ListAgentMemoriesInputs(BaseModel):
    """Inputs for list_agent_memories."""
    key_prefix: Optional[str] = Field(default=None, description="Optional prefix filter (e.g. report_)")


class ListAgentMemoriesOutputs(BaseModel):
    """Outputs from list_agent_memories."""
    memory_keys: List[str] = Field(description="List of keys")
    count: int = Field(description="Number of keys")
    formatted: str = Field(description="Human-readable summary")


async def list_agent_memories_tool(
    key_prefix: Optional[str] = None,
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """List keys in this agent's memory, optionally filtered by prefix."""
    uid, profile_id = _user_and_profile(_pipeline_metadata)
    if not profile_id:
        return {
            "memory_keys": [],
            "count": 0,
            "formatted": "Agent memory requires agent_profile_id (run in an agent playbook).",
        }
    client = await get_backend_tool_client()
    keys = await client.list_agent_memories(
        user_id=uid,
        agent_profile_id=profile_id,
        key_prefix=key_prefix,
    )
    keys = keys or []
    return {
        "memory_keys": keys,
        "count": len(keys),
        "formatted": f"Found {len(keys)} key(s): {', '.join(keys)}",
    }


class DeleteAgentMemoryInputs(BaseModel):
    """Inputs for delete_agent_memory."""
    memory_key: str = Field(description="Key to delete")


class DeleteAgentMemoryOutputs(BaseModel):
    """Outputs from delete_agent_memory."""
    success: bool = Field(description="True if deleted or key did not exist")
    formatted: str = Field(description="Human-readable summary")


async def delete_agent_memory_tool(
    memory_key: str,
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Delete a key from this agent's memory."""
    uid, profile_id = _user_and_profile(_pipeline_metadata)
    if not profile_id:
        return {
            "success": False,
            "formatted": "Agent memory requires agent_profile_id (run in an agent playbook).",
        }
    client = await get_backend_tool_client()
    ok = await client.delete_agent_memory(
        user_id=uid,
        agent_profile_id=profile_id,
        memory_key=memory_key,
    )
    return {
        "success": ok,
        "formatted": f"Deleted '{memory_key}'." if ok else "Failed to delete.",
    }


class AppendAgentMemoryInputs(BaseModel):
    """Inputs for append_agent_memory."""
    memory_key: str = Field(description="Log key to append to")
    entry: Dict[str, Any] = Field(description="Entry object to append (e.g. {timestamp, message})")


class AppendAgentMemoryOutputs(BaseModel):
    """Outputs from append_agent_memory."""
    success: bool = Field(description="True if appended")
    formatted: str = Field(description="Human-readable summary")


async def append_agent_memory_tool(
    memory_key: str,
    entry: Dict[str, Any],
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Append an entry to a log-type memory (creates the key as a list if missing)."""
    uid, profile_id = _user_and_profile(_pipeline_metadata)
    if not profile_id:
        return {
            "success": False,
            "formatted": "Agent memory requires agent_profile_id (run in an agent playbook).",
        }
    client = await get_backend_tool_client()
    ok = await client.append_agent_memory(
        user_id=uid,
        agent_profile_id=profile_id,
        memory_key=memory_key,
        entry=entry,
    )
    return {
        "success": ok,
        "formatted": f"Appended to '{memory_key}'." if ok else "Failed to append.",
    }


def register_agent_memory_actions():
    """Register agent memory tools with the action I/O registry."""
    register_action(
        name="get_agent_memory",
        category="memory",
        description="Read a value from this agent's persistent memory (key-value).",
        inputs_model=GetAgentMemoryInputs,
        outputs_model=GetAgentMemoryOutputs,
        tool_function=get_agent_memory_tool,
    )
    register_action(
        name="set_agent_memory",
        category="memory",
        description="Write a value to this agent's persistent memory.",
        inputs_model=SetAgentMemoryInputs,
        params_model=SetAgentMemoryParams,
        outputs_model=SetAgentMemoryOutputs,
        tool_function=set_agent_memory_tool,
    )
    register_action(
        name="list_agent_memories",
        category="memory",
        description="List keys in this agent's memory, optionally filtered by prefix.",
        inputs_model=ListAgentMemoriesInputs,
        outputs_model=ListAgentMemoriesOutputs,
        tool_function=list_agent_memories_tool,
    )
    register_action(
        name="delete_agent_memory",
        category="memory",
        description="Delete a key from this agent's memory.",
        inputs_model=DeleteAgentMemoryInputs,
        outputs_model=DeleteAgentMemoryOutputs,
        tool_function=delete_agent_memory_tool,
    )
    register_action(
        name="append_agent_memory",
        category="memory",
        description="Append an entry to a log-type memory (list).",
        inputs_model=AppendAgentMemoryInputs,
        outputs_model=AppendAgentMemoryOutputs,
        tool_function=append_agent_memory_tool,
    )


register_agent_memory_actions()
