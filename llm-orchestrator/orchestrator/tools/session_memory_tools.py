"""
Session-scoped memory tools for multi-step workflows.

Clipboard store/retrieve keyed by user_id so steps can pass named values.
"""

import logging
from typing import Any, Dict

from pydantic import BaseModel, Field

from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)

_clipboard_store: Dict[str, Dict[str, str]] = {}


class ClipboardStoreInputs(BaseModel):
    key: str = Field(description="Name to store the value under")
    value: str = Field(description="String value to store")


class ClipboardStoreOutputs(BaseModel):
    success: bool = Field(description="Whether the value was stored")
    key: str = Field(description="Key used")
    formatted: str = Field(description="Human-readable confirmation")


class ClipboardGetInputs(BaseModel):
    key: str = Field(description="Name the value was stored under")


class ClipboardGetOutputs(BaseModel):
    found: bool = Field(description="Whether the key was found")
    key: str = Field(description="Key requested")
    value: str = Field(default="", description="Stored value if found")
    formatted: str = Field(description="Human-readable result")


def clipboard_store_tool(key: str, value: str, user_id: str = "system") -> Dict[str, Any]:
    """Store a named value in session-scoped memory. Returns dict with success, key, formatted."""
    if user_id not in _clipboard_store:
        _clipboard_store[user_id] = {}
    _clipboard_store[user_id][key] = value
    msg = f"Stored under key: {key}"
    return {"success": True, "key": key, "formatted": msg}


def clipboard_get_tool(key: str, user_id: str = "system") -> Dict[str, Any]:
    """Retrieve a value from session clipboard. Returns dict with found, key, value, formatted."""
    user_store = _clipboard_store.get(user_id, {})
    if key not in user_store:
        msg = f"No value stored for key: {key}"
        return {"found": False, "key": key, "value": "", "formatted": msg}
    return {"found": True, "key": key, "value": user_store[key], "formatted": user_store[key]}


register_action(
    name="clipboard_store",
    category="session",
    description="Store a named value in session memory for later steps.",
    inputs_model=ClipboardStoreInputs,
    outputs_model=ClipboardStoreOutputs,
    tool_function=clipboard_store_tool,
)
register_action(
    name="clipboard_get",
    category="session",
    description="Retrieve a value stored by clipboard_store.",
    inputs_model=ClipboardGetInputs,
    outputs_model=ClipboardGetOutputs,
    tool_function=clipboard_get_tool,
)
