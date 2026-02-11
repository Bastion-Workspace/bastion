"""
Session-scoped memory tools for multi-step workflows.

Clipboard store/retrieve keyed by user_id so steps can pass named values.
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)

_clipboard_store: Dict[str, Dict[str, str]] = {}


def clipboard_store_tool(key: str, value: str, user_id: str = "system") -> str:
    """
    Store a named value in session-scoped memory for use by later steps.

    Args:
        key: Name to store the value under.
        value: String value to store.
        user_id: User ID (injected by engine if omitted); used to scope the clipboard.

    Returns:
        Confirmation message.
    """
    if user_id not in _clipboard_store:
        _clipboard_store[user_id] = {}
    _clipboard_store[user_id][key] = value
    return f"Stored under key: {key}"


def clipboard_get_tool(key: str, user_id: str = "system") -> str:
    """
    Retrieve a value previously stored with clipboard_store_tool.

    Args:
        key: Name the value was stored under.
        user_id: User ID (injected by engine if omitted).

    Returns:
        The stored value, or a message if not found.
    """
    user_store = _clipboard_store.get(user_id, {})
    if key not in user_store:
        return f"No value stored for key: {key}"
    return user_store[key]
