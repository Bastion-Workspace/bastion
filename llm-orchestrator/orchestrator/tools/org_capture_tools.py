"""
Org Capture Tools - Org-mode inbox via backend gRPC
"""

import logging
from typing import List, Optional, Union

from orchestrator.backend_tool_client import get_backend_tool_client

logger = logging.getLogger(__name__)


def _to_tag_list(value: Optional[Union[List[str], str]]) -> Optional[List[str]]:
    """Accept comma-separated string or list for tags."""
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [s.strip() for s in value.split(",") if s.strip()] or None
    return None


async def add_org_inbox_item_tool(
    user_id: str = "system",
    text: str = "",
    kind: str = "todo",
    schedule: Optional[str] = None,
    tags: Optional[Union[List[str], str]] = None,
) -> str:
    """
    Add an item to the user's org-mode inbox. Call once per distinct item; do not call multiple times with the same text.

    Args:
        user_id: User ID (injected by engine if omitted).
        text: Item text (e.g. "Get groceries", "Call mom").
        kind: "todo", "note", "checkbox", "event", or "contact". Default "todo". Use "note" for longer-form notes (headline without TODO).
        schedule: Optional org timestamp (e.g. "<2026-02-05 Thu>").
        tags: Optional tags, comma-separated string or list (e.g. "work", "urgent").

    Returns:
        Success message with line preview or error.
    """
    try:
        if not text or not text.strip():
            return "Error: text is required."
        tag_list = _to_tag_list(tags)
        logger.info("add_org_inbox_item: kind=%s text=%s", kind, text[:80])
        client = await get_backend_tool_client()
        result = await client.add_org_inbox_item(
            user_id=user_id,
            text=text.strip(),
            kind=kind,
            schedule=schedule,
            repeater=None,
            tags=tag_list,
        )
        if not result.get("success"):
            return result.get("error", "Failed to add item.")
        return result.get("message", "Item added to inbox.")
    except Exception as e:
        logger.error("add_org_inbox_item_tool error: %s", e)
        return f"Error: {str(e)}"
