"""
RSS Tools - RSS feed management via backend gRPC
"""

import logging
from typing import Any, Dict

from orchestrator.backend_tool_client import get_backend_tool_client

logger = logging.getLogger(__name__)


def _format_rss_result(data: Dict[str, Any], default_msg: str = "Done.") -> str:
    """Format RSS result dict into a readable string for the LLM."""
    if not data:
        return default_msg
    if not data.get("success", True):
        return data.get("error", "Operation failed.")
    parts = []
    if "message" in data and data["message"]:
        parts.append(data["message"])
    if "feed_id" in data and data["feed_id"]:
        parts.append(f"Feed ID: {data['feed_id']}")
    if "task_id" in data and data["task_id"]:
        parts.append(f"Task ID: {data['task_id']}")
    if "feeds" in data:
        feeds = data["feeds"]
        count = data.get("count", len(feeds))
        parts.append(f"Feeds ({count}):")
        for i, f in enumerate(feeds[:20], 1):
            name = f.get("feed_name", "?")
            url = f.get("feed_url", "")
            fid = f.get("feed_id", "")
            parts.append(f"  {i}. {name} | {url} | ID: {fid}")
    return "\n".join(parts) if parts else default_msg


async def add_rss_feed_tool(
    user_id: str = "system",
    feed_url: str = "",
    feed_name: str = "",
    category: str = "",
    is_global: bool = False,
) -> str:
    """
    Add an RSS feed to the user's or global feeds.

    Args:
        user_id: User ID (injected by engine if omitted).
        feed_url: URL of the RSS feed (required).
        feed_name: Optional display name (defaults to URL).
        category: Optional category.
        is_global: If true, add as global feed (admin only).

    Returns:
        Success message with feed_id or error.
    """
    try:
        if not feed_url:
            return "Error: feed_url is required."
        logger.info("add_rss_feed: url=%s name=%s", feed_url[:80], feed_name[:50] if feed_name else "")
        client = await get_backend_tool_client()
        result = await client.add_rss_feed(
            user_id=user_id,
            feed_url=feed_url,
            feed_name=feed_name,
            category=category,
            is_global=is_global,
        )
        return _format_rss_result(result, "Feed added.")
    except Exception as e:
        logger.error("add_rss_feed_tool error: %s", e)
        return f"Error: {str(e)}"


async def list_rss_feeds_tool(
    user_id: str = "system",
    scope: str = "user",
) -> str:
    """
    List RSS feeds for the user or globally.

    Args:
        user_id: User ID (injected by engine if omitted).
        scope: "user" for my feeds, "global" for shared feeds.

    Returns:
        Formatted list of feeds with IDs and URLs.
    """
    try:
        logger.info("list_rss_feeds: scope=%s", scope)
        client = await get_backend_tool_client()
        result = await client.list_rss_feeds(user_id=user_id, scope=scope)
        return _format_rss_result(result, "No feeds found.")
    except Exception as e:
        logger.error("list_rss_feeds_tool error: %s", e)
        return f"Error: {str(e)}"


async def refresh_rss_feed_tool(
    user_id: str = "system",
    feed_name: str = "",
    feed_id: str = "",
) -> str:
    """
    Trigger a refresh for an RSS feed by name or feed ID.

    Args:
        user_id: User ID (injected by engine if omitted).
        feed_name: Display name of the feed (from list_rss_feeds).
        feed_id: Alternatively, feed ID.

    Returns:
        Success message with task_id or error.
    """
    try:
        if not feed_name and not feed_id:
            return "Error: feed_name or feed_id is required."
        logger.info("refresh_rss_feed: name=%s id=%s", feed_name[:50] if feed_name else "", feed_id[:50] if feed_id else "")
        client = await get_backend_tool_client()
        result = await client.refresh_rss_feed(
            user_id=user_id,
            feed_name=feed_name,
            feed_id=feed_id,
        )
        return _format_rss_result(result, "Refresh triggered.")
    except Exception as e:
        logger.error("refresh_rss_feed_tool error: %s", e)
        return f"Error: {str(e)}"
