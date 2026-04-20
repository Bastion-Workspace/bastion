"""
Conversation cache search stub for gRPC / tool runners.

Former unified local search (`UnifiedSearchTools`, `search_local`) was removed; document search
uses `search_documents` via document-service. This module keeps `search_conversation_cache` only.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


_CONVERSATION_CACHE_UNAVAILABLE_MESSAGE = (
    "Conversation cache search is not implemented: prior messages in this thread are already "
    "included in the model context. Use search_documents and related tools to search the "
    "knowledge base."
)


async def search_conversation_cache(
    query: str,
    conversation_id: str = None,
    freshness_hours: int = 24,
    user_id: str = None,
) -> Dict[str, Any]:
    """
    Stub: conversation-scoped cache search is not implemented (orchestrator holds thread context).

    Returns a dict for gRPC and any caller that expects structured tool output; ``formatted`` is
    included for legacy LangGraph tool runners that read a unified dict shape.
    """
    try:
        logger.info(
            "Conversation cache search requested (not implemented): query=%s",
            (query or "")[:120],
        )
        return {
            "cache_hit": False,
            "entries": [],
            "message": _CONVERSATION_CACHE_UNAVAILABLE_MESSAGE,
            "formatted": _CONVERSATION_CACHE_UNAVAILABLE_MESSAGE,
        }
    except Exception as e:
        logger.error("Conversation cache search failed: %s", e)
        err = f"Conversation cache search failed: {e}"
        return {
            "cache_hit": False,
            "entries": [],
            "message": err,
            "formatted": err,
        }
