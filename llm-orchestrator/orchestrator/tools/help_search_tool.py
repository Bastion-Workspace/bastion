"""
Help Search Tool - Search app help documentation for Agent Factory.
"""

import logging
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)


class SearchHelpDocsInputs(BaseModel):
    """Required inputs for help docs search."""
    query: str = Field(description="Natural language question about the app (e.g. how to create a folder, what the research agent does)")


class SearchHelpDocsParams(BaseModel):
    """Optional parameters."""
    limit: int = Field(default=5, description="Max number of help sections to return")


class SearchHelpDocsOutputs(BaseModel):
    """Outputs for search_help_docs tool."""
    results: List[Dict[str, Any]] = Field(description="List of help results with topic_id, title, content, score")
    count: int = Field(description="Number of results")
    query_used: str = Field(description="The query that was executed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def search_help_docs_tool(
    query: str,
    limit: int = 5,
    user_id: str = "system",
) -> Dict[str, Any]:
    """
    Search app help documentation for how-to questions about Bastion features.
    Use when the user asks how to do something in the app, what a feature does, or where to find something.
    Returns structured dict with results, count, query_used, and formatted.
    """
    try:
        query = (query or "").strip()
        if not query:
            return {
                "results": [],
                "count": 0,
                "query_used": "",
                "formatted": "No search query provided.",
            }
        client = await get_backend_tool_client()
        result = await client.search_help_docs(query=query, user_id=user_id, limit=limit)
        if "error" in result:
            return {
                "results": [],
                "count": 0,
                "query_used": query,
                "formatted": "Help docs search failed: %s" % result["error"],
            }
        raw = result.get("results", [])
        total = result.get("total_count", 0)
        parts = []
        for i, r in enumerate(raw, 1):
            title = r.get("title", "")
            topic_id = r.get("topic_id", "")
            content = (r.get("content") or "").strip()
            score = r.get("score", 0.0)
            parts.append(f"**{i}. {title}** (topic: {topic_id})")
            parts.append("Relevance: %.2f" % score)
            parts.append(content[:2000])
            parts.append("")
        formatted = "\n".join(parts).strip() if parts else "No help topics matched your query."
        return {
            "results": raw,
            "count": total,
            "query_used": query,
            "formatted": formatted,
        }
    except Exception as e:
        logger.warning("search_help_docs_tool error: %s", e)
        return {
            "results": [],
            "count": 0,
            "query_used": query if query else "",
            "formatted": "Help documentation search is temporarily unavailable.",
        }


register_action(
    name="search_help_docs",
    category="search",
    description="Search app help documentation for how-to questions about Bastion (e.g. how to create a folder, what the research agent does, where to find settings).",
    inputs_model=SearchHelpDocsInputs,
    params_model=SearchHelpDocsParams,
    outputs_model=SearchHelpDocsOutputs,
    tool_function=search_help_docs_tool,
)
