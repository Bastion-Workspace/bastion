"""
Enhancement Tools - Query expansion and conversation caching
"""

import logging
from typing import Dict, Any, Optional, List

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)


# ── I/O models for enhancement tools ─────────────────────────────────────────

class EnhanceQueryInputs(BaseModel):
    """Required inputs for enhance_query_tool."""
    query: str = Field(description="Original query to enhance or expand")


class EnhanceQueryParams(BaseModel):
    """Optional parameters."""
    mode: str = Field(default="basic", description="basic = expand only; project_aware = analyze needs + project-aware queries")
    project_context: Optional[Dict[str, Any]] = Field(default=None, description="Project context for mode=project_aware")
    query_type: str = Field(default="research", description="Query type for mode=project_aware")
    num_variations: int = Field(default=3, description="Number of variations (basic mode)")
    conversation_context: Optional[str] = Field(default=None, description="Optional conversation context (basic mode)")


class EnhanceQueryOutputs(BaseModel):
    """Typed outputs for enhance_query_tool."""
    queries: List[str] = Field(description="Query strings to use for search")
    expanded_queries: List[str] = Field(description="Same as queries (backward compat)")
    information_needs: Optional[Dict[str, Any]] = Field(default=None, description="When mode=project_aware")
    search_queries: Optional[List[Dict[str, Any]]] = Field(default=None, description="List of {query, priority, focus} when project_aware")
    key_entities: List[Any] = Field(default_factory=list, description="Key entities (basic mode)")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class ExpandQueryInputs(BaseModel):
    """Required inputs for expand_query_tool (internal)."""
    query: str = Field(description="Original query to expand")


class ExpandQueryParams(BaseModel):
    """Optional parameters."""
    num_variations: int = Field(default=3, description="Number of variations to generate")
    conversation_context: Optional[str] = Field(default=None, description="Optional conversation context")


class ExpandQueryOutputs(BaseModel):
    """Typed outputs for expand_query_tool."""
    original_query: str = Field(description="Original query")
    expanded_queries: List[str] = Field(description="List of expanded query variations")
    key_entities: List[Any] = Field(default_factory=list, description="Key entities extracted")
    expansion_count: int = Field(description="Number of variations generated")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class SearchConversationCacheInputs(BaseModel):
    """Required inputs for search_conversation_cache_tool."""
    query: str = Field(description="Search query")


class SearchConversationCacheParams(BaseModel):
    """Optional parameters."""
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID to scope search")
    freshness_hours: int = Field(default=24, description="How recent to search (hours)")


class SearchConversationCacheOutputs(BaseModel):
    """Typed outputs for search_conversation_cache_tool."""
    cache_hit: bool = Field(description="Whether cache had matching entries")
    entries: List[Dict[str, Any]] = Field(default_factory=list, description="Cached entries found")
    entries_count: int = Field(description="Number of cached entries found")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def _expand_query_impl(
    query: str,
    num_variations: int = 3,
    conversation_context: Optional[str] = None
) -> Dict[str, Any]:
    """Internal: expand query with semantic variations (used by enhance_query_tool)."""
    try:
        client = await get_backend_tool_client()
        result = await client.expand_query(
            query=query,
            num_variations=num_variations,
            conversation_context=conversation_context
        )
        expanded = result.get("expanded_queries", [query])
        expansion_count = result.get("expansion_count", len(expanded))
        formatted = f"Expanded query into {len(expanded)} variation(s): " + "; ".join(expanded[:5])
        if len(expanded) > 5:
            formatted += f" ... and {len(expanded) - 5} more"
        return {
            "original_query": result.get("original_query", query),
            "expanded_queries": expanded,
            "key_entities": result.get("key_entities", []),
            "expansion_count": expansion_count,
            "formatted": formatted
        }
    except Exception as e:
        logger.error(f"Query expansion error: {e}")
        return {
            "original_query": query,
            "expanded_queries": [query],
            "key_entities": [],
            "expansion_count": 1,
            "formatted": f"Query expansion failed, using original query: {query}"
        }


async def enhance_query_tool(
    query: str,
    user_id: str = "system",
    project_context: Optional[Dict[str, Any]] = None,
    mode: str = "basic",
    query_type: str = "research",
    get_llm_func: Optional[Any] = None,
    num_variations: int = 3,
    conversation_context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Unified query enhancement: basic expansion or project-aware (analyze needs + generate queries).
    mode='basic': backend expand_query only.
    mode='project_aware': analyze_information_needs then generate_project_aware_queries; fallback to expand on failure.
    Returns queries (list of strings), optional information_needs and search_queries (when project_aware), formatted.
    """
    if mode != "project_aware" or not get_llm_func or not project_context:
        # Basic mode or missing project_aware params
        result = await _expand_query_impl(
            query=query,
            num_variations=num_variations,
            conversation_context=conversation_context
        )
        expanded = result.get("expanded_queries", [query])
        return {
            "queries": expanded,
            "expanded_queries": expanded,
            "information_needs": None,
            "search_queries": None,
            "key_entities": result.get("key_entities", []),
            "formatted": result.get("formatted", ""),
        }
    try:
        from orchestrator.tools.information_analysis_tools import (
            analyze_information_needs_tool,
            generate_project_aware_queries_tool,
        )
        info_result = await analyze_information_needs_tool(
            query=query,
            query_type=query_type,
            project_context=project_context,
            context_keys=None,
            domain_name="project",
            user_id=user_id,
            get_llm_func=get_llm_func,
        )
        gen_result = await generate_project_aware_queries_tool(
            query=query,
            query_type=query_type,
            information_needs=info_result,
            project_context=project_context,
            user_id=user_id,
            num_queries=max(3, num_variations),
            get_llm_func=get_llm_func,
        )
        search_queries = gen_result.get("search_queries", [])
        queries = [q.get("query", "") for q in search_queries if q.get("query")]
        if not queries:
            queries = [query]
        return {
            "queries": queries,
            "expanded_queries": queries,
            "information_needs": info_result,
            "search_queries": search_queries,
            "key_entities": info_result.get("relevant_entities", []),
            "formatted": gen_result.get("formatted", f"Generated {len(queries)} query(ies)."),
        }
    except Exception as e:
        logger.warning(f"Project-aware query enhancement failed, falling back to expand: {e}")
        result = await _expand_query_impl(query=query, num_variations=num_variations, conversation_context=conversation_context)
        expanded = result.get("expanded_queries", [query])
        return {
            "queries": expanded,
            "expanded_queries": expanded,
            "information_needs": None,
            "search_queries": None,
            "key_entities": result.get("key_entities", []),
            "formatted": result.get("formatted", ""),
        }




async def search_conversation_cache_tool(
    query: str,
    conversation_id: str = None,
    freshness_hours: int = 24
) -> Dict[str, Any]:
    """
    Search conversation cache for previous research
    
    Args:
        query: Search query
        conversation_id: Conversation ID (optional)
        freshness_hours: How recent to search
        
    Returns:
        Dict with cache_hit and entries
    """
    try:
        logger.info(f"Searching conversation cache: {query[:100]}")
        
        client = await get_backend_tool_client()
        result = await client.search_conversation_cache(
            query=query,
            conversation_id=conversation_id,
            freshness_hours=freshness_hours
        )
        
        entries = result.get("entries", [])
        entries_count = len(entries) if isinstance(entries, list) else 0
        if result.get("cache_hit"):
            logger.info(f"Cache hit! Found {entries_count} cached entries")
            formatted = f"Found {entries_count} cached entr(y/ies) for this query."
        else:
            logger.info("Cache miss - no previous research found")
            formatted = "No cached research found for this query."
        return {
            **result,
            "entries_count": entries_count,
            "formatted": formatted
        }
        
    except Exception as e:
        logger.error(f"Cache search tool error: {e}")
        return {
            "cache_hit": False,
            "entries": [],
            "entries_count": 0,
            "formatted": "Conversation cache search failed."
        }


register_action(
    name="enhance_query",
    category="search",
    description="Expand or enhance query: basic (semantic variations) or project_aware (analyze needs + targeted queries)",
    inputs_model=EnhanceQueryInputs,
    params_model=EnhanceQueryParams,
    outputs_model=EnhanceQueryOutputs,
    tool_function=enhance_query_tool,
)
register_action(
    name="search_conversation_cache",
    category="search",
    description="Search conversation cache for previous research",
    inputs_model=SearchConversationCacheInputs,
    params_model=SearchConversationCacheParams,
    outputs_model=SearchConversationCacheOutputs,
    tool_function=search_conversation_cache_tool,
)


# Tool registry
ENHANCEMENT_TOOLS = {
    'enhance_query': enhance_query_tool,
    'search_conversation_cache': search_conversation_cache_tool
}

