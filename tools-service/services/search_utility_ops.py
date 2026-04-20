"""
Search utility operations for ToolService gRPC (non-protobuf layer).

Co-occurrence KG lookup, query expansion, and conversation cache normalization.
Imports only leaf `services.*` modules — not grpc_handlers.
"""

import json
from typing import Any, Dict, List, Optional


async def find_co_occurring_entities(
    entity_names: List[str],
    min_co_occurrences: int,
) -> List[Dict[str, Any]]:
    """
    Return co-occurring entities from the knowledge graph service, if connected.

    Each item: name, type, co_occurrence_count (aligned with EntityInfo mapping).
    """
    from services.service_container import get_service_container

    container = await get_service_container()
    kg_service = getattr(container, "knowledge_graph_service", None)
    if not kg_service or not kg_service.is_connected():
        return []

    co_occurring = await kg_service.find_co_occurring_entities(
        list(entity_names),
        min_co_occurrences=min_co_occurrences,
    )
    out: List[Dict[str, Any]] = []
    for entity in co_occurring or []:
        out.append(
            {
                "name": entity["name"],
                "type": entity["type"],
                "co_occurrence_count": entity["co_occurrence_count"],
            }
        )
    return out


async def expand_query_for_rpc(
    *,
    original_query: str,
    num_variations: int,
    conversation_context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run query expansion and return a dict suitable for QueryExpansionResponse.

    Keys: original_query, expanded_queries (list[str]), key_entities (list).
    """
    from services.langgraph_tools.query_expansion_tool import expand_query

    result_json = await expand_query(
        original_query=original_query,
        num_expansions=num_variations,
        conversation_context=conversation_context,
    )
    try:
        parsed = json.loads(result_json)
    except (json.JSONDecodeError, TypeError):
        return {
            "original_query": original_query,
            "expanded_queries": [],
            "key_entities": [],
        }

    if not isinstance(parsed, dict):
        return {
            "original_query": original_query,
            "expanded_queries": [],
            "key_entities": [],
        }

    expanded = parsed.get("expanded_queries") or []
    if not isinstance(expanded, list):
        expanded = []
    key_entities = parsed.get("key_entities") or []
    if not isinstance(key_entities, list):
        key_entities = []

    return {
        "original_query": parsed.get("original_query", original_query),
        "expanded_queries": expanded,
        "key_entities": key_entities,
    }


async def search_conversation_cache_for_rpc(
    *,
    query: str,
    conversation_id: Optional[str],
    freshness_hours: int,
) -> Dict[str, Any]:
    """
    Run conversation cache search and normalize to cache_hit + entries.

    Each entry dict: content, timestamp, agent_name, relevance_score.
    """
    from services.langgraph_tools.unified_search_tools import search_conversation_cache

    result = await search_conversation_cache(
        query=query,
        conversation_id=conversation_id,
        freshness_hours=freshness_hours,
    )

    if isinstance(result, dict):
        cache_hit = bool(result.get("cache_hit", False))
        entries: List[Dict[str, Any]] = []
        for raw in list(result.get("entries") or []):
            if isinstance(raw, dict):
                entries.append(
                    {
                        "content": raw.get("content", ""),
                        "timestamp": str(raw.get("timestamp", "")),
                        "agent_name": str(raw.get("agent_name", "")),
                        "relevance_score": float(raw.get("relevance_score", 0.0)),
                    }
                )
        msg = (result.get("message") or "").strip()
        if not entries and msg:
            entries.append(
                {
                    "content": msg,
                    "timestamp": "",
                    "agent_name": "",
                    "relevance_score": 0.0,
                }
            )
        return {"cache_hit": cache_hit, "entries": entries}

    text = str(result).strip() if result is not None else ""
    if text:
        return {
            "cache_hit": False,
            "entries": [
                {
                    "content": text,
                    "timestamp": "",
                    "agent_name": "",
                    "relevance_score": 0.0,
                }
            ],
        }
    return {"cache_hit": False, "entries": []}
