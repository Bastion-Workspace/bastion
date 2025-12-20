"""
Enhancement Tools - Query expansion and conversation caching
"""

import logging
from typing import Dict, Any, Optional

from orchestrator.backend_tool_client import get_backend_tool_client

logger = logging.getLogger(__name__)


async def expand_query_tool(
    query: str,
    num_variations: int = 3,
    conversation_context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Expand query with semantic variations
    
    Args:
        query: Original query
        num_variations: Number of variations to generate
        conversation_context: Optional conversation context (last 2 messages) to help resolve vague references
        
    Returns:
        Dict with expanded queries and entities
    """
    try:
        logger.info(f"Expanding query: {query[:100]}")
        
        client = await get_backend_tool_client()
        result = await client.expand_query(
            query=query, 
            num_variations=num_variations,
            conversation_context=conversation_context
        )
        
        logger.info(f"Generated {result['expansion_count']} query variations")
        return result
        
    except Exception as e:
        logger.error(f"Query expansion tool error: {e}")
        return {
            'original_query': query,
            'expanded_queries': [query],
            'key_entities': [],
            'expansion_count': 1
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
        
        if result['cache_hit']:
            logger.info(f"Cache hit! Found {len(result['entries'])} cached entries")
        else:
            logger.info("Cache miss - no previous research found")
        
        return result
        
    except Exception as e:
        logger.error(f"Cache search tool error: {e}")
        return {'cache_hit': False, 'entries': []}


# Tool registry
ENHANCEMENT_TOOLS = {
    'expand_query': expand_query_tool,
    'search_conversation_cache': search_conversation_cache_tool
}

