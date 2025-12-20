"""
Query Expansion Tool - Universal Query Expansion for LangGraph Agents
Standalone LangGraph tool for generating alternative search queries

Agents use this tool to explicitly request query expansions when needed,
giving them full control over search strategy.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class QueryExpansionInput(BaseModel):
    """Input schema for query expansion tool"""
    original_query: str = Field(..., description="Original search query to expand")
    num_expansions: int = Field(default=2, ge=1, le=5, description="Number of alternative queries to generate")
    expansion_type: str = Field(default="semantic", description="Type of expansion: semantic, synonym, broader, narrower")


class QueryExpansionResult(BaseModel):
    """Result from query expansion"""
    original_query: str = Field(..., description="Original query")
    expanded_queries: List[str] = Field(..., description="Generated alternative queries")
    all_queries: List[str] = Field(..., description="Original + expanded queries combined")
    expansion_count: int = Field(..., description="Number of expansions generated")


async def expand_query(original_query: str, num_expansions: int = 2, expansion_type: str = "semantic", conversation_context: Optional[str] = None, user_id: Optional[str] = None) -> str:
    """
    Generate alternative search queries for better recall across local and web searches.
    
    This tool creates semantically related queries to improve search coverage.
    Results can be used by both local and web search tools.
    
    Args:
        original_query: The query to expand
        num_expansions: Number of alternative queries (1-5)
        expansion_type: Type of expansion (currently informational only)
        conversation_context: Optional conversation context (last 2 messages) to help resolve vague references
        
    Returns:
        JSON string with original query, expanded queries, and combined list
    """
    try:
        logger.info(f"Expanding query: '{original_query[:50]}...' ({num_expansions} variations)")
        
        # Use the standalone Query Expansion Service
        from services.query_expansion_service import get_query_expansion_service
        
        service = await get_query_expansion_service()
        expanded_queries = await service.expand_query(
            query_text=original_query,
            num_expansions=num_expansions,
            conversation_context=conversation_context,
            user_id=user_id
        )
        
        # Create comprehensive result
        all_queries = [original_query] + expanded_queries
        
        result = QueryExpansionResult(
            original_query=original_query,
            expanded_queries=expanded_queries,
            all_queries=all_queries,
            expansion_count=len(expanded_queries)
        )
        
        logger.info(f"✅ Generated {len(expanded_queries)} query expansions")
        for i, query in enumerate(expanded_queries, 1):
            logger.info(f"   {i}. {query}")
        
        # Return JSON string for LangGraph compatibility
        return json.dumps(result.dict(), indent=2)
        
    except Exception as e:
        logger.error(f"Query expansion failed: {e}")
        # Return minimal result with just original query
        fallback_result = QueryExpansionResult(
            original_query=original_query,
            expanded_queries=[],
            all_queries=[original_query],
            expansion_count=0
        )
        return json.dumps(fallback_result.dict(), indent=2)


# Global function for tool registry integration
async def expand_query_universal(original_query: str, num_expansions: int = 2, expansion_type: str = "semantic", user_id: str = None) -> str:
    """
    Universal query expansion function for tool registry
    
    This is the entry point called by the centralized tool registry.
    The user_id parameter is used for timezone-aware date/time context.
    """
    return await expand_query(original_query, num_expansions, expansion_type, user_id=user_id)
