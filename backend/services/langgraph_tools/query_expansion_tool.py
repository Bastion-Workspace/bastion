"""
Query Expansion Tool - Roosevelt's Universal Query Expansion
Standalone LangGraph tool for generating alternative search queries
"""

import logging
import json
import asyncio
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
# Removed LangChain tool decorator - using simple async function instead

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


async def expand_query(original_query: str, num_expansions: int = 2, expansion_type: str = "semantic") -> str:
    """
    Generate alternative search queries for better recall across local and web searches.
    
    This tool creates semantically related queries to improve search coverage.
    Results can be used by both local and web search tools.
    """
    try:
        logger.info(f"🔍 ROOSEVELT'S QUERY EXPANSION: Expanding '{original_query[:50]}...' ({num_expansions} variations)")
        
        # Get expanded queries using the same logic as embedding_manager
        expanded_queries = await _generate_query_expansions_universal(
            original_query, 
            num_expansions,
            expansion_type
        )
        
        # Create comprehensive result
        all_queries = [original_query] + expanded_queries
        
        result = QueryExpansionResult(
            original_query=original_query,
            expanded_queries=expanded_queries,
            all_queries=all_queries,
            expansion_count=len(expanded_queries)
        )
        
        logger.info(f"✅ QUERY EXPANSION: Generated {len(expanded_queries)} alternatives")
        for i, query in enumerate(expanded_queries, 1):
            logger.info(f"   {i}. {query}")
        
        # Return JSON string for LangGraph compatibility
        return json.dumps(result.dict(), indent=2)
        
    except Exception as e:
        logger.error(f"❌ Query expansion failed: {e}")
        # Return minimal result with just original query
        fallback_result = QueryExpansionResult(
            original_query=original_query,
            expanded_queries=[],
            all_queries=[original_query],
            expansion_count=0
        )
        return json.dumps(fallback_result.dict(), indent=2)


async def _generate_query_expansions_universal(
    query_text: str, 
    num_expansions: int = 2,
    expansion_type: str = "semantic"
) -> List[str]:
    """
    Universal query expansion using Intent Classification model
    Extracted from embedding_manager and made standalone
    """
    try:
        # Import chat service for LLM access (same pattern as capability-based classifier)
        from services.chat_service import ChatService
        from config import settings
        
        chat_service = ChatService()
        
        # Initialize chat service if needed
        if not chat_service.openai_client:
            await chat_service.initialize()
        
        # **ROOSEVELT FIX**: Use user-configured classification model from Settings, not config.py
        from services.settings_service import settings_service
        expansion_model = await settings_service.get_classification_model()
        
        # Enhanced prompt based on the existing embedding_manager logic
        expansion_prompt = f"""Create {num_expansions} alternative search queries for: "{query_text}"

EXPANSION TYPE: {expansion_type}

Consider these strategies:
1. If it looks like a filename or document ID, try variations without extensions
2. If it's a number, try different formats (e.g., "795254" -> "795254.pdf", "document 795254")
3. If it's a title, try synonyms and related terms
4. If it's a topic, try broader and narrower terms

Generate different ways to search for the same information using:
- Synonyms and related terms
- Different word order
- Abbreviations and acronyms
- Broader and narrower concepts
- File extensions and document types

Return only a JSON array of the {num_expansions} alternative queries:
["alternative query 1", "alternative query 2"]"""

        logger.info(f"🔍 Using expansion model: {expansion_model}")
        
        # Add datetime context for consistency
        from utils.system_prompt_utils import add_datetime_context_to_system_prompt
        
        system_prompt = add_datetime_context_to_system_prompt(
            "You generate alternative search queries. Return only a valid JSON array with no additional text or explanations."
        )
        
        # Make LLM call with timeout
        response = await asyncio.wait_for(
            chat_service.openai_client.chat.completions.create(
                model=expansion_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": expansion_prompt}
                ],
                max_tokens=300,
                temperature=0.7
            ),
            timeout=30.0
        )
        
        if not response.choices or not response.choices[0].message.content:
            logger.warning("🔍 LLM returned empty response for query expansion")
            return []
        
        expansions_text = response.choices[0].message.content.strip()
        logger.info(f"🔍 Raw expansion response: {expansions_text}")
        
        # Parse JSON array
        try:
            # Extract JSON if embedded in other text
            json_start = expansions_text.find('[')
            json_end = expansions_text.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = expansions_text[json_start:json_end]
                raw_queries = json.loads(json_text)
                
                # Filter out duplicates and original query
                expansions = [q for q in raw_queries if q.lower().strip() != query_text.lower().strip()]
                return expansions[:num_expansions]
            else:
                logger.warning(f"🔍 No JSON array found in response: '{expansions_text}'")
                return []
                
        except json.JSONDecodeError as e:
            logger.warning(f"🔍 Failed to parse JSON from LLM: {e}")
            logger.warning(f"🔍 Raw response: '{expansions_text}'")
            return []
        
    except Exception as e:
        logger.error(f"❌ Query expansion generation failed: {e}")
        return []


# Global function for tool registry integration
async def expand_query_universal(original_query: str, num_expansions: int = 2, expansion_type: str = "semantic", user_id: str = None) -> str:
    """Universal query expansion function for tool registry"""
    # user_id parameter is passed by the tool registry but not used for expansion
    return await expand_query(original_query, num_expansions, expansion_type)
