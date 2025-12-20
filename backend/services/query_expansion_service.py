"""
Query Expansion Service - Standalone service for generating alternative search queries

This service was extracted from EmbeddingManager to follow clean architecture principles.
Agents use this service through the LangGraph query_expansion_tool to explicitly
decide when to expand queries, rather than having it happen automatically.
"""

import asyncio
import json
import logging
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, validator

from config import settings
from utils.system_prompt_utils import add_datetime_context_to_system_prompt

logger = logging.getLogger(__name__)


class QueryExpansions(BaseModel):
    """Pydantic model for validating query expansion responses"""
    queries: List[str]
    
    @validator('queries')
    def validate_queries(cls, v):
        """Filter out invalid queries"""
        valid_queries = []
        for query in v:
            if isinstance(query, str) and len(query.strip()) > 5:
                valid_queries.append(query.strip())
        return valid_queries[:5]  # Limit to 5 expansions max


class QueryExpansionService:
    """
    Service for generating alternative search queries using LLM
    
    This service provides query expansion capabilities for agents to improve
    search recall. Agents explicitly request expansions when needed, giving
    them full control over search strategy.
    """
    
    def __init__(self):
        self.client: Optional[AsyncOpenAI] = None
        self.query_expansion_cache: Dict[str, List[str]] = {}
        self.expansion_cache_timestamps: Dict[str, float] = {}
        self.expansion_cache_ttl: int = 3600  # 1 hour TTL
        self._initialized = False
    
    async def initialize(self):
        """Initialize the query expansion service"""
        if self._initialized:
            return
        
        logger.info("Initializing Query Expansion Service...")
        
        if not settings.OPENROUTER_API_KEY:
            logger.warning("OPENROUTER_API_KEY not set - query expansion will be unavailable")
            return
        
        # Initialize OpenRouter client with automatic reasoning support
        from utils.openrouter_client import get_openrouter_client
        self.client = get_openrouter_client()
        
        self._initialized = True
        logger.info("✅ Query Expansion Service initialized")
    
    async def expand_query(
        self, 
        query_text: str, 
        num_expansions: int = 2,
        expansion_model: Optional[str] = None,
        conversation_context: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[str]:
        """
        Generate alternative query formulations using LLM with caching
        
        Args:
            query_text: Original search query
            num_expansions: Number of alternative queries to generate
            expansion_model: Model to use (defaults to settings.FAST_MODEL)
            conversation_context: Optional conversation context (last 2 messages) to help resolve vague references
            
        Returns:
            List of alternative query strings
        """
        if not self._initialized:
            await self.initialize()
        
        if not self.client:
            logger.warning("Query expansion client not available")
            return []
        
        try:
            # Check cache first (but include context in cache key if provided)
            cache_key = query_text
            if conversation_context:
                cache_key = f"{query_text}|||{conversation_context[:100]}"
            cached_expansions = self._get_cached_expansions(cache_key)
            if cached_expansions is not None:
                return cached_expansions[:num_expansions]
            
            # Use provided model or fall back to classification model from settings service
            if expansion_model:
                model_to_use = expansion_model
            else:
                # Get classification model from settings service (user-configured fast model)
                try:
                    from services.settings_service import settings_service
                    if not hasattr(settings_service, '_initialized') or not settings_service._initialized:
                        await settings_service.initialize()
                    model_to_use = await settings_service.get_classification_model()
                except Exception as e:
                    logger.warning(f"Failed to get classification model from settings service: {e}, falling back to FAST_MODEL")
                    model_to_use = settings.FAST_MODEL
            
            if not model_to_use:
                logger.warning("No model specified for query expansion and no fast model configured")
                return []
            
            # Build expansion prompt with optional conversation context
            context_section = ""
            if conversation_context:
                context_section = f"""

**PREVIOUS CONVERSATION CONTEXT**:
{conversation_context}

**IMPORTANT**: The current query may contain vague references like "this", "that", "it", "they", etc. Use the conversation context above to understand what these references mean. When generating alternative queries, incorporate the relevant context from the previous conversation to make the queries more specific and meaningful.

For example, if the context shows a discussion about "Rob Reiner investigation" and the current query is "Were the Obamas involved in this?", generate queries that explicitly include "Rob Reiner" or "Rob Reiner investigation" rather than just generic "Obama involvement" queries.
"""
            
            # Enhanced prompt for better document discovery
            expansion_prompt = f"""Create {num_expansions} alternative search queries for: "{query_text}"{context_section}

Consider these strategies:
1. If it looks like a filename or document ID, try variations without extensions
2. If it's a number, try different formats (e.g., "795254" -> "795254.pdf", "document 795254")
3. If it's a title, try synonyms and related terms
4. If it's a topic, try broader and narrower terms
5. If the query contains vague references (this, that, it, they), use the conversation context to resolve them

Generate different ways to search for the same information using:
- Synonyms and related terms
- Different word order
- Abbreviations and acronyms
- Broader and narrower concepts
- File extensions and document types
- Context from previous conversation (if provided)

Return only a JSON array of the {num_expansions} alternative queries:
["alternative query 1", "alternative query 2"]"""

            logger.info(f"Generating query expansions with model: {model_to_use}")
            if conversation_context:
                logger.info(f"Using conversation context for query expansion ({len(conversation_context)} chars)")
            
            # Use user's timezone for date/time context
            from utils.system_prompt_utils import add_datetime_context_to_system_prompt_for_user
            system_prompt = await add_datetime_context_to_system_prompt_for_user(
                "You generate alternative search queries. Return only a valid JSON array with no additional text or explanations.",
                user_id=user_id
            )
            
            # Make LLM call with timeout
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=model_to_use,
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
                logger.warning("LLM returned empty response for query expansion")
                return []
            
            expansions_text = response.choices[0].message.content.strip()
            
            if not expansions_text:
                logger.warning("LLM returned empty content for query expansion")
                return []
            
            # Parse JSON array format
            try:
                # Extract JSON if embedded in other text
                json_start = expansions_text.find('[')
                json_end = expansions_text.rfind(']') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_text = expansions_text[json_start:json_end]
                    raw_queries = json.loads(json_text)
                    
                    # Validate with Pydantic
                    validated = QueryExpansions(queries=raw_queries)
                    expansions = [q for q in validated.queries if q.lower() != query_text.lower()]
                else:
                    logger.warning(f"No JSON array found in response: '{expansions_text}'")
                    expansions = []
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from LLM: {e}")
                expansions = []
            except Exception as e:
                logger.warning(f"Failed to validate expansions: {e}")
                expansions = []
            
            # Filter out expansions that are too similar or empty
            filtered_expansions = []
            for expansion in expansions:
                if (expansion.lower() != query_text.lower() and 
                    len(expansion) > 3 and 
                    expansion not in filtered_expansions):
                    filtered_expansions.append(expansion)
            
            final_expansions = filtered_expansions[:5]  # Limit to 5 total
            
            # Cache the expansions (using cache_key that includes context if provided)
            if final_expansions:
                self._cache_expansions(cache_key, final_expansions)
            
            logger.info(f"✅ Generated {len(final_expansions)} query expansions")
            return final_expansions[:num_expansions]
            
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return []
    
    def _get_cached_expansions(self, query_text: str) -> Optional[List[str]]:
        """Get cached query expansions if available and not expired"""
        current_time = time.time()
        
        if query_text in self.query_expansion_cache:
            cache_time = self.expansion_cache_timestamps.get(query_text, 0)
            if current_time - cache_time < self.expansion_cache_ttl:
                logger.debug(f"Using cached query expansions for: '{query_text[:50]}...'")
                return self.query_expansion_cache[query_text]
            else:
                # Cache expired, remove it
                del self.query_expansion_cache[query_text]
                del self.expansion_cache_timestamps[query_text]
        
        return None
    
    def _cache_expansions(self, query_text: str, expansions: List[str]):
        """Cache query expansions with timestamp"""
        self.query_expansion_cache[query_text] = expansions
        self.expansion_cache_timestamps[query_text] = time.time()
        logger.debug(f"Cached {len(expansions)} query expansions for: '{query_text[:50]}...'")
    
    def clear_cache(self):
        """Clear the query expansion cache"""
        cache_size = len(self.query_expansion_cache)
        self.query_expansion_cache.clear()
        self.expansion_cache_timestamps.clear()
        logger.info(f"Cleared query expansion cache ({cache_size} entries)")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the query expansion cache"""
        current_time = time.time()
        active_entries = 0
        expired_entries = 0
        
        for query_text, cache_time in self.expansion_cache_timestamps.items():
            if current_time - cache_time < self.expansion_cache_ttl:
                active_entries += 1
            else:
                expired_entries += 1
        
        return {
            "total_entries": len(self.query_expansion_cache),
            "active_entries": active_entries,
            "expired_entries": expired_entries,
            "cache_ttl_seconds": self.expansion_cache_ttl
        }


# Singleton instance
_query_expansion_service: Optional[QueryExpansionService] = None


async def get_query_expansion_service() -> QueryExpansionService:
    """Get or create singleton query expansion service"""
    global _query_expansion_service
    
    if _query_expansion_service is None:
        _query_expansion_service = QueryExpansionService()
        await _query_expansion_service.initialize()
    
    return _query_expansion_service

