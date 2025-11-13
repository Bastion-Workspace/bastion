"""
Pending Query Manager - Roosevelt's Hybrid Clarification System
Manages ambiguous queries awaiting clarification with intelligent resolution
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

import redis.asyncio as redis
from config import settings

logger = logging.getLogger(__name__)


class QueryState(Enum):
    """States for pending queries"""
    PENDING = "pending"
    CLARIFIED = "clarified"
    RESOLVED = "resolved"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class PendingQuery:
    """A query waiting for clarification"""
    query_id: str
    original_query: str
    conversation_id: str
    user_id: str
    ambiguities: List[str]
    context_snapshot: Dict[str, Any]
    created_at: datetime
    expires_at: datetime
    state: QueryState = QueryState.PENDING
    clarification_attempts: int = 0
    partial_clarifications: List[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.partial_clarifications is None:
            self.partial_clarifications = []


@dataclass
class ResolvedQuery:
    """A successfully resolved query with full context"""
    original_query: str
    resolved_query: str
    clarification_data: Dict[str, Any]
    resolution_confidence: float
    context_enhancement: Dict[str, Any]


@dataclass
class CancellationResult:
    """Result when research planning is cancelled"""
    type: str = "research_cancelled"
    message: str = ""
    cancelled_queries: List[str] = None
    suggested_alternative: str = ""
    
    def __post_init__(self):
        if self.cancelled_queries is None:
            self.cancelled_queries = []


class PendingQueryManager:
    """
    Roosevelt's Hybrid Clarification System
    
    Manages the full lifecycle of ambiguous queries:
    1. Storage of pending queries awaiting clarification
    2. Intelligent resolution when clarification arrives
    3. Context enhancement for seamless conversation flow
    4. Memory storage for improved future interactions
    5. LLM-based cancellation detection for graceful de-escalation
    """
    
    def __init__(self, redis_client=None, mcp_chat_service=None):
        self.redis_client = redis_client
        self.mcp_chat_service = mcp_chat_service
        self.query_ttl = 1800  # 30 minutes for clarification timeout
        
    async def initialize(self):
        """Initialize the pending query manager"""
        if not self.redis_client:
            try:
                self.redis_client = redis.from_url(settings.REDIS_URL)
                await self.redis_client.ping()
                logger.info("✅ PendingQueryManager Redis connection established")
            except Exception as e:
                logger.error(f"❌ PendingQueryManager Redis connection failed: {e}")
                raise
    
    async def store_pending_query(
        self, 
        original_query: str,
        conversation_id: str,
        user_id: str,
        ambiguities: List[str],
        context_snapshot: Dict[str, Any]
    ) -> str:
        """
        Store a query that needs clarification
        Returns the query_id for tracking
        """
        try:
            query_id = f"pq_{conversation_id}_{int(time.time())}"
            
            pending_query = PendingQuery(
                query_id=query_id,
                original_query=original_query,
                conversation_id=conversation_id,
                user_id=user_id,
                ambiguities=ambiguities,
                context_snapshot=context_snapshot,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(seconds=self.query_ttl)
            )
            
            # Store in Redis with TTL
            redis_key = f"pending_query:{query_id}"
            query_data = self._serialize_pending_query(pending_query)
            
            await self.redis_client.setex(
                redis_key,
                self.query_ttl,
                json.dumps(query_data)
            )
            
            # Also store by conversation for easy lookup
            conv_key = f"pending_queries:{conversation_id}"
            await self.redis_client.sadd(conv_key, query_id)
            await self.redis_client.expire(conv_key, self.query_ttl)
            
            logger.info(f"✅ Stored pending query: {query_id} for '{original_query[:50]}...'")
            return query_id
            
        except Exception as e:
            logger.error(f"❌ Failed to store pending query: {e}")
            raise
    
    async def get_pending_query(self, query_id: str) -> Optional[PendingQuery]:
        """Retrieve a pending query by ID"""
        try:
            redis_key = f"pending_query:{query_id}"
            query_data = await self.redis_client.get(redis_key)
            
            if not query_data:
                return None
            
            data = json.loads(query_data)
            return self._deserialize_pending_query(data)
            
        except Exception as e:
            logger.error(f"❌ Failed to get pending query {query_id}: {e}")
            return None
    
    async def get_conversation_pending_queries(self, conversation_id: str) -> List[PendingQuery]:
        """Get all pending queries for a conversation"""
        try:
            conv_key = f"pending_queries:{conversation_id}"
            query_ids = await self.redis_client.smembers(conv_key)
            
            pending_queries = []
            for query_id in query_ids:
                query = await self.get_pending_query(query_id.decode())
                if query and query.state == QueryState.PENDING:
                    pending_queries.append(query)
            
            # Sort by creation time (most recent first)
            pending_queries.sort(key=lambda q: q.created_at, reverse=True)
            return pending_queries
            
        except Exception as e:
            logger.error(f"❌ Failed to get pending queries for conversation {conversation_id}: {e}")
            return []
    
    async def resolve_query_with_clarification(
        self,
        conversation_id: str,
        clarification_text: str,
        context_enhancement: Dict[str, Any] = None
    ) -> Optional[ResolvedQuery | CancellationResult]:
        """
        Roosevelt's Intelligent Query Resolution
        
        Attempts to resolve pending queries using the clarification.
        Uses sophisticated matching to find the most relevant pending query.
        """
        try:
            logger.info(f"🔍 Resolving queries for conversation {conversation_id} with: '{clarification_text[:50]}...'")
            
            # Get all pending queries for this conversation
            pending_queries = await self.get_conversation_pending_queries(conversation_id)
            
            if not pending_queries:
                logger.info("ℹ️ No pending queries found for resolution")
                return None
            
            # STEP 1: LLM-based cancellation detection
            cancellation_result = await self._assess_cancellation_intent(
                clarification_text, pending_queries, context_enhancement or {}
            )
            
            if cancellation_result:
                logger.info(f"🚫 Research cancellation detected: {cancellation_result.message}")
                await self._cancel_pending_queries(conversation_id)
                return cancellation_result
            
            # Find the best matching query using intelligent scoring
            best_match = await self._find_best_matching_query(
                pending_queries, clarification_text, context_enhancement or {}
            )
            
            if not best_match:
                logger.info("ℹ️ No suitable query match found for clarification")
                return None
            
            # Resolve the query
            resolved_query = await self._resolve_query_with_entities(
                best_match['query'], clarification_text, best_match['confidence']
            )
            
            # Mark query as resolved
            await self._update_query_state(best_match['query'].query_id, QueryState.RESOLVED)
            
            logger.info(f"✅ Resolved query: '{best_match['query'].original_query}' → '{resolved_query.resolved_query}'")
            return resolved_query
            
        except Exception as e:
            logger.error(f"❌ Failed to resolve query with clarification: {e}")
            return None
    
    async def _find_best_matching_query(
        self, 
        pending_queries: List[PendingQuery], 
        clarification_text: str,
        context_enhancement: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Find the best matching pending query using intelligent scoring
        """
        best_match = None
        best_score = 0.0
        
        for query in pending_queries:
            score = self._calculate_match_score(query, clarification_text, context_enhancement)
            
            if score > best_score and score > 0.3:  # Minimum confidence threshold
                best_score = score
                best_match = {
                    'query': query,
                    'confidence': score
                }
        
        return best_match
    
    def _calculate_match_score(
        self, 
        pending_query: PendingQuery, 
        clarification_text: str,
        context_enhancement: Dict[str, Any]
    ) -> float:
        """
        Calculate how well a clarification matches a pending query
        """
        score = 0.0
        
        # Time relevance (newer queries score higher)
        time_diff = (datetime.utcnow() - pending_query.created_at).total_seconds()
        time_score = max(0, 1.0 - (time_diff / 3600))  # Decay over 1 hour
        score += time_score * 0.3
        
        # Ambiguity relevance (check if clarification addresses ambiguities)
        clarification_lower = clarification_text.lower()
        ambiguity_matches = 0
        for ambiguity in pending_query.ambiguities:
            if any(keyword in clarification_lower for keyword in ambiguity.lower().split()):
                ambiguity_matches += 1
        
        if pending_query.ambiguities:
            ambiguity_score = ambiguity_matches / len(pending_query.ambiguities)
            score += ambiguity_score * 0.4
        
        # Context relevance (entities, topics)
        if context_enhancement:
            context_score = self._calculate_context_relevance(
                pending_query.context_snapshot, context_enhancement
            )
            score += context_score * 0.3
        
        return min(1.0, score)
    
    def _calculate_context_relevance(
        self, 
        stored_context: Dict[str, Any], 
        current_context: Dict[str, Any]
    ) -> float:
        """Calculate relevance between stored and current context"""
        relevance = 0.0
        
        # Entity overlap
        stored_entities = set(stored_context.get('recent_subjects', []))
        current_entities = set(current_context.get('entities', []))
        
        if stored_entities and current_entities:
            overlap = len(stored_entities.intersection(current_entities))
            entity_relevance = overlap / len(stored_entities.union(current_entities))
            relevance += entity_relevance * 0.5
        
        # Topic overlap
        stored_topics = set(stored_context.get('topics', []))
        current_topics = set(current_context.get('topics', []))
        
        if stored_topics and current_topics:
            overlap = len(stored_topics.intersection(current_topics))
            topic_relevance = overlap / len(stored_topics.union(current_topics))
            relevance += topic_relevance * 0.5
        
        return relevance
    
    async def _resolve_query_with_entities(
        self, 
        pending_query: PendingQuery, 
        clarification_text: str,
        confidence: float
    ) -> ResolvedQuery:
        """
        Resolve a pending query by substituting entities/pronouns with clarification
        """
        original_query = pending_query.original_query
        resolved_query = original_query
        
        # Simple pronoun/reference resolution
        pronouns = ['he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'hers', 'its', 'their']
        
        for pronoun in pronouns:
            # Replace pronouns with clarification (case-sensitive)
            resolved_query = resolved_query.replace(f' {pronoun} ', f' {clarification_text} ')
            resolved_query = resolved_query.replace(f' {pronoun.capitalize()} ', f' {clarification_text} ')
            
            # Handle start/end of sentence
            if resolved_query.startswith(f'{pronoun} '):
                resolved_query = f'{clarification_text} ' + resolved_query[len(pronoun)+1:]
            if resolved_query.startswith(f'{pronoun.capitalize()} '):
                resolved_query = f'{clarification_text} ' + resolved_query[len(pronoun)+1:]
        
        # Handle vague references
        vague_refs = ['the person', 'the company', 'the organization', 'that person', 'this person']
        for ref in vague_refs:
            resolved_query = resolved_query.replace(ref, clarification_text)
            resolved_query = resolved_query.replace(ref.capitalize(), clarification_text)
        
        return ResolvedQuery(
            original_query=original_query,
            resolved_query=resolved_query.strip(),
            clarification_data={'entity': clarification_text, 'method': 'pronoun_substitution'},
            resolution_confidence=confidence,
            context_enhancement=pending_query.context_snapshot
        )
    
    async def _update_query_state(self, query_id: str, new_state: QueryState):
        """Update the state of a pending query"""
        try:
            query = await self.get_pending_query(query_id)
            if query:
                query.state = new_state
                
                redis_key = f"pending_query:{query_id}"
                query_data = self._serialize_pending_query(query)
                
                await self.redis_client.setex(
                    redis_key,
                    self.query_ttl,
                    json.dumps(query_data)
                )
                
        except Exception as e:
            logger.error(f"❌ Failed to update query state: {e}")
    
    async def cleanup_expired_queries(self):
        """Clean up expired pending queries"""
        try:
            # Redis TTL handles most cleanup, but we can do manual cleanup here
            # This could be called periodically
            pass
        except Exception as e:
            logger.error(f"❌ Failed to cleanup expired queries: {e}")
    
    async def _assess_cancellation_intent(
        self, 
        clarification_text: str, 
        pending_queries: List[PendingQuery],
        context_enhancement: Dict[str, Any]
    ) -> Optional[CancellationResult]:
        """
        Roosevelt's LLM-based Cancellation Assessment
        
        Uses the LLM to intelligently detect when user wants to cancel research planning
        and gracefully return to chat mode.
        """
        try:
            if not self.mcp_chat_service or not self.mcp_chat_service.openrouter_client:
                logger.warning("⚠️ No LLM available for cancellation assessment")
                return None
            
            # Build context about pending queries
            pending_context = []
            for query in pending_queries[:3]:  # Limit to most recent 3
                pending_context.append({
                    "query": query.original_query,
                    "ambiguities": query.ambiguities,
                    "age_minutes": int((datetime.utcnow() - query.created_at).total_seconds() / 60)
                })
            
            # Create cancellation assessment prompt
            cancellation_prompt = f"""Analyze if this user response indicates they want to CANCEL their research planning and return to regular chat.

USER'S RESPONSE: "{clarification_text}"

PENDING RESEARCH QUERIES:
{json.dumps(pending_context, indent=2)}

CONTEXT: The user was asked for clarification about ambiguous research queries. Now they've responded with the above text.

CANCELLATION INDICATORS:
- Explicit abandonment: "never mind", "cancel", "forget it", "let's move on"
- Topic changes: "actually, I want to ask about..." (different topic)
- Mode changes: "let's just chat", "I don't need research"
- Frustration: "this is too complicated", "I changed my mind"
- Dismissive: "skip this", "not important", "doesn't matter"

CLARIFICATION INDICATORS:
- Providing entities: names of people, companies, concepts
- Providing context: dates, locations, specifications
- Asking for help: "what do you mean?", "can you clarify?"

Respond with valid JSON only:
{{
    "is_cancellation": boolean,
    "confidence": 0.0-1.0,
    "cancellation_type": "explicit|implicit|topic_change|mode_change",
    "reasoning": "brief explanation",
    "suggested_alternative": "suggested chat response or empty string"
}}"""

            # Get the classification model
            from services.settings_service import settings_service
            try:
                model = await settings_service.get_classification_model()
                if not model:
                    model = await settings_service.get_llm_model()
            except:
                model = await settings_service.get_llm_model()
            
            if not model:
                logger.warning("⚠️ No model available for cancellation assessment")
                return None
            
            response = await self.mcp_chat_service.openrouter_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert at understanding user intent and detecting when they want to cancel or abandon research planning."},
                    {"role": "user", "content": cancellation_prompt}
                ],
                max_tokens=200,
                temperature=0.1  # Low temperature for consistent analysis
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                result = json.loads(response_text)
                
                if result.get("is_cancellation", False) and result.get("confidence", 0) > 0.6:
                    # Generate appropriate cancellation message
                    cancellation_message = self._generate_cancellation_message(
                        result.get("cancellation_type", "explicit"),
                        result.get("suggested_alternative", ""),
                        pending_queries
                    )
                    
                    return CancellationResult(
                        type="research_cancelled",
                        message=cancellation_message,
                        cancelled_queries=[q.query_id for q in pending_queries],
                        suggested_alternative=result.get("suggested_alternative", "")
                    )
                
            except json.JSONDecodeError:
                logger.warning(f"⚠️ Invalid JSON response from cancellation assessment: {response_text}")
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to assess cancellation intent: {e}")
            return None
    
    def _generate_cancellation_message(
        self, 
        cancellation_type: str, 
        suggested_alternative: str,
        cancelled_queries: List[PendingQuery]
    ) -> str:
        """Generate appropriate cancellation message based on context"""
        
        if cancellation_type == "explicit":
            return "No problem! I've cancelled the research planning. How can I help you instead?"
        
        elif cancellation_type == "topic_change":
            return "I understand you'd like to move to a different topic. I've cancelled the previous research. What would you like to explore?"
        
        elif cancellation_type == "mode_change":
            return "Got it! Let's continue with regular conversation instead. What's on your mind?"
        
        elif cancellation_type == "implicit":
            return "I sense you'd prefer to move on from that research. No worries! What else can I help you with?"
        
        else:
            return "Research planning cancelled. Let's chat about something else - what interests you?"
    
    async def _cancel_pending_queries(self, conversation_id: str):
        """Cancel all pending queries for a conversation"""
        try:
            pending_queries = await self.get_conversation_pending_queries(conversation_id)
            
            for query in pending_queries:
                await self._update_query_state(query.query_id, QueryState.CANCELLED)
            
            # Clean up conversation pending list
            conv_key = f"pending_queries:{conversation_id}"
            await self.redis_client.delete(conv_key)
            
            logger.info(f"✅ Cancelled {len(pending_queries)} pending queries for conversation {conversation_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to cancel pending queries: {e}")
    
    def _serialize_pending_query(self, query: PendingQuery) -> Dict[str, Any]:
        """Serialize PendingQuery to dict for Redis storage"""
        data = asdict(query)
        data['created_at'] = query.created_at.isoformat()
        data['expires_at'] = query.expires_at.isoformat()
        data['state'] = query.state.value
        return data
    
    def _deserialize_pending_query(self, data: Dict[str, Any]) -> PendingQuery:
        """Deserialize dict to PendingQuery from Redis storage"""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        data['state'] = QueryState(data['state'])
        return PendingQuery(**data)
