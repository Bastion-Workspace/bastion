"""
Conversation Intelligence Service - Roosevelt's "Memory Palace" 
Manages conversation context, caching, and intelligent analysis using LangGraph native patterns
"""

import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from pydantic import ValidationError

from services.langgraph_enhanced_state import (
    ConversationIntelligence, CachedResult, CachedResultType
)

logger = logging.getLogger(__name__)


class ConversationIntelligenceService:
    """
    Roosevelt's Conversation Intelligence Service
    
    Manages conversation context and caching using LangGraph native state patterns.
    No tools needed - operates directly on ConversationState for maximum performance.
    """
    
    def __init__(self):
        self._embedding_manager = None
        
    async def _get_embedding_manager(self):
        """Lazy load embedding service wrapper for semantic analysis"""
        if self._embedding_manager is None:
            from services.embedding_service_wrapper import get_embedding_service
            self._embedding_manager = await get_embedding_service()
        return self._embedding_manager
    
    async def analyze_and_update_intelligence(
        self, 
        state: Dict[str, Any], 
        agent_type: str, 
        agent_output: str,
        agent_results: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        ROOSEVELT'S INTELLIGENCE ANALYSIS: Update conversation intelligence with new agent output
        
        This is called by the orchestrator after each agent execution to maintain
        conversation context without requiring tool calls.
        """
        try:
            logger.info(f"🧠 INTELLIGENCE UPDATE: Processing {agent_type} output ({len(agent_output)} chars)")
            
            # Get or create conversation intelligence
            conversation_intel = state.get("conversation_intelligence", {})
            if not isinstance(conversation_intel, dict):
                conversation_intel = ConversationIntelligence().dict()
            
            # ROOSEVELT'S DEFENSIVE PROGRAMMING: Ensure required fields exist
            if "cached_results" not in conversation_intel:
                conversation_intel["cached_results"] = {}
            if "agent_outputs" not in conversation_intel:
                conversation_intel["agent_outputs"] = {}
            if "collaboration_suggestions" not in conversation_intel:
                conversation_intel["collaboration_suggestions"] = []
            if "query_coverage_cache" not in conversation_intel:
                conversation_intel["query_coverage_cache"] = {}
            
            # Add this agent's output to the cache
            content_hash = self._generate_content_hash(agent_output)
            
            cached_result = CachedResult(
                content=agent_output,
                result_type=self._determine_result_type(agent_type, agent_output),
                source_agent=agent_type,
                timestamp=datetime.now().isoformat(),
                confidence_score=agent_results.get("confidence", 0.8) if agent_results else 0.8,
                topics=await self._extract_topics(agent_output),
                citations=self._extract_citations(agent_output, agent_results)
            )
            
            conversation_intel["cached_results"][content_hash] = cached_result.dict()
            
            # Update agent outputs by type
            agent_outputs = conversation_intel.get("agent_outputs", {})
            agent_outputs.setdefault(agent_type, []).append(content_hash)
            # Keep only last 5 outputs per agent
            agent_outputs[agent_type] = agent_outputs[agent_type][-5:]
            conversation_intel["agent_outputs"] = agent_outputs
            
            # Update topic tracking
            await self._update_topic_tracking(conversation_intel, agent_output, agent_type)
            
            # Update collaboration suggestions if any
            if agent_results and agent_results.get("collaboration_suggestion"):
                collaboration_suggestions = conversation_intel.get("collaboration_suggestions", [])
                collaboration_suggestions.append({
                    "agent": agent_type,
                    "suggestion": agent_results["collaboration_suggestion"],
                    "confidence": agent_results.get("collaboration_confidence", 0.5),
                    "timestamp": datetime.now().isoformat()
                })
                # Keep only last 3 suggestions
                conversation_intel["collaboration_suggestions"] = collaboration_suggestions[-3:]
            
            # Update research cache if this was research
            if agent_type == "research_agent":
                await self._update_research_cache(conversation_intel, agent_output, agent_results)
            
            # Pre-compute coverage analysis for common follow-up patterns
            await self._precompute_coverage_analysis(conversation_intel)
            
            # Update metadata
            conversation_intel["last_updated"] = datetime.now().isoformat()
            
            # Store back in state
            state["conversation_intelligence"] = conversation_intel
            
            logger.info(f"✅ INTELLIGENCE UPDATED: {len(conversation_intel['cached_results'])} cached results, topic: {conversation_intel.get('current_topic', 'unknown')}")
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Intelligence analysis failed: {e}")
            # Don't fail the conversation - just skip intelligence update
            return state
    
    async def get_relevant_context(
        self, 
        query: str, 
        conversation_intelligence: Dict[str, Any],
        agent_type: str = None
    ) -> Dict[str, Any]:
        """
        ROOSEVELT'S CONTEXT EXTRACTION: Get relevant context for current query
        
        This replaces the cache tool - agents get context automatically from state.
        """
        try:
            logger.info(f"🧠 CONTEXT EXTRACTION: Analyzing relevance for '{query[:50]}...'")
            
            # Semantic analysis for better coverage assessment
            coverage_analysis = await self._analyze_query_coverage(query, conversation_intelligence)
            
            if coverage_analysis["coverage_score"] > 0.7:
                logger.info(f"🏆 HIGH COVERAGE: {coverage_analysis['coverage_score']:.2f} - recommend using cached context")
                return {
                    "use_cache": True,
                    "coverage_score": coverage_analysis["coverage_score"],
                    "relevant_results": coverage_analysis["relevant_results"],
                    "recommendation": "use_cached_context"
                }
            elif coverage_analysis["coverage_score"] > 0.3:
                logger.info(f"🔄 PARTIAL COVERAGE: {coverage_analysis['coverage_score']:.2f} - supplement with targeted search")
                return {
                    "use_cache": True,
                    "coverage_score": coverage_analysis["coverage_score"],
                    "relevant_results": coverage_analysis["relevant_results"],
                    "recommendation": "supplement_cache"
                }
            else:
                logger.info(f"❌ LOW COVERAGE: {coverage_analysis['coverage_score']:.2f} - recommend fresh search")
                return {
                    "use_cache": False,
                    "coverage_score": coverage_analysis["coverage_score"],
                    "recommendation": "fresh_search"
                }
                
        except Exception as e:
            logger.error(f"❌ Context extraction failed: {e}")
            return {"use_cache": False, "coverage_score": 0.0, "recommendation": "fresh_search"}
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate hash for content indexing"""
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _determine_result_type(self, agent_type: str, content: str) -> CachedResultType:
        """Determine the type of cached result based on agent and content"""
        if agent_type == "research_agent":
            return CachedResultType.RESEARCH_FINDINGS
        elif agent_type == "chat_agent":
            return CachedResultType.CHAT_OUTPUT
        elif "collaboration" in content.lower():
            return CachedResultType.AGENT_COLLABORATION
        else:
            return CachedResultType.CHAT_OUTPUT
    
    async def _extract_topics(self, content: str) -> List[str]:
        """Extract key topics from content using enhanced keyword analysis for trip planning"""
        try:
            # Enhanced keyword extraction for trip planning context
            words = content.lower().split()
            
            # Look for capitalized words and common topic indicators
            topics = []
            topic_indicators = [
                # Trip planning
                "trip", "travel", "vacation", "journey", "visit", "explore", "day_trip", "weekend_trip",
                "planning", "itinerary", "schedule", "route", "destination", "origin", "departure",
                "arrival", "timing", "date", "time", "morning", "afternoon", "evening",
                
                # Weather and conditions
                "weather", "forecast", "temperature", "climate", "conditions", "precipitation",
                
                # Activities and attractions
                "activities", "attractions", "sights", "restaurants", "shopping", "entertainment",
                "outdoor", "indoor", "adventure", "relaxation", "culture", "history",
                
                # Research and analysis
                "research", "study", "analysis", "investigation", "information", "details",
                "history", "timeline", "development", "origin", "evolution",
                
                # Technical topics
                "code", "programming", "development", "software", "technical",
                
                # Conversation topics
                "chat", "conversation", "discussion", "talk", "question", "answer", "help", "assist"
            ]
            
            for word in words:
                if word in topic_indicators:
                    topics.append(word)
            
            # Extract potential proper nouns (simplified)
            sentences = content.split('.')
            for sentence in sentences[:5]:  # Check first 5 sentences
                words = sentence.split()
                for word in words:
                    if word.istitle() and len(word) > 2:
                        topics.append(word.lower())
            
            # Extract specific trip planning context
            if any(word in content.lower() for word in ["trip", "travel", "visit", "going", "planning"]):
                # Look for origin-destination patterns
                import re
                origin_patterns = [
                    r"i'm in (\w+)",
                    r"from (\w+)",
                    r"starting from (\w+)",
                    r"departing from (\w+)"
                ]
                destination_patterns = [
                    r"going to (\w+)",
                    r"visiting (\w+)",
                    r"destination (\w+)",
                    r"headed to (\w+)"
                ]
                
                for pattern in origin_patterns:
                    matches = re.findall(pattern, content.lower())
                    for match in matches:
                        topics.append(f"origin_{match}")
                
                for pattern in destination_patterns:
                    matches = re.findall(pattern, content.lower())
                    for match in matches:
                        topics.append(f"destination_{match}")
            
            return list(set(topics))[:10]  # Return unique topics, max 10
            
        except Exception as e:
            logger.error(f"❌ Topic extraction failed: {e}")
            return []
    
    def _extract_citations(self, content: str, agent_results: Dict[str, Any] = None) -> List[str]:
        """Extract citations from content and agent results"""
        try:
            citations = []
            
            # Extract URLs from content
            import re
            url_pattern = r'https?://[^\s\)]+' 
            urls = re.findall(url_pattern, content)
            citations.extend(urls[:5])  # Max 5 URLs
            
            # Extract from agent results if available
            if agent_results and "citations" in agent_results:
                agent_citations = agent_results["citations"]
                if isinstance(agent_citations, list):
                    for citation in agent_citations[:5]:
                        if isinstance(citation, dict) and "url" in citation:
                            citations.append(citation["url"])
                        elif isinstance(citation, str):
                            citations.append(citation)
            
            return list(set(citations))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"❌ Citation extraction failed: {e}")
            return []
    
    async def _update_topic_tracking(
        self, 
        conversation_intel: Dict[str, Any], 
        content: str, 
        agent_type: str
    ):
        """Update topic continuity tracking"""
        try:
            # Simple topic detection for now
            # TODO: Enhance with semantic topic modeling
            
            topics = await self._extract_topics(content)
            if topics:
                current_topic = topics[0]  # Use first topic as primary
                
                # Check if topic changed
                previous_topic = conversation_intel.get("current_topic")
                if previous_topic and previous_topic != current_topic:
                    # Record topic transition
                    topic_transitions = conversation_intel.get("topic_transitions", [])
                    topic_transitions.append({
                        "from_topic": previous_topic,
                        "to_topic": current_topic,
                        "agent": agent_type,
                        "timestamp": datetime.now().isoformat()
                    })
                    conversation_intel["topic_transitions"] = topic_transitions[-5:]  # Keep last 5
                
                # Update current topic and history
                conversation_intel["current_topic"] = current_topic
                topic_history = conversation_intel.get("topic_history", [])
                if current_topic not in topic_history:
                    topic_history.append(current_topic)
                conversation_intel["topic_history"] = topic_history[-10:]  # Keep last 10
                
        except Exception as e:
            logger.error(f"❌ Topic tracking update failed: {e}")
    
    async def _update_research_cache(
        self, 
        conversation_intel: Dict[str, Any], 
        research_output: str,
        agent_results: Dict[str, Any] = None
    ):
        """Update research cache with new findings"""
        try:
            research_cache = conversation_intel.get("research_cache", {})
            
            # Use query as cache key
            query = agent_results.get("query", "research") if agent_results else "research"
            query_key = query[:50]  # First 50 chars as key
            
            research_cache[query_key] = {
                "findings": research_output,
                "timestamp": datetime.now().isoformat(),
                "confidence": agent_results.get("confidence", 0.8) if agent_results else 0.8,
                "sources": agent_results.get("citations", []) if agent_results else [],
                "tools_used": agent_results.get("tools_used", []) if agent_results else []
            }
            
            conversation_intel["research_cache"] = research_cache
            
            # Also update source cache if we have URLs
            if agent_results and "citations" in agent_results:
                await self._update_source_cache(conversation_intel, agent_results["citations"])
                
        except Exception as e:
            logger.error(f"❌ Research cache update failed: {e}")
    
    async def _update_source_cache(self, conversation_intel: Dict[str, Any], citations: List):
        """Update source cache with citation content"""
        try:
            source_cache = conversation_intel.get("source_cache", {})
            
            for citation in citations[:10]:  # Max 10 sources
                if isinstance(citation, dict) and "url" in citation:
                    url = citation["url"]
                    source_cache[url] = {
                        "title": citation.get("title", ""),
                        "last_accessed": datetime.now().isoformat(),
                        "access_count": source_cache.get(url, {}).get("access_count", 0) + 1,
                        "content_summary": citation.get("excerpt", "")[:500]
                    }
            
            conversation_intel["source_cache"] = source_cache
            
        except Exception as e:
            logger.error(f"❌ Source cache update failed: {e}")
    
    async def _analyze_query_coverage(
        self, 
        query: str, 
        conversation_intelligence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze how well cached results cover the new query"""
        try:
            # Get cached results
            cached_results = conversation_intelligence.get("cached_results", {})
            
            if not cached_results:
                return {"coverage_score": 0.0, "relevant_results": []}
            
            # Simple keyword-based analysis for now
            # TODO: Replace with semantic similarity using embeddings
            query_words = set(query.lower().split())
            
            relevant_results = []
            max_coverage = 0.0
            
            for result_hash, result_data in cached_results.items():
                content = result_data.get("content", "")
                content_words = set(content.lower().split())
                
                # Calculate word overlap
                word_overlap = len(query_words.intersection(content_words)) / len(query_words) if query_words else 0.0
                
                if word_overlap > 0.2:  # Minimum relevance threshold
                    relevant_results.append({
                        "hash": result_hash,
                        "data": result_data,
                        "relevance_score": word_overlap
                    })
                    max_coverage = max(max_coverage, word_overlap)
            
            # Sort by relevance
            relevant_results.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            return {
                "coverage_score": min(max_coverage, 1.0),
                "relevant_results": relevant_results[:5],  # Top 5 most relevant
                "total_cached_results": len(cached_results),
                "analysis_method": "keyword_overlap"  # TODO: upgrade to semantic
            }
            
        except Exception as e:
            logger.error(f"❌ Query coverage analysis failed: {e}")
            return {"coverage_score": 0.0, "relevant_results": []}
    
    async def _precompute_coverage_analysis(self, conversation_intel: Dict[str, Any]):
        """Pre-compute coverage for common follow-up patterns"""
        try:
            # Common follow-up patterns to pre-analyze
            common_patterns = [
                "timeline", "chronological", "history", "development",
                "table", "format", "organize", "structure",
                "more details", "expand", "elaborate", "explain further",
                "examples", "model answers", "questions", "brainstorm"
            ]
            
            coverage_cache = conversation_intel.get("query_coverage_cache", {})
            cached_results = conversation_intel.get("cached_results", {})
            
            for pattern in common_patterns:
                if pattern not in coverage_cache:
                    analysis = await self._analyze_query_coverage(pattern, conversation_intel)
                    coverage_cache[pattern] = analysis["coverage_score"]
            
            conversation_intel["query_coverage_cache"] = coverage_cache
            
        except Exception as e:
            logger.error(f"❌ Pre-compute coverage analysis failed: {e}")
    
    def get_collaboration_context(self, conversation_intelligence: Dict[str, Any]) -> Dict[str, Any]:
        """Get recent collaboration suggestions for intent classification"""
        try:
            collaboration_suggestions = conversation_intelligence.get("collaboration_suggestions", [])
            
            # Check for recent suggestions (last 5 minutes)
            recent_suggestions = []
            current_time = datetime.now()
            
            for suggestion in collaboration_suggestions:
                suggestion_time = datetime.fromisoformat(suggestion["timestamp"])
                if (current_time - suggestion_time).total_seconds() < 300:  # 5 minutes
                    recent_suggestions.append(suggestion)
            
            return {
                "has_recent_suggestions": len(recent_suggestions) > 0,
                "suggestions": recent_suggestions,
                "most_recent": recent_suggestions[-1] if recent_suggestions else None
            }
            
        except Exception as e:
            logger.error(f"❌ Collaboration context extraction failed: {e}")
            return {"has_recent_suggestions": False, "suggestions": []}
    
    def extract_cached_content_for_agent(
        self, 
        query: str,
        conversation_intelligence: Dict[str, Any], 
        agent_type: str
    ) -> Optional[str]:
        """
        Extract cached content relevant to agent's current query
        
        This replaces tool calls - agents get context directly from state
        """
        try:
            # Get relevant context analysis
            coverage_analysis = self._analyze_query_coverage_sync(query, conversation_intelligence)
            
            if coverage_analysis["coverage_score"] > 0.7:
                # High coverage - format cached results for agent use
                relevant_results = coverage_analysis["relevant_results"]
                
                formatted_context = f"🏆 **CONVERSATION CONTEXT** (Coverage: {coverage_analysis['coverage_score']:.1%})\n\n"
                
                for i, result in enumerate(relevant_results[:3], 1):
                    result_data = result["data"]
                    agent = result_data.get("source_agent", "unknown")
                    content = result_data.get("content", "")
                    
                    formatted_context += f"**{i}. Previous {agent} work:**\n"
                    formatted_context += f"{content[:400]}...\n\n"
                
                formatted_context += "✅ **Use this context to build upon previous work instead of starting fresh.**"
                
                return formatted_context
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Cache content extraction failed: {e}")
            return None
    
    def _analyze_query_coverage_sync(self, query: str, conversation_intelligence: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous version of coverage analysis for immediate use"""
        try:
            # Check pre-computed cache first
            query_coverage_cache = conversation_intelligence.get("query_coverage_cache", {})
            for pattern, score in query_coverage_cache.items():
                if pattern.lower() in query.lower():
                    logger.info(f"🏆 PRE-COMPUTED COVERAGE: {pattern} → {score:.2f}")
                    # Get relevant results for this pattern
                    cached_results = conversation_intelligence.get("cached_results", {})
                    relevant_results = []
                    
                    for result_hash, result_data in cached_results.items():
                        content = result_data.get("content", "").lower()
                        if pattern.lower() in content:
                            relevant_results.append({
                                "hash": result_hash,
                                "data": result_data,
                                "relevance_score": score
                            })
                    
                    return {
                        "coverage_score": score,
                        "relevant_results": relevant_results[:5],
                        "analysis_method": "pre_computed"
                    }
            
            # Fallback to keyword analysis
            cached_results = conversation_intelligence.get("cached_results", {})
            query_words = set(query.lower().split())
            
            relevant_results = []
            max_coverage = 0.0
            
            for result_hash, result_data in cached_results.items():
                content = result_data.get("content", "")
                content_words = set(content.lower().split())
                
                word_overlap = len(query_words.intersection(content_words)) / len(query_words) if query_words else 0.0
                
                if word_overlap > 0.2:
                    relevant_results.append({
                        "hash": result_hash,
                        "data": result_data,
                        "relevance_score": word_overlap
                    })
                    max_coverage = max(max_coverage, word_overlap)
            
            relevant_results.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            return {
                "coverage_score": min(max_coverage, 1.0),
                "relevant_results": relevant_results[:5],
                "analysis_method": "keyword_overlap"
            }
            
        except Exception as e:
            logger.error(f"❌ Sync coverage analysis failed: {e}")
            return {"coverage_score": 0.0, "relevant_results": []}


# Global service instance
_conversation_intelligence_service: Optional[ConversationIntelligenceService] = None


async def get_conversation_intelligence_service() -> ConversationIntelligenceService:
    """Get global conversation intelligence service instance"""
    global _conversation_intelligence_service
    if _conversation_intelligence_service is None:
        _conversation_intelligence_service = ConversationIntelligenceService()
    return _conversation_intelligence_service
