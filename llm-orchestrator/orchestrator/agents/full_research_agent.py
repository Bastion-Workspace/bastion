"""
Full Research Agent - Complete replication of backend clean_research_agent
Multi-round research with gap analysis, caching, and sophisticated synthesis
"""

import logging
import json
import re
from typing import Dict, Any, List, TypedDict, Optional
from datetime import datetime
from enum import Enum

from pydantic import ValidationError
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from orchestrator.tools import (
    search_documents_tool,
    get_document_content_tool,
    search_web_tool,
    search_and_crawl_tool,
    expand_query_tool,
    search_conversation_cache_tool
)
from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.formatting_detection import detect_formatting_need
from orchestrator.models import ResearchAssessmentResult, ResearchGapAnalysis, QuickAnswerAssessment, QueryTypeDetection
from orchestrator.agents.base_agent import BaseAgent
from config.settings import settings

logger = logging.getLogger(__name__)


class ResearchRound(str, Enum):
    """Research round tracking"""
    QUICK_ANSWER_CHECK = "quick_answer_check"
    CACHE_CHECK = "cache_check"
    INITIAL_LOCAL = "initial_local"
    ROUND_2_GAP_FILLING = "round_2_gap_filling"
    WEB_ROUND_1 = "web_round_1"
    ASSESS_WEB_ROUND_1 = "assess_web_round_1"
    GAP_ANALYSIS_WEB = "gap_analysis_web"
    WEB_ROUND_2 = "web_round_2"
    FINAL_SYNTHESIS = "final_synthesis"


class ResearchState(TypedDict):
    """Complete state for sophisticated research workflow"""
    # Query info
    query: str
    original_query: str
    expanded_queries: List[str]
    key_entities: List[str]
    
    # Messages
    messages: List[Any]
    
    # Shared memory for cross-agent communication
    shared_memory: Dict[str, Any]
    
    # Quick answer tracking
    quick_answer_provided: bool
    quick_answer_content: str
    skip_quick_answer: bool
    
    # Research rounds
    current_round: str
    cache_hit: bool
    cached_context: str
    
    # Round 1 - Initial local search
    round1_results: Dict[str, Any]
    round1_sufficient: bool
    
    # Gap analysis
    gap_analysis: Dict[str, Any]
    identified_gaps: List[str]
    
    # Round 2 - Gap filling
    round2_results: Dict[str, Any]
    round2_sufficient: bool
    
    # Web search Round 1
    web_round1_results: Dict[str, Any]
    web_round1_sufficient: bool
    web_permission_granted: bool  # For future HITL
    
    # Web gap analysis
    web_gap_analysis: Dict[str, Any]
    web_identified_gaps: List[str]
    
    # Web search Round 2
    web_round2_results: Dict[str, Any]
    
    # Legacy field for backward compatibility
    web_search_results: Dict[str, Any]
    
    # Query type detection
    query_type: Optional[str]  # "objective", "subjective", or "mixed"
    query_type_detection: Dict[str, Any]  # Full detection result
    should_present_options: bool
    num_options: Optional[int]
    
    # Final synthesis
    final_response: str
    citations: List[Dict[str, Any]]
    sources_used: List[str]
    routing_recommendation: Optional[str]
    
    # Control
    research_complete: bool
    error: str


class FullResearchAgent(BaseAgent):
    """
    Sophisticated research agent replicating clean_research_agent capabilities
    
    Workflow:
    1. Cache check - Look for previous research
    2. Query expansion - Generate variations
    3. Round 1 - Initial local search (documents + entities)
    4. Quality assessment - Evaluate sufficiency
    5. Gap analysis - Identify missing information
    6. Round 2 - Targeted gap filling
    7. Web search - If local insufficient (no permission needed now)
    8. Final synthesis - Comprehensive answer with citations
    """
    
    def __init__(self):
        super().__init__("full_research_agent")
        # LLMs will be created lazily using _get_llm() to respect user model preferences
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """Build sophisticated multi-round research workflow"""
        
        workflow = StateGraph(ResearchState)
        
        # Add quick answer check node (first node - entry point)
        workflow.add_node("quick_answer_check", self._quick_answer_check_node)
        
        # Add nodes for each stage
        workflow.add_node("cache_check", self._cache_check_node)
        workflow.add_node("query_expansion", self._query_expansion_node)
        workflow.add_node("round1_parallel_search", self._round1_parallel_search_node)
        workflow.add_node("assess_combined_round1", self._assess_combined_round1_node)
        workflow.add_node("gap_analysis", self._gap_analysis_node)
        workflow.add_node("round2_gap_filling", self._round2_gap_filling_node)
        workflow.add_node("web_round1", self._web_round1_node)
        workflow.add_node("assess_web_round1", self._assess_web_round1_node)
        workflow.add_node("gap_analysis_web", self._gap_analysis_web_node)
        workflow.add_node("web_round2", self._web_round2_node)
        workflow.add_node("detect_query_type", self._detect_query_type_node)
        workflow.add_node("final_synthesis", self._final_synthesis_node)
        
        # Entry point - quick answer check first
        workflow.set_entry_point("quick_answer_check")
        
        # Quick answer check routing
        workflow.add_conditional_edges(
            "quick_answer_check",
            self._route_from_quick_answer,
            {
                "quick_answer": END,  # Short-circuit with quick answer
                "full_research": "cache_check"  # Continue normal research
            }
        )
        
        # Cache check routing
        workflow.add_conditional_edges(
            "cache_check",
            self._route_from_cache,
            {
                "use_cache": "detect_query_type",
                "do_research": "query_expansion"
            }
        )
        
        # Query expansion always goes to parallel round 1 search
        workflow.add_edge("query_expansion", "round1_parallel_search")
        
        # Round 1 parallel search goes to combined assessment
        workflow.add_edge("round1_parallel_search", "assess_combined_round1")
        
        # Combined Round 1 assessment routing (now has both local and web data)
        workflow.add_conditional_edges(
            "assess_combined_round1",
            self._route_from_combined_round1,
            {
                "sufficient": "detect_query_type",
                "needs_gap_filling": "gap_analysis",
                "needs_web_round2": "web_round2"
            }
        )
        
        # Gap analysis routes to Round 2 Local or Web Round 1
        workflow.add_conditional_edges(
            "gap_analysis",
            self._route_from_gap_analysis,
            {
                "round2_local": "round2_gap_filling",
                "needs_web": "web_round1"
            }
        )
        
        # Round 2 Local routes to either Web Round 1 or query type detection
        workflow.add_conditional_edges(
            "round2_gap_filling",
            self._route_from_round2,
            {
                "sufficient": "detect_query_type",
                "needs_web": "web_round1"
            }
        )
        
        # Web Round 1 goes to assessment
        workflow.add_edge("web_round1", "assess_web_round1")
        
        # Web Round 1 assessment routes to query type detection or gap analysis
        workflow.add_conditional_edges(
            "assess_web_round1",
            self._route_from_web_round1,
            {
                "sufficient": "detect_query_type",
                "needs_web_gap_analysis": "gap_analysis_web"
            }
        )
        
        # Web gap analysis routes to Round 2 Web or query type detection
        workflow.add_conditional_edges(
            "gap_analysis_web",
            self._route_from_web_gap_analysis,
            {
                "web_round2": "web_round2",
                "sufficient": "detect_query_type"
            }
        )
        
        # Web Round 2 goes to query type detection, then synthesis
        workflow.add_edge("web_round2", "detect_query_type")
        
        # Query type detection always goes to synthesis
        workflow.add_edge("detect_query_type", "final_synthesis")
        
        # Add data formatting node (if formatting is needed)
        workflow.add_node("format_data", self._format_data_node)
        
        # After synthesis, check if formatting is needed
        workflow.add_conditional_edges(
            "final_synthesis",
            self._route_from_synthesis,
            {
                "format": "format_data",
                "complete": END
            }
        )
        
        # Formatting node goes to end
        workflow.add_edge("format_data", END)
        
        return workflow.compile(checkpointer=checkpointer)
    
    # ===== Routing Functions =====
    
    def _route_from_cache(self, state: ResearchState) -> str:
        """Route based on cache hit"""
        if state.get("cache_hit") and state.get("cached_context"):
            logger.info("Cache hit - using cached research")
            return "use_cache"
        logger.info("Cache miss - proceeding with research")
        return "do_research"
    
    def _route_from_combined_round1(self, state: ResearchState) -> str:
        """Route based on combined Round 1 assessment (local + web)"""
        if state.get("round1_sufficient"):
            logger.info("Combined Round 1 sufficient - proceeding to synthesis")
            return "sufficient"
        
        round1_assessment = state.get("round1_assessment", {})
        needs_more_local = round1_assessment.get("needs_more_local", False)
        needs_more_web = round1_assessment.get("needs_more_web", False)
        best_source = round1_assessment.get("best_source", "both")
        
        # If we need more web data, go to web round 2
        if needs_more_web:
            logger.info("Combined Round 1: Needs more web data - routing to Web Round 2")
            return "needs_web_round2"
        
        # If we need more local data, do gap analysis for Round 2 local
        if needs_more_local:
            logger.info("Combined Round 1: Needs more local data - routing to gap analysis")
            return "needs_gap_filling"
        
        # Default: do gap analysis to determine next steps
        logger.info("Combined Round 1 insufficient - doing gap analysis")
        return "needs_gap_filling"
    
    def _route_from_round1(self, state: ResearchState) -> str:
        """Route based on Round 1 sufficiency (legacy - redirects to combined routing)"""
        return self._route_from_combined_round1(state)
    
    def _route_from_gap_analysis(self, state: ResearchState) -> str:
        """Route from gap analysis based on severity and web search needs"""
        gap_analysis = state.get("gap_analysis", {})
        severity = gap_analysis.get("gap_severity", "moderate")
        needs_web_search = gap_analysis.get("needs_web_search", False)
        
        # Severe gaps + needs web search → skip Round 2 Local, go to Web Round 1
        if severity == "severe" and needs_web_search:
            logger.info("Gap analysis: Severe gaps + web search needed - skipping Round 2 Local, going to Web Round 1")
            return "needs_web"
        
        # Minor/Moderate gaps → try Round 2 Local first
        if severity in ["minor", "moderate"]:
            logger.info(f"Gap analysis: {severity} gaps - trying Round 2 Local first")
            return "round2_local"
        
        # Default: try Round 2 Local
        logger.info("Gap analysis: Default - trying Round 2 Local")
        return "round2_local"
    
    def _route_from_round2(self, state: ResearchState) -> str:
        """Route based on Round 2 Local sufficiency"""
        round2_sufficient = state.get("round2_sufficient", False)
        
        if round2_sufficient:
            logger.info("Round 2 Local sufficient - proceeding to synthesis")
            return "sufficient"
        
        logger.info("Round 2 Local insufficient - proceeding to Web Round 1")
        return "needs_web"
    
    def _route_from_web_round1(self, state: ResearchState) -> str:
        """Route based on Web Round 1 sufficiency"""
        if state.get("web_round1_sufficient"):
            logger.info("Web Round 1 sufficient - proceeding to synthesis")
            return "sufficient"
        
        logger.info("Web Round 1 insufficient - doing web gap analysis for Round 2 Web")
        return "needs_web_gap_analysis"
    
    def _route_from_web_gap_analysis(self, state: ResearchState) -> str:
        """Route from web gap analysis - decide if Round 2 Web is needed"""
        web_gap_analysis = state.get("web_gap_analysis", {})
        needs_web_round2 = web_gap_analysis.get("needs_web_round2", False)
        
        if needs_web_round2:
            logger.info("Web gap analysis: Round 2 Web needed")
            return "web_round2"
        
        logger.info("Web gap analysis: Round 2 Web not needed - proceeding to synthesis")
        return "sufficient"
    
    # ===== Workflow Nodes =====
    
    async def _quick_answer_check_node(self, state: ResearchState) -> Dict[str, Any]:
        """Check if query can be answered quickly from general knowledge without searching"""
        try:
            query = state["query"]
            skip_quick_answer = state.get("skip_quick_answer", False)
            
            # If we're skipping quick answer (follow-up request), proceed to full research
            if skip_quick_answer:
                logger.info("Skipping quick answer check - proceeding to full research")
                return {
                    "quick_answer_provided": False,
                    "quick_answer_content": "",
                    "current_round": ResearchRound.QUICK_ANSWER_CHECK.value
                }
            
            logger.info(f"Quick answer check for: {query}")
            
            # Use LLM to evaluate if query can be answered from general knowledge
            evaluation_prompt = f"""Evaluate whether this query can be answered accurately from general knowledge without searching documents or the web.

USER QUERY: {query}

Consider:
1. Is this a well-known, factual query? (e.g., "What is the best water temperature for tea?")
2. Can it be answered accurately from general knowledge?
3. Does it require specific, current, or user-specific information that would need searching?
4. Is the answer likely to be stable and well-established? (not time-sensitive or controversial)

Examples of queries that CAN be answered quickly:
- "What is the best water temperature for tea?" (well-known fact)
- "What is the capital of France?" (common knowledge)
- "How many days are in a year?" (established fact)

Examples of queries that CANNOT be answered quickly:
- "What was the relationship between Dan Reingold and Bernie Ebbers?" (requires specific research)
- "What are the latest developments in AI?" (time-sensitive, needs current sources)
- "What documents do I have about project X?" (user-specific, needs document search)

STRUCTURED OUTPUT REQUIRED - Respond with ONLY valid JSON matching this exact schema:
{{
    "can_answer_quickly": boolean,
    "confidence": number (0.0-1.0),
    "quick_answer": "string (provide the answer if can_answer_quickly=true, otherwise null)",
    "reasoning": "brief explanation of why this can or cannot be answered quickly"
}}

Example for "What is the best water temperature for tea?":
{{
    "can_answer_quickly": true,
    "confidence": 0.95,
    "quick_answer": "The best water temperature for tea depends on the type of tea. Generally: white and green teas (175-185°F / 80-85°C), oolong teas (185-205°F / 85-96°C), black teas (200-212°F / 93-100°C), and herbal teas (212°F / 100°C). Water that's too hot can make tea bitter, while water that's too cool won't extract the full flavor.",
    "reasoning": "This is a well-established fact about tea preparation that doesn't require current sources or user-specific information."
}}

Example for "What was the relationship between Dan Reingold and Bernie Ebbers?":
{{
    "can_answer_quickly": false,
    "confidence": 0.9,
    "quick_answer": null,
    "reasoning": "This requires specific information about historical relationships between individuals that would need research from documents or web sources."
}}"""
            
            llm = self._get_llm(temperature=0.3, state=state)
            response = await llm.ainvoke([
                SystemMessage(content="You are a query evaluator. Always respond with valid JSON matching the exact schema provided."),
                HumanMessage(content=evaluation_prompt)
            ])
            
            # Parse response with Pydantic validation
            try:
                # Clean response content - strip markdown code fences
                text = response.content.strip()
                if '```json' in text:
                    m = re.search(r'```json\s*\n([\s\S]*?)\n```', text)
                    if m:
                        text = m.group(1).strip()
                elif '```' in text:
                    m = re.search(r'```\s*\n([\s\S]*?)\n```', text)
                    if m:
                        text = m.group(1).strip()
                
                # Try to find JSON object if still wrapped in other text
                json_match = re.search(r'\{[\s\S]*\}', text)
                if json_match:
                    text = json_match.group(0)
                
                # Validate with Pydantic
                assessment = QuickAnswerAssessment.parse_raw(text)
                
                if assessment.can_answer_quickly and assessment.quick_answer:
                    # Format the quick answer with offer for deeper research
                    formatted_answer = f"""{assessment.quick_answer}

---
*Would you like me to perform a deeper search for more detailed information, sources, or alternative perspectives? Just let me know!*"""
                    
                    logger.info(f"Quick answer provided: confidence={assessment.confidence}")
                    logger.info(f"Reasoning: {assessment.reasoning}")
                    
                    # Update shared_memory to track agent selection for conversation continuity
                    shared_memory = state.get("shared_memory", {}) or {}
                    shared_memory["primary_agent_selected"] = "full_research_agent"
                    shared_memory["last_agent"] = "full_research_agent"
                    
                    return {
                        "quick_answer_provided": True,
                        "quick_answer_content": formatted_answer,
                        "final_response": formatted_answer,
                        "research_complete": True,
                        "current_round": ResearchRound.QUICK_ANSWER_CHECK.value,
                        "shared_memory": shared_memory
                    }
                else:
                    logger.info(f"Query requires full research: {assessment.reasoning}")
                    return {
                        "quick_answer_provided": False,
                        "quick_answer_content": "",
                        "current_round": ResearchRound.QUICK_ANSWER_CHECK.value
                    }
                    
            except (json.JSONDecodeError, ValidationError, Exception) as e:
                logger.warning(f"Failed to parse quick answer assessment: {e}")
                logger.warning(f"Raw response: {response.content[:500]}")
                # Fallback: proceed to full research if parsing fails
                logger.info("Quick answer assessment parsing failed - proceeding to full research")
                return {
                    "quick_answer_provided": False,
                    "quick_answer_content": "",
                    "current_round": ResearchRound.QUICK_ANSWER_CHECK.value
                }
            
        except Exception as e:
            logger.error(f"Quick answer check error: {e}")
            # On error, proceed to full research
            return {
                "quick_answer_provided": False,
                "quick_answer_content": "",
                "current_round": ResearchRound.QUICK_ANSWER_CHECK.value
            }
    
    def _route_from_quick_answer(self, state: ResearchState) -> str:
        """Route from quick answer check: provide quick answer or proceed to full research"""
        if state.get("quick_answer_provided") and state.get("quick_answer_content"):
            logger.info("Quick answer provided - short-circuiting to response")
            return "quick_answer"
        logger.info("Proceeding to full research workflow")
        return "full_research"
    
    async def _cache_check_node(self, state: ResearchState) -> Dict[str, Any]:
        """Check conversation cache for previous research"""
        try:
            query = state["query"]
            logger.info(f"Checking cache for: {query}")
            
            # Search cache
            cache_result = await search_conversation_cache_tool(query=query, freshness_hours=24)
            
            if cache_result["cache_hit"] and cache_result["entries"]:
                # Found cached research
                cached_context = "\n\n".join([
                    f"[{entry['agent_name']}]: {entry['content']}"
                    for entry in cache_result["entries"]
                ])
                
                logger.info(f"Cache HIT - found {len(cache_result['entries'])} cached entries")
                
                return {
                    "cache_hit": True,
                    "cached_context": cached_context,
                    "current_round": ResearchRound.CACHE_CHECK.value
                }
            
            logger.info("Cache MISS - no previous research found")
            return {
                "cache_hit": False,
                "cached_context": "",
                "current_round": ResearchRound.CACHE_CHECK.value
            }
            
        except Exception as e:
            logger.error(f"Cache check error: {e}")
            return {
                "cache_hit": False,
                "cached_context": "",
                "current_round": ResearchRound.CACHE_CHECK.value
            }
    
    async def _query_expansion_node(self, state: ResearchState) -> Dict[str, Any]:
        """Expand query with semantic variations"""
        try:
            query = state["query"]
            logger.info(f"Expanding query: {query}")
            
            # Expand query
            expansion_result = await expand_query_tool(query=query, num_variations=3)
            
            expanded_queries = expansion_result.get("expanded_queries", [query])
            key_entities = expansion_result.get("key_entities", [])
            
            logger.info(f"Generated {len(expanded_queries)} query variations, {len(key_entities)} entities")
            
            return {
                "expanded_queries": expanded_queries,
                "key_entities": key_entities,
                "original_query": query
            }
            
        except Exception as e:
            logger.error(f"Query expansion error: {e}")
            return {
                "expanded_queries": [state["query"]],
                "key_entities": [],
                "original_query": state["query"]
            }
    
    async def _round1_parallel_search_node(self, state: ResearchState) -> Dict[str, Any]:
        """Round 1: Parallel local and web search for faster results and better decision-making"""
        try:
            query = state["query"]
            expanded_queries = state.get("expanded_queries", [query])
            
            logger.info(f"Round 1: Parallel search - local + web with {len(expanded_queries)} queries")
            
            # Run local and web search in parallel
            import asyncio
            from orchestrator.tools.web_tools import search_and_crawl_tool
            
            async def local_search_task():
                """Local search task"""
                try:
                    all_results = []
                    for q in expanded_queries[:3]:  # Limit to top 3
                        result = await search_documents_tool(query=q, limit=10)
                        all_results.append(result)
                    combined_results = "\n\n".join(all_results)
                    return {
                        "search_results": combined_results,
                        "queries_used": expanded_queries[:3],
                        "result_count": len(all_results)
                    }
                except Exception as e:
                    logger.error(f"Local search error: {e}")
                    return {"error": str(e), "search_results": ""}
            
            async def web_search_task():
                """Web search task"""
                try:
                    web_result = await search_and_crawl_tool(query=query, max_results=10)
                    return {
                        "content": web_result,
                        "query_used": query
                    }
                except Exception as e:
                    logger.error(f"Web search error: {e}")
                    return {"error": str(e), "content": ""}
            
            # Execute both searches in parallel
            local_result, web_result = await asyncio.gather(
                local_search_task(),
                web_search_task(),
                return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(local_result, Exception):
                logger.error(f"Local search exception: {local_result}")
                local_result = {"error": str(local_result), "search_results": ""}
            
            if isinstance(web_result, Exception):
                logger.error(f"Web search exception: {web_result}")
                web_result = {"error": str(web_result), "content": ""}
            
            logger.info(f"✅ Parallel search complete: local={bool(local_result.get('search_results'))}, web={bool(web_result.get('content'))}")
            
            return {
                "round1_results": local_result,
                "web_round1_results": web_result,
                "current_round": ResearchRound.INITIAL_LOCAL.value
            }
            
        except Exception as e:
            logger.error(f"Round 1 parallel search error: {e}")
            return {
                "round1_results": {"error": str(e), "search_results": ""},
                "web_round1_results": {"error": str(e), "content": ""},
                "current_round": ResearchRound.INITIAL_LOCAL.value
            }
    
    async def _round1_local_search_node(self, state: ResearchState) -> Dict[str, Any]:
        """Round 1: Initial local search (legacy - kept for backward compatibility)"""
        return await self._round1_parallel_search_node(state)
    
    async def _assess_combined_round1_node(self, state: ResearchState) -> Dict[str, Any]:
        """Assess combined Round 1 results (local + web) for quality and sufficiency"""
        try:
            query = state["query"]
            round1_results = state.get("round1_results", {})
            web_round1_results = state.get("web_round1_results", {})
            
            local_results = round1_results.get("search_results", "")
            web_results = web_round1_results.get("content", "")
            
            logger.info("Assessing combined Round 1 results (local + web)")
            
            # Use LLM to assess quality with structured output - now includes both local and web
            assessment_prompt = f"""Assess the quality and sufficiency of these combined search results (local documents + web search) for answering the user's query.

USER QUERY: {query}

LOCAL DOCUMENT RESULTS:
{local_results[:1500] if local_results else "No local results found."}

WEB SEARCH RESULTS:
{web_results[:1500] if web_results else "No web results found."}

Evaluate:
1. Do the results (local + web combined) contain relevant information?
2. Is there enough detail to answer the query comprehensively?
3. What information is still missing (if any)?
4. Which source (local vs web) provides better information?

STRUCTURED OUTPUT REQUIRED - Respond with ONLY valid JSON matching this exact schema:
{{
    "sufficient": boolean,
    "has_relevant_info": boolean,
    "missing_info": ["list", "of", "specific", "gaps"],
    "confidence": number (0.0-1.0),
    "reasoning": "brief explanation of assessment",
    "best_source": "local" | "web" | "both",
    "needs_more_local": boolean,
    "needs_more_web": boolean
}}

Example:
{{
    "sufficient": true,
    "has_relevant_info": true,
    "missing_info": [],
    "confidence": 0.9,
    "reasoning": "Combined local and web results provide comprehensive information to answer the query",
    "best_source": "both",
    "needs_more_local": false,
    "needs_more_web": false
}}"""
            
            llm = self._get_llm(temperature=0.7, state=state)
            response = await llm.ainvoke([
                SystemMessage(content="You are a research quality assessor. Always respond with valid JSON."),
                HumanMessage(content=assessment_prompt)
            ])
            
            # Parse response with Pydantic validation
            try:
                # Clean response content - strip markdown code fences
                text = response.content.strip()
                if '```json' in text:
                    m = re.search(r'```json\s*\n([\s\S]*?)\n```', text)
                    if m:
                        text = m.group(1).strip()
                elif '```' in text:
                    m = re.search(r'```\s*\n([\s\S]*?)\n```', text)
                    if m:
                        text = m.group(1).strip()
                
                # Try to find JSON object if still wrapped in other text
                json_match = re.search(r'\{[\s\S]*\}', text)
                if json_match:
                    text = json_match.group(0)
                
                # Validate with Pydantic
                assessment = ResearchAssessmentResult.parse_raw(text)
                sufficient = assessment.sufficient
                
                # Extract additional fields if present
                assessment_dict = json.loads(text) if isinstance(text, str) else text
                best_source = assessment_dict.get("best_source", "both")
                needs_more_local = assessment_dict.get("needs_more_local", False)
                needs_more_web = assessment_dict.get("needs_more_web", False)
                
                logger.info(f"Combined Round 1 assessment: sufficient={sufficient}, confidence={assessment.confidence}, best_source={best_source}")
                logger.info(f"Assessment reasoning: {assessment.reasoning}")
                
            except (json.JSONDecodeError, ValidationError, Exception) as e:
                logger.warning(f"Failed to parse assessment with Pydantic: {e}")
                logger.warning(f"Raw response: {response.content[:500]}")
                # Fallback: conservative assumption that more research needed
                sufficient = False
                best_source = "both"
                needs_more_local = True
                needs_more_web = False
                logger.info(f"Combined Round 1 assessment (fallback): sufficient={sufficient}")
            
            return {
                "round1_sufficient": sufficient,
                "round1_assessment": {
                    "sufficient": assessment.sufficient if 'assessment' in locals() else sufficient,
                    "has_relevant_info": assessment.has_relevant_info if 'assessment' in locals() else True,
                    "confidence": assessment.confidence if 'assessment' in locals() else 0.5,
                    "reasoning": assessment.reasoning if 'assessment' in locals() else response.content[:200],
                    "best_source": best_source if 'best_source' in locals() else "both",
                    "needs_more_local": needs_more_local if 'needs_more_local' in locals() else False,
                    "needs_more_web": needs_more_web if 'needs_more_web' in locals() else False
                },
                "gap_analysis": {
                    "has_local_gaps": not sufficient,
                    "assessment_text": response.content
                }
            }
            
        except Exception as e:
            logger.error(f"Combined Round 1 assessment error: {e}")
            return {
                "round1_sufficient": False,
                "round1_assessment": {
                    "sufficient": False,
                    "has_relevant_info": False,
                    "confidence": 0.0,
                    "reasoning": f"Assessment error: {str(e)}",
                    "best_source": "both",
                    "needs_more_local": True,
                    "needs_more_web": False
                },
                "gap_analysis": {"has_local_gaps": True}
            }
    
    async def _assess_round1_node(self, state: ResearchState) -> Dict[str, Any]:
        """Assess Round 1 results quality and sufficiency (legacy - redirects to combined assessment)"""
        return await self._assess_combined_round1_node(state)
    
    async def _gap_analysis_node(self, state: ResearchState) -> Dict[str, Any]:
        """Analyze gaps in Round 1 results"""
        try:
            query = state["query"]
            round1_results = state.get("round1_results", {})
            
            logger.info("Performing gap analysis")
            
            # Use LLM to identify specific gaps with structured output
            gap_prompt = f"""Analyze what information is missing from the search results to fully answer the query.

USER QUERY: {query}

RESULTS SO FAR: {round1_results.get('search_results', '')[:1500]}

Identify:
1. Specific missing entities, people, facts, or concepts
2. Targeted search queries that could fill those specific gaps
3. Whether web search would be beneficial
4. How severe the information gaps are

STRUCTURED OUTPUT REQUIRED - Respond with ONLY valid JSON matching this exact schema:
{{
    "missing_entities": ["specific", "missing", "entities"],
    "suggested_queries": ["targeted search query 1", "targeted search query 2"],
    "needs_web_search": boolean,
    "gap_severity": "minor" | "moderate" | "severe",
    "reasoning": "explanation of gaps and how to fill them"
}}

Example:
{{
    "missing_entities": ["Dan Reingold's employer", "specific dates of interactions", "nature of professional relationship"],
    "suggested_queries": ["Dan Reingold analyst firm WorldCom", "Reingold Ebbers professional relationship timeline"],
    "needs_web_search": false,
    "gap_severity": "moderate",
    "reasoning": "Local results provide context but lack specific details about their professional interactions and Reingold's role"
}}"""
            
            llm = self._get_llm(temperature=0.7, state=state)
            response = await llm.ainvoke([
                SystemMessage(content="You are a research gap analyst. Always respond with valid JSON."),
                HumanMessage(content=gap_prompt)
            ])
            
            logger.info("Gap analysis complete")
            
            # Parse the LLM response with Pydantic validation
            identified_gaps = []
            try:
                # Clean response content - strip markdown code fences
                text = response.content.strip()
                if '```json' in text:
                    m = re.search(r'```json\s*\n([\s\S]*?)\n```', text)
                    if m:
                        text = m.group(1).strip()
                elif '```' in text:
                    m = re.search(r'```\s*\n([\s\S]*?)\n```', text)
                    if m:
                        text = m.group(1).strip()
                
                # Try to find JSON object if still wrapped in other text
                json_match = re.search(r'\{[\s\S]*\}', text)
                if json_match:
                    text = json_match.group(0)
                
                # Validate with Pydantic
                gap_analysis = ResearchGapAnalysis.parse_raw(text)
                
                # Extract suggested queries first (most specific)
                if gap_analysis.suggested_queries:
                    identified_gaps = gap_analysis.suggested_queries
                elif gap_analysis.missing_entities:
                    # Use missing entities as search terms
                    identified_gaps = gap_analysis.missing_entities
                else:
                    # Fallback to original query if no specific gaps found
                    identified_gaps = [query]
                    
                logger.info(f"Gap analysis: severity={gap_analysis.gap_severity}, web_search={gap_analysis.needs_web_search}")
                logger.info(f"Missing entities: {gap_analysis.missing_entities}")
                logger.info(f"Suggested queries: {gap_analysis.suggested_queries}")
                logger.info(f"Reasoning: {gap_analysis.reasoning}")
                
                # Store needs_web_search flag in state for routing decision
                return {
                    "gap_analysis": {
                        "analysis": response.content,
                        "has_local_gaps": True,
                        "needs_web_search": gap_analysis.needs_web_search,
                        "gap_severity": gap_analysis.gap_severity
                    },
                    "identified_gaps": identified_gaps
                }
                
            except (json.JSONDecodeError, ValidationError, Exception) as e:
                logger.warning(f"Failed to parse gap analysis with Pydantic: {e}")
                logger.warning(f"Raw response: {response.content[:500]}")
                # Fallback: use original query and assume web search needed
                identified_gaps = [query]
                logger.info(f"Gap analysis (fallback): using original query, assuming web search needed")
                return {
                    "gap_analysis": {
                        "analysis": response.content if 'response' in locals() else "",
                        "has_local_gaps": True,
                        "needs_web_search": True,  # Default to web search on fallback
                        "gap_severity": "severe"
                    },
                    "identified_gaps": identified_gaps
                }
            
        except Exception as e:
            logger.error(f"Gap analysis error: {e}")
            return {
                "gap_analysis": {"has_local_gaps": False},
                "identified_gaps": []
            }
    
    async def _round2_gap_filling_node(self, state: ResearchState) -> Dict[str, Any]:
        """Round 2: Targeted search to fill identified gaps"""
        try:
            query = state["query"]
            identified_gaps = state.get("identified_gaps", [])
            
            logger.info(f"Round 2: Gap filling for {len(identified_gaps)} gaps")
            
            # Search for gaps
            gap_results = []
            for gap in identified_gaps[:3]:
                result = await search_documents_tool(query=gap, limit=5)
                gap_results.append(result)
            
            combined_gap_results = "\n\n".join(gap_results)
            
            # Simple sufficiency check
            has_results = len(combined_gap_results) > 100
            
            return {
                "round2_results": {
                    "gap_results": combined_gap_results,
                    "gaps_addressed": len(gap_results)
                },
                "round2_sufficient": has_results,
                "current_round": ResearchRound.ROUND_2_GAP_FILLING.value
            }
            
        except Exception as e:
            logger.error(f"Round 2 gap filling error: {e}")
            return {
                "round2_results": {"error": str(e)},
                "round2_sufficient": False,
                "current_round": ResearchRound.ROUND_2_GAP_FILLING.value
            }
    
    async def _web_round1_node(self, state: ResearchState) -> Dict[str, Any]:
        """Web Round 1: Initial web search when local sources insufficient"""
        try:
            query = state["query"]
            expanded_queries = state.get("expanded_queries", [query])
            
            logger.info(f"Web Round 1: {query}")
            
            # Use search_and_crawl for comprehensive web research
            web_result = await search_and_crawl_tool(query=query, max_results=10)
            
            return {
                "web_round1_results": {
                    "content": web_result,
                    "query_used": query
                },
                "web_search_results": {  # Legacy field
                    "content": web_result,
                    "query_used": query
                },
                "web_permission_granted": True,  # No HITL currently
                "current_round": ResearchRound.WEB_ROUND_1.value
            }
            
        except Exception as e:
            logger.error(f"Web Round 1 error: {e}")
            return {
                "web_round1_results": {"error": str(e)},
                "web_search_results": {"error": str(e)},
                "current_round": ResearchRound.WEB_ROUND_1.value
            }
    
    async def _assess_web_round1_node(self, state: ResearchState) -> Dict[str, Any]:
        """Assess Web Round 1 results quality and sufficiency"""
        try:
            query = state["query"]
            web_round1_results = state.get("web_round1_results", {})
            web_content = web_round1_results.get("content", "")
            
            # Extract text content if it's a formatted string
            if isinstance(web_content, str):
                search_results = web_content
            else:
                search_results = str(web_content)
            
            logger.info("Assessing Web Round 1 results quality")
            
            # Use LLM to assess quality with structured output
            assessment_prompt = f"""Assess the quality and sufficiency of these web search results for answering the user's query.

USER QUERY: {query}

WEB SEARCH RESULTS:
{search_results[:2000]}

Evaluate:
1. Do the results contain relevant information?
2. Is there enough detail to answer the query comprehensively?
3. What information is missing (if any)?

STRUCTURED OUTPUT REQUIRED - Respond with ONLY valid JSON matching this exact schema:
{{
    "sufficient": boolean,
    "has_relevant_info": boolean,
    "missing_info": ["list", "of", "specific", "gaps"],
    "confidence": number (0.0-1.0),
    "reasoning": "brief explanation of assessment"
}}"""
            
            llm = self._get_llm(temperature=0.7, state=state)
            response = await llm.ainvoke([
                SystemMessage(content="You are a research quality assessor. Always respond with valid JSON."),
                HumanMessage(content=assessment_prompt)
            ])
            
            # Parse response with Pydantic validation
            try:
                text = response.content.strip()
                if '```json' in text:
                    m = re.search(r'```json\s*\n([\s\S]*?)\n```', text)
                    if m:
                        text = m.group(1).strip()
                elif '```' in text:
                    m = re.search(r'```\s*\n([\s\S]*?)\n```', text)
                    if m:
                        text = m.group(1).strip()
                
                json_match = re.search(r'\{[\s\S]*\}', text)
                if json_match:
                    text = json_match.group(0)
                
                assessment = ResearchAssessmentResult.parse_raw(text)
                sufficient = assessment.sufficient
                
                logger.info(f"Web Round 1 assessment: sufficient={sufficient}, confidence={assessment.confidence}, relevant={assessment.has_relevant_info}")
                logger.info(f"Assessment reasoning: {assessment.reasoning}")
                
            except (json.JSONDecodeError, ValidationError, Exception) as e:
                logger.warning(f"Failed to parse web assessment with Pydantic: {e}")
                sufficient = False
            
            return {
                "web_round1_sufficient": sufficient,
                "web_round1_assessment": {
                    "sufficient": sufficient,
                    "has_relevant_info": assessment.has_relevant_info if 'assessment' in locals() else True,
                    "confidence": assessment.confidence if 'assessment' in locals() else 0.5,
                    "reasoning": assessment.reasoning if 'assessment' in locals() else "Assessment parsing failed"
                }
            }
            
        except Exception as e:
            logger.error(f"Web Round 1 assessment error: {e}")
            return {
                "web_round1_sufficient": False,
                "web_round1_assessment": {
                    "sufficient": False,
                    "has_relevant_info": False,
                    "confidence": 0.0,
                    "reasoning": f"Assessment error: {str(e)}"
                }
            }
    
    async def _gap_analysis_web_node(self, state: ResearchState) -> Dict[str, Any]:
        """Analyze gaps in Web Round 1 results"""
        try:
            query = state["query"]
            web_round1_results = state.get("web_round1_results", {})
            web_content = web_round1_results.get("content", "")
            
            if isinstance(web_content, str):
                search_results = web_content
            else:
                search_results = str(web_content)
            
            logger.info("Performing web gap analysis")
            
            # Use LLM to identify specific gaps with structured output
            gap_prompt = f"""Analyze what information is missing from the web search results to fully answer the query.

USER QUERY: {query}

WEB SEARCH RESULTS SO FAR:
{search_results[:1500]}

Identify:
1. Specific missing entities, people, facts, or concepts
2. Whether a second round of web search would be beneficial
3. How severe the information gaps are

STRUCTURED OUTPUT REQUIRED - Respond with ONLY valid JSON matching this exact schema:
{{
    "missing_entities": ["specific", "missing", "entities"],
    "suggested_queries": ["targeted web search query 1", "targeted web search query 2"],
    "needs_web_round2": boolean,
    "gap_severity": "minor" | "moderate" | "severe",
    "reasoning": "explanation of gaps and whether Round 2 Web would help"
}}"""
            
            llm = self._get_llm(temperature=0.7, state=state)
            response = await llm.ainvoke([
                SystemMessage(content="You are a research gap analyst. Always respond with valid JSON."),
                HumanMessage(content=gap_prompt)
            ])
            
            logger.info("Web gap analysis complete")
            
            # Parse the LLM response with Pydantic validation
            web_identified_gaps = []
            try:
                text = response.content.strip()
                if '```json' in text:
                    m = re.search(r'```json\s*\n([\s\S]*?)\n```', text)
                    if m:
                        text = m.group(1).strip()
                elif '```' in text:
                    m = re.search(r'```\s*\n([\s\S]*?)\n```', text)
                    if m:
                        text = m.group(1).strip()
                
                json_match = re.search(r'\{[\s\S]*\}', text)
                if json_match:
                    text = json_match.group(0)
                
                gap_analysis = ResearchGapAnalysis.parse_raw(text)
                
                # Extract suggested queries
                if gap_analysis.suggested_queries:
                    web_identified_gaps = gap_analysis.suggested_queries
                elif gap_analysis.missing_entities:
                    web_identified_gaps = gap_analysis.missing_entities
                else:
                    web_identified_gaps = [query]
                
                logger.info(f"Web gap analysis: severity={gap_analysis.gap_severity}, needs_round2={gap_analysis.needs_web_search}")
                logger.info(f"Web missing entities: {gap_analysis.missing_entities}")
                logger.info(f"Web suggested queries: {gap_analysis.suggested_queries}")
                
                return {
                    "web_gap_analysis": {
                        "analysis": response.content,
                        "needs_web_round2": gap_analysis.needs_web_search,  # Reuse needs_web_search field
                        "gap_severity": gap_analysis.gap_severity
                    },
                    "web_identified_gaps": web_identified_gaps
                }
                
            except (json.JSONDecodeError, ValidationError, Exception) as e:
                logger.warning(f"Failed to parse web gap analysis: {e}")
                return {
                    "web_gap_analysis": {
                        "needs_web_round2": False,  # Default to no Round 2 if parsing fails
                        "gap_severity": "minor"
                    },
                    "web_identified_gaps": []
                }
            
        except Exception as e:
            logger.error(f"Web gap analysis error: {e}")
            return {
                "web_gap_analysis": {"needs_web_round2": False},
                "web_identified_gaps": []
            }
    
    async def _web_round2_node(self, state: ResearchState) -> Dict[str, Any]:
        """Web Round 2: Targeted web search to fill identified gaps"""
        try:
            query = state["query"]
            web_identified_gaps = state.get("web_identified_gaps", [])
            
            logger.info(f"Web Round 2: Gap filling for {len(web_identified_gaps)} gaps")
            
            # Use the first identified gap query, or original query
            search_query = web_identified_gaps[0] if web_identified_gaps else query
            
            logger.info(f"Web Round 2 search: {search_query}")
            
            # Use search_and_crawl for targeted web research
            web_result = await search_and_crawl_tool(query=search_query, max_results=10)
            
            return {
                "web_round2_results": {
                    "content": web_result,
                    "query_used": search_query
                },
                "current_round": ResearchRound.WEB_ROUND_2.value
            }
            
        except Exception as e:
            logger.error(f"Web Round 2 error: {e}")
            return {
                "web_round2_results": {"error": str(e)},
                "current_round": ResearchRound.WEB_ROUND_2.value
            }
    
    async def _detect_query_type_node(self, state: ResearchState) -> Dict[str, Any]:
        """Detect query type to determine synthesis approach: objective (synthesize) vs subjective (present options)"""
        try:
            query = state["query"]
            
            logger.info(f"Detecting query type for: {query}")
            
            # Use LLM to detect query type with structured output
            detection_prompt = f"""Analyze this query to determine whether it should receive a synthesized single answer or multiple distinct options.

USER QUERY: {query}

Consider:
1. **Objective queries** (synthesize single answer):
   - Factual questions: "What is the boiling point of water?"
   - Process questions: "How do I change a tire?"
   - Historical facts: "What happened at the Battle of Hastings?"
   - Scientific facts: "What is photosynthesis?"
   - These have clear, objective answers that benefit from synthesis

2. **Subjective queries** (present 2-3 options):
   - Preference-based: "Perfect hot cocoa recipe" (taste is subjective)
   - Style choices: "Best color scheme for a website" (depends on context)
   - Personal decisions: "What laptop should I buy?" (depends on needs/budget)
   - Creative projects: "How to decorate my living room?" (personal taste)
   - Recipe requests: "Best chocolate chip cookie recipe" (many valid approaches)
   - These have multiple valid answers and benefit from presenting options

3. **Mixed queries** (synthesize with alternatives mentioned):
   - Queries that have a primary answer but notable alternatives
   - Example: "How should I structure a research paper?" (standard approach + alternatives)

STRUCTURED OUTPUT REQUIRED - Respond with ONLY valid JSON matching this exact schema:
{{
    "query_type": "objective" | "subjective" | "mixed",
    "confidence": number (0.0-1.0),
    "reasoning": "brief explanation of why this query type was chosen",
    "should_present_options": boolean,
    "num_options": number (2-3, only relevant if should_present_options=true)
}}

Example for "What is the perfect hot cocoa recipe?":
{{
    "query_type": "subjective",
    "confidence": 0.95,
    "reasoning": "Recipe preferences are highly subjective - some prefer dark chocolate, others milk chocolate; some like marshmallows, others whipped cream. Multiple valid approaches exist.",
    "should_present_options": true,
    "num_options": 3
}}

Example for "What is the boiling point of water?":
{{
    "query_type": "objective",
    "confidence": 0.99,
    "reasoning": "This is a factual scientific question with a single correct answer that benefits from synthesis.",
    "should_present_options": false,
    "num_options": null
}}

Example for "How should I structure a research paper?":
{{
    "query_type": "mixed",
    "confidence": 0.85,
    "reasoning": "There is a standard research paper structure, but notable variations exist (IMRaD vs. narrative vs. argumentative). Should synthesize the standard approach while mentioning alternatives.",
    "should_present_options": false,
    "num_options": null
}}"""
            
            llm = self._get_llm(temperature=0.3, state=state)
            response = await llm.ainvoke([
                SystemMessage(content="You are a query type classifier. Always respond with valid JSON matching the exact schema provided."),
                HumanMessage(content=detection_prompt)
            ])
            
            # Parse response with Pydantic validation
            try:
                # Clean response content - strip markdown code fences
                text = response.content.strip()
                if '```json' in text:
                    m = re.search(r'```json\s*\n([\s\S]*?)\n```', text)
                    if m:
                        text = m.group(1).strip()
                elif '```' in text:
                    m = re.search(r'```\s*\n([\s\S]*?)\n```', text)
                    if m:
                        text = m.group(1).strip()
                
                # Try to find JSON object if still wrapped in other text
                json_match = re.search(r'\{[\s\S]*\}', text)
                if json_match:
                    text = json_match.group(0)
                
                # Validate with Pydantic
                detection = QueryTypeDetection.parse_raw(text)
                
                logger.info(f"Query type detected: {detection.query_type}, confidence={detection.confidence}")
                logger.info(f"Should present options: {detection.should_present_options}, num_options={detection.num_options}")
                logger.info(f"Reasoning: {detection.reasoning}")
                
                return {
                    "query_type": detection.query_type,
                    "query_type_detection": {
                        "query_type": detection.query_type,
                        "confidence": detection.confidence,
                        "reasoning": detection.reasoning,
                        "should_present_options": detection.should_present_options,
                        "num_options": detection.num_options
                    },
                    "should_present_options": detection.should_present_options,
                    "num_options": detection.num_options
                }
                
            except (json.JSONDecodeError, ValidationError, Exception) as e:
                logger.warning(f"Failed to parse query type detection: {e}")
                logger.warning(f"Raw response: {response.content[:500]}")
                # Fallback: default to objective (synthesize) if parsing fails
                logger.info("Query type detection parsing failed - defaulting to objective (synthesize)")
                return {
                    "query_type": "objective",
                    "query_type_detection": {
                        "query_type": "objective",
                        "confidence": 0.5,
                        "reasoning": "Detection parsing failed - defaulting to objective",
                        "should_present_options": False,
                        "num_options": None
                    },
                    "should_present_options": False,
                    "num_options": None
                }
            
        except Exception as e:
            logger.error(f"Query type detection error: {e}")
            # Fallback: default to objective (synthesize) on error
            return {
                "query_type": "objective",
                "query_type_detection": {
                    "query_type": "objective",
                    "confidence": 0.5,
                    "reasoning": f"Detection error: {str(e)}",
                    "should_present_options": False,
                    "num_options": None
                },
                "should_present_options": False,
                "num_options": None
            }
    
    async def _final_synthesis_node(self, state: ResearchState) -> Dict[str, Any]:
        """Final synthesis with all gathered information"""
        try:
            query = state["query"]
            
            # Gather all results
            cached_context = state.get("cached_context", "")
            round1_results = state.get("round1_results", {})
            round2_results = state.get("round2_results", {})
            web_round1_results = state.get("web_round1_results", {})
            web_round2_results = state.get("web_round2_results", {})
            # Legacy field for backward compatibility
            web_results = state.get("web_search_results", {}) or web_round1_results
            
            logger.info("Synthesizing final response from all sources")
            
            # Build comprehensive context
            context_parts = []
            
            if cached_context:
                context_parts.append(f"CACHED RESEARCH:\n{cached_context}")
            
            if round1_results:
                context_parts.append(f"LOCAL SEARCH ROUND 1:\n{round1_results.get('search_results', '')[:2000]}")
            
            if round2_results:
                context_parts.append(f"LOCAL SEARCH ROUND 2:\n{round2_results.get('gap_results', '')[:1500]}")
            
            if web_round1_results:
                web_content = web_round1_results.get("content", "")
                if isinstance(web_content, str):
                    context_parts.append(f"WEB SEARCH ROUND 1:\n{web_content[:2000]}")
                else:
                    context_parts.append(f"WEB SEARCH ROUND 1:\n{str(web_content)[:2000]}")
            
            if web_round2_results:
                web2_content = web_round2_results.get("content", "")
                if isinstance(web2_content, str):
                    context_parts.append(f"WEB SEARCH ROUND 2:\n{web2_content[:1500]}")
                else:
                    context_parts.append(f"WEB SEARCH ROUND 2:\n{str(web2_content)[:1500]}")
            
            full_context = "\n\n".join(context_parts)
            
            # Get query type detection results
            query_type = state.get("query_type", "objective")
            should_present_options = state.get("should_present_options", False)
            num_options = state.get("num_options", 3)
            query_type_detection = state.get("query_type_detection", {})
            reasoning = query_type_detection.get("reasoning", "")
            
            logger.info(f"Synthesizing response with query_type={query_type}, should_present_options={should_present_options}")
            
            # Generate different prompts based on query type
            if should_present_options and query_type in ["subjective", "mixed"]:
                # Subjective query: Present 2-3 distinct options
                synthesis_prompt = f"""Based on all available research, present {num_options} distinct, well-researched approaches to the user's query.

USER QUERY: {query}

RESEARCH FINDINGS:
{full_context}

REASONING FOR PRESENTING OPTIONS:
{reasoning}

Provide a well-organized response that:
1. Presents {num_options} distinct approaches/options (each with clear title/name)
2. For each option, include:
   - Key characteristics and features
   - Advantages and trade-offs
   - When this approach works best
   - Specific details from research
3. Highlight key differences between the options
4. Let the user choose based on their preferences/needs
5. Cite sources where appropriate
6. Use clear, professional language

Format as:
## Option 1: [Name]
[Description, characteristics, advantages, trade-offs, when to use]

## Option 2: [Name]
[Description, characteristics, advantages, trade-offs, when to use]

## Option 3: [Name] (if num_options=3)
[Description, characteristics, advantages, trade-offs, when to use]

Your response with {num_options} distinct options:"""
            
            elif query_type == "mixed":
                # Mixed query: Synthesize primary answer but mention alternatives
                synthesis_prompt = f"""Based on all available research, provide a comprehensive answer to the user's query, with the primary approach synthesized and notable alternatives mentioned.

USER QUERY: {query}

RESEARCH FINDINGS:
{full_context}

REASONING:
{reasoning}

Provide a well-organized response that:
1. Synthesizes the primary/standard approach into a comprehensive answer
2. Mentions notable alternative approaches or variations
3. Explains when alternatives might be preferred
4. Directly answers the query
5. Synthesizes information from all sources
6. Cites sources where appropriate
7. Uses clear, professional language

Your comprehensive response:"""
            
            else:
                # Objective query: Standard synthesis (default)
                synthesis_prompt = f"""Based on all available research, provide a comprehensive answer to the user's query.

USER QUERY: {query}

RESEARCH FINDINGS:
{full_context}

Provide a well-organized, thorough response that:
1. Directly answers the query
2. Synthesizes information from all sources
3. Cites sources where appropriate
4. Acknowledges any remaining gaps
5. Uses clear, professional language

Your comprehensive response:"""
            
            synthesis_llm = self._get_llm(temperature=0.3, state=state)
            response = await synthesis_llm.ainvoke([
                SystemMessage(content="You are an expert research synthesizer."),
                HumanMessage(content=synthesis_prompt)
            ])
            
            final_response = response.content
            
            logger.info(f"Synthesis complete: {len(final_response)} characters")
            
            # Detect if data formatting would improve the response
            routing_recommendation = await detect_formatting_need(query, final_response)
            
            if routing_recommendation:
                logger.info(f"📊 Formatting agent recommended for this research")
            
            # Extract citations (simplified)
            sources_used = []
            if round1_results:
                sources_used.append("local_round1")
            if round2_results:
                sources_used.append("local_round2")
            if web_round1_results:
                sources_used.append("web_round1")
            if web_round2_results:
                sources_used.append("web_round2")
            
            # Update shared_memory to track agent selection for conversation continuity
            shared_memory = state.get("shared_memory", {}) or {}
            shared_memory["primary_agent_selected"] = "full_research_agent"
            shared_memory["last_agent"] = "full_research_agent"
            
            return {
                "final_response": final_response,
                "research_complete": True,
                "sources_used": sources_used,
                "current_round": ResearchRound.FINAL_SYNTHESIS.value,
                "routing_recommendation": routing_recommendation,
                "citations": [],  # TODO: Implement citation extraction
                "shared_memory": shared_memory
            }
            
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            # Update shared_memory even on error to track agent selection
            shared_memory = state.get("shared_memory", {}) or {}
            shared_memory["primary_agent_selected"] = "full_research_agent"
            shared_memory["last_agent"] = "full_research_agent"
            return {
                "final_response": f"Research completed but synthesis failed: {str(e)}",
                "research_complete": True,
                "error": str(e),
                "shared_memory": shared_memory
            }
    
    def _route_from_synthesis(self, state: ResearchState) -> str:
        """Route from synthesis: check if formatting is needed"""
        routing_recommendation = state.get("routing_recommendation")
        if routing_recommendation == "data_formatting":
            logger.info("📊 Routing to data formatting agent")
            return "format"
        return "complete"
    
    async def _format_data_node(self, state: ResearchState) -> Dict[str, Any]:
        """Format research results using data formatting agent"""
        try:
            logger.info("📊 Formatting research results with data formatting agent...")
            
            # Import data formatting agent
            from orchestrator.agents import DataFormattingAgent
            
            # Get the research response and query
            query = state.get("query", "")
            final_response = state.get("final_response", "")
            
            # Build formatting request
            formatting_query = f"Format the following research results into a well-organized structure:\n\n{final_response}"
            
            # Get messages from state for context
            messages = state.get("messages", [])
            
            # Create data formatting agent instance
            formatting_agent = DataFormattingAgent()
            
            # Call data formatting agent
            formatting_result = await formatting_agent.process(
                query=formatting_query,
                metadata={"user_id": "system"},
                messages=messages
            )
            
            # Extract formatted output
            formatted_output = formatting_result.get("response", final_response)
            format_type = formatting_result.get("format_type", "structured_text")
            
            logger.info(f"✅ Data formatting complete: {format_type}")
            
            return {
                "final_response": formatted_output,
                "format_type": format_type,
                "formatted": True
            }
            
        except Exception as e:
            logger.error(f"❌ Data formatting failed: {e}")
            # Return original response if formatting fails
            return {
                "final_response": state.get("final_response", ""),
                "formatted": False,
                "formatting_error": str(e)
            }
    
    async def process(self, query: str, metadata: Dict[str, Any] = None, messages: List[Any] = None) -> Dict[str, Any]:
        """
        Process research request with follow-up detection for quick answer short-circuit
        
        Args:
            query: User query string
            metadata: Optional metadata dictionary (user_id, conversation_id, etc.)
            messages: Optional conversation history
            
        Returns:
            Dictionary with research response
        """
        try:
            metadata = metadata or {}
            messages = messages or []
            
            # Extract user_id and conversation_id from metadata
            user_id = metadata.get("user_id", "system")
            conversation_id = metadata.get("conversation_id")
            
            logger.info(f"Research Agent processing: {query[:80]}...")
            
            # Get workflow and checkpoint config for follow-up detection
            workflow = await self._get_workflow()
            config = self._get_checkpoint_config(metadata)
            
            # Load shared_memory from checkpoint if available
            checkpoint_state = await workflow.aget_state(config)
            existing_shared_memory = {}
            if checkpoint_state and checkpoint_state.values:
                existing_shared_memory = checkpoint_state.values.get("shared_memory", {})
            
            # Merge with any shared_memory from metadata
            shared_memory = metadata.get("shared_memory", {}) or {}
            shared_memory.update(existing_shared_memory)
            
            # Check if this is a follow-up to a quick answer
            skip_quick_answer = False
            try:
                if checkpoint_state and checkpoint_state.values:
                    previous_quick_answer = checkpoint_state.values.get("quick_answer_provided", False)
                    if previous_quick_answer:
                        # Check if current query is an affirmative response to the quick answer offer
                        query_lower = query.lower().strip()
                        affirmative_keywords = [
                            "yes", "y", "ok", "okay", "sure", "go ahead", "proceed",
                            "do it", "search more", "deeper search", "more information",
                            "find more", "tell me more", "more details", "search deeper"
                        ]
                        
                        # Check if query is short and affirmative (likely a follow-up)
                        is_affirmative = (
                            any(keyword in query_lower for keyword in affirmative_keywords) and
                            len(query_lower.split()) <= 5
                        ) or any(phrase in query_lower for phrase in [
                            "do a deeper search", "perform a deeper search",
                            "search for more", "find more information"
                        ])
                        
                        if is_affirmative:
                            logger.info("Follow-up detected: User requested deeper research after quick answer")
                            skip_quick_answer = True
            except Exception as e:
                logger.debug(f"Could not check checkpoint state for follow-up detection: {e}")
            
            # Call research method with skip_quick_answer flag and shared_memory
            result = await self.research(
                query=query,
                conversation_id=conversation_id,
                skip_quick_answer=skip_quick_answer,
                shared_memory=shared_memory
            )
            
            # Format response in standard agent format
            final_response = result.get("final_response", "")
            research_complete = result.get("research_complete", False)
            
            return {
                "task_status": "complete" if research_complete else "incomplete",
                "response": final_response,
                "agent_type": "full_research_agent",
                "sources_used": result.get("sources_used", []),
                "quick_answer_provided": result.get("quick_answer_provided", False)
            }
            
        except Exception as e:
            logger.error(f"Research Agent process failed: {e}")
            return {
                "task_status": "error",
                "response": f"Research failed: {str(e)}",
                "error_message": str(e)
            }
    
    async def research(self, query: str, conversation_id: str = None, skip_quick_answer: bool = False, shared_memory: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute complete research workflow
        
        Args:
            query: Research query
            conversation_id: Optional conversation ID for caching
            skip_quick_answer: If True, skip quick answer check and proceed directly to full research
            shared_memory: Optional shared memory dictionary for cross-agent communication
            
        Returns:
            Complete research results with answer and metadata
        """
        try:
            logger.info(f"Starting sophisticated research for: {query}")
            
            # Initialize state
            initial_state = {
                "query": query,
                "original_query": query,
                "expanded_queries": [],
                "key_entities": [],
                "messages": [HumanMessage(content=query)],
                "shared_memory": shared_memory or {},
                "quick_answer_provided": False,
                "quick_answer_content": "",
                "skip_quick_answer": skip_quick_answer,
                "current_round": "",
                "cache_hit": False,
                "cached_context": "",
                "round1_results": {},
                "round1_sufficient": False,
                "round1_assessment": {},
                "gap_analysis": {},
                "identified_gaps": [],
                "round2_results": {},
                "round2_sufficient": False,
                "web_round1_results": {},
                "web_round1_sufficient": False,
                "web_round1_assessment": {},
                "web_gap_analysis": {},
                "web_identified_gaps": [],
                "web_round2_results": {},
                "web_search_results": {},  # Legacy field
                "web_permission_granted": False,
                "query_type": None,
                "query_type_detection": {},
                "should_present_options": False,
                "num_options": None,
                "final_response": "",
                "citations": [],
                "sources_used": [],
                "research_complete": False,
                "routing_recommendation": None,
                "error": ""
            }
            
            # Get workflow and checkpoint config
            workflow = await self._get_workflow()
            
            # Create metadata for checkpoint config
            metadata = {"conversation_id": conversation_id} if conversation_id else {}
            config = self._get_checkpoint_config(metadata)
            
            # Run workflow with checkpointing
            result = await workflow.ainvoke(initial_state, config=config)
            
            logger.info("Research workflow complete")
            
            return result
            
        except Exception as e:
            logger.error(f"Research failed: {e}")
            return {
                "query": query,
                "final_response": f"Research failed: {str(e)}",
                "research_complete": True,
                "error": str(e)
            }


# Global agent instance
_full_research_agent = None


def get_full_research_agent() -> FullResearchAgent:
    """Get or create global research agent instance"""
    global _full_research_agent
    if _full_research_agent is None:
        _full_research_agent = FullResearchAgent()
    return _full_research_agent

