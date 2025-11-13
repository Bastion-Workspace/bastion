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
from orchestrator.models import ResearchAssessmentResult, ResearchGapAnalysis
from config.settings import settings

logger = logging.getLogger(__name__)


class ResearchRound(str, Enum):
    """Research round tracking"""
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
    
    # Final synthesis
    final_response: str
    citations: List[Dict[str, Any]]
    sources_used: List[str]
    routing_recommendation: Optional[str]
    
    # Control
    research_complete: bool
    error: str


class FullResearchAgent:
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
        self.llm = ChatOpenAI(
            model=settings.DEFAULT_MODEL,
            openai_api_key=settings.OPENROUTER_API_KEY,
            openai_api_base=settings.OPENROUTER_BASE_URL,
            temperature=0.7
        )
        
        self.synthesis_llm = ChatOpenAI(
            model=settings.DEFAULT_MODEL,
            openai_api_key=settings.OPENROUTER_API_KEY,
            openai_api_base=settings.OPENROUTER_BASE_URL,
            temperature=0.3  # Lower temp for synthesis
        )
        
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build sophisticated multi-round research workflow"""
        
        workflow = StateGraph(ResearchState)
        
        # Add nodes for each stage
        workflow.add_node("cache_check", self._cache_check_node)
        workflow.add_node("query_expansion", self._query_expansion_node)
        workflow.add_node("round1_local_search", self._round1_local_search_node)
        workflow.add_node("assess_round1", self._assess_round1_node)
        workflow.add_node("gap_analysis", self._gap_analysis_node)
        workflow.add_node("round2_gap_filling", self._round2_gap_filling_node)
        workflow.add_node("web_round1", self._web_round1_node)
        workflow.add_node("assess_web_round1", self._assess_web_round1_node)
        workflow.add_node("gap_analysis_web", self._gap_analysis_web_node)
        workflow.add_node("web_round2", self._web_round2_node)
        workflow.add_node("final_synthesis", self._final_synthesis_node)
        
        # Entry point
        workflow.set_entry_point("cache_check")
        
        # Cache check routing
        workflow.add_conditional_edges(
            "cache_check",
            self._route_from_cache,
            {
                "use_cache": "final_synthesis",
                "do_research": "query_expansion"
            }
        )
        
        # Query expansion always goes to round 1
        workflow.add_edge("query_expansion", "round1_local_search")
        
        # Round 1 assessment routing
        workflow.add_conditional_edges(
            "assess_round1",
            self._route_from_round1,
            {
                "sufficient": "final_synthesis",
                "needs_gap_filling": "gap_analysis",
                "needs_web": "web_round1"
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
        
        # Round 2 Local routes to either Web Round 1 or synthesis
        workflow.add_conditional_edges(
            "round2_gap_filling",
            self._route_from_round2,
            {
                "sufficient": "final_synthesis",
                "needs_web": "web_round1"
            }
        )
        
        # Web Round 1 goes to assessment
        workflow.add_edge("web_round1", "assess_web_round1")
        
        # Web Round 1 assessment routes to synthesis or gap analysis
        workflow.add_conditional_edges(
            "assess_web_round1",
            self._route_from_web_round1,
            {
                "sufficient": "final_synthesis",
                "needs_web_gap_analysis": "gap_analysis_web"
            }
        )
        
        # Web gap analysis routes to Round 2 Web or synthesis
        workflow.add_conditional_edges(
            "gap_analysis_web",
            self._route_from_web_gap_analysis,
            {
                "web_round2": "web_round2",
                "sufficient": "final_synthesis"
            }
        )
        
        # Web Round 2 goes to synthesis
        workflow.add_edge("web_round2", "final_synthesis")
        
        # Synthesis is the end
        workflow.add_edge("final_synthesis", END)
        
        # Round 1 goes to assessment
        workflow.add_edge("round1_local_search", "assess_round1")
        
        return workflow.compile()
    
    # ===== Routing Functions =====
    
    def _route_from_cache(self, state: ResearchState) -> str:
        """Route based on cache hit"""
        if state.get("cache_hit") and state.get("cached_context"):
            logger.info("Cache hit - using cached research")
            return "use_cache"
        logger.info("Cache miss - proceeding with research")
        return "do_research"
    
    def _route_from_round1(self, state: ResearchState) -> str:
        """Route based on Round 1 sufficiency with early exit detection"""
        if state.get("round1_sufficient"):
            logger.info("Round 1 sufficient - proceeding to synthesis")
            return "sufficient"
        
        # Check for early exit: empty results or no relevant info
        round1_results = state.get("round1_results", {})
        search_results = round1_results.get("search_results", "")
        round1_assessment = state.get("round1_assessment", {})
        
        # Early exit: Zero or very minimal results
        if not search_results or len(search_results.strip()) < 50:
            logger.info("Round 1: Empty results detected - skipping Round 2 Local, going to Web Round 1")
            return "needs_web"
        
        # Early exit: No relevant information found
        if not round1_assessment.get("has_relevant_info", True):
            logger.info("Round 1: No relevant info detected - skipping Round 2 Local, going to Web Round 1")
            return "needs_web"
        
        # Otherwise, do gap analysis to decide Round 2 Local vs Web
        logger.info("Round 1 insufficient - doing gap analysis")
        return "needs_gap_filling"
    
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
    
    async def _round1_local_search_node(self, state: ResearchState) -> Dict[str, Any]:
        """Round 1: Initial local search"""
        try:
            query = state["query"]
            expanded_queries = state.get("expanded_queries", [query])
            
            logger.info(f"Round 1: Local search with {len(expanded_queries)} queries")
            
            # Search using all query variations
            all_results = []
            for q in expanded_queries[:3]:  # Limit to top 3
                result = await search_documents_tool(query=q, limit=10)
                all_results.append(result)
            
            # Combine results
            combined_results = "\n\n".join(all_results)
            
            return {
                "round1_results": {
                    "search_results": combined_results,
                    "queries_used": expanded_queries[:3],
                    "result_count": len(all_results)
                },
                "current_round": ResearchRound.INITIAL_LOCAL.value
            }
            
        except Exception as e:
            logger.error(f"Round 1 search error: {e}")
            return {
                "round1_results": {"error": str(e)},
                "current_round": ResearchRound.INITIAL_LOCAL.value
            }
    
    async def _assess_round1_node(self, state: ResearchState) -> Dict[str, Any]:
        """Assess Round 1 results quality and sufficiency"""
        try:
            query = state["query"]
            round1_results = state.get("round1_results", {})
            search_results = round1_results.get("search_results", "")
            
            logger.info("Assessing Round 1 results quality")
            
            # Use LLM to assess quality with structured output
            assessment_prompt = f"""Assess the quality and sufficiency of these search results for answering the user's query.

USER QUERY: {query}

SEARCH RESULTS:
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
}}

Example:
{{
    "sufficient": false,
    "has_relevant_info": true,
    "missing_info": ["specific dates of events", "financial details"],
    "confidence": 0.7,
    "reasoning": "Results mention the relationship but lack specific details about interactions and timeline"
}}"""
            
            response = await self.llm.ainvoke([
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
                
                logger.info(f"Round 1 assessment: sufficient={sufficient}, confidence={assessment.confidence}, relevant={assessment.has_relevant_info}")
                logger.info(f"Assessment reasoning: {assessment.reasoning}")
                
            except (json.JSONDecodeError, ValidationError, Exception) as e:
                logger.warning(f"Failed to parse assessment with Pydantic: {e}")
                logger.warning(f"Raw response: {response.content[:500]}")
                # Fallback: conservative assumption that more research needed
                sufficient = False
                logger.info(f"Round 1 assessment (fallback): sufficient={sufficient}")
            
            return {
                "round1_sufficient": sufficient,
                "round1_assessment": {
                    "sufficient": assessment.sufficient if 'assessment' in locals() else sufficient,
                    "has_relevant_info": assessment.has_relevant_info if 'assessment' in locals() else True,
                    "confidence": assessment.confidence if 'assessment' in locals() else 0.5,
                    "reasoning": assessment.reasoning if 'assessment' in locals() else response.content[:200]
                },
                "gap_analysis": {
                    "has_local_gaps": not sufficient,
                    "assessment_text": response.content
                }
            }
            
        except Exception as e:
            logger.error(f"Round 1 assessment error: {e}")
            return {
                "round1_sufficient": False,
                "round1_assessment": {
                    "sufficient": False,
                    "has_relevant_info": False,
                    "confidence": 0.0,
                    "reasoning": f"Assessment error: {str(e)}"
                },
                "gap_analysis": {"has_local_gaps": True}
            }
    
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
            
            response = await self.llm.ainvoke([
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
            web_result = await search_and_crawl_tool(query=query, num_results=10)
            
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
            
            response = await self.llm.ainvoke([
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
            
            response = await self.llm.ainvoke([
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
            web_result = await search_and_crawl_tool(query=search_query, num_results=10)
            
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
            
            # Generate comprehensive answer
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
            
            response = await self.synthesis_llm.ainvoke([
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
            
            return {
                "final_response": final_response,
                "research_complete": True,
                "sources_used": sources_used,
                "current_round": ResearchRound.FINAL_SYNTHESIS.value,
                "routing_recommendation": routing_recommendation,
                "citations": []  # TODO: Implement citation extraction
            }
            
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return {
                "final_response": f"Research completed but synthesis failed: {str(e)}",
                "research_complete": True,
                "error": str(e)
            }
    
    async def research(self, query: str, conversation_id: str = None) -> Dict[str, Any]:
        """
        Execute complete research workflow
        
        Args:
            query: Research query
            conversation_id: Optional conversation ID for caching
            
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
                "final_response": "",
                "citations": [],
                "sources_used": [],
                "research_complete": False,
                "routing_recommendation": None,
                "error": ""
            }
            
            # Run workflow
            result = await self.workflow.ainvoke(initial_state)
            
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

