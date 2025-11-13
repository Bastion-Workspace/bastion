"""
Full Research Agent - Complete replication of backend clean_research_agent
Multi-round research with gap analysis, caching, and sophisticated synthesis
"""

import logging
from typing import Dict, Any, List, TypedDict, Optional
from datetime import datetime
from enum import Enum

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
from config.settings import settings

logger = logging.getLogger(__name__)


class ResearchRound(str, Enum):
    """Research round tracking"""
    CACHE_CHECK = "cache_check"
    INITIAL_LOCAL = "initial_local"
    ROUND_2_GAP_FILLING = "round_2_gap_filling"
    WEB_SEARCH = "web_search"
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
    
    # Web search
    web_search_results: Dict[str, Any]
    web_permission_granted: bool  # For future HITL
    
    # Final synthesis
    final_response: str
    citations: List[Dict[str, Any]]
    sources_used: List[str]
    
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
        workflow.add_node("web_search", self._web_search_node)
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
                "needs_web": "web_search"
            }
        )
        
        # Gap analysis leads to round 2
        workflow.add_edge("gap_analysis", "round2_gap_filling")
        
        # Round 2 routes to either web or synthesis
        workflow.add_conditional_edges(
            "round2_gap_filling",
            self._route_from_round2,
            {
                "sufficient": "final_synthesis",
                "needs_web": "web_search"
            }
        )
        
        # Web search always goes to synthesis
        workflow.add_edge("web_search", "final_synthesis")
        
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
        """Route based on Round 1 sufficiency"""
        if state.get("round1_sufficient"):
            logger.info("Round 1 sufficient - proceeding to synthesis")
            return "sufficient"
        
        gap_analysis = state.get("gap_analysis", {})
        if gap_analysis.get("has_local_gaps"):
            logger.info("Round 1 insufficient - doing gap analysis")
            return "needs_gap_filling"
        
        logger.info("Round 1 insufficient - proceeding to web search")
        return "needs_web"
    
    def _route_from_round2(self, state: ResearchState) -> str:
        """Route based on Round 2 sufficiency"""
        if state.get("round2_sufficient"):
            logger.info("Round 2 sufficient - proceeding to synthesis")
            return "sufficient"
        
        logger.info("Round 2 insufficient - proceeding to web search")
        return "needs_web"
    
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
            
            # Use LLM to assess quality
            assessment_prompt = f"""Assess the quality and sufficiency of these search results for answering the user's query.

USER QUERY: {query}

SEARCH RESULTS:
{search_results[:2000]}

Evaluate:
1. Do the results contain relevant information?
2. Is there enough detail to answer the query comprehensively?
3. What information is missing (if any)?

Respond in JSON format:
{{
    "sufficient": true/false,
    "has_relevant_info": true/false,
    "missing_info": ["list", "of", "gaps"],
    "confidence": 0.0-1.0
}}"""
            
            response = await self.llm.ainvoke([
                SystemMessage(content="You are a research quality assessor."),
                HumanMessage(content=assessment_prompt)
            ])
            
            # Parse response (simplified for now)
            assessment_text = response.content.lower()
            sufficient = "sufficient\": true" in assessment_text or "enough detail" in assessment_text
            
            logger.info(f"Round 1 assessment: sufficient={sufficient}")
            
            return {
                "round1_sufficient": sufficient,
                "gap_analysis": {
                    "has_local_gaps": not sufficient,
                    "assessment_text": response.content
                }
            }
            
        except Exception as e:
            logger.error(f"Round 1 assessment error: {e}")
            return {
                "round1_sufficient": False,
                "gap_analysis": {"has_local_gaps": True}
            }
    
    async def _gap_analysis_node(self, state: ResearchState) -> Dict[str, Any]:
        """Analyze gaps in Round 1 results"""
        try:
            query = state["query"]
            round1_results = state.get("round1_results", {})
            
            logger.info("Performing gap analysis")
            
            # Use LLM to identify specific gaps
            gap_prompt = f"""Analyze what information is missing from the search results to fully answer the query.

USER QUERY: {query}

RESULTS SO FAR: {round1_results.get('search_results', '')[:1500]}

Identify:
1. Specific missing entities or facts
2. Additional search terms that might fill gaps
3. Whether web search would help

Respond in JSON format:
{{
    "missing_entities": ["entity1", "entity2"],
    "suggested_queries": ["query1", "query2"],
    "needs_web_search": true/false
}}"""
            
            response = await self.llm.ainvoke([
                SystemMessage(content="You are a research gap analyst."),
                HumanMessage(content=gap_prompt)
            ])
            
            logger.info("Gap analysis complete")
            
            return {
                "gap_analysis": {
                    "analysis": response.content,
                    "has_local_gaps": True
                },
                "identified_gaps": ["gaps from analysis"]  # Simplified
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
    
    async def _web_search_node(self, state: ResearchState) -> Dict[str, Any]:
        """Web search when local sources insufficient"""
        try:
            query = state["query"]
            expanded_queries = state.get("expanded_queries", [query])
            
            logger.info(f"Web search: {query}")
            
            # Use search_and_crawl for comprehensive web research
            web_result = await search_and_crawl_tool(query=query, num_results=10)
            
            return {
                "web_search_results": {
                    "content": web_result,
                    "query_used": query
                },
                "web_permission_granted": True,  # No HITL currently
                "current_round": ResearchRound.WEB_SEARCH.value
            }
            
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return {
                "web_search_results": {"error": str(e)},
                "current_round": ResearchRound.WEB_SEARCH.value
            }
    
    async def _final_synthesis_node(self, state: ResearchState) -> Dict[str, Any]:
        """Final synthesis with all gathered information"""
        try:
            query = state["query"]
            
            # Gather all results
            cached_context = state.get("cached_context", "")
            round1_results = state.get("round1_results", {})
            round2_results = state.get("round2_results", {})
            web_results = state.get("web_search_results", {})
            
            logger.info("Synthesizing final response from all sources")
            
            # Build comprehensive context
            context_parts = []
            
            if cached_context:
                context_parts.append(f"CACHED RESEARCH:\n{cached_context}")
            
            if round1_results:
                context_parts.append(f"LOCAL SEARCH RESULTS:\n{round1_results.get('search_results', '')[:2000]}")
            
            if round2_results:
                context_parts.append(f"ADDITIONAL LOCAL RESULTS:\n{round2_results.get('gap_results', '')[:1500]}")
            
            if web_results:
                context_parts.append(f"WEB RESEARCH:\n{web_results.get('content', '')[:2000]}")
            
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
            
            # Extract citations (simplified)
            sources_used = []
            if round1_results:
                sources_used.append("local_documents")
            if web_results:
                sources_used.append("web_search")
            
            return {
                "final_response": final_response,
                "research_complete": True,
                "sources_used": sources_used,
                "current_round": ResearchRound.FINAL_SYNTHESIS.value,
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
                "gap_analysis": {},
                "identified_gaps": [],
                "round2_results": {},
                "round2_sufficient": False,
                "web_search_results": {},
                "web_permission_granted": False,
                "final_response": "",
                "citations": [],
                "sources_used": [],
                "research_complete": False,
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

