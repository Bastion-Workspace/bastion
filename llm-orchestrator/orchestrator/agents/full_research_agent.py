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
    get_document_content_tool,
    search_web_tool,
    crawl_web_content_tool,
    expand_query_tool,
    search_conversation_cache_tool
)
from orchestrator.tools.dynamic_tool_analyzer import analyze_tool_needs_for_research
from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.formatting_detection import detect_formatting_need, detect_post_processing_needs
from orchestrator.models import ResearchAssessmentResult, ResearchGapAnalysis, QuickAnswerAssessment, QueryTypeDetection
from orchestrator.agents.base_agent import BaseAgent
from orchestrator.subgraphs import (
    build_research_workflow_subgraph, 
    build_gap_analysis_subgraph,
    build_web_research_subgraph,
    build_assessment_subgraph,
    build_data_formatting_subgraph,
    build_visualization_subgraph
)
from config.settings import settings

logger = logging.getLogger(__name__)


class ResearchRound(str, Enum):
    """Research round tracking"""
    QUICK_ANSWER_CHECK = "quick_answer_check"
    CACHE_CHECK = "cache_check"
    INITIAL_LOCAL = "initial_local"
    ROUND_2_GAP_FILLING = "round_2_gap_filling"  # Legacy - kept for backward compatibility
    ROUND_2_PARALLEL = "round_2_parallel"
    WEB_ROUND_1 = "web_round_1"  # Legacy - kept for backward compatibility
    ASSESS_WEB_ROUND_1 = "assess_web_round_1"  # Legacy - kept for backward compatibility
    GAP_ANALYSIS_WEB = "gap_analysis_web"  # Legacy - kept for backward compatibility
    WEB_ROUND_2 = "web_round_2"  # Legacy - kept for backward compatibility
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
    quick_vector_results: List[Dict[str, Any]]  # Vector search results from quick check
    quick_vector_relevance: Optional[str]  # "high" | "medium" | "low" | "none"
    
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
    
    # Post-processing recommendations
    formatting_recommendations: Optional[Dict[str, Any]]
    visualization_results: Optional[Dict[str, Any]]
    formatted_output: Optional[str]
    format_type: Optional[str]
    
    # Full document analysis
    full_doc_analysis_needed: bool
    document_ids_to_analyze: List[str]
    analysis_queries: List[str]
    full_doc_insights: Dict[str, Any]
    documents_analyzed: List[str]
    full_doc_decision_reasoning: str
    
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
        super().__init__("research_agent")
        # LLMs will be created lazily using _get_llm() to respect user model preferences
        self._research_subgraphs = {}  # Cache subgraphs by (skip_cache, skip_expansion) config
        self._full_doc_analysis_subgraph = None  # Full document analysis subgraph
        self._web_research_subgraph = None  # Web research subgraph
        self._assessment_subgraph = None  # Assessment subgraph
    
    def _get_research_subgraph(self, checkpointer, skip_cache: bool = False, skip_expansion: bool = False):
        """Get or build research workflow subgraph"""
        # For Round 2, we need a different subgraph config (skip cache, skip expansion)
        cache_key = (skip_cache, skip_expansion)
        if not hasattr(self, '_research_subgraphs'):
            self._research_subgraphs = {}
        
        if cache_key not in self._research_subgraphs:
            self._research_subgraphs[cache_key] = build_research_workflow_subgraph(
                checkpointer, 
                skip_cache=skip_cache, 
                skip_expansion=skip_expansion
            )
        return self._research_subgraphs[cache_key]
    
    def _get_web_research_subgraph(self, checkpointer):
        """Get or build web research subgraph"""
        if self._web_research_subgraph is None:
            self._web_research_subgraph = build_web_research_subgraph(checkpointer)
        return self._web_research_subgraph
    
    def _get_assessment_subgraph(self, checkpointer):
        """Get or build assessment subgraph"""
        if self._assessment_subgraph is None:
            self._assessment_subgraph = build_assessment_subgraph(checkpointer)
        return self._assessment_subgraph
    
    def _get_data_formatting_subgraph(self, checkpointer):
        """Get or build data formatting subgraph"""
        if not hasattr(self, '_data_formatting_subgraph') or self._data_formatting_subgraph is None:
            self._data_formatting_subgraph = build_data_formatting_subgraph(checkpointer)
        return self._data_formatting_subgraph
    
    def _get_visualization_subgraph(self, checkpointer):
        """Get or build visualization subgraph"""
        if not hasattr(self, '_visualization_subgraph') or self._visualization_subgraph is None:
            self._visualization_subgraph = build_visualization_subgraph(checkpointer)
        return self._visualization_subgraph
    
    async def _call_research_subgraph_node(self, state: ResearchState) -> Dict[str, Any]:
        """Call research workflow subgraph to perform core research"""
        try:
            logger.info("🔬 Calling research workflow subgraph")
            
            workflow = await self._get_workflow()
            checkpointer = workflow.checkpointer
            research_sg = self._get_research_subgraph(checkpointer, skip_cache=False, skip_expansion=False)
            
            # Prepare subgraph state from agent state - ensure query is explicitly passed
            query = state.get("query", "")
            messages = state.get("messages", [])
            shared_memory = state.get("shared_memory", {})
            subgraph_state = {
                "query": query,
                "original_query": query,
                "shared_memory": shared_memory,
                "messages": messages,
                "user_id": shared_memory.get("user_id", "system"),  # Get user_id from shared_memory
                "metadata": state.get("metadata", {})
            }
            
            logger.info(f"🔬 Passing query to research subgraph: '{query[:100]}'")
            logger.info(f"🔍 DEBUG: Passing {len(messages)} messages to research subgraph")
            
            # Run subgraph
            config = self._get_checkpoint_config(state.get("metadata", {}))
            result = await research_sg.ainvoke(subgraph_state, config)
            
            logger.info("✅ Research subgraph completed")
            
            # Map subgraph results back to ResearchState format
            research_findings = result.get("research_findings", {})
            sources_found = result.get("sources_found", [])
            citations = result.get("citations", [])
            research_sufficient = result.get("research_sufficient", False)
            round1_sufficient = result.get("round1_sufficient", False)
            
            # CRITICAL: First try to get round1_results and web_round1_results directly from subgraph result
            # (they should be preserved by synthesize_findings_node)
            subgraph_round1_results = result.get("round1_results", {})
            subgraph_web_round1_results = result.get("web_round1_results", {})
            
            # Extract round1 results - prefer direct from subgraph, fallback to research_findings
            if subgraph_round1_results and subgraph_round1_results.get("search_results"):
                round1_results = subgraph_round1_results
                logger.info(f"✅ Using round1_results from subgraph: {len(round1_results.get('search_results', ''))} chars")
            else:
                round1_results = {
                    "search_results": research_findings.get("local_results", ""),
                    "documents_found": len([s for s in sources_found if s.get("source") == "local"]),
                    "round1_document_ids": [s.get("document_id") for s in sources_found if s.get("source") == "local" and s.get("document_id")]
                }
                logger.info(f"⚠️ Using fallback round1_results from research_findings: {len(round1_results.get('search_results', ''))} chars")
            
            # Extract web_round1_results - prefer direct from subgraph, fallback to research_findings
            if subgraph_web_round1_results and subgraph_web_round1_results.get("content"):
                web_round1_results = subgraph_web_round1_results
                logger.info(f"✅ Using web_round1_results from subgraph: {len(web_round1_results.get('content', ''))} chars")
            else:
                web_round1_results = {
                    "content": research_findings.get("web_results", ""),
                    "sources_found": [s for s in sources_found if s.get("source") == "web"]
                }
                logger.info(f"⚠️ Using fallback web_round1_results from research_findings: {len(web_round1_results.get('content', ''))} chars")
            
            # Extract round1 assessment if available
            round1_assessment = result.get("round1_assessment", {})
            
            # Extract gap analysis if available (from subgraph's gap_analysis node)
            gap_analysis = result.get("gap_analysis", {})
            identified_gaps = result.get("identified_gaps", [])
            
            # Check if we had a cache hit
            cache_hit = result.get("cache_hit", False)
            cached_context = result.get("cached_context", "")
            
            return {
                "round1_results": round1_results,
                "web_round1_results": web_round1_results,
                "sources_found": sources_found,
                "citations": citations,
                "round1_sufficient": round1_sufficient or research_sufficient,
                "round1_assessment": round1_assessment,
                "gap_analysis": gap_analysis,
                "identified_gaps": identified_gaps,
                "cache_hit": cache_hit,
                "cached_context": cached_context,
                "current_round": ResearchRound.ROUND_2_PARALLEL.value if not (round1_sufficient or research_sufficient) else ResearchRound.FINAL_SYNTHESIS.value,
                "research_findings": research_findings
            }
            
        except Exception as e:
            logger.error(f"❌ Research subgraph error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "round1_results": {"error": str(e), "search_results": ""},
                "web_round1_results": {"error": str(e), "content": ""},
                "sources_found": [],
                "citations": [],
                "round1_sufficient": False,
                "cache_hit": False,
                "cached_context": "",
                "current_round": ResearchRound.ROUND_2_PARALLEL.value,
                "error": str(e)
            }
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """Build sophisticated multi-round research workflow"""
        
        workflow = StateGraph(ResearchState)
        
        # Add quick answer check node (first node - entry point)
        workflow.add_node("quick_answer_check", self._quick_answer_check_node)
        
        # Add research subgraph node (replaces cache_check, query_expansion, round1_parallel_search, assess_combined_round1, gap_analysis)
        workflow.add_node("research_subgraph", self._call_research_subgraph_node)
        
        # Add full document analysis nodes
        workflow.add_node("full_doc_analysis_decision", self._full_document_analysis_decision_node)
        workflow.add_node("full_doc_analysis_subgraph", self._call_full_document_analysis_subgraph_node)
        workflow.add_node("gap_analysis_check", self._gap_analysis_check_node)
        
        # Add nodes for additional research rounds (unique to FullResearchAgent)
        workflow.add_node("round2_parallel", self._round2_parallel_node)
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
                "full_research": "research_subgraph"  # Continue with research subgraph
            }
        )
        
        # Research subgraph routing - check if cache hit or if we need more research
        workflow.add_conditional_edges(
            "research_subgraph",
            self._route_from_research_subgraph,
            {
                "use_cache": "detect_query_type",  # Cache hit - go straight to synthesis
                "sufficient": "full_doc_analysis_decision",  # Research sufficient - check if full docs needed
                "needs_round2": "full_doc_analysis_decision"  # Need Round 2 - check if full docs needed first
            }
        )
        
        # Full document analysis decision routing
        workflow.add_conditional_edges(
            "full_doc_analysis_decision",
            self._route_from_full_doc_decision,
            {
                "analyze_full_docs": "full_doc_analysis_subgraph",  # Need full doc analysis
                "skip_full_docs": "gap_analysis_check"  # Skip full doc analysis
            }
        )
        
        # Full document analysis subgraph always goes to gap analysis check
        workflow.add_edge("full_doc_analysis_subgraph", "gap_analysis_check")
        
        # Gap analysis check - determine if we need round 2 or can proceed to synthesis
        workflow.add_conditional_edges(
            "gap_analysis_check",
            self._route_from_gap_analysis_check,
            {
                "needs_round2": "round2_parallel",  # Need Round 2
                "proceed_to_synthesis": "detect_query_type"  # Can proceed to synthesis
            }
        )
        
        # Round 2 Parallel always proceeds to query type detection then synthesis
        workflow.add_edge("round2_parallel", "detect_query_type")
        
        # Query type detection always goes to synthesis
        workflow.add_edge("detect_query_type", "final_synthesis")
        
        # Add post-processing node (for formatting and/or visualization)
        workflow.add_node("post_process", self._post_process_results_node)
        
        # After synthesis, check if post-processing is needed
        workflow.add_conditional_edges(
            "final_synthesis",
            self._route_from_synthesis,
            {
                "post_process": "post_process",
                "complete": END
            }
        )
        
        # Post-processing node goes to end
        workflow.add_edge("post_process", END)
        
        return workflow.compile(checkpointer=checkpointer)
    
    # ===== Routing Functions =====
    
    def _route_from_research_subgraph(self, state: ResearchState) -> str:
        """Route after research subgraph completes"""
        # Check if we had a cache hit
        if state.get("cache_hit") and state.get("cached_context"):
            logger.info("Research subgraph: Cache hit - using cached research")
            return "use_cache"
        
        # Check if research is sufficient
        if state.get("round1_sufficient"):
            logger.info("Research subgraph: Research sufficient - checking if full docs needed")
            return "sufficient"
        
        # Check gap analysis flags (both are independent)
        gap_analysis = state.get("gap_analysis", {})
        needs_local = gap_analysis.get("needs_local_search", False)
        needs_web = gap_analysis.get("needs_web_search", False)
        
        # If either local or web is needed, check if full docs needed first
        if needs_local or needs_web:
            logger.info(f"Research subgraph: Needs Round 2 (local={needs_local}, web={needs_web}) - checking if full docs needed")
            return "needs_round2"
        
        # Default: proceed to full doc analysis decision
        logger.info("Research subgraph: No additional searches needed - checking if full docs needed")
        return "sufficient"
    
    def _route_from_full_doc_decision(self, state: ResearchState) -> str:
        """Route after full document analysis decision"""
        if state.get("full_doc_analysis_needed"):
            logger.info("Full doc decision: Will analyze full documents")
            return "analyze_full_docs"
        else:
            logger.info("Full doc decision: Skipping full document analysis")
            return "skip_full_docs"
    
    def _route_from_gap_analysis_check(self, state: ResearchState) -> str:
        """Route after gap analysis check (with or without full doc insights)"""
        # Check gap analysis flags
        gap_analysis = state.get("gap_analysis", {})
        needs_local = gap_analysis.get("needs_local_search", False)
        needs_web = gap_analysis.get("needs_web_search", False)
        
        # If either local or web is needed, go to parallel Round 2
        if needs_local or needs_web:
            logger.info(f"Gap analysis check: Needs Round 2 (local={needs_local}, web={needs_web})")
            return "needs_round2"
        
        # Default: proceed to synthesis
        logger.info("Gap analysis check: No additional searches needed - proceeding to synthesis")
        return "proceed_to_synthesis"
    
    # ===== Workflow Nodes =====
    
    async def _full_document_analysis_decision_node(self, state: ResearchState) -> Dict[str, Any]:
        """
        Hybrid decision: rules + LLM to determine if full docs needed
        
        Phase 1: Fast rule-based pre-screening
        Phase 2: LLM decision (only if rules pass) to generate targeted queries
        """
        try:
            query = state.get("query", "")
            round1_results = state.get("round1_results", {})
            sources_found = state.get("sources_found", [])
            
            # Configuration constants
            MIN_QUALITY_CHUNKS = 3
            MIN_CHUNKS_PER_DOC = 3
            MIN_CHUNK_SCORE = 0.7
            MAX_DOC_TOKENS = 100000  # 100k tokens
            LLM_CONFIDENCE_THRESHOLD = 0.6
            MAX_DOCS_TO_ANALYZE = 2
            MAX_PARALLEL_QUERIES = 4
            
            logger.info("🔍 Evaluating if full document analysis is needed")
            
            # ============================================
            # PHASE 1: Fast Rule-Based Pre-Screening
            # ============================================
            
            # Skip if query is too short/simple
            if len(query.split()) < 5:
                logger.info("❌ Skip full doc: Query too simple (< 5 words)")
                return {"full_doc_analysis_needed": False}
            
            # Extract document IDs from sources
            document_ids = []
            for source in sources_found:
                if source.get("type") == "document" and source.get("document_id"):
                    document_ids.append(source.get("document_id"))
            
            if not document_ids:
                logger.info("❌ Skip full doc: No document IDs found in sources")
                return {"full_doc_analysis_needed": False}
            
            # Get chunk information from round1_results
            # Try to extract from round1_document_ids and re-query for chunk details
            round1_document_ids = round1_results.get("round1_document_ids", [])
            if not round1_document_ids:
                # Fallback to sources
                round1_document_ids = document_ids[:5]
            
            # Re-query to get chunk details for analysis
            # We need to check if documents have multiple high-quality chunks
            from orchestrator.subgraphs.intelligent_document_retrieval_subgraph import retrieve_documents_intelligently
            
            shared_memory = state.get("shared_memory", {})
            user_id = shared_memory.get("user_id", "system") if shared_memory else "system"
            
            # Quick retrieval to get chunk structure
            try:
                chunk_result = await retrieve_documents_intelligently(
                    query=query,
                    user_id=user_id,
                    mode="fast",
                    max_results=5,
                    small_doc_threshold=15000
                )
                
                retrieved_docs = chunk_result.get("retrieved_documents", [])
                
                if not retrieved_docs:
                    logger.info("❌ Skip full doc: No documents retrieved")
                    return {"full_doc_analysis_needed": False}
                
                # Analyze chunk patterns
                doc_chunk_counts = {}
                high_quality_chunks = []
                cross_ref_signals = []
                
                for doc in retrieved_docs:
                    doc_id = doc.get("document_id")
                    if not doc_id:
                        continue
                    
                    # Check retrieval strategy
                    strategy = doc.get("retrieval_strategy", "")
                    
                    # If full_document, it's already small - skip
                    if strategy == "full_document":
                        continue
                    
                    # If multi_chunk, count chunks
                    if strategy == "multi_chunk":
                        chunks = doc.get("chunks", [])
                        high_quality = [c for c in chunks if c.get("relevance_score", 0.0) >= MIN_CHUNK_SCORE]
                        if len(high_quality) >= MIN_CHUNKS_PER_DOC:
                            doc_chunk_counts[doc_id] = len(high_quality)
                            high_quality_chunks.extend(high_quality)
                            
                            # Check for cross-reference signals
                            for chunk in high_quality:
                                content = chunk.get("content", "").lower()
                                if any(signal in content for signal in ["see section", "as discussed", "mentioned earlier", "refer to chapter", "in part"]):
                                    cross_ref_signals.append(doc_id)
                                    break
                
                # RULE: Need at least one document with 3+ high-quality chunks
                promising_docs = {
                    doc_id: count for doc_id, count in doc_chunk_counts.items() 
                    if count >= MIN_CHUNKS_PER_DOC
                }
                
                if not promising_docs:
                    logger.info("❌ Skip full doc: No document with 3+ quality chunks")
                    return {"full_doc_analysis_needed": False}
                
                # Check document sizes (skip if > 100k tokens)
                from orchestrator.backend_tool_client import get_backend_tool_client
                client = await get_backend_tool_client()
                
                reasonable_docs = {}
                for doc_id in promising_docs.keys():
                    try:
                        doc_size = await client.get_document_size(doc_id, user_id)
                        estimated_tokens = doc_size // 4  # Rough estimate
                        if estimated_tokens < MAX_DOC_TOKENS:
                            reasonable_docs[doc_id] = promising_docs[doc_id]
                    except Exception as e:
                        logger.warning(f"Could not check size for doc {doc_id}: {e}")
                        # Assume reasonable if we can't check
                        reasonable_docs[doc_id] = promising_docs[doc_id]
                
                if not reasonable_docs:
                    logger.info("❌ Skip full doc: All promising docs too large")
                    return {"full_doc_analysis_needed": False}
                
                # Check total high-quality chunks
                if len(high_quality_chunks) < MIN_QUALITY_CHUNKS:
                    logger.info(f"❌ Skip full doc: Too few quality chunks ({len(high_quality_chunks)} < {MIN_QUALITY_CHUNKS})")
                    return {"full_doc_analysis_needed": False}
                
            except Exception as e:
                logger.warning(f"Error checking chunk patterns: {e}, skipping full doc analysis")
                return {"full_doc_analysis_needed": False}
            
            # ============================================
            # PHASE 2: LLM Decision (only if rules passed)
            # ============================================
            logger.info(f"✅ Rules passed: {len(reasonable_docs)} promising docs, asking LLM...")
            
            # Build context for LLM
            chunk_preview = "\n\n".join([
                f"[Doc {c.get('document_id', 'unknown')[:8]}] Score: {c.get('relevance_score', 0.0):.2f}\n{c.get('content', '')[:200]}..."
                for c in high_quality_chunks[:5]
            ])
            
            top_doc_id = max(reasonable_docs.items(), key=lambda x: x[1])[0]
            top_doc_count = reasonable_docs[top_doc_id]
            
            decision_prompt = f"""You are evaluating whether to retrieve full documents for deeper analysis.

QUERY: {query}

CHUNK ANALYSIS:
- Found {len(high_quality_chunks)} high-quality chunks across {len(reasonable_docs)} documents
- Top document has {top_doc_count} relevant chunks
- Cross-reference signals found: {len(cross_ref_signals)} document(s)

CHUNK PREVIEW:
{chunk_preview}

EVALUATION CRITERIA:
1. Do chunks appear fragmentary or incomplete?
2. Does the query require synthesis across sections? (e.g., "relationship between X and Y", "how did X evolve")
3. Would full document context significantly improve answer quality?
4. Are chunks referencing other sections ("see chapter X", "as discussed earlier")?

Respond with ONLY valid JSON matching this exact schema:
{{
    "should_retrieve_full_docs": boolean,
    "confidence": number (0.0-1.0),
    "reasoning": "brief explanation",
    "suggested_queries": ["query1", "query2", "query3"]  // 3-4 specific questions to ask of full docs
}}

Examples where full docs help:
- "What was the relationship between Dan Reingold and Bernie Ebbers?" (needs full narrative)
- "How did the author's perspective evolve throughout the book?" (needs full arc)
- "Compare the methodology in sections 2 and 5" (needs cross-referencing)

Examples where chunks are sufficient:
- "What is the definition of X?" (factual, single answer)
- "List the key findings" (can be extracted from chunks)
- "What year did X happen?" (simple fact)

Your decision:"""
            
            llm = self._get_llm(temperature=0.3, state=state)
            datetime_context = self._get_datetime_context()
            
            decision_messages = [
                SystemMessage(content="You are a document analysis decision maker. Always respond with valid JSON matching the exact schema provided."),
                SystemMessage(content=datetime_context)
            ]
            
            # Include conversation history if available
            conversation_messages = state.get("messages", [])
            if conversation_messages:
                decision_messages.extend(conversation_messages)
            
            decision_messages.append(HumanMessage(content=decision_prompt))
            
            response = await llm.ainvoke(decision_messages)
            
            # Parse LLM response
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
                
                decision = json.loads(text)
                
                if decision.get("should_retrieve_full_docs") and decision.get("confidence", 0.0) >= LLM_CONFIDENCE_THRESHOLD:
                    suggested_queries = decision.get("suggested_queries", [])
                    if not suggested_queries:
                        # Fallback: generate queries from original query
                        suggested_queries = [
                            f"What specific details about {query}?",
                            f"How does this relate to the main topic?",
                            f"What context is missing from the chunks?"
                        ]
                    
                    # Select top documents (by chunk count)
                    sorted_docs = sorted(reasonable_docs.items(), key=lambda x: x[1], reverse=True)
                    doc_ids_to_analyze = [doc_id for doc_id, _ in sorted_docs[:MAX_DOCS_TO_ANALYZE]]
                    
                    logger.info(f"✅ LLM decided YES (confidence={decision.get('confidence', 0.0):.2f}): {decision.get('reasoning', '')}")
                    logger.info(f"📄 Will analyze {len(doc_ids_to_analyze)} documents with {len(suggested_queries[:MAX_PARALLEL_QUERIES])} queries")
                    
                    return {
                        "full_doc_analysis_needed": True,
                        "document_ids_to_analyze": doc_ids_to_analyze,
                        "analysis_queries": suggested_queries[:MAX_PARALLEL_QUERIES],
                        "full_doc_decision_reasoning": decision.get("reasoning", "")
                    }
                else:
                    logger.info(f"❌ LLM decided NO (confidence={decision.get('confidence', 0.0):.2f}): {decision.get('reasoning', '')}")
                    return {"full_doc_analysis_needed": False}
                    
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"LLM decision parsing failed: {e}, defaulting to NO")
                return {"full_doc_analysis_needed": False}
            
        except Exception as e:
            logger.error(f"Full document analysis decision error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"full_doc_analysis_needed": False}
    
    async def _gap_analysis_check_node(self, state: ResearchState) -> Dict[str, Any]:
        """Simple pass-through node to check gap analysis after full doc analysis"""
        # This node just passes through state - routing logic handles the decision
        return {}
    
    async def _call_full_document_analysis_subgraph_node(self, state: ResearchState) -> Dict[str, Any]:
        """Call full document analysis subgraph to analyze full documents"""
        try:
            logger.info("📚 Calling full document analysis subgraph")
            
            document_ids = state.get("document_ids_to_analyze", [])
            analysis_queries = state.get("analysis_queries", [])
            original_query = state.get("query", "")
            
            if not document_ids or not analysis_queries:
                logger.warning("Missing required inputs for full document analysis")
                return {
                    "full_doc_insights": {},
                    "documents_analyzed": [],
                    "synthesis": ""
                }
            
            workflow = await self._get_workflow()
            checkpointer = workflow.checkpointer
            
            # Get or build full document analysis subgraph
            if not hasattr(self, '_full_doc_analysis_subgraph') or self._full_doc_analysis_subgraph is None:
                from orchestrator.subgraphs import build_full_document_analysis_subgraph
                self._full_doc_analysis_subgraph = build_full_document_analysis_subgraph(checkpointer)
            
            # Prepare subgraph state
            shared_memory = state.get("shared_memory", {})
            user_id = shared_memory.get("user_id", "system") if shared_memory else "system"
            
            subgraph_state = {
                "document_ids": document_ids,
                "analysis_queries": analysis_queries,
                "original_query": original_query,
                "chunk_context": [],  # Could pass chunk results if needed
                "user_id": user_id,
                "messages": state.get("messages", []),
                "metadata": state.get("metadata", {})
            }
            
            logger.info(f"📚 Analyzing {len(document_ids)} documents with {len(analysis_queries)} queries")
            
            # Run subgraph
            config = self._get_checkpoint_config(state.get("metadata", {}))
            result = await self._full_doc_analysis_subgraph.ainvoke(subgraph_state, config)
            
            logger.info("✅ Full document analysis subgraph completed")
            
            return {
                "full_doc_insights": result.get("full_doc_insights", {}),
                "documents_analyzed": result.get("documents_analyzed", []),
                "synthesis": result.get("synthesis", "")
            }
            
        except Exception as e:
            logger.error(f"❌ Full document analysis subgraph error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "full_doc_insights": {},
                "documents_analyzed": [],
                "synthesis": ""
            }
    
    async def _quick_vector_search(self, query: str, limit: int = 8, user_id: str = "system") -> List[Dict[str, Any]]:
        """
        Perform fast vector search for quick answer context using intelligent retrieval subgraph
        
        Args:
            query: Search query
            limit: Maximum number of results (default: 8 for speed)
            user_id: User ID for searching user-specific documents (default: "system")
            
        Returns:
            List of document results with scores, or empty list on error
        """
        try:
            import asyncio
            from orchestrator.subgraphs.intelligent_document_retrieval_subgraph import retrieve_documents_intelligently
            
            # Use intelligent retrieval subgraph with fast mode and timeout
            try:
                result = await asyncio.wait_for(
                    retrieve_documents_intelligently(
                        query=query,
                        user_id=user_id,
                        mode="fast",  # Quick retrieval for quick answer check
                        max_results=limit,
                        small_doc_threshold=5000
                    ),
                    timeout=2.0
                )
                
                # Extract documents from subgraph result
                retrieved_docs = result.get("retrieved_documents", [])
                
                # Format results for relevance analysis (maintain compatibility)
                formatted_results = []
                for doc in retrieved_docs:
                    formatted_results.append({
                        'document_id': doc.get('document_id'),
                        'title': doc.get('title', doc.get('filename', 'Unknown')),
                        'filename': doc.get('filename', ''),
                        'content_preview': doc.get('content_preview', ''),
                        'relevance_score': doc.get('relevance_score', 0.0),
                        'metadata': doc.get('metadata', {})
                    })
                
                logger.info(f"Quick vector search found {len(formatted_results)} results via intelligent retrieval")
                return formatted_results
                
            except asyncio.TimeoutError:
                logger.warning("Quick vector search timed out after 2 seconds - falling back to basic search")
                # Fallback to basic search
                from orchestrator.tools import search_documents_structured
                search_result = await search_documents_structured(query=query, limit=limit, user_id=user_id)
                results = search_result.get('results', [])
                formatted_results = []
                for doc in results:
                    formatted_results.append({
                        'document_id': doc.get('document_id'),
                        'title': doc.get('title', doc.get('filename', 'Unknown')),
                        'filename': doc.get('filename', ''),
                        'content_preview': doc.get('content_preview', ''),
                        'relevance_score': doc.get('relevance_score', 0.0),
                        'metadata': doc.get('metadata', {})
                    })
                return formatted_results
            
        except Exception as e:
            logger.warning(f"Quick vector search failed: {e} - continuing without vector results")
            return []
    
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
                    "current_round": ResearchRound.QUICK_ANSWER_CHECK.value,
                    "quick_vector_results": [],
                    "quick_vector_relevance": None
                }
            
            logger.info(f"Quick answer check for: {query}")
            
            # Run LLM evaluation and vector search in parallel for speed
            import asyncio
            
            # Prepare LLM evaluation prompt
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
            datetime_context = self._get_datetime_context()
            
            # Check for agent handoff context
            shared_memory = state.get("shared_memory", {})
            handoff_context = shared_memory.get("handoff_context", {})
            handoff_note = ""
            
            if handoff_context:
                source_agent = handoff_context.get("source_agent", "unknown")
                reference_doc = handoff_context.get("reference_document", {})
                
                if reference_doc.get("has_content"):
                    ref_content = reference_doc.get("content", "")[:1000]  # First 1000 chars for context
                    ref_filename = reference_doc.get("filename", "unknown")
                    handoff_note = f"""

**AGENT HANDOFF CONTEXT**:
- Delegated by: {source_agent}
- User has reference document: {ref_filename}
- Document preview: {ref_content}{"..." if len(reference_doc.get("content", "")) > 1000 else ""}

When answering, you can reference data from the user's document above."""
                    logger.info(f"🔗 Handoff context detected from {source_agent}")
            
            # Build messages including conversation history for context
            messages_for_llm = [
                SystemMessage(content="You are a query evaluator. Always respond with valid JSON matching the exact schema provided."),
                SystemMessage(content=datetime_context)
            ]
            
            # Include conversation history if available (critical for follow-up queries)
            conversation_messages = state.get("messages", [])
            if conversation_messages:
                # Add conversation history so LLM has context
                messages_for_llm.extend(conversation_messages)
                logger.info(f"🔍 Including {len(conversation_messages)} conversation messages for context")
            
            # Add the evaluation prompt with handoff context
            full_prompt = evaluation_prompt + handoff_note
            messages_for_llm.append(HumanMessage(content=full_prompt))
            
            # Run LLM evaluation and vector search in parallel
            async def llm_evaluation_task():
                """LLM evaluation task"""
                return await llm.ainvoke(messages_for_llm)
            
            async def vector_search_task():
                """Vector search task"""
                # Extract user_id from shared_memory (stored in process method)
                user_id = shared_memory.get("user_id", "system") if shared_memory else "system"
                logger.debug(f"Quick vector search using user_id: {user_id}")
                return await self._quick_vector_search(query, limit=8, user_id=user_id)
            
            # Initialize vector_results to empty list in case of early errors
            vector_results = []
            
            # Execute both in parallel
            logger.info("Running LLM evaluation and vector search in parallel...")
            try:
                llm_response, vector_results = await asyncio.gather(
                    llm_evaluation_task(),
                    vector_search_task(),
                    return_exceptions=True
                )
                
                # Handle exceptions
                if isinstance(llm_response, Exception):
                    logger.error(f"LLM evaluation failed: {llm_response}")
                    raise llm_response
                if isinstance(vector_results, Exception):
                    logger.warning(f"Vector search failed: {vector_results} - continuing without vector results")
                    vector_results = []
                
                response = llm_response
            except Exception as e:
                logger.error(f"Parallel execution failed: {e}")
                # If parallel execution fails, try LLM only
                try:
                    response = await llm.ainvoke(messages_for_llm)
                    vector_results = []  # Ensure it's defined
                except Exception as llm_error:
                    logger.error(f"LLM fallback also failed: {llm_error}")
                    raise llm_error
            
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
                
                # Analyze vector search results for relevance
                vector_relevance = "none"
                high_relevance_docs = []
                medium_relevance_docs = []
                
                if vector_results:
                    for doc in vector_results:
                        score = doc.get('relevance_score', 0.0)
                        if score >= 0.7:
                            high_relevance_docs.append(doc)
                        elif score >= 0.5:
                            medium_relevance_docs.append(doc)
                    
                    if high_relevance_docs:
                        vector_relevance = "high"
                    elif medium_relevance_docs:
                        vector_relevance = "medium"
                    elif vector_results:
                        vector_relevance = "low"
                    
                    logger.info(f"Vector search relevance: {vector_relevance} ({len(high_relevance_docs)} high, {len(medium_relevance_docs)} medium)")
                
                if assessment.can_answer_quickly and assessment.quick_answer:
                    # Format the quick answer with intelligent merging based on vector results
                    formatted_answer = assessment.quick_answer
                    
                    # High relevance: Include citations in answer
                    if vector_relevance == "high" and high_relevance_docs:
                        citations_text = "\n\n**Sources from your documents:**\n"
                        for doc in high_relevance_docs[:3]:  # Top 3 most relevant
                            title = doc.get('title', doc.get('filename', 'Unknown'))
                            filename = doc.get('filename', '')
                            citations_text += f"- {title}"
                            if filename and filename != title:
                                citations_text += f" ({filename})"
                            citations_text += "\n"
                        formatted_answer += citations_text
                        logger.info(f"Included {len(high_relevance_docs[:3])} high-relevance document citations")
                    
                    # Medium relevance: Mention docs are available
                    elif vector_relevance == "medium" and medium_relevance_docs:
                        doc_mention = f"\n\n*Note: I found {len(medium_relevance_docs)} potentially relevant document(s) in your knowledge base. "
                        doc_mention += "Would you like me to search deeper for more specific information from your documents?*"
                        formatted_answer += doc_mention
                        logger.info(f"Mentioned {len(medium_relevance_docs)} medium-relevance documents")
                    
                    # Add standard offer for deeper research
                    formatted_answer += "\n\n---\n*Would you like me to perform a deeper search for more detailed information, sources, or alternative perspectives? Just let me know!*"
                    
                    logger.info(f"Quick answer provided: confidence={assessment.confidence}, vector_relevance={vector_relevance}")
                    logger.info(f"Reasoning: {assessment.reasoning}")
                    
                    # Update shared_memory to track agent selection for conversation continuity
                    shared_memory = state.get("shared_memory", {}) or {}
                    shared_memory["primary_agent_selected"] = "research_agent"
                    shared_memory["last_agent"] = "research_agent"
                    
                    return {
                        "quick_answer_provided": True,
                        "quick_answer_content": formatted_answer,
                        "final_response": formatted_answer,
                        "research_complete": True,
                        "current_round": ResearchRound.QUICK_ANSWER_CHECK.value,
                        "quick_vector_results": vector_results,
                        "quick_vector_relevance": vector_relevance,
                        "shared_memory": shared_memory
                    }
                else:
                    logger.info(f"Query requires full research: {assessment.reasoning}")
                    # Store vector results even if proceeding to full research (may be useful later)
                    vector_relevance = "none"
                    if vector_results:
                        has_high = any(doc.get('relevance_score', 0.0) >= 0.7 for doc in vector_results)
                        has_medium = any(doc.get('relevance_score', 0.0) >= 0.5 for doc in vector_results)
                        if has_high:
                            vector_relevance = "high"
                        elif has_medium:
                            vector_relevance = "medium"
                        else:
                            vector_relevance = "low"
                    
                    return {
                        "quick_answer_provided": False,
                        "quick_answer_content": "",
                        "current_round": ResearchRound.QUICK_ANSWER_CHECK.value,
                        "quick_vector_results": vector_results,
                        "quick_vector_relevance": vector_relevance
                    }
                    
            except (json.JSONDecodeError, ValidationError, Exception) as e:
                logger.warning(f"Failed to parse quick answer assessment: {e}")
                logger.warning(f"Raw response: {response.content[:500]}")
                # Fallback: proceed to full research if parsing fails
                logger.info("Quick answer assessment parsing failed - proceeding to full research")
                # Determine vector relevance for state
                vector_relevance = "none"
                if vector_results:
                    has_high = any(doc.get('relevance_score', 0.0) >= 0.7 for doc in vector_results)
                    has_medium = any(doc.get('relevance_score', 0.0) >= 0.5 for doc in vector_results)
                    if has_high:
                        vector_relevance = "high"
                    elif has_medium:
                        vector_relevance = "medium"
                    else:
                        vector_relevance = "low"
                
                return {
                    "quick_answer_provided": False,
                    "quick_answer_content": "",
                    "current_round": ResearchRound.QUICK_ANSWER_CHECK.value,
                    "quick_vector_results": vector_results if 'vector_results' in locals() else [],
                    "quick_vector_relevance": vector_relevance
                }
            
        except Exception as e:
            logger.error(f"Quick answer check error: {e}")
            # On error, proceed to full research
            return {
                "quick_answer_provided": False,
                "quick_answer_content": "",
                "current_round": ResearchRound.QUICK_ANSWER_CHECK.value,
                "quick_vector_results": [],
                "quick_vector_relevance": "none"
            }
    
    def _route_from_quick_answer(self, state: ResearchState) -> str:
        """Route from quick answer check: provide quick answer or proceed to full research"""
        if state.get("quick_answer_provided") and state.get("quick_answer_content"):
            logger.info("Quick answer provided - short-circuiting to response")
            return "quick_answer"
        logger.info("Proceeding to full research workflow")
        return "full_research"
    
    # ===== Workflow Nodes =====
    # Note: The following nodes are now handled by the research workflow subgraph:
    # - cache_check -> research_subgraph
    # - query_expansion -> research_subgraph  
    # - round1_parallel_search -> research_subgraph
    # - assess_combined_round1 -> research_subgraph
    # - gap_analysis -> research_subgraph (via universal gap_analysis_subgraph)
    
    async def _round2_parallel_node(self, state: ResearchState) -> Dict[str, Any]:
        """Round 2: Parallel local and web gap-filling searches based on gap analysis flags"""
        try:
            query = state["query"]
            gap_analysis = state.get("gap_analysis", {})
            identified_gaps = state.get("identified_gaps", [])
            
            needs_local = gap_analysis.get("needs_local_search", False)
            needs_web = gap_analysis.get("needs_web_search", False)
            
            logger.info(f"Round 2 Parallel: local={needs_local}, web={needs_web}, gaps={len(identified_gaps)}")
            
            # Prepare gap queries
            gap_queries = identified_gaps[:3] if identified_gaps else [query]
            
            async def local_search_task():
                """Local gap-filling search using Research Subgraph"""
                if not needs_local:
                    logger.info("Skipping Round 2 Local - not needed per gap analysis")
                    return None
                
                try:
                    logger.info(f"Round 2 Local: Searching for {len(gap_queries)} gaps")
                    
                    # Get workflow and research subgraph
                    workflow = await self._get_workflow()
                    checkpointer = workflow.checkpointer
                    research_sg = self._get_research_subgraph(checkpointer, skip_cache=True, skip_expansion=True)
                    
                    # Prepare subgraph state with gap queries
                    shared_memory = state.get("shared_memory", {})
                    subgraph_state = {
                        "query": gap_queries[0],
                        "provided_queries": gap_queries,
                        "shared_memory": shared_memory,
                        "messages": state.get("messages", []),
                        "user_id": shared_memory.get("user_id", "system"),  # Get user_id from shared_memory
                        "metadata": state.get("metadata", {})
                    }
                    
                    # Run subgraph
                    config = self._get_checkpoint_config(state.get("metadata", {}))
                    result = await research_sg.ainvoke(subgraph_state, config)
                    
                    logger.info("✅ Round 2 Local complete")
                    return result
                    
                except Exception as e:
                    logger.error(f"Round 2 Local error: {e}")
                    return {"error": str(e)}
            
            async def web_search_task():
                """Web gap-filling search using Web Research Subgraph"""
                if not needs_web:
                    logger.info("Skipping Round 2 Web - not needed per gap analysis")
                    return None
                
                try:
                    logger.info(f"Round 2 Web: Searching for {len(gap_queries)} gaps")
                    
                    # Get web research subgraph
                    workflow = await self._get_workflow()
                    checkpointer = workflow.checkpointer if workflow else None
                    web_research_sg = self._get_web_research_subgraph(checkpointer)
                    
                    # Prepare subgraph state with gap queries
                    web_subgraph_state = {
                        "query": gap_queries[0],
                        "queries": gap_queries if len(gap_queries) > 1 else None,
                        "max_results": 10,
                        "crawl_top_n": 5,  # Default to 5, but will crawl more if many high-relevance results
                        "shared_memory": state.get("shared_memory", {}),
                        "messages": state.get("messages", []),
                        "metadata": state.get("metadata", {})
                    }
                    
                    # Run subgraph
                    config = self._get_checkpoint_config(state.get("metadata", {}))
                    result = await web_research_sg.ainvoke(web_subgraph_state, config)
                    
                    logger.info("✅ Round 2 Web complete")
                    return result
                    
                except Exception as e:
                    logger.error(f"Round 2 Web error: {e}")
                    return {"error": str(e)}
            
            # Execute searches in parallel
            import asyncio
            local_result, web_result = await asyncio.gather(
                local_search_task(),
                web_search_task(),
                return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(local_result, Exception):
                logger.error(f"Local search exception: {local_result}")
                local_result = None
            
            if isinstance(web_result, Exception):
                logger.error(f"Web search exception: {web_result}")
                web_result = None
            
            # Combine results
            combined_sources = []
            combined_findings = {}
            
            if local_result:
                local_sources = local_result.get("sources_found", [])
                combined_sources.extend(local_sources)
                combined_findings["local_round2"] = local_result.get("research_findings", {}).get("local_results", "")
            
            if web_result:
                web_sources = web_result.get("sources_found", [])
                combined_sources.extend(web_sources)
                combined_findings["web_round2"] = web_result.get("web_results", {}).get("content", "")
            
            logger.info(f"✅ Round 2 Parallel complete: {len(combined_sources)} total sources")
            
            return {
                "round2_results": {
                    "local_result": local_result,
                    "web_result": web_result,
                    "combined_sources": len(combined_sources)
                },
                "sources_found": combined_sources,
                "research_findings": {
                    **state.get("research_findings", {}),
                    **combined_findings
                },
                "round2_sufficient": True,  # Always proceed to synthesis after Round 2
                "current_round": ResearchRound.FINAL_SYNTHESIS.value
            }
            
        except Exception as e:
            logger.error(f"Round 2 Parallel error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "round2_results": {"error": str(e)},
                "sources_found": [],
                "round2_sufficient": True,  # Proceed to synthesis even on error
                "current_round": ResearchRound.FINAL_SYNTHESIS.value
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
            datetime_context = self._get_datetime_context()
            
            # Build messages including conversation history for context
            detection_messages = [
                SystemMessage(content="You are a query type classifier. Always respond with valid JSON matching the exact schema provided."),
                SystemMessage(content=datetime_context)
            ]
            
            # Include conversation history if available (critical for follow-up queries)
            conversation_messages = state.get("messages", [])
            if conversation_messages:
                detection_messages.extend(conversation_messages)
            
            detection_messages.append(HumanMessage(content=detection_prompt))
            
            response = await llm.ainvoke(detection_messages)
            
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
            # Full document analysis insights
            full_doc_insights = state.get("full_doc_insights", {})
            full_doc_synthesis = full_doc_insights.get("synthesis", "")
            
            # Log what we have for synthesis
            round1_content_len = len(round1_results.get("search_results", "")) if round1_results else 0
            round1_entity_len = len(round1_results.get("entity_graph_results", "")) if round1_results else 0
            round2_content_len = len(round2_results.get("gap_results", "")) if round2_results else 0
            web1_content_len = len(web_round1_results.get("content", "")) if web_round1_results else 0
            web2_content_len = len(web_round2_results.get("content", "")) if web_round2_results else 0
            full_doc_content_len = len(full_doc_synthesis) if full_doc_synthesis else 0
            logger.info(f"📊 Synthesis node received: round1={round1_content_len} chars, round1_entity={round1_entity_len} chars, round2={round2_content_len} chars, web1={web1_content_len} chars, web2={web2_content_len} chars, full_doc={full_doc_content_len} chars")
            
            logger.info("Synthesizing final response from all sources")
            
            # Build comprehensive context
            context_parts = []
            
            if cached_context:
                context_parts.append(f"CACHED RESEARCH:\n{cached_context}")
            
            if round1_results:
                local_content = round1_results.get('search_results', '')
                entity_graph_content = round1_results.get('entity_graph_results', '')  # NEW
                
                if local_content:
                    context_parts.append(f"LOCAL SEARCH ROUND 1:\n{local_content[:20000]}")
                
                if entity_graph_content:  # NEW
                    context_parts.append(f"ENTITY GRAPH SEARCH (Knowledge Graph):\n{entity_graph_content[:15000]}")
            
            if round2_results:
                context_parts.append(f"LOCAL SEARCH ROUND 2:\n{round2_results.get('gap_results', '')[:20000]}")
            
            if web_round1_results:
                web_content = web_round1_results.get("content", "")
                if isinstance(web_content, str):
                    context_parts.append(f"WEB SEARCH ROUND 1:\n{web_content}")
                else:
                    context_parts.append(f"WEB SEARCH ROUND 1:\n{str(web_content)}")
            
            if web_round2_results:
                web2_content = web_round2_results.get("content", "")
                if isinstance(web2_content, str):
                    context_parts.append(f"WEB SEARCH ROUND 2:\n{web2_content}")
                else:
                    context_parts.append(f"WEB SEARCH ROUND 2:\n{str(web2_content)}")
            
            if full_doc_synthesis:
                context_parts.append(f"FULL DOCUMENT ANALYSIS:\n{full_doc_synthesis}")
            
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

**CRITICAL**: Only include options that are directly relevant to the query. Do NOT include:
- Unrelated results (e.g., different restaurants with similar names, different people with similar names)
- Tangential information that doesn't answer the query
- Results that are clearly about different entities or topics
- Low-relevance matches that don't contribute to answering the query

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

**CRITICAL**: Only include information that is directly relevant to the query. Do NOT include:
- Unrelated results (e.g., different restaurants with similar names, different people with similar names)
- Tangential information that doesn't answer the query
- Results that are clearly about different entities or topics
- Low-relevance matches that don't contribute to answering the query

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

**CRITICAL**: Only include information that is directly relevant to the query. Do NOT include:
- Unrelated results (e.g., different restaurants with similar names, different people with similar names)
- Tangential information that doesn't answer the query
- Results that are clearly about different entities or topics
- Low-relevance matches that don't contribute to answering the query

If no relevant information was found, simply state that clearly without listing unrelated results.

Your comprehensive response:"""
            
            synthesis_llm = self._get_llm(temperature=0.3, state=state)
            datetime_context = self._get_datetime_context()
            
            # Check for agent handoff context
            shared_memory = state.get("shared_memory", {})
            handoff_context = shared_memory.get("handoff_context", {})
            handoff_note = ""
            
            if handoff_context:
                source_agent = handoff_context.get("source_agent", "unknown")
                reference_doc = handoff_context.get("reference_document", {})
                
                if reference_doc.get("has_content"):
                    ref_content = reference_doc.get("content", "")
                    ref_filename = reference_doc.get("filename", "unknown")
                    handoff_note = f"""

**AGENT HANDOFF CONTEXT**:
- Delegated by: {source_agent}
- User has reference document: {ref_filename}

**USER'S REFERENCE DOCUMENT CONTENT**:
{ref_content}

When synthesizing your answer, integrate information from the user's reference document with your research findings."""
                    logger.info(f"🔗 Handoff context available for synthesis from {source_agent}")
            
            # Build messages including conversation history for context
            synthesis_messages = [
                SystemMessage(content="You are an expert research synthesizer."),
                SystemMessage(content=datetime_context)
            ]
            
            # Include conversation history if available (critical for follow-up queries)
            conversation_messages = state.get("messages", [])
            if conversation_messages:
                # Add conversation history so synthesis understands context
                synthesis_messages.extend(conversation_messages)
                logger.info(f"🔍 Including {len(conversation_messages)} conversation messages for synthesis context")
            
            # Add the synthesis prompt with handoff context
            full_synthesis_prompt = synthesis_prompt + handoff_note
            synthesis_messages.append(HumanMessage(content=full_synthesis_prompt))
            
            response = await synthesis_llm.ainvoke(synthesis_messages)
            
            final_response = response.content
            
            logger.info(f"Synthesis complete: {len(final_response)} characters")
            
            # Detect what post-processing would improve the response
            formatting_recommendations = await detect_post_processing_needs(query, final_response)
            
            if formatting_recommendations:
                logger.info(f"Post-processing recommended: table={formatting_recommendations.get('table_recommended')}, "
                           f"chart={formatting_recommendations.get('chart_recommended')}, "
                           f"timeline={formatting_recommendations.get('timeline_recommended')}")
                if formatting_recommendations.get('chart_recommended'):
                    logger.info(f"📊 Visualization requested for query: '{query}' - will route to visualization subgraph")
            
            # Legacy routing_recommendation for backward compatibility
            routing_recommendation = None
            if formatting_recommendations and (formatting_recommendations.get("table_recommended") or 
                                               formatting_recommendations.get("timeline_recommended")):
                routing_recommendation = "data_formatting"
            
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
            shared_memory["primary_agent_selected"] = "research_agent"
            shared_memory["last_agent"] = "research_agent"
            
            return {
                "final_response": final_response,
                "research_complete": True,
                "sources_used": sources_used,
                "current_round": ResearchRound.FINAL_SYNTHESIS.value,
                "routing_recommendation": routing_recommendation,
                "formatting_recommendations": formatting_recommendations,
                "citations": [],  # TODO: Implement citation extraction
                "shared_memory": shared_memory
            }
            
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            # Update shared_memory even on error to track agent selection
            shared_memory = state.get("shared_memory", {}) or {}
            shared_memory["primary_agent_selected"] = "research_agent"
            shared_memory["last_agent"] = "research_agent"
            return {
                "final_response": f"Research completed but synthesis failed: {str(e)}",
                "research_complete": True,
                "error": str(e),
                "shared_memory": shared_memory
            }
    
    def _route_from_synthesis(self, state: ResearchState) -> str:
        """Route from synthesis: check if post-processing is needed"""
        formatting_recommendations = state.get("formatting_recommendations")
        if formatting_recommendations and (formatting_recommendations.get("table_recommended") or 
                                           formatting_recommendations.get("chart_recommended") or 
                                           formatting_recommendations.get("timeline_recommended")):
            logger.info("Routing to post-processing (formatting and/or visualization)")
            return "post_process"
        return "complete"
    
    async def _post_process_results_node(self, state: ResearchState) -> Dict[str, Any]:
        """Post-process research results using formatting and/or visualization subgraphs in parallel"""
        try:
            import asyncio
            
            formatting_recommendations = state.get("formatting_recommendations", {})
            query = state.get("query", "")
            final_response = state.get("final_response", "")
            
            # Get workflow and checkpointer
            workflow = await self._get_workflow()
            checkpointer = workflow.checkpointer
            metadata = state.get("metadata", {})
            messages = state.get("messages", [])
            shared_memory = state.get("shared_memory", {})
            config = self._get_checkpoint_config(metadata)
            
            # Determine what post-processing is needed
            needs_formatting = formatting_recommendations.get("table_recommended") or formatting_recommendations.get("timeline_recommended")
            needs_visualization = formatting_recommendations.get("chart_recommended")
            
            logger.info(f"Post-processing: formatting={needs_formatting}, visualization={needs_visualization}")
            
            # Prepare tasks for parallel execution
            formatting_task = None
            visualization_task = None
            
            if needs_formatting:
                formatting_sg = self._get_data_formatting_subgraph(checkpointer)
                formatting_query = f"Format the following research results into a well-organized structure:\n\n{final_response}"
                formatting_state = {
                    "query": formatting_query,
                    "messages": messages,
                    "shared_memory": shared_memory,
                    "user_id": state.get("user_id", shared_memory.get("user_id", "system")),
                    "metadata": metadata
                }
                formatting_task = formatting_sg.ainvoke(formatting_state, config)
            
            if needs_visualization:
                visualization_sg = self._get_visualization_subgraph(checkpointer)
                
                # **BULLY!** For follow-up visualization requests, ensure we have research data!
                # If final_response is short or empty, try to extract from previous messages
                research_data_for_viz = final_response
                if not research_data_for_viz or len(research_data_for_viz) < 200:
                    # Look for previous research results in conversation history
                    for msg in reversed(messages[-10:]):
                        if hasattr(msg, 'content'):
                            content = msg.content
                            is_assistant = (hasattr(msg, 'type') and msg.type == "ai") or (hasattr(msg, 'role') and msg.role == "assistant")
                            if is_assistant and len(content) > 200:
                                research_data_for_viz = content
                                logger.info(f"Using previous research data from conversation ({len(content)} chars) for visualization")
                                break
                
                visualization_state = {
                    "query": query,
                    "messages": messages,
                    "research_data": research_data_for_viz,  # Use extracted data if available
                    "shared_memory": shared_memory,
                    "user_id": state.get("user_id", shared_memory.get("user_id", "system")),
                    "metadata": metadata
                }
                logger.info(f"📊 Calling visualization subgraph with {len(research_data_for_viz)} chars of research data")
                visualization_task = visualization_sg.ainvoke(visualization_state, config)
            
            # Execute tasks in parallel
            tasks = []
            task_indices = {}
            if formatting_task:
                task_indices['formatting'] = len(tasks)
                tasks.append(formatting_task)
            if visualization_task:
                task_indices['visualization'] = len(tasks)
                tasks.append(visualization_task)
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                results = []
            
            # Extract results
            formatting_result = None
            visualization_result = None
            
            if 'formatting' in task_indices:
                idx = task_indices['formatting']
                if idx < len(results) and not isinstance(results[idx], Exception):
                    formatting_result = results[idx]
            
            if 'visualization' in task_indices:
                idx = task_indices['visualization']
                if idx < len(results) and not isinstance(results[idx], Exception):
                    visualization_result = results[idx]
            
            # Handle formatting result
            formatted_output = final_response
            format_type = None
            if formatting_result:
                formatted_output = formatting_result.get("formatted_output", final_response)
                format_type = formatting_result.get("format_type", "structured_text")
                logger.info(f"Data formatting complete: {format_type}")
            
            # Handle visualization result
            visualization_data = None
            chart_type = None
            chart_output_format = None
            static_visualization_data = None
            static_format = None
            
            if visualization_result:
                viz_result = visualization_result.get("visualization_result", {})
                if viz_result.get("success"):
                    visualization_data = viz_result.get("chart_data")
                    chart_type = viz_result.get("chart_type")
                    chart_output_format = viz_result.get("output_format")
                    static_visualization_data = viz_result.get("static_visualization_data")
                    static_format = viz_result.get("static_format")
                    logger.info(f"Visualization complete: {chart_type}, format: {chart_output_format}, static: {bool(static_visualization_data)}")
                else:
                    logger.warning(f"Visualization failed: {viz_result.get('error')}")
            
            # Combine results
            combined_response = self._combine_post_processed_results(
                formatted_output, format_type, visualization_data, chart_type, chart_output_format
            )
            
            return {
                "final_response": combined_response,
                "format_type": format_type,
                "formatted": needs_formatting and formatting_result is not None,
                "visualization_results": {
                    "success": visualization_result is not None and visualization_result.get("visualization_result", {}).get("success", False),
                    "chart_type": chart_type,
                    "chart_data": visualization_data,
                    "output_format": chart_output_format,
                    "static_visualization_data": static_visualization_data,
                    "static_format": static_format
                } if visualization_result else None,
                "static_visualization_data": static_visualization_data,
                "static_format": static_format,
                # Preserve critical state keys
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.error(f"Post-processing failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return original response if post-processing fails
            return {
                "final_response": state.get("final_response", ""),
                "formatted": False,
                "post_processing_error": str(e),
                # Preserve critical state keys even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    def _combine_post_processed_results(
        self, 
        formatted_output: str, 
        format_type: Optional[str],
        visualization_data: Optional[str],
        chart_type: Optional[str],
        chart_output_format: Optional[str]
    ) -> str:
        """Combine formatted output and visualization into final response"""
        combined = formatted_output
        
        # Append visualization if available
        if visualization_data and chart_type:
            combined += "\n\n"
            if chart_output_format == "html":
                # **BULLY!** Wrap interactive HTML in special chart code block for frontend rendering!
                combined += f"## Chart: {chart_type.title()}\n\n"
                combined += f"```html:chart\n{visualization_data}\n```"
            elif chart_output_format == "base64_png":
                combined += f"## Chart: {chart_type.title()}\n\n"
                combined += f"![Chart: {chart_type}]({visualization_data})"
            else:
                combined += f"## Chart: {chart_type.title()}\n\n"
                combined += f"Chart generated successfully (format: {chart_output_format})"
        
        return combined
    
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
            
            # Prepare new messages (current query)
            new_messages = self._prepare_messages_with_query(messages, query)
            
            # Load and merge checkpointed messages to preserve conversation history
            conversation_messages = await self._load_and_merge_checkpoint_messages(
                workflow, config, new_messages, look_back_limit=10
            )
            
            # Load shared_memory from checkpoint if available
            checkpoint_state = await workflow.aget_state(config)
            existing_shared_memory = {}
            if checkpoint_state and checkpoint_state.values:
                existing_shared_memory = checkpoint_state.values.get("shared_memory", {})
            
            # Merge with any shared_memory from metadata
            shared_memory = metadata.get("shared_memory", {}) or {}
            shared_memory.update(existing_shared_memory)
            
            # Ensure user_chat_model is in metadata if it's in shared_memory (for subgraph nodes)
            if "user_chat_model" in shared_memory and "user_chat_model" not in metadata:
                metadata["user_chat_model"] = shared_memory["user_chat_model"]
            
            # Store user_id in shared_memory so it's available in state
            if user_id and user_id != "system":
                shared_memory["user_id"] = user_id
            
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
            
            # Call research method with skip_quick_answer flag, shared_memory, conversation messages, and metadata
            result = await self.research(
                query=query,
                conversation_id=conversation_id,
                skip_quick_answer=skip_quick_answer,
                shared_memory=shared_memory,
                messages=conversation_messages,
                metadata=metadata  # Pass metadata to preserve user_chat_model
            )
            
            # Format response in standard agent format
            final_response = result.get("final_response", "")
            research_complete = result.get("research_complete", False)
            
            return {
                "task_status": "complete" if research_complete else "incomplete",
                "response": final_response,
                "agent_type": "research_agent",
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
    
    async def research(self, query: str, conversation_id: str = None, skip_quick_answer: bool = False, shared_memory: Dict[str, Any] = None, messages: List[Any] = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute complete research workflow
        
        Args:
            query: Research query
            conversation_id: Optional conversation ID for caching
            skip_quick_answer: If True, skip quick answer check and proceed directly to full research
            shared_memory: Optional shared memory dictionary for cross-agent communication
            messages: Optional conversation history (for context in research)
            metadata: Optional metadata dictionary (preserves user_chat_model, etc.)
            
        Returns:
            Complete research results with answer and metadata
        """
        try:
            logger.info(f"Starting sophisticated research for: {query}")
            
            # Analyze tool needs for dynamic loading (Kiro-style)
            conversation_context = {
                "previous_tools_used": shared_memory.get("previous_tools_used", []) if shared_memory else []
            }
            tool_analysis = analyze_tool_needs_for_research(query, conversation_context)
            
            logger.info(
                f"🎯 Dynamic tool analysis: {tool_analysis['tool_count']} tools needed "
                f"(core: {tool_analysis['core_count']}, conditional: {tool_analysis['conditional_count']})"
            )
            logger.info(f"🎯 Categories: {', '.join(tool_analysis['categories'])}")
            logger.info(f"🎯 Reasoning: {tool_analysis['reasoning']}")
            
            # Store tool analysis in shared_memory for tracking
            if shared_memory is None:
                shared_memory = {}
            shared_memory["tool_analysis"] = tool_analysis
            shared_memory["dynamic_tools_loaded"] = tool_analysis["all_tools"]
            
            # Prepare messages for state (use conversation history if provided, otherwise just current query)
            if messages:
                # Use provided conversation history (includes previous messages from checkpoint)
                state_messages = list(messages)
            else:
                # Fallback: just current query
                from langchain_core.messages import HumanMessage
                state_messages = [HumanMessage(content=query)]
            
            # Preserve metadata (including user_chat_model) for state and subgraphs
            state_metadata = metadata or {}
            if conversation_id:
                state_metadata["conversation_id"] = conversation_id
            
            # Ensure user_chat_model is in both metadata and shared_memory (bidirectional sync)
            if shared_memory is None:
                shared_memory = {}
            # Copy from shared_memory to metadata if not present
            if "user_chat_model" in shared_memory and "user_chat_model" not in state_metadata:
                state_metadata["user_chat_model"] = shared_memory["user_chat_model"]
            # Copy from metadata to shared_memory if not present
            if "user_chat_model" in state_metadata and "user_chat_model" not in shared_memory:
                shared_memory["user_chat_model"] = state_metadata["user_chat_model"]
            
            # Initialize state
            initial_state = {
                "query": query,
                "original_query": query,
                "expanded_queries": [],
                "key_entities": [],
                "messages": state_messages,
                "shared_memory": shared_memory or {},
                "metadata": state_metadata,  # Include full metadata in state
                "user_id": state_metadata.get("user_id", "system"),
                "quick_answer_provided": False,
                "quick_answer_content": "",
                "skip_quick_answer": skip_quick_answer,
                "quick_vector_results": [],
                "quick_vector_relevance": None,
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
                "full_doc_analysis_needed": False,
                "document_ids_to_analyze": [],
                "analysis_queries": [],
                "full_doc_insights": {},
                "documents_analyzed": [],
                "full_doc_decision_reasoning": "",
                "final_response": "",
                "citations": [],
                "sources_used": [],
                "research_complete": False,
                "routing_recommendation": None,
                "error": ""
            }
            
            # Get workflow and checkpoint config
            workflow = await self._get_workflow()
            
            # Use preserved metadata for checkpoint config
            config = self._get_checkpoint_config(state_metadata)
            
            # Run workflow with checkpointing
            result = await workflow.ainvoke(initial_state, config=config)
            
            # Log final dynamic tool usage summary
            final_shared_memory = result.get("shared_memory", {})
            tool_analysis = final_shared_memory.get("tool_analysis", {})
            previous_tools = final_shared_memory.get("previous_tools_used", [])
            
            if tool_analysis:
                logger.info(
                    f"🎯 Dynamic tool usage summary: "
                    f"{len(previous_tools)} tools used out of {tool_analysis.get('tool_count', 0)} available "
                    f"(core: {tool_analysis.get('core_count', 0)}, conditional: {tool_analysis.get('conditional_count', 0)})"
                )
                logger.info(f"🎯 Tools actually used: {', '.join(previous_tools) if previous_tools else 'none'}")
            
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

