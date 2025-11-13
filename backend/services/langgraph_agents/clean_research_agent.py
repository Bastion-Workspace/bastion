"""
Clean Research Agent - Roosevelt's "Lean Mean Research Machine"
Simple, focused research workflow: local search → evaluate → ask permission → web search → answer
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from .base_agent import BaseAgent
from models.agent_response_models import ResearchTaskResult, TaskStatus
from .research_routing_utils import detect_formatting_request
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)


class CleanResearchAgent(BaseAgent):
    """
    Roosevelt's "Lean Research Agent" - follows the simple workflow:
    1. Search local sources (documents, entities)
    2. Evaluate if information is sufficient
    3. If not sufficient, request permission for web search
    4. If permission granted, search web and combine results
    5. Provide answer with citations
    """
    
    def __init__(self):
        super().__init__("research_agent")  # Use existing enum mapping
        
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Main research workflow - Roosevelt's Simple but Effective Strategy"""
        try:
            logger.info("🔬 BULLY! Clean Research Agent charging forward...")
            
            # CRITICAL: Get the original query, NOT the latest message (which might be permission response)
            query = self._get_original_research_query(state)
            if not query.strip():
                return self._create_error_response(state, "No research query provided")
            
            # ROOSEVELT'S CONVERSATION INTELLIGENCE: Check cache first before any searches
            cached_context = await self._get_relevant_context(query, state)
            if cached_context and self._should_use_cached_context(query, state):
                logger.info("🏆 CACHE-FIRST RESEARCH: Using conversation intelligence instead of new searches")
                return await self._process_with_cached_context(query, cached_context, state)
            
            # ROOSEVELT'S DIRECT RESEARCH: Execute comprehensive research with both local and web sources
            logger.info("🌐 Executing comprehensive research with local + web sources...")
            
            # **ROOSEVELT'S SMART SEARCH STRATEGY**: Search first, expand only if needed!
            # This saves time and only generates expansions when actually necessary
            shared_memory = state.get("shared_memory", {})
            
            # Check if we already have expanded queries from a previous iteration
            expanded_queries_data = shared_memory.get("expanded_queries")
            if expanded_queries_data:
                logger.info(f"✅ Using preserved expanded queries: {expanded_queries_data.get('expansion_count', 0)} variations")
                return await self._execute_comprehensive_research(state, query)
            
            # **SEARCH FIRST STRATEGY**: Try initial local search before expanding
            logger.info("🔍 SMART SEARCH: Starting with initial local search...")
            initial_search_result = await self._do_initial_local_search(state, query)
            
            # ROOSEVELT'S SEMANTIC GAP ANALYSIS: Check if results are truly sufficient
            # Not just quantitatively (scores) but semantically (coverage)
            logger.info("🧠 SEMANTIC GAP ANALYSIS: Evaluating Round 1 coverage...")
            gap_analysis = await self._analyze_semantic_gaps(state, query, initial_search_result)
            
            if not gap_analysis.get("needs_round_2"):
                logger.info("✅ ROUND 1 SUFFICIENT: Semantic coverage complete!")
                # Store the good results and proceed directly to synthesis
                shared_memory["initial_search_results"] = initial_search_result
                return await self._execute_research_with_sufficient_results(state, query, initial_search_result)
            
            # Gap Analysis says we need Round 2!
            logger.info(f"🔍 ROUND 2 NEEDED: Filling gaps for {gap_analysis.get('entities_missing', [])}")
            logger.info(f"🎯 GAPS IDENTIFIED: {gap_analysis.get('gaps_identified', [])}")
            
            # ROUND 2: Execute gap-filling search
            round2_results = await self._execute_round_2_gap_filling(
                state,
                query,
                gap_analysis,
                initial_search_result
            )
            
            # MULTI-ROUND SYNTHESIS: Combine Round 1 + Round 2
            logger.info("🎨 SYNTHESIS: Combining Round 1 + Round 2 results...")
            return await self._synthesize_multi_round_results(
                state,
                query,
                initial_search_result,
                round2_results,
                gap_analysis
            )
                
        except Exception as e:
            logger.error(f"❌ Research agent failed: {e}")
            return self._create_error_response(state, str(e))
    
    # REMOVED: _execute_local_research - No longer needed since we do comprehensive research directly
    
    async def _process_with_cached_context(self, query: str, cached_context: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """ROOSEVELT'S CACHE-FIRST PROCESSING: Use conversation intelligence instead of new searches"""
        try:
            logger.info("🏆 PROCESSING WITH CACHED CONTEXT: Using conversation intelligence")
            
            # Build response using cached context
            system_prompt = f"""You are Roosevelt's Research Agent using CACHED CONVERSATION INTELLIGENCE.

**MISSION**: Answer the user's query using the cached context from this conversation instead of performing new searches.

**USER QUERY**: {query}

**CACHED CONTEXT FROM CONVERSATION**:
{cached_context}

**INSTRUCTIONS**:
1. **USE CACHED CONTEXT**: Build your response using the provided conversation context
2. **NO NEW SEARCHES**: Do not request additional searches - work with what's provided
3. **REFERENCE PREVIOUS WORK**: Acknowledge and build upon previous research/chat outputs
4. **MAINTAIN QUALITY**: Provide comprehensive response using cached intelligence

**STRUCTURED OUTPUT REQUIRED**:
You MUST respond with valid JSON matching this schema:
{{
    "task_status": "complete",
    "findings": "Your comprehensive response using cached context",
    "sources_searched": ["conversation_cache"],
    "confidence_level": 0.9,
    "permission_request": null,
    "next_steps": null
}}

**By George!** Use the conversation intelligence to provide an excellent response without redundant searches!
"""

            # Get LLM response using cached context
            chat_service = await self._get_chat_service()
            model_name = await self._get_model_name()
            
            from langchain_openai import ChatOpenAI
            from langchain.schema import SystemMessage, HumanMessage
            
            client = ChatOpenAI(
                model=model_name,
                temperature=0.3,
                openai_api_key=chat_service.openai_client.api_key,
                openai_api_base=str(chat_service.openai_client.base_url)
            )
            
            response = await client.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=query)
            ])
            
            # Parse structured response
            response_content = response.content if hasattr(response, 'content') else str(response)
            structured_result = self._parse_structured_response(response_content, state)
            
            # Create successful result with cache indicator
            state["agent_results"] = {
                "structured_response": structured_result.dict(),
                "research_mode": "cached_context",
                "citations": [],  # Citations already in cached context
                "cache_used": True,
                "timestamp": datetime.now().isoformat(),
                "processing_time": 0.5,  # Fast cache-based processing
            }
            
            state["latest_response"] = structured_result.findings
            
            # ROOSEVELT'S PURE LANGGRAPH: Add research response to LangGraph state messages (cached context - no new citations)
            if structured_result.findings and structured_result.findings.strip():
                from langchain_core.messages import AIMessage
                cached_message = AIMessage(
                    content=structured_result.findings,
                    additional_kwargs={
                        "citations": [],  # No new citations for cached responses
                        "research_mode": "cached_context",
                        "cache_used": True,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                state.setdefault("messages", []).append(cached_message)
                logger.info(f"✅ RESEARCH AGENT: Added cached research response to LangGraph messages")
            
            state["is_complete"] = True
            
            logger.info("✅ CACHE-FIRST RESEARCH: Completed using conversation intelligence")
            return state
            
        except Exception as e:
            logger.error(f"❌ Cache-first processing failed: {e}")
            # Fallback to smart search flow
            logger.info("🔄 FALLBACK: Proceeding with smart search strategy")
            # Generate expansions and proceed to comprehensive research
            expanded_queries_data = await self._expand_query_for_research(state, query)
            return await self._execute_comprehensive_research(state, query)
    
    async def _execute_comprehensive_research(self, state: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Execute comprehensive research with web tools"""
        try:
            logger.info("🌐 Executing comprehensive research (local + web)...")
            
            # Get OpenAI-compatible tools asynchronously (includes both local and web)
            all_tools = await self._get_agent_tools_async()
            
            # Get shared memory for context
            shared_memory = state.get("shared_memory", {})
            
            # ROOSEVELT'S EXPANSION REUSE: Check if we already have expanded queries
            expanded_queries_data = shared_memory.get("expanded_queries")
            if not expanded_queries_data or not isinstance(expanded_queries_data, dict):
                # Generate them now if missing or corrupted (safety fallback)
                logger.info("🔍 No valid expanded queries found - generating them now...")
                expanded_queries_data = await self._expand_query_for_research(state, query)
            else:
                expansion_count = expanded_queries_data.get('expansion_count', 0) if isinstance(expanded_queries_data, dict) else 0
                logger.info(f"✅ Reusing existing expanded queries: {expansion_count} variations")
            
            # **ROOSEVELT TAG DETECTION**: Detect tags from ORIGINAL query before expansion!
            # This ensures we don't lose tag references during query expansion
            if not shared_memory.get("detected_filters"):
                try:
                    from services.langgraph_tools.tag_detection_service import get_tag_detection_service
                    from repositories.document_repository import DocumentRepository
                    
                    logger.info(f"🔍 TAG DETECTION: Checking original query for tag/category references...")
                    
                    # Get available tags and categories
                    doc_repo = DocumentRepository()
                    await doc_repo.initialize()
                    available_tags = await doc_repo.get_all_tags()
                    available_categories = await doc_repo.get_all_categories()
                    
                    # Detect tags from ORIGINAL user query
                    tag_service = get_tag_detection_service()
                    detection_result = await tag_service.detect_and_match_filters(
                        query, available_tags, available_categories
                    )
                    
                    # Store in shared memory for tool calls to use
                    shared_memory["detected_filters"] = detection_result
                    
                    if detection_result["should_filter"]:
                        filter_msg = tag_service.format_filter_message(detection_result)
                        logger.info(f"✅ TAG DETECTION: {filter_msg}")
                    else:
                        logger.info("🔍 TAG DETECTION: No filters detected, will search all documents")
                        
                except Exception as e:
                    logger.warning(f"⚠️ TAG DETECTION: Failed, continuing without filters: {e}")
                    shared_memory["detected_filters"] = {"should_filter": False}
            else:
                logger.info("✅ TAG DETECTION: Reusing pre-detected filters from shared memory")
            
            # **ROOSEVELT FIX**: Use web-focused prompt for comprehensive research
            # No need to differentiate - we do comprehensive research regardless
            system_prompt = self._build_web_focused_research_prompt(query, expanded_queries_data, shared_memory)
            
            # Execute research with all tools
            result = await self._execute_research_with_tools(
                state, query, system_prompt, all_tools, "comprehensive"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Comprehensive research failed: {e}")
            return self._create_error_response(state, f"Comprehensive research failed: {str(e)}")
    
    async def _execute_round_2_gap_filling(
        self,
        state: Dict[str, Any],
        query: str,
        gap_analysis: Dict[str, Any],
        round1_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        **ROOSEVELT'S ROUND 2 GAP-FILLING SEARCH**: Targeted search to fill identified gaps
        
        Strategy based on gap analysis:
        - search_local_unfiltered: Search local docs without tag filters
        - search_web: Search the web for missing information
        - both: Try unfiltered local first, then web if needed
        """
        try:
            logger.info(f"🔍 ROUND 2 GAP-FILLING: Strategy = {gap_analysis.get('round_2_strategy')}")
            logger.info(f"🎯 ROUND 2 TARGETS: {gap_analysis.get('entities_missing', [])}")
            
            strategy = gap_analysis.get('round_2_strategy', 'search_web')
            suggested_queries = gap_analysis.get('suggested_queries_for_round_2', [query])
            missing_entities = gap_analysis.get('entities_missing', [])
            
            # Prepare Round 2 search queries
            if suggested_queries:
                round2_query = suggested_queries[0]  # Use first suggested query
            elif missing_entities:
                # Construct query from missing entities
                round2_query = " ".join(missing_entities) + " " + query.split()[-2:]  # Add context from original
            else:
                round2_query = query
            
            logger.info(f"🔍 ROUND 2 QUERY: {round2_query}")
            
            # Get shared memory
            shared_memory = state.get("shared_memory", {})
            
            # ROUND 2 EXECUTION
            if strategy == "search_local_unfiltered":
                # Search local WITHOUT tag filters
                logger.info("📚 ROUND 2: Searching local documents WITHOUT filters...")
                from services.langgraph_tools.unified_search_tools import unified_local_search
                
                user_id = state.get("user_id")
                round2_search_results = await unified_local_search(
                    query=round2_query,
                    limit=100,
                    search_types=["vector", "entities"],
                    user_id=user_id,
                    filter_tags=None,  # NO FILTERS!
                    filter_category=None
                )
                
                round2_quality = self._analyze_search_quality(round2_search_results, round2_query)
                
            elif strategy == "search_web":
                # Search the web for missing information
                logger.info("🌐 ROUND 2: Searching web for gap-filling information...")
                from services.langgraph_tools.web_content_tools import search_and_crawl
                
                round2_search_results = await search_and_crawl(
                    query=round2_query,
                    max_results=5
                )
                
                # Web search quality
                round2_quality = {
                    "sufficient": True,  # Assume web search provides useful data
                    "result_count": 5,
                    "top_score": 0.8,
                    "summary": "Web search executed for gap filling"
                }
                
            else:  # "both"
                # Try unfiltered local first, then web
                logger.info("🔄 ROUND 2: Trying unfiltered local, then web if needed...")
                from services.langgraph_tools.unified_search_tools import unified_local_search
                
                user_id = state.get("user_id")
                local_results = await unified_local_search(
                    query=round2_query,
                    limit=100,
                    search_types=["vector", "entities"],
                    user_id=user_id,
                    filter_tags=None,
                    filter_category=None
                )
                
                local_quality = self._analyze_search_quality(local_results, round2_query)
                
                if local_quality["sufficient"]:
                    logger.info("✅ ROUND 2: Unfiltered local search sufficient!")
                    round2_search_results = local_results
                    round2_quality = local_quality
                else:
                    logger.info("🌐 ROUND 2: Local insufficient, searching web...")
                    from services.langgraph_tools.web_content_tools import search_and_crawl
                    
                    web_results = await search_and_crawl(
                        query=round2_query,
                        max_results=5
                    )
                    
                    # Combine local + web
                    round2_search_results = local_results + "\n\n" + web_results
                    round2_quality = {
                        "sufficient": True,
                        "result_count": local_quality["result_count"] + 5,
                        "top_score": max(local_quality["top_score"], 0.8),
                        "summary": "Combined unfiltered local + web search"
                    }
            
            logger.info(f"✅ ROUND 2 COMPLETE: {round2_quality['summary']}")
            
            return {
                "results": round2_search_results,
                "quality_assessment": round2_quality['summary'],
                "result_count": round2_quality['result_count'],
                "top_score": round2_quality['top_score'],
                "strategy_used": strategy,
                "queries_executed": [round2_query],
                "gaps_filled": missing_entities
            }
            
        except Exception as e:
            logger.error(f"❌ Round 2 gap-filling failed: {e}")
            # Return empty results on failure
            return {
                "results": "",
                "quality_assessment": f"Round 2 failed: {str(e)}",
                "result_count": 0,
                "top_score": 0.0,
                "strategy_used": "failed",
                "queries_executed": [],
                "gaps_filled": []
            }
    
    async def _synthesize_multi_round_results(
        self,
        state: Dict[str, Any],
        query: str,
        round1_results: Dict[str, Any],
        round2_results: Dict[str, Any],
        gap_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        **ROOSEVELT'S MULTI-ROUND SYNTHESIS**: Combine Round 1 + Round 2 into cohesive answer
        
        Weaves together filtered results (Round 1) and gap-filling results (Round 2)
        into one comprehensive, natural response
        """
        try:
            logger.info("🎨 MULTI-ROUND SYNTHESIS: Combining Round 1 + Round 2...")
            
            # ROOSEVELT'S BUG FIX: Ensure results are strings before slicing
            round1_results_text = str(round1_results.get('results', ''))[:2000]
            round2_results_text = str(round2_results.get('results', ''))[:2000]
            
            synthesis_prompt = f"""You are Roosevelt's Research Synthesizer!

**MISSION**: Combine research from TWO search rounds into ONE comprehensive, cohesive answer.

**ORIGINAL QUERY**: {query}

**ROUND 1 RESULTS** (Filtered/Initial Search):
{round1_results_text}

Round 1 Metadata:
- Found: {round1_results.get('result_count', 0)} documents
- Top score: {round1_results.get('top_score', 0.0):.3f}
- Filters: {round1_results.get('filter_tags', [])}

**ROUND 2 RESULTS** (Gap-Filling Search):
{round2_results_text}

Round 2 Metadata:
- Strategy: {round2_results.get('strategy_used', 'unknown')}
- Gaps filled: {round2_results.get('gaps_filled', [])}
- Found: {round2_results.get('result_count', 0)} results

**GAP ANALYSIS CONTEXT**:
Entities requested: {gap_analysis.get('entities_requested', [])}
Entities from Round 1: {gap_analysis.get('entities_found_in_round1', [])}
Entities from Round 2: {gap_analysis.get('entities_missing', [])}

**SYNTHESIS INSTRUCTIONS**:

1. **Weave, Don't Segregate**: Create ONE integrated answer, not "Round 1 said... Round 2 said..."
2. **Comprehensive Coverage**: Address ALL entities/aspects from the query
3. **Natural Flow**: Make it read as if it came from one unified research effort
4. **Cite Appropriately**: Reference sources naturally from both rounds
5. **Comparative Analysis**: If query asks for comparison, synthesize insights from both rounds
6. **Acknowledge Depth**: Show the comprehensive multi-round research adds value

**EXAMPLE SYNTHESIS** (for "Compare Enron and WorldCom"):
"Both the Enron and WorldCom corporate scandals of the early 2000s share striking commonalities in their methods, scale, and impact on corporate governance reform.

**Scale and Methods**: WorldCom, under CEO Bernie Ebbers, inflated assets by approximately $11 billion through fraudulent accounting practices [Round 1 local docs]. Similarly, Enron executives engaged in massive accounting fraud, using special purpose entities to hide billions in debt [Round 2 web sources]. Both companies manipulated financial statements to present false pictures of profitability.

**Leadership Failures**: In both cases, charismatic CEOs (Ebbers at WorldCom, Kenneth Lay and Jeff Skilling at Enron) presided over cultures where financial manipulation was normalized..."

**STRUCTURED OUTPUT REQUIRED** (valid JSON only):
{{
    "task_status": "complete",
    "findings": "Your comprehensive synthesized answer weaving Round 1 + Round 2 naturally",
    "sources_searched": ["local_documents_filtered", "local_documents_unfiltered", "web"],
    "confidence_level": 0.9,
    "next_steps": null,
    "synthesis_metadata": {{
        "rounds_used": 2,
        "round1_contribution": "WorldCom details from tagged documents",
        "round2_contribution": "Enron details from web search"
    }}
}}

**By George!** Create a masterful synthesis that shows the power of multi-round research!"""

            # Get LLM to synthesize
            chat_service = await self._get_chat_service()
            model_name = await self._get_model_name()
            
            from langchain_openai import ChatOpenAI
            from langchain.schema import SystemMessage, HumanMessage
            
            client = ChatOpenAI(
                model=model_name,
                temperature=0.3,
                openai_api_key=chat_service.openai_client.api_key,
                openai_api_base=str(chat_service.openai_client.base_url)
            )
            
            response = await client.ainvoke([
                SystemMessage(content=synthesis_prompt),
                HumanMessage(content="Synthesize the multi-round research into one comprehensive answer.")
            ])
            
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse structured response
            structured_result = self._parse_structured_response(response_content, state)
            
            # ROOSEVELT'S UNIFIED CITATION EXTRACTION: Same method for both rounds!
            round1_citations = self._extract_citations_unified(round1_results, "Round 1")
            round2_citations = self._extract_citations_unified(round2_results, "Round 2")
            
            # Combine and renumber citations
            all_citations = round1_citations + round2_citations
            for idx, citation in enumerate(all_citations, 1):
                citation["id"] = idx
            
            logger.info(f"🔗 MULTI-ROUND CITATIONS: {len(all_citations)} total ({len(round1_citations)} R1 + {len(round2_citations)} R2)")
            
            # Store in state
            state["agent_results"] = {
                "structured_response": structured_result.dict(),
                "research_mode": "multi_round",
                "citations": all_citations,
                "search_rounds": 2,
                "round1_metadata": {
                    "result_count": round1_results.get('result_count', 0),
                    "filters_applied": round1_results.get('filter_tags', [])
                },
                "round2_metadata": {
                    "strategy": round2_results.get('strategy_used', 'unknown'),
                    "gaps_filled": round2_results.get('gaps_filled', [])
                },
                "timestamp": datetime.now().isoformat(),
            }
            
            state["latest_response"] = structured_result.findings
            
            # Add to LangGraph messages
            if structured_result.findings and structured_result.findings.strip():
                from langchain_core.messages import AIMessage
                research_message = AIMessage(
                    content=structured_result.findings,
                    additional_kwargs={
                        "citations": all_citations,
                        "research_mode": "multi_round",
                        "rounds": 2,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                state.setdefault("messages", []).append(research_message)
                logger.info(f"✅ MULTI-ROUND SYNTHESIS: Completed with {len(all_citations)} citations from 2 search rounds!")
            
            state["is_complete"] = True
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Multi-round synthesis failed: {e}")
            # Fallback to single-round if synthesis fails
            logger.info("⚠️ Falling back to Round 1 results only...")
            return await self._execute_research_with_sufficient_results(state, query, round1_results)
    
    async def _do_initial_local_search(self, state: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        **ROOSEVELT'S SMART SEARCH**: Perform initial local search with tag detection
        
        Returns:
            Dict with:
                - sufficient: bool (whether results are good enough to answer)
                - results: str (search results)
                - result_count: int
                - top_score: float
                - quality_assessment: str
        """
        try:
            shared_memory = state.get("shared_memory", {})
            
            # **TAG DETECTION**: Detect tags from original query
            if not shared_memory.get("detected_filters"):
                try:
                    from services.langgraph_tools.tag_detection_service import get_tag_detection_service
                    from repositories.document_repository import DocumentRepository
                    
                    logger.info(f"🔍 TAG DETECTION: Analyzing query for tag/category references...")
                    
                    doc_repo = DocumentRepository()
                    await doc_repo.initialize()
                    available_tags = await doc_repo.get_all_tags()
                    available_categories = await doc_repo.get_all_categories()
                    
                    tag_service = get_tag_detection_service()
                    detection_result = await tag_service.detect_and_match_filters(
                        query, available_tags, available_categories
                    )
                    
                    shared_memory["detected_filters"] = detection_result
                    
                    if detection_result["should_filter"]:
                        filter_msg = tag_service.format_filter_message(detection_result)
                        logger.info(f"✅ TAG DETECTION: {filter_msg}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ TAG DETECTION: Failed, continuing without filters: {e}")
                    shared_memory["detected_filters"] = {"should_filter": False}
            
            # **INITIAL SEARCH**: Search local documents with detected filters
            from services.langgraph_tools.unified_search_tools import unified_local_search
            
            detected_filters = shared_memory.get("detected_filters", {})
            filter_tags = detected_filters.get("filter_tags", []) if detected_filters.get("should_filter") else None
            filter_category = detected_filters.get("filter_category") if detected_filters.get("should_filter") else None
            
            user_id = state.get("user_id")
            
            logger.info(f"🔍 Performing initial local search (tags: {filter_tags}, category: {filter_category})...")
            search_results = await unified_local_search(
                query=query,
                limit=200,
                search_types=["vector", "entities"],
                user_id=user_id,
                filter_tags=filter_tags,
                filter_category=filter_category
            )
            
            # **QUALITY ANALYSIS**: Parse and analyze results
            quality_assessment = self._analyze_search_quality(search_results, query)
            
            logger.info(f"📊 QUALITY ASSESSMENT: {quality_assessment['summary']}")
            
            return {
                "sufficient": quality_assessment["sufficient"],
                "results": search_results,
                "result_count": quality_assessment["result_count"],
                "top_score": quality_assessment["top_score"],
                "quality_assessment": quality_assessment["summary"],
                "filter_tags": filter_tags,
                "filter_category": filter_category
            }
            
        except Exception as e:
            logger.error(f"❌ Initial local search failed: {e}")
            return {
                "sufficient": False,
                "results": "",
                "result_count": 0,
                "top_score": 0.0,
                "quality_assessment": f"Search failed: {str(e)}"
            }
    
    def _analyze_search_quality(self, search_results: str, query: str) -> Dict[str, Any]:
        """
        **ROOSEVELT'S SEMANTIC-AWARE QUALITY ANALYZER**: Determine if search results are sufficient
        
        Criteria for "sufficient":
        - QUANTITATIVE: At least 20 results with top score >= 0.5, OR 5+ excellent matches (≥0.6)
        - SEMANTIC: Results must cover all key entities/topics mentioned in the query
        - BOTH must be true for true sufficiency!
        """
        try:
            import re
            
            # QUANTITATIVE ANALYSIS
            # Count results
            result_pattern = r'\*\*\d+\.\s+Document'
            matches = re.findall(result_pattern, search_results)
            result_count = len(matches)
            
            # Extract top score
            score_pattern = r'Score:\s+([\d.]+)'
            scores = re.findall(score_pattern, search_results)
            scores_float = [float(s) for s in scores if s]
            top_score = max(scores_float) if scores_float else 0.0
            
            # Count excellent matches (score >= 0.6)
            excellent_matches = len([s for s in scores_float if s >= 0.6])
            
            # Quantitative sufficiency check
            quantitative_sufficient = (
                (result_count >= 20 and top_score >= 0.5) or 
                (excellent_matches >= 5)
            )
            
            # SEMANTIC ANALYSIS
            # Extract key entities/topics from query (simple approach)
            query_entities = self._extract_key_entities_from_query(query)
            result_entities = self._extract_entities_from_results(search_results)
            
            # Check coverage (case-insensitive comparison)
            result_entities_lower = [e.lower() for e in result_entities]
            missing_entities = [
                e for e in query_entities 
                if e.lower() not in result_entities_lower
            ]
            semantic_sufficient = len(missing_entities) == 0
            
            # COMBINED SUFFICIENCY
            # Both quantitative AND semantic must be true
            sufficient = quantitative_sufficient and semantic_sufficient
            
            # Build reason message
            if not quantitative_sufficient and not semantic_sufficient:
                reason = f"❌ Low scores AND missing coverage for: {missing_entities}"
            elif not semantic_sufficient:
                reason = f"⚠️ Good scores ({top_score:.3f}) BUT missing coverage for: {missing_entities}"
            elif not quantitative_sufficient:
                reason = f"⚠️ Found {query_entities} BUT low result quality (count: {result_count}, top: {top_score:.3f})"
            else:
                reason = f"✅ {result_count} results with full semantic coverage of {query_entities}"
            
            return {
                "sufficient": sufficient,
                "quantitative_sufficient": quantitative_sufficient,
                "semantic_sufficient": semantic_sufficient,
                "query_entities": query_entities,
                "result_entities": result_entities,
                "missing_entities": missing_entities,
                "result_count": result_count,
                "top_score": top_score,
                "excellent_matches": excellent_matches,
                "summary": reason
            }
            
        except Exception as e:
            logger.error(f"❌ Quality analysis failed: {e}")
            return {
                "sufficient": False,
                "quantitative_sufficient": False,
                "semantic_sufficient": False,
                "query_entities": [],
                "result_entities": [],
                "missing_entities": [],
                "result_count": 0,
                "top_score": 0.0,
                "excellent_matches": 0,
                "summary": f"Analysis failed: {str(e)}"
            }
    
    async def _execute_research_with_sufficient_results(self, state: Dict[str, Any], query: str, search_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        **ROOSEVELT'S EFFICIENT PATH**: Process research when initial results are sufficient
        Skips expansion and additional searches - goes straight to synthesis
        """
        try:
            logger.info("🎯 EFFICIENT PATH: Synthesizing answer from sufficient initial results...")
            
            # Build a synthesis-focused prompt (no tool calls needed)
            system_prompt = f"""**BULLY!** You are Roosevelt's Research Agent conducting efficient research!

**SITUATION**: 
You have already obtained LOCAL search results that are SUFFICIENT to answer the user's question.
Your task is to synthesize a comprehensive answer using ONLY these results.

**INITIAL SEARCH RESULTS**:
{search_result['results']}

**QUALITY ASSESSMENT**:
{search_result['quality_assessment']}

**INSTRUCTIONS**:
1. **NO ADDITIONAL SEARCHES**: The results above are sufficient - do NOT call any search tools
2. **SYNTHESIZE COMPREHENSIVELY**: Provide a complete answer using the local results
3. **CITE SOURCES**: Reference the documents you're using
4. **ACKNOWLEDGE SCOPE**: You searched local documents{' with filters: ' + str(search_result.get('filter_tags', [])) if search_result.get('filter_tags') else ''}

**STRUCTURED OUTPUT REQUIRED**:
{{
    "task_status": "complete",
    "findings": "Your comprehensive synthesized answer with citations",
    "sources_searched": ["local_documents"],
    "confidence_level": 0.8,
    "permission_request": null,
    "next_steps": null
}}

**USER QUERY**: {query}

**By George!** Provide a thorough answer using the excellent local results we found!
"""

            # Get LLM to synthesize response (no tools)
            chat_service = await self._get_chat_service()
            model_name = await self._get_model_name()
            
            from langchain_openai import ChatOpenAI
            from langchain.schema import SystemMessage, HumanMessage
            
            client = ChatOpenAI(
                model=model_name,
                temperature=0.3,
                openai_api_key=chat_service.openai_client.api_key,
                openai_api_base=str(chat_service.openai_client.base_url)
            )
            
            response = await client.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=query)
            ])
            
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse structured response
            structured_result = self._parse_structured_response(response_content, state)
            
            # **ROOSEVELT'S UNIFIED CITATIONS**: Extract citations using unified extractor
            local_citations = self._extract_citations_unified(search_result, "Efficient Path")
            logger.info(f"📄 EFFICIENT PATH: Extracted {len(local_citations)} local document citations")
            
            # Store results in state
            state["agent_results"] = {
                "structured_response": structured_result.dict(),
                "research_mode": "efficient_local",
                "citations": local_citations,
                "search_rounds": 1,  # Only one initial search!
                "timestamp": datetime.now().isoformat(),
            }
            
            state["latest_response"] = structured_result.findings
            
            # Add to LangGraph messages
            if structured_result.findings and structured_result.findings.strip():
                from langchain_core.messages import AIMessage
                research_message = AIMessage(
                    content=structured_result.findings,
                    additional_kwargs={
                        "citations": local_citations,  # Include local document citations!
                        "research_mode": "efficient_local",
                        "timestamp": datetime.now().isoformat()
                    }
                )
                state.setdefault("messages", []).append(research_message)
                logger.info(f"✅ EFFICIENT RESEARCH: Completed in 1 search round with {len(local_citations)} citations!")
            
            state["is_complete"] = True
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Efficient synthesis failed: {e}")
            # Fallback to full comprehensive research
            logger.info("⚠️ Falling back to comprehensive research...")
            expanded_queries_data = await self._expand_query_for_research(state, query)
            return await self._execute_comprehensive_research(state, query)
    
    async def _analyze_semantic_gaps(
        self, 
        state: Dict[str, Any], 
        query: str, 
        round1_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        **ROOSEVELT'S SEMANTIC GAP ANALYZER**: Intelligent analysis of Round 1 coverage
        
        Uses LLM to determine if Round 1 results comprehensively answer the query
        or if Round 2 is needed to fill gaps
        """
        try:
            logger.info("🧠 SEMANTIC GAP ANALYSIS: Evaluating Round 1 coverage...")
            
            # Prepare context for gap analysis
            round1_summary = round1_results.get('results', '')[:3000]  # First 3000 chars
            quality_assessment = round1_results.get('quality_assessment', 'Unknown')
            filters_used = round1_results.get('filter_tags', []) or round1_results.get('filter_category')
            
            gap_analysis_prompt = f"""You are Roosevelt's Research Gap Analyzer!

**MISSION**: Determine if Round 1 search results COMPREHENSIVELY answer the user's query, or if Round 2 is needed.

**ORIGINAL QUERY**: {query}

**ROUND 1 SEARCH RESULTS**:
{round1_summary}

**ROUND 1 METADATA**:
- Documents found: {round1_results.get('result_count', 0)}
- Top relevance score: {round1_results.get('top_score', 0.0):.3f}
- Quality assessment: {quality_assessment}
- Filters applied: {filters_used if filters_used else 'None (searched all documents)'}

**CRITICAL SEMANTIC ANALYSIS**:

1. **Entity Coverage Check**:
   - What entities/topics does the query ask about? (e.g., "Enron AND WorldCom" = 2 entities)
   - Which entities are covered in Round 1 results?
   - Which entities are MISSING from Round 1 results?

2. **Aspect Coverage Check**:
   - Does query ask for comparisons? ("commonalities", "differences", "versus")
   - Does query ask for multiple aspects? (causes, effects, timeline, etc.)
   - Are ALL requested aspects covered in Round 1?

3. **Completeness Assessment**:
   - Can we provide a COMPREHENSIVE answer with ONLY Round 1 results?
   - What specific information is missing or insufficient?

4. **Filter Impact Analysis**:
   - If filters were applied, did they exclude important information?
   - Example: If searching for "Enron AND WorldCom" but filtered to "worldcom" tag, Enron info is missing!

**DECISION CRITERIA**:
- ✅ **SUFFICIENT**: All entities covered, all aspects addressed, comprehensive answer possible
- ❌ **NEED ROUND 2**: Missing entities, missing aspects, or insufficient depth

**STRUCTURED OUTPUT REQUIRED** (valid JSON only):
{{
    "sufficient": false,
    "confidence_in_sufficiency": 0.4,
    "entities_requested": ["Enron", "WorldCom"],
    "entities_found_in_round1": ["WorldCom"],
    "entities_missing": ["Enron"],
    "aspects_requested": ["commonalities", "comparison"],
    "aspects_covered_in_round1": ["WorldCom details"],
    "aspects_missing": ["Enron details", "comparative analysis"],
    "gaps_identified": [
        "No information about Enron scandal found in Round 1",
        "Cannot compare Enron and WorldCom without Enron data"
    ],
    "needs_round_2": true,
    "round_2_strategy": "search_web",
    "suggested_queries_for_round_2": [
        "Enron scandal corporate fraud accounting",
        "Enron collapse timeline causes"
    ],
    "reasoning": "Round 1 only found WorldCom documents due to tag filtering. Query asks about BOTH Enron and WorldCom commonalities, but we have zero Enron information. Round 2 must search for Enron (unfiltered local OR web)."
}}

**By George!** Analyze carefully - we need honest assessment of coverage!"""

            # Get LLM analysis
            chat_service = await self._get_chat_service()
            model_name = await self._get_model_name()
            
            from langchain_openai import ChatOpenAI
            from langchain.schema import SystemMessage, HumanMessage
            
            client = ChatOpenAI(
                model=model_name,
                temperature=0.1,  # Low temperature for analytical task
                openai_api_key=chat_service.openai_client.api_key,
                openai_api_base=str(chat_service.openai_client.base_url)
            )
            
            response = await client.ainvoke([
                SystemMessage(content=gap_analysis_prompt),
                HumanMessage(content="Analyze the coverage and determine if Round 2 is needed.")
            ])
            
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse structured JSON response
            import json
            import re
            
            # Extract JSON from response
            json_text = response_content.strip()
            if '```json' in json_text:
                match = re.search(r'```json\s*\n(.*?)\n```', json_text, re.DOTALL)
                if match:
                    json_text = match.group(1).strip()
            elif '```' in json_text:
                match = re.search(r'```\s*\n(.*?)\n```', json_text, re.DOTALL)
                if match:
                    json_text = match.group(1).strip()
            
            # Find JSON object if mixed with text
            if not json_text.startswith('{'):
                match = re.search(r'\{.*\}', json_text, re.DOTALL)
                if match:
                    json_text = match.group(0)
            
            gap_analysis = json.loads(json_text)
            
            # Log results
            if gap_analysis.get("needs_round_2"):
                logger.info(f"🔍 GAP ANALYSIS: Round 2 NEEDED - Missing: {gap_analysis.get('entities_missing', [])}")
                logger.info(f"🔍 GAPS: {gap_analysis.get('gaps_identified', [])}")
                logger.info(f"🔍 STRATEGY: {gap_analysis.get('round_2_strategy', 'unknown')}")
            else:
                logger.info(f"✅ GAP ANALYSIS: Round 1 SUFFICIENT - Full coverage achieved")
                logger.info(f"✅ COVERED: {gap_analysis.get('entities_found_in_round1', [])}")
            
            return gap_analysis
            
        except Exception as e:
            logger.error(f"❌ Semantic gap analysis failed: {e}")
            logger.error(f"❌ Response content: {response_content[:500] if 'response_content' in locals() else 'N/A'}")
            # Conservative fallback - assume we need Round 2 if analysis fails
            return {
                "sufficient": False,
                "needs_round_2": True,
                "round_2_strategy": "search_web",
                "suggested_queries_for_round_2": [query],
                "reasoning": f"Gap analysis failed: {str(e)}. Defaulting to Round 2 for safety.",
                "gaps_identified": ["Gap analysis failed - proceeding with Round 2 as safety measure"],
                "entities_requested": [],
                "entities_missing": [],
                "entities_found_in_round1": []
            }
    
    async def _execute_research_with_tools(
        self, 
        state: Dict[str, Any], 
        query: str, 
        system_prompt: str, 
        tools: List[Any], 
        research_mode: str
    ) -> Dict[str, Any]:
        """Universal research execution method using OpenAI client like all other agents"""
        try:
            logger.info(f"🎯 Executing {research_mode} research with {len(tools)} tools...")
            
            # Prepare messages using BaseAgent infrastructure
            # ROOSEVELT'S CONVERSATION HISTORY: Provide full conversation history to LLM
            # Use full context to maintain conversation continuity
            messages = await self._prepare_messages_with_full_context(state, system_prompt)
            
            # Get OpenAI client configuration (consistent with other agents)
            chat_service = await self._get_chat_service()
            model_name = await self._get_model_name()
            
            # Execute LLM call with OpenAI client (like ChatAgent, CodingAgent)
            logger.info(f"🤖 Calling OpenAI client for {research_mode} research...")
            start_time = datetime.now()
            response = await chat_service.openai_client.chat.completions.create(
                messages=messages,
                model=model_name,
                tools=tools,  # BaseAgent tools are already OpenAI-compatible
                tool_choice="auto",
                temperature=0.1
            )
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # BEST PRACTICE: Comprehensive response validation and debugging
            logger.info(f"🔍 RESPONSE DEBUG: Type={type(response)}")
            if hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                logger.info(f"🔍 RESPONSE CONTENT: {str(choice.message.content)[:300]}...")
                if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                    logger.info(f"🔍 TOOL CALLS: {len(choice.message.tool_calls)} calls")
            
            # Check if LLM wants to use tools first - USE BASE AGENT INFRASTRUCTURE
            has_tool_calls = (
                hasattr(response, 'choices') and 
                response.choices and 
                hasattr(response.choices[0].message, 'tool_calls') and 
                response.choices[0].message.tool_calls
            )
            
            if has_tool_calls:
                logger.info(f"🔧 LLM requested tool calls - using BaseAgent infrastructure directly...")
                
                # Use the existing BaseAgent tool execution infrastructure (no conversion needed!)
                final_answer, tools_used = await self._execute_tool_calls(response, messages, state)
                
                # Ask LLM for structured JSON response based on tool results
                final_structured_response = await self._get_final_structured_response_openai(
                    state, tools_used, research_mode
                )
                content_to_parse = final_structured_response
            else:
                # Extract content for parsing - OpenAI format
                content_to_parse = None
                if hasattr(response, 'choices') and response.choices:
                    choice = response.choices[0]
                    if hasattr(choice.message, 'content') and choice.message.content:
                        content_to_parse = choice.message.content
            
            logger.info(f"🎯 CONTENT TO PARSE: {str(content_to_parse)[:200]}...")
            
            if content_to_parse and str(content_to_parse).strip():
                # Parse the structured JSON response with citation processing
                structured_result = self._parse_structured_response(content_to_parse, state)
                
                # Extract citations from tool calls if any (OpenAI format)
                citations = self._extract_citations_from_response_openai(response, research_mode)
                
                # Extract citations from tool results in shared_memory
                extracted_citations = self._extract_citations_from_tool_results(state, research_mode)
                all_citations = citations + extracted_citations
                
                # ROOSEVELT'S LLM FORMATTING DETECTION: Let AI decide if data should be formatted
                routing_recommendation = await detect_formatting_request(state, structured_result, self._get_latest_user_message)
                
                # ROOSEVELT'S COLLABORATION INTELLIGENCE: Detect opportunities for research→weather collaboration
                # ROOSEVELT'S NATURAL COLLABORATION: Let the LLM handle collaboration decisions
                
                # **ROOSEVELT FIX**: Ensure citations are serializable before storing
                try:
                    import json
                    safe_citations = json.loads(json.dumps(all_citations, default=str))
                except Exception as e:
                    logger.warning(f"⚠️ Citation serialization failed: {e}")
                    safe_citations = []
                
                # Store results in state
                state["agent_results"] = {
                    "structured_response": structured_result.dict(),
                    "research_mode": research_mode,
                    "citations": safe_citations,  # Use serializable citations
                    "tool_calls_made": len(getattr(response.choices[0].message, 'tool_calls', [])) if response.choices else 0,
                    "timestamp": datetime.now().isoformat(),
                    "processing_time": processing_time,
                    "routing_recommendation": routing_recommendation,  # ROOSEVELT'S SMART ROUTING
                    # ROOSEVELT'S NATURAL COLLABORATION: Removed collaboration suggestion field
                }
                
                # Update shared memory with search results
                self._update_shared_memory_with_results(state, structured_result, research_mode)
                
                # PHASE 1: Add collaboration suggestion to final response (no routing changes)
                final_response = structured_result.findings
                # ROOSEVELT'S NATURAL COLLABORATION: Let the LLM handle collaboration naturally
                logger.info(f"🤝 PHASE 1: Research response completed")
                
                # CRITICAL: Also update the structured_response.findings that orchestrator reads
                structured_result.findings = final_response
                # Update the stored structured response
                state["agent_results"]["structured_response"] = structured_result.dict()
                
                # Store final response with collaboration suggestion
                state["latest_response"] = final_response
                
                # ROOSEVELT'S PURE LANGGRAPH: Add research response to LangGraph state messages WITH CITATIONS
                if final_response and final_response.strip():
                    from langchain_core.messages import AIMessage
                    
                    # ROOSEVELT'S CITATION ATTACHMENT: Store citations in AIMessage metadata
                    # Use safe_citations from above to avoid duplication
                    research_message = AIMessage(
                        content=final_response,
                        additional_kwargs={
                            "citations": safe_citations,  # Attach serializable citations
                            "research_mode": research_mode,
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                    state.setdefault("messages", []).append(research_message)
                    logger.info(f"✅ RESEARCH AGENT: Added research response with {len(safe_citations)} citations to LangGraph messages")
                
                # ROOSEVELT'S CONTEXTUAL INTELLIGENCE: Store research findings for other agents
                shared_memory = state.get("shared_memory", {})
                shared_memory.setdefault("research_findings", {})
                
                # Store research findings with query as key for other agents to access
                original_query = self._get_original_research_query(state)
                query_key = original_query[:50] if original_query else "research"
                
                shared_memory["research_findings"][query_key] = {
                    "findings": structured_result.findings,
                    "confidence": structured_result.confidence_level,
                    "timestamp": datetime.now().isoformat(),
                    "sources": structured_result.sources_searched
                }
                
                state["shared_memory"] = shared_memory
                logger.info(f"📚 SHARED MEMORY: Stored research findings for Weather Agent context: '{query_key}'")
                
                state["is_complete"] = True
                return state
            else:
                # BEST PRACTICE: Graceful error handling with detailed logging
                logger.error(f"❌ No parseable content found in response")
                logger.error(f"❌ Response attributes: {dir(response)}")
                raise ValueError("LLM response contains no parseable content")
                
        except Exception as e:
            logger.error(f"❌ Research execution failed: {e}")
            raise
    
    async def _get_final_structured_response_openai(
        self, 
        state: Dict[str, Any], 
        tools_used: List[str], 
        research_mode: str
    ) -> str:
        """Get final structured JSON response after tool execution using OpenAI client"""
        try:
            logger.info("🤖 ROOSEVELT'S FINAL RESPONSE: Getting structured JSON after tool execution...")
            
            # Get actual tool results from shared memory (CRITICAL FOR VALIDATION)
            shared_memory = state.get("shared_memory", {})
            search_results = shared_memory.get("search_results", {})
            tool_results = shared_memory.get("tool_results", [])
            
            # Extract actual search result content for validation
            search_result_content = ""
            no_results_found = True
            
            for tool_result in tool_results:
                tool_name = tool_result.get("tool_name", "")
                result = tool_result.get("result", "")
                
                # ROOSEVELT'S DEBUG: Log tool result details for investigation
                logger.info(f"🔍 TOOL RESULT DEBUG: {tool_name} returned type={type(result)}, length={len(str(result))}")
                
                if tool_name == "search_local":
                    search_result_content += f"\n**Local Search Result:** {result[:500]}..."
                    # Check if search actually found results
                    if "No results found" not in str(result) and "0 unique chunks" not in str(result):
                        no_results_found = False
                        
                elif tool_name == "get_document":
                    # ROOSEVELT'S CRITICAL FIX: Include FULL document content for deep analysis!
                    # Handle both string and dict formats
                    if isinstance(result, str):
                        # String format from get_document_content
                        search_result_content += f"\n**Full Document Retrieved:** {result}"
                        if result and result.strip() and "failed" not in result.lower():
                            no_results_found = False
                            logger.info(f"📄 FULL DOCUMENT CONTENT (STRING): get_document returned {len(result)} chars for analysis")
                    elif isinstance(result, dict):
                        # Dict format - extract content
                        content = result.get("content", "")
                        if content:
                            search_result_content += f"\n**Full Document Retrieved:** {content}"
                            no_results_found = False
                            logger.info(f"📄 FULL DOCUMENT CONTENT (DICT): get_document returned {len(content)} chars for analysis")
                        else:
                            logger.warning(f"⚠️ get_document returned dict but no content: {result}")
                    else:
                        logger.warning(f"⚠️ get_document returned unexpected type: {type(result)}")
                        
                elif tool_name in ["search_and_crawl", "crawl_web_content", "search_web"]:
                    # ROOSEVELT'S FIX: Include web search results in prompt!
                    search_result_content += f"\n**Web Search Result ({tool_name}):** {str(result)[:1000]}..."
                    # Web results should never be considered "no results"
                    if result and str(result).strip() and "failed" not in str(result).lower():
                        no_results_found = False
                        logger.info(f"🌐 WEB RESULTS DETECTED: {tool_name} returned {len(str(result))} chars")
            
            # ROOSEVELT'S COMPREHENSIVE MODE: Always in comprehensive mode now with local + web access
            if no_results_found:
                logger.warning("⚠️ NO RESULTS DETECTED: Tools returned no results - may indicate tool execution problem")
                search_result_content += "\n**IMPORTANT**: The search tools returned no results or empty data. This may indicate a tool execution problem. Provide the best answer possible based on available information and acknowledge if information is limited."
            
            # ROOSEVELT'S COMPREHENSIVE RESEARCH LOGIC: Always have web access now
            web_tools_used = [tool for tool in tools_used if tool in ['search_web', 'crawl_web_content', 'search_and_crawl']]
            has_web_results = len(web_tools_used) > 0
            
            # Always complete - we have access to all tools (local + web)
            decision_logic = """5. **DECISION LOGIC (COMPREHENSIVE RESEARCH)**:
   - You have access to ALL research tools (local + web)
   - If search returned relevant results → task_status: "complete"
   - If search found comprehensive information → task_status: "complete"
   - If results are limited, acknowledge this but still mark as complete
   - CRITICAL: Always complete the task - you have full research access!"""
            
            # ROOSEVELT'S CONTENT DEBUG: Log what content is being passed to final LLM
            try:
                logger.info(f"🔍 FINAL PROMPT CONTENT DEBUG: search_result_content length = {len(search_result_content or '')}")
            except Exception:
                logger.info("🔍 FINAL PROMPT CONTENT DEBUG: search_result_content length = unknown (non-string)")
            logger.info(f"🔍 FINAL PROMPT PREVIEW: {search_result_content[:1000]}...")
            
            final_prompt = f"""Based on the ACTUAL tool execution results below, provide your final research answer.

QUERY CONTEXT: Research mode: {research_mode}
TOOLS USED: {', '.join(tools_used)}
WEB TOOLS EXECUTED: {', '.join(web_tools_used) if web_tools_used else "None"}

ACTUAL SEARCH RESULTS:
{search_result_content if search_result_content else "No search results available"}

INTELLIGENT EVALUATION RULES:
1. **RELEVANCE ASSESSMENT**: Evaluate if search results are actually relevant to the user's query (not just keyword matches)
2. **CONTENT QUALITY**: Assess if available information adequately answers the query 
3. **COMPLETENESS CHECK**: Determine if the response would be helpful and comprehensive
4. **CURRENCY NEEDS**: Consider if the query requires current/real-time information
{decision_logic}
6. **HONESTY PRINCIPLE**: NEVER claim to have found information not in the actual search results
7. **NO HALLUCINATION**: Base findings ONLY on actual search result content

RESPONSE FORMATTING REQUIREMENTS:
- **USE PROPER MARKDOWN**: Format your responses with clear headers (##, ###), lists, and emphasis
- **CLEAR STRUCTURE**: Organize information with logical headers and subheadings
- **READABLE FORMAT**: Use bullet points, numbered lists, and proper spacing
- **BOLD KEY POINTS**: Emphasize important information with **bold text**
- **IN-LINE CITATIONS**: Use numbered citations (1), (2), (3) format when referencing sources
- **CITATION INTEGRATION**: Smoothly integrate citations into sentences, e.g., "According to recent studies (1), this approach shows promise."
- **REGIONAL BREAKDOWN**: When query asks for regional analysis, organize by geographic regions with clear headers
- **COMPARATIVE ANALYSIS**: When comparing topics, use clear sections for each item being compared
- **COMPREHENSIVE COVERAGE**: Address ALL aspects requested in the user's query

CRITICAL: You MUST respond with ONLY valid JSON. No markdown, no explanation, just pure JSON.

Required JSON format:
{{
    "task_status": "complete",
    "findings": "Your well-formatted findings with proper headers and structure based ONLY on the actual search results shown above",
    "sources_searched": {tools_used if tools_used else ["documents", "entities", "web"]},
    "confidence_level": 0.8,
    "next_steps": null
}}"""
            
            # Get OpenAI client
            chat_service = await self._get_chat_service()
            model_name = await self._get_model_name()
            
            # Create messages for final response
            messages = [
                {"role": "system", "content": final_prompt},
                {"role": "user", "content": f"Synthesize the research results for the {research_mode} research task."}
            ]
            
            logger.info("🤖 Getting final structured response using OpenAI client...")
            final_response = await chat_service.openai_client.chat.completions.create(
                messages=messages,
                model=model_name,
                temperature=0.1
                # No tools needed for final synthesis
            )
            
            # Extract the final content (OpenAI format)
            if (hasattr(final_response, 'choices') and 
                final_response.choices and 
                hasattr(final_response.choices[0].message, 'content') and 
                final_response.choices[0].message.content):
                
                content = final_response.choices[0].message.content
                logger.info(f"✅ Got final response: {str(content)[:200]}...")
                return content
            else:
                logger.error("❌ Final response was empty")
                return '{"task_status": "error", "findings": "Tool execution completed but final response was empty", "sources_searched": ["error"], "confidence_level": 0.1}'
                
        except Exception as e:
            logger.error(f"❌ Final response generation failed: {e}")
            return f'{{"task_status": "error", "findings": "Final response generation failed: {str(e)}", "sources_searched": ["error"], "confidence_level": 0.1}}'
    
    async def _get_agent_tools_async(self) -> List[Dict[str, Any]]:
        """Get agent tools asynchronously to avoid event loop conflicts"""
        try:
            from services.langgraph_tools.centralized_tool_registry import get_tool_objects_for_agent
            
            # Get tool objects in OpenAI format (not just tool names)
            agent_tools = await get_tool_objects_for_agent(self.agent_type_enum)
            
            # Log available tools
            tool_names = [tool["function"]["name"] for tool in agent_tools]
            logger.info(f"🔧 {self.agent_type} has access to tools: {tool_names}")
            
            return agent_tools
            
        except Exception as e:
            logger.error(f"❌ Failed to get agent tools for {self.agent_type}: {e}")
            return []
    
    def _parse_structured_response(self, content: str, state: Dict[str, Any] = None) -> 'ResearchTaskResult':
        """Parse structured response - Roosevelt's Citation-Enhanced JSON Parser"""
        try:
            from models.agent_response_models import ResearchTaskResult
            from services.citation_formatting_service import get_citation_formatting_service
            import json
            import re
            
            logger.info(f"🎯 ROOSEVELT'S CITATION-ENHANCED PARSER: Processing response length: {len(str(content))}")
            
            # If it's already a ResearchTaskResult object, return it
            if hasattr(content, 'task_status'):
                return content
            
            # If it's a dict, convert directly but process citations
            if isinstance(content, dict):
                # Extract tool results for citation processing
                tool_results = []
                if state:
                    shared_memory = state.get("shared_memory", {})
                    tool_results = shared_memory.get("tool_results", [])
                
                # Process citations if findings exist
                if "findings" in content and tool_results:
                    citation_service = get_citation_formatting_service()
                    formatted_findings, numbered_citations = citation_service.format_research_response_with_citations(
                        content["findings"], 
                        tool_results
                    )
                    content["findings"] = formatted_findings
                    content["citations"] = [citation.dict() for citation in numbered_citations]
                
                return ResearchTaskResult(**content)
            
            # If it's a string, clean and parse
            if isinstance(content, str):
                json_text = content.strip()
                
                # Roosevelt's Universal JSON Cleaning - handle any wrapping
                if '```json' in json_text:
                    # Extract from markdown code blocks
                    match = re.search(r'```json\s*\n(.*?)\n```', json_text, re.DOTALL)
                    if match:
                        json_text = match.group(1).strip()
                elif '```' in json_text:
                    # Extract from generic code blocks
                    match = re.search(r'```\s*\n(.*?)\n```', json_text, re.DOTALL)
                    if match:
                        json_text = match.group(1).strip()
                
                # Find JSON object if mixed with other text
                if not json_text.startswith('{'):
                    match = re.search(r'\{.*\}', json_text, re.DOTALL)
                    if match:
                        json_text = match.group(0)
                
                logger.info(f"🎯 CLEANED JSON: {json_text[:200]}...")
                
                # Parse and validate with citation processing; if not JSON, wrap as fallback
                try:
                    data = json.loads(json_text)
                except Exception:
                    # Fallback to structured response using plain content
                    from models.agent_response_models import ResearchTaskResult, TaskStatus
                    logger.info("⚠️ Non-JSON LLM response detected; returning structured fallback")
                    return ResearchTaskResult(
                        task_status=TaskStatus.COMPLETE,
                        findings=content,
                        sources_searched=["documents", "web"],
                        confidence_level=0.7
                    )
                
                # Extract tool results for citation processing
                tool_results = []
                if state:
                    shared_memory = state.get("shared_memory", {})
                    tool_results = shared_memory.get("tool_results", [])
                
                # Process citations if findings exist
                if "findings" in data and tool_results:
                    citation_service = get_citation_formatting_service()
                    formatted_findings, numbered_citations = citation_service.format_research_response_with_citations(
                        data["findings"], 
                        tool_results
                    )
                    data["findings"] = formatted_findings
                    data["citations"] = [citation.dict() for citation in numbered_citations]
                
                result = ResearchTaskResult(**data)
                logger.info(f"✅ PARSED SUCCESSFULLY: task_status={result.task_status}")
                return result
            
            # Fallback
            return content
                
        except Exception as e:
            logger.error(f"❌ JSON parsing failed: {e}")
            logger.error(f"❌ Raw content: {str(content)[:500]}...")
            
            # Create error response
            from models.agent_response_models import ResearchTaskResult, TaskStatus
            return ResearchTaskResult(
                task_status=TaskStatus.ERROR,
                findings=f"Failed to parse JSON response: {str(e)}. Raw content: {str(content)[:200]}",
                sources_searched=["error"],
                confidence_level=0.1
            )
    
    # REMOVED: _create_permission_request - No longer needed since we do comprehensive research directly without permission
    
    def _create_error_response(self, state: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        from models.agent_response_models import ResearchTaskResult, TaskStatus
        
        error_result = ResearchTaskResult(
            task_status=TaskStatus.ERROR,
            findings=f"Research failed: {error_message}",
            sources_searched=["error"],
            confidence_level=0.0
        )
        
        state["agent_results"] = {
            "structured_response": error_result.dict(),
            "error": True,
            "error_message": error_message,
            "timestamp": datetime.now().isoformat()
        }
        state["is_complete"] = True
        return state
    
    def _extract_citations_unified(
        self, 
        search_results: Dict[str, Any], 
        source_label: str = "unknown"
    ) -> List[Dict[str, Any]]:
        """
        **ROOSEVELT'S UNIFIED CITATION EXTRACTOR**
        
        Handles ALL citation types with consistent CitationSource-compliant output:
        - Local documents (from vector search - string format)
        - Web search results (from search_and_crawl - dict format)
        - Mixed results (from comprehensive search)
        
        Args:
            search_results: Dict containing 'results' key with either string or dict format
            source_label: Human-readable label for logging (e.g., "Round 1", "Round 2")
        
        Returns:
            List of standardized CitationSource-compatible dicts
        """
        citations = []
        
        try:
            results_data = search_results.get('results', None)
            
            if results_data is None:
                logger.info(f"📚 CITATION EXTRACTION ({source_label}): No results data found")
                return citations
            
            # Detect format and delegate to appropriate parser
            if isinstance(results_data, dict):
                # Web format: {results: {search_results: [...], ...}}
                logger.info(f"🌐 CITATION EXTRACTION ({source_label}): Detected WEB format (dict)")
                citations = self._parse_web_citations(results_data)
                
            elif isinstance(results_data, str):
                # Local format: {results: "**1. Document name** (Score: 0.5)..."}
                logger.info(f"📚 CITATION EXTRACTION ({source_label}): Detected LOCAL format (string)")
                citations = self._parse_local_citations(results_data)
                
            else:
                logger.warning(f"⚠️ CITATION EXTRACTION ({source_label}): Unknown format type: {type(results_data)}")
            
            logger.info(f"✅ EXTRACTED {len(citations)} citations from {source_label}")
            return citations
            
        except Exception as e:
            logger.error(f"❌ Citation extraction failed for {source_label}: {e}")
            return []
    
    def _parse_web_citations(self, web_results_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        **ROOSEVELT'S WEB CITATION PARSER**
        
        Parses web search results from search_and_crawl format:
        {
            "search_results": [
                {"title": "...", "url": "...", "snippet": "...", ...},
                ...
            ]
        }
        
        Returns: List of CitationSource-compatible dicts
        """
        citations = []
        
        try:
            search_results_list = web_results_dict.get('search_results', [])
            
            for idx, result in enumerate(search_results_list, 1):
                citation = {
                    "id": idx,
                    "title": result.get("title", "Web Result"),
                    "type": "webpage",
                    "url": result.get("url", ""),
                    "author": result.get("author"),
                    "date": result.get("published_date") or result.get("accessed_date"),
                    "excerpt": self._extract_excerpt(result)
                }
                citations.append(citation)
                logger.info(f"🌐 WEB CITATION ({idx}): {citation['title'][:50]}... - {citation['url'][:50]}...")
                
        except Exception as e:
            logger.warning(f"⚠️ Web citation parsing failed: {e}")
        
        return citations
    
    def _parse_local_citations(self, local_results_string: str) -> List[Dict[str, Any]]:
        """
        **ROOSEVELT'S LOCAL CITATION PARSER**
        
        Parses local document search results from string format:
        "**1. Document doc_name** (Score: 0.567, Collection: global)
         Content: ...
         **2. Document another_doc** (Score: 0.558, Collection: user)
         ..."
        
        Returns: List of CitationSource-compatible dicts
        """
        citations = []
        cited_documents = set()
        
        try:
            import re
            
            # Pattern: **N. Document doc_name** (Score: X.XXX, Collection: xxx)
            doc_pattern = r'\*\*(\d+)\.\s+Document\s+(.+?)\*\*\s+\(Score:\s+([\d.]+),\s+Collection:\s+(\w+)\)'
            
            for match in re.finditer(doc_pattern, local_results_string):
                doc_name = match.group(2).strip()
                score = float(match.group(3))
                collection = match.group(4)
                
                # Avoid duplicate citations
                if doc_name in cited_documents:
                    continue
                
                cited_documents.add(doc_name)
                
                citation = {
                    "id": len(citations) + 1,
                    "title": doc_name,
                    "type": "document",
                    "url": f"/documents/{doc_name}",  # Internal document link
                    "author": None,  # Not available in search format
                    "date": None,  # Not available in search format
                    "excerpt": self._extract_excerpt_from_local_result(local_results_string, doc_name),
                    "collection": collection,
                    "relevance_score": score
                }
                citations.append(citation)
                logger.info(f"📄 LOCAL CITATION ({citation['id']}): {doc_name} - {collection} (score: {score:.3f})")
                
                # Limit to top 10 citations
                if len(citations) >= 10:
                    break
                    
        except Exception as e:
            logger.warning(f"⚠️ Local citation parsing failed: {e}")
        
        return citations
    
    def _extract_excerpt(self, result: Dict[str, Any]) -> str:
        """Extract excerpt from various result formats"""
        # Try different possible fields
        if result.get("content"):
            return result["content"][:200]
        elif result.get("snippet"):
            return result["snippet"][:200]
        elif result.get("description"):
            return result["description"][:200]
        return ""
    
    def _extract_excerpt_from_local_result(self, results_string: str, doc_name: str) -> str:
        """Extract content excerpt for a specific document from search results"""
        try:
            # Find the document section
            import re
            # Look for content after the document header
            pattern = rf'\*\*\d+\.\s+Document\s+{re.escape(doc_name)}\*\*.*?\n\s*Content:(.*?)(?=\*\*\d+\.|$)'
            match = re.search(pattern, results_string, re.DOTALL)
            if match:
                content = match.group(1).strip()
                return content[:200]
        except Exception:
            pass
        return ""
    
    def _extract_local_citations_from_search_result(self, search_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        **DEPRECATED**: Use _extract_citations_unified() instead
        
        This method is kept for backward compatibility but should not be used for new code.
        The unified extractor handles both local and web formats automatically.
        """
        citations = []
        citation_id = 1
        cited_documents = set()
        
        try:
            # Parse the formatted search results string
            results_text = search_result.get("results", "")
            if not results_text:
                return citations
            
            # Extract document entries using regex
            import re
            # Pattern: **N. Document doc_name** (Score: X.XXX, Collection: xxx)
            doc_pattern = r'\*\*(\d+)\.\s+Document\s+(.+?)\*\*\s+\(Score:\s+([\d.]+),\s+Collection:\s+(\w+)\)'
            
            for match in re.finditer(doc_pattern, results_text):
                doc_name = match.group(2).strip()
                score = float(match.group(3))
                collection = match.group(4)
                
                # Try to find document_id in the content section (if available)
                # For now, use doc_name as a proxy
                if doc_name in cited_documents:
                    continue
                
                cited_documents.add(doc_name)
                
                citation = {
                    "id": citation_id,
                    "title": doc_name,
                    "url": f"/documents/{doc_name}",  # Will be enhanced with real ID
                    "type": "document",
                    "collection": collection,
                    "relevance_score": score
                }
                citations.append(citation)
                logger.info(f"📄 EFFICIENT CITATION: ({citation_id}) {doc_name} - {collection} (score: {score:.3f})")
                citation_id += 1
                
                # Limit to top 10 citations for efficiency
                if citation_id > 10:
                    break
                    
        except Exception as e:
            logger.warning(f"⚠️ Failed to extract local citations from search result: {e}")
        
        return citations
    
    def _extract_citations_from_tool_results(self, state: Dict[str, Any], research_mode: str) -> List[Dict[str, Any]]:
        """Extract citations from tool results stored in shared_memory - Roosevelt's NUMBERED Citation System!"""
        citations = []
        
        try:
            shared_memory = state.get("shared_memory", {})
            search_results = shared_memory.get("search_results", {})
            
            # ROOSEVELT'S CITATION COUNTER: Start numbering from 1
            citation_id = 1
            
            # **ROOSEVELT'S LOCAL DOCUMENT CITATIONS**: Extract from search_local results first!
            local_search_data = search_results.get("local")
            if local_search_data and isinstance(local_search_data, dict):
                local_results = local_search_data.get("results", [])
                # Track unique documents to avoid duplicate citations
                cited_documents = set()
                
                for result in local_results[:20]:  # Top 20 local results for citations
                    document_id = result.get("document_id", "")
                    if not document_id or document_id in cited_documents:
                        continue
                    
                    cited_documents.add(document_id)
                    
                    # Extract metadata
                    metadata = result.get("metadata", {})
                    title = result.get("citation_title") or metadata.get("title") or result.get("filename", "Local Document")
                    collection = result.get("collection", "unknown")
                    score = result.get("score", 0.0)
                    
                    # Create citation for local document
                    citation = {
                        "id": citation_id,
                        "title": title,
                        "url": f"/documents/{document_id}",  # Internal document link
                        "type": "document",  # Local document type
                        "author": metadata.get("author"),
                        "date": metadata.get("created_at") or metadata.get("date"),
                        "excerpt": result.get("content", "")[:200],  # First 200 chars
                        "document_id": document_id,
                        "collection": collection,
                        "relevance_score": score
                    }
                    citations.append(citation)
                    logger.info(f"📄 LOCAL CITATION: ({citation_id}) {title} - {collection} collection (score: {score:.3f})")
                    citation_id += 1
            
            # Extract citations from search_and_crawl results (web)
            search_and_crawl_data = search_results.get("search_and_crawl")
            if search_and_crawl_data and isinstance(search_and_crawl_data, dict):
                search_results_list = search_and_crawl_data.get("search_results", [])
                for result in search_results_list:
                    # ROOSEVELT'S CITATION FIX: Match CitationSource Pydantic model field names
                    citation = {
                        "id": citation_id,  # FIX: Add ID for numbered citations (1), (2), etc.
                        "title": result.get("title", "Web Result"),  # FIX: title not source_title
                        "url": result.get("url", ""),  # FIX: url not source_url
                        "type": "webpage",  # CitationSource expects "document", "webpage", or "book"
                        "author": result.get("author"),  # Optional author
                        "date": result.get("published_date") or result.get("accessed_date"),  # Optional date
                        "excerpt": result.get("snippet", "")  # Optional excerpt
                    }
                    citations.append(citation)
                    logger.info(f"🔗 WEB CITATION: ({citation_id}) {result.get('title', 'Unknown')} - {result.get('url', 'No URL')}")
                    citation_id += 1
                    
            # Extract citations from tool_results (crawled URLs)
            tool_results = shared_memory.get("tool_results", [])
            for tool_result in tool_results:
                if tool_result.get("tool_name") == "crawl_web_content":
                    result_content = tool_result.get("result", "")
                    # Extract URL from crawl result content (format: "🕷️ **Crawled 1 URLs:**\n\n**Title**\nURL: https://...")
                    import re
                    url_match = re.search(r'URL: (https?://[^\s\n]+)', str(result_content))
                    title_match = re.search(r'\*\*(.*?)\*\*\nURL:', str(result_content))
                    
                    if url_match:
                        # ROOSEVELT'S CITATION FIX: Match CitationSource Pydantic model field names
                        citation = {
                            "id": citation_id,  # FIX: Add ID for numbered citations
                            "title": title_match.group(1) if title_match else "Crawled Web Content",  # FIX: title not source_title
                            "url": url_match.group(1),  # FIX: url not source_url
                            "type": "webpage",  # CitationSource expects "document", "webpage", or "book"
                            "author": None,  # No author for crawled content
                            "date": datetime.now().strftime("%Y-%m-%d"),  # Accessed date
                            "excerpt": "Full content extracted via Crawl4AI"  # FIX: excerpt not quote_text
                        }
                        citations.append(citation)
                        logger.info(f"🕷️ CRAWL CITATION EXTRACTED: ({citation_id}) {citation['title']} - {citation['url']}")
                        citation_id += 1
                        
        except Exception as e:
            logger.warning(f"⚠️ Citation extraction from tool results failed: {e}")
        
        # Count citation types for logging
        local_count = sum(1 for c in citations if c.get("type") == "document")
        web_count = sum(1 for c in citations if c.get("type") == "webpage")
        
        logger.info(f"🔗 TOTAL CITATIONS: {len(citations)} sources ({local_count} local documents, {web_count} web pages)")
        return citations

    def _extract_citations_from_response_openai(self, response, research_mode: str) -> List[Dict[str, Any]]:
        """Extract citations from tool calls and response content (OpenAI format)"""
        citations = []
        
        try:
            # Extract from tool calls if available (OpenAI format)
            if (hasattr(response, 'choices') and 
                response.choices and 
                hasattr(response.choices[0].message, 'tool_calls') and 
                response.choices[0].message.tool_calls):
                
                for tool_call in response.choices[0].message.tool_calls:
                    import json
                    tool_name = tool_call.function.name if hasattr(tool_call, 'function') else "unknown"
                    try:
                        tool_args = json.loads(tool_call.function.arguments) if hasattr(tool_call, 'function') else {}
                    except:
                        tool_args = {}
                    
                    citations.append({
                        "type": "tool_usage",
                        "tool_name": tool_name,
                        "search_query": tool_args.get("query", ""),
                        "research_mode": research_mode
                    })
            
            # Extract URLs from content if any (OpenAI format)
            if (hasattr(response, 'choices') and 
                response.choices and 
                hasattr(response.choices[0].message, 'content') and 
                response.choices[0].message.content):
                
                import re
                content = response.choices[0].message.content
                urls = re.findall(r'https?://[^\s\)>\]"}]+', str(content))
                for url in urls:
                    citations.append({
                        "type": "web_reference",
                        "url": url,
                        "research_mode": research_mode
                    })
                    
        except Exception as e:
            logger.warning(f"⚠️ Citation extraction failed: {e}")
        
        return citations
    
    def _update_shared_memory_with_results(self, state: Dict[str, Any], result: 'ResearchTaskResult', mode: str):
        """Update shared memory with research results"""
        try:
            shared_memory = state.get("shared_memory", {})
            if "search_results" not in shared_memory:
                shared_memory["search_results"] = {}
            
            shared_memory["search_results"][mode] = {
                "findings": result.findings,
                "sources": result.sources_searched,
                "confidence": result.confidence_level,
                "timestamp": datetime.now().isoformat()
            }
            
            state["shared_memory"] = shared_memory
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to update shared memory: {e}")
    
    # REMOVED: _build_local_research_prompt - No longer needed since we do comprehensive research directly
    
    # REMOVED: _build_comprehensive_research_prompt - Using _build_web_focused_research_prompt instead
    
    def _build_web_focused_research_prompt(self, query: str, expanded_queries_data: Dict[str, Any] = None, shared_memory: Dict[str, Any] = None) -> str:
        """Build Grok-style iterative research prompt with local-first priority - ENHANCED PATTERN"""
        
        # **ROOSEVELT'S EFFICIENCY**: Check if we already did initial local search
        initial_search_done = shared_memory.get("initial_local_search_done", False) if shared_memory else False
        initial_search_results = shared_memory.get("initial_search_results", {}) if shared_memory else {}
        
        # Build initial search context if available
        initial_search_context = ""
        if initial_search_done and initial_search_results:
            initial_search_context = f"""
🎯 **ROOSEVELT'S EFFICIENCY - LOCAL SEARCH ALREADY COMPLETED!**

We already performed an initial local search with the following results:
- **Results Found**: {initial_search_results.get('result_count', 0)} documents
- **Top Relevance Score**: {initial_search_results.get('top_score', 0):.3f}
- **Quality Assessment**: {initial_search_results.get('quality_assessment', 'Unknown')}

**IMPORTANT**: You DON'T need to search local documents again with the same query!
The local search was deemed insufficient (not enough high-quality results).
Your job: Either refine local search with DIFFERENT angles OR go to web sources to fill gaps.

"""
        
        # Build expanded query context
        expanded_query_context = ""
        if expanded_queries_data and expanded_queries_data.get("expanded_queries"):
            expanded_queries = expanded_queries_data.get("expanded_queries", [])
            all_queries = expanded_queries_data.get("all_queries", [query])
            
            expanded_query_context = f"""
ROOSEVELT'S INTELLIGENT QUERY EXPANSION:
Your query has been intelligently expanded to improve search coverage:

ORIGINAL QUERY: {query}

EXPANDED QUERIES AVAILABLE:
"""
            for i, exp_query in enumerate(expanded_queries, 1):
                expanded_query_context += f"{i}. {exp_query}\n"
            
            expanded_query_context += f"""
ALL QUERIES TO CONSIDER: {all_queries}

"""
        
        # **ROOSEVELT TAG FILTERING**: Build filter instructions if tags were detected
        filter_instructions = ""
        if shared_memory:
            detected_filters = shared_memory.get("detected_filters", {})
            if detected_filters.get("should_filter"):
                filter_category = detected_filters.get("filter_category")
                filter_tags = detected_filters.get("filter_tags", [])
                confidence = detected_filters.get("confidence", "none")
                
                filter_instructions = "\n🏷️ **ROOSEVELT'S SMART FILTERING DETECTED!**\n\n"
                filter_instructions += f"Your query contains references to document tags/categories! (Confidence: {confidence})\n"
                
                if filter_tags:
                    filter_instructions += f"**Detected Tags**: {', '.join(filter_tags)}\n"
                if filter_category:
                    filter_instructions += f"**Detected Category**: {filter_category}\n"
                
                filter_instructions += "\n**CRITICAL**: When calling search_local, you MUST include these parameters:\n"
                if filter_tags:
                    filter_instructions += f"  filter_tags={filter_tags}\n"
                if filter_category:
                    filter_instructions += f"  filter_category='{filter_category}'\n"
                
                filter_instructions += "\nExample: search_local(query='your search', "
                if filter_tags:
                    filter_instructions += f"filter_tags={filter_tags}"
                if filter_category:
                    if filter_tags:
                        filter_instructions += ", "
                    filter_instructions += f"filter_category='{filter_category}'"
                filter_instructions += ")\n\n"
        
        # Build adaptive strategy based on whether initial search was done
        if initial_search_done:
            strategy_section = f"""🎯 ROOSEVELT'S ADAPTIVE RESEARCH STRATEGY:

You have UP TO 8 ROUNDS of tool calling. Use them wisely!

**STARTING POINT - INITIAL LOCAL SEARCH ALREADY DONE:**
We already searched local documents with the original query (see results above).
The results were insufficient, so NOW:

**ROUND 1 - FILL THE GAPS:**
1. Analyze what's missing from initial local results
2. Either:
   a) Search local with DIFFERENT angle/expanded queries, OR
   b) Go directly to web with search_and_crawl for missing information
3. Don't repeat the same local search - we already did that!"""
        else:
            strategy_section = """🎯 ROOSEVELT'S GROK-STYLE ITERATIVE RESEARCH STRATEGY:

You have UP TO 8 ROUNDS of tool calling. Use them wisely for iterative refinement!

**ROUND 1 - LOCAL-FIRST SEARCH:**
1. Start with search_local to check YOUR KNOWLEDGE BASE FIRST
2. Use get_document for promising local documents
3. Analyze what you found - is it sufficient?"""
        
        return f"""You are Roosevelt's Research Agent with GROK-STYLE ITERATIVE RESEARCH CAPABILITY!
{initial_search_context}{expanded_query_context}{filter_instructions}

TASK: Execute iterative research with LOCAL-FIRST priority and adaptive refinement.

QUERY: {query}

AVAILABLE TOOLS:
- **search_local**: Search local documents and entities
  - OPTIONAL: filter_tags (list of strings) - Filter results by document tags
  - OPTIONAL: filter_category (string) - Filter results by document category
- **get_document**: Retrieve full content from local documents
- **search_and_crawl**: 🌟 PRIMARY WEB TOOL - Search AND crawl top results with Crawl4AI for full content
- **crawl_web_content**: Extract full content from specific URLs

{strategy_section}

**AFTER EACH ROUND - GAP ANALYSIS:**
After seeing tool results, ask yourself:
- Do I have enough information to answer comprehensively?
- What specific gaps remain (dates, details, perspectives)?
- What additional searches would fill these gaps?

**ITERATIVE REFINEMENT:**
If gaps exist after local search:
- Round 2: Use search_and_crawl for WEB content on specific gaps
- Round 3: Refine with more targeted queries if needed
- Round 4+: Keep refining until you have comprehensive coverage OR reach iteration limit

**MULTI-ANGLE SEARCH STRATEGY:**
- Use ORIGINAL query for broad coverage
- Use EXPANDED variations (listed above) for specific angles
- Refine queries based on what you learn in each round
- Don't repeat the same search - learn and adapt!

**LOCAL-FIRST PRIORITY:**
✅ ALWAYS search local sources BEFORE going to web
✅ Prioritize user's documents - they chose to save them for a reason
✅ Only go to web if local sources are insufficient
✅ Combine local + web for comprehensive answers

**WHEN TO STOP:**
✅ You have comprehensive information covering all query aspects
✅ Additional searches unlikely to add significant value
✅ You've reached 8 iterations (synthesis time!)

**EXAMPLE ITERATIVE FLOW:**
Round 1: search_local("AI ethics") → Found 50 docs
         Analyze: Good overview, but missing recent 2024 developments
Round 2: search_and_crawl("AI ethics 2024 developments") → Web results
         Analyze: Good recent info, but missing deontological perspective
Round 3: search_local("deontological ethics") → Found philosophy books
         Analyze: Excellent! Now have complete picture
Round 4: Synthesize final comprehensive answer

STRUCTURED OUTPUT FORMAT:
- task_status: **"complete"** (when you have sufficient information)
- findings: Your comprehensive answer combining all search rounds
- sources_searched: ["documents", "entities", "web"] (list what you actually used)
- confidence_level: Based on information quality and coverage
- next_steps: null (task complete)

RESPONSE FORMATTING REQUIREMENTS:
- **USE PROPER MARKDOWN**: Format with clear headers (##, ###), lists, and emphasis
- **CLEAR STRUCTURE**: Organize with logical headers and subheadings
- **READABLE FORMAT**: Use bullet points, numbered lists, proper spacing
- **BOLD KEY POINTS**: Emphasize important information
- **CITE SOURCES**: Mention where information came from (local docs vs web)
- **COMPREHENSIVE COVERAGE**: Address ALL aspects of the query

Required JSON format:
{{
    "task_status": "complete",
    "findings": "Your comprehensive, well-formatted answer from iterative research",
    "sources_searched": ["documents", "entities", "web"],
    "confidence_level": 0.9,
    "next_steps": null
}}

BULLY! Use iterative refinement to build comprehensive research - LOCAL FIRST, then WEB as needed!"""
    
    # REMOVED: _build_web_expansion_prompt - No longer needed since we do comprehensive research directly without local-first
    
    def _get_original_research_query(self, state: Dict[str, Any]) -> str:
        """Get the original research query, avoiding permission response contamination and topic contamination"""
        try:
            # ROOSEVELT'S ENHANCED QUERY EXTRACTION: Prioritize shared memory for resumed conversations
            shared_memory = state.get("shared_memory", {})
            
            # 1. PRIORITY: Try original query from shared memory (for resumed requests after permission)
            original_query = shared_memory.get("original_user_query", "").strip()
            if original_query and original_query.lower() not in ["yes", "no", "y", "n", "ok", "proceed", "approved"]:
                logger.info(f"🎯 QUERY EXTRACTION: Using preserved original query from shared memory: '{original_query[:50]}...'")
                return original_query
            
            # 2. Try expanded queries data for original query (backup location)
            expanded_queries_data = shared_memory.get("expanded_queries", {})
            if isinstance(expanded_queries_data, dict):
                expansion_original = expanded_queries_data.get("original_query", "").strip()
                if expansion_original and expansion_original.lower() not in ["yes", "no", "y", "n", "ok", "proceed", "approved"]:
                    logger.info(f"🎯 QUERY EXTRACTION: Using original query from expansion data: '{expansion_original[:50]}...'")
                    return expansion_original
            
            # 3. Try current_query field (for fresh requests)
            current_query = state.get("current_query", "").strip()
            if current_query and current_query.lower() not in ["yes", "no", "y", "n", "ok", "proceed", "approved"]:
                logger.info(f"🎯 QUERY EXTRACTION: Using current_query field: '{current_query[:50]}...'")
                return current_query
                
            # 4. Fall back to latest non-permission user message (reverse order for latest)
            messages = state.get("messages", [])
            for msg in reversed(messages):  # Go backwards to get the latest user message
                if hasattr(msg, 'type') and msg.type == "human":
                    content = msg.content.strip()
                    # Skip common permission responses
                    if content.lower() not in ["yes", "no", "y", "n", "ok", "proceed", "approved", "deny", "denied"]:
                        logger.info(f"🎯 QUERY EXTRACTION: Using message fallback: '{content[:50]}...'")
                        return content
                elif isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content", "").strip()
                    if content.lower() not in ["yes", "no", "y", "n", "ok", "proceed", "approved", "deny", "denied"]:
                        logger.info(f"🎯 QUERY EXTRACTION: Using dict message fallback: '{content[:50]}...'")
                        return content
            
            logger.warning("⚠️ QUERY EXTRACTION: No valid query found in any location!")
            return ""
        except Exception as e:
            logger.error(f"❌ Failed to get original research query: {e}")
            return ""

    def _get_latest_user_message(self, state: Dict[str, Any]) -> str:
        """Extract the latest user message from state"""
        try:
            messages = state.get("messages", [])
            for msg in reversed(messages):
                if hasattr(msg, 'type') and msg.type == "human":
                    return msg.content
                elif isinstance(msg, dict) and msg.get("role") == "user":
                    return msg.get("content", "")
            return ""
        except Exception:
            return ""
    
    async def _expand_query_for_research(self, state: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        ROOSEVELT'S QUERY EXPANSION: Generate alternative queries before research
        Stores expanded queries in shared_memory for reuse across local and web search
        """
        try:
            logger.info(f"🔍 QUERY EXPANSION: Expanding query '{query[:50]}...'")
            
            # Use the new expand_query tool via centralized registry
            from services.langgraph_tools.centralized_tool_registry import get_tool_registry, AgentType
            tool_registry = await get_tool_registry()
            
            # Get the expand_query function directly
            expand_function = tool_registry.get_tool_function("expand_query", AgentType.RESEARCH_AGENT)
            if expand_function:
                expansion_result = await expand_function(
                    original_query=query,
                    num_expansions=2,
                    expansion_type="semantic"
                )
            else:
                logger.warning("🔍 expand_query tool not available - using fallback")
                expansion_result = None
            
            # Parse the JSON result
            import json
            if expansion_result and isinstance(expansion_result, str):
                try:
                    expansion_data = json.loads(expansion_result)
                    logger.info(f"✅ EXPANSION SUCCESS: Parsed {expansion_data.get('expansion_count', 0)} alternative queries")
                except json.JSONDecodeError as e:
                    logger.warning(f"🔍 Failed to parse expansion result: {e}")
                    expansion_data = None
            else:
                logger.warning(f"🔍 Invalid expansion result type: {type(expansion_result)}")
                expansion_data = None
                
            # Fallback if parsing failed
            if not expansion_data:
                logger.info("🔍 Using fallback expansion with original query only")
                expansion_data = {
                    "original_query": query,
                    "expanded_queries": [],
                    "all_queries": [query],
                    "expansion_count": 0
                }
            
            # Store in shared_memory for reuse by web search
            shared_memory = state.get("shared_memory", {})
            shared_memory["expanded_queries"] = expansion_data
            
            # Update state (additive pattern)
            updated_state = state.copy()
            updated_state["shared_memory"] = shared_memory
            
            logger.info(f"✅ QUERY EXPANSION: Generated {expansion_data.get('expansion_count', 0)} alternatives")
            for i, expanded in enumerate(expansion_data.get("expanded_queries", []), 1):
                logger.info(f"   {i}. {expanded}")
            
            return expansion_data
            
        except Exception as e:
            logger.error(f"❌ Query expansion failed: {e}")
            # Return minimal data with just original query
            fallback_data = {
                "original_query": query,
                "expanded_queries": [],
                "all_queries": [query], 
                "expansion_count": 0
            }
            
            # Still store in shared_memory for consistency
            shared_memory = state.get("shared_memory", {})
            shared_memory["expanded_queries"] = fallback_data
            updated_state = state.copy()
            updated_state["shared_memory"] = shared_memory
            
            return fallback_data
    
    # ROOSEVELT'S NATURAL COLLABORATION: Removed collaboration detection method
    # Let the LLM handle collaboration decisions with full conversation context
    
    def _extract_key_entities_from_query(self, query: str) -> List[str]:
        """
        **ROOSEVELT'S ENTITY EXTRACTOR**: Extract key entities/topics from query
        
        Simple pattern-based extraction for common entities:
        - Proper nouns (capitalized words)
        - Company names, brands
        - Known entities (Enron, WorldCom, etc.)
        """
        import re
        
        # Extract capitalized words/phrases (likely entities)
        # Pattern: One or more capitalized words
        entity_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        matches = re.findall(entity_pattern, query)
        
        # Filter out common question words and articles
        stop_words = {'What', 'Where', 'When', 'Who', 'Why', 'How', 'Which', 'The', 'This', 'That', 'These', 'Those'}
        entities = [m for m in matches if m not in stop_words]
        
        # Also check for known multi-word entities
        known_entities = ['Enron', 'WorldCom', 'World Com', 'Bernie Ebbers', 'Kenneth Lay', 'Jeff Skilling']
        for known in known_entities:
            if known.lower() in query.lower() and known not in entities:
                entities.append(known)
        
        # Deduplicate and sort
        entities = sorted(list(set(entities)))
        
        logger.info(f"🔍 ENTITY EXTRACTION: Found {len(entities)} entities in query: {entities}")
        return entities
    
    def _extract_entities_from_results(self, search_results: str) -> List[str]:
        """
        **ROOSEVELT'S RESULT ENTITY EXTRACTOR**: Extract entities mentioned in search results
        
        Looks for capitalized words/phrases in the result text
        """
        import re
        
        # Extract capitalized words/phrases from results
        entity_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        matches = re.findall(entity_pattern, search_results)
        
        # Filter out common words
        stop_words = {'Document', 'Score', 'Collection', 'Global', 'The', 'This', 'That', 'These', 'Those', 'Content'}
        entities = [m for m in matches if m not in stop_words]
        
        # Deduplicate
        entities = sorted(list(set(entities)))
        
        logger.info(f"🔍 RESULT ENTITIES: Found {len(entities)} entities in results: {entities[:10]}...")
        return entities
    
    def _extract_locations_from_research(self, research_findings: str) -> List[str]:
        """Extract location names from research findings for weather collaboration"""
        import re
        
        # Simple location extraction patterns
        location_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:,\s*[A-Z]{2})?)\b',  # City, State format
            r'\b\d{5}\b',  # ZIP codes
        ]
        
        locations = []
        for pattern in location_patterns:
            matches = re.findall(pattern, research_findings)
            locations.extend(matches)
        
        # Filter out common non-location words
        filtered_locations = [
            loc for loc in locations 
            if len(loc) > 2 and loc not in ["The", "And", "For", "With", "This", "That"]
        ]
        
        return filtered_locations[:3]  # Return top 3 locations
    
    # ROOSEVELT'S NATURAL COLLABORATION: Removed collaboration suggestion generation
    # Let the LLM handle collaboration decisions with full conversation context
