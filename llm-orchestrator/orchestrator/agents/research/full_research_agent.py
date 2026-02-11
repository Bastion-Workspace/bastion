"""
Full Research Agent - Multi-round research with gap analysis and synthesis.
Implementation lives in research/ package; state, helpers, routing, and attachment nodes are in separate modules.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from orchestrator.tools.dynamic_tool_analyzer import analyze_tool_needs_for_research
from orchestrator.agents.base_agent import BaseAgent
from orchestrator.models.agent_response_contract import AgentResponse
from orchestrator.subgraphs import (
    build_research_workflow_subgraph,
    build_web_research_subgraph,
    build_assessment_subgraph,
    build_data_formatting_subgraph,
    build_visualization_subgraph,
)

from orchestrator.agents.research.research_state import ResearchState
from orchestrator.agents.research.research_helpers import is_local_search_intent
from orchestrator.agents.research.research_routing import (
    route_from_research_subgraph,
    route_from_full_doc_decision,
    route_from_gap_analysis_check,
    route_from_quick_answer,
    route_from_synthesis,
)
from orchestrator.agents.research.research_skill_config import get_research_skill_config

logger = logging.getLogger(__name__)


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
        from .research_core_nodes import call_research_subgraph_node
        return await call_research_subgraph_node(self, state)

    def _build_workflow(self, checkpointer) -> StateGraph:
        """Build sophisticated multi-round research workflow"""
        
        workflow = StateGraph(ResearchState)

        # Attachment detection (entry point) and analysis subgraph
        workflow.add_node("detect_attachments", self._detect_attachments_node)
        workflow.add_node("attachment_analysis", self._attachment_analysis_node)

        # Add quick answer check node
        workflow.add_node("quick_answer_check", self._quick_answer_check_node)

        # Tier detection and fast path (simple existence queries)
        workflow.add_node("tier_detection", self._tier_detection_node)
        workflow.add_node("fast_path", self._fast_path_node)

        # Legacy attachment processing (kept for backward compatibility)
        workflow.add_node("process_attachments", self._process_attachments_node)
        
        # Add research subgraph node (replaces cache_check, query_expansion, round1_local_search, assess_local_results, round1_web_search, assess_combined_round1, gap_analysis)
        workflow.add_node("research_subgraph", self._call_research_subgraph_node)
        
        # Add full document analysis nodes
        workflow.add_node("full_doc_analysis_decision", self._full_document_analysis_decision_node)
        workflow.add_node("full_doc_analysis_subgraph", self._call_full_document_analysis_subgraph_node)
        workflow.add_node("gap_analysis_check", self._gap_analysis_check_node)
        
        # Add nodes for additional research rounds (unique to FullResearchAgent)
        workflow.add_node("round2_parallel", self._round2_parallel_node)
        workflow.add_node("detect_query_type", self._detect_query_type_node)
        workflow.add_node("final_synthesis", self._final_synthesis_node)
        
        # Entry point - attachment detection first
        workflow.set_entry_point("detect_attachments")

        # Route: if image attachments, run attachment analysis; else quick answer check
        workflow.add_conditional_edges(
            "detect_attachments",
            lambda s: "attachment_analysis" if s.get("has_attachments") else "quick_answer_check",
            {
                "attachment_analysis": "attachment_analysis",
                "quick_answer_check": "quick_answer_check",
            }
        )

        # After attachment analysis, go to final synthesis
        workflow.add_edge("attachment_analysis", "final_synthesis")

        # Quick answer check routing
        workflow.add_conditional_edges(
            "quick_answer_check",
            lambda s: route_from_quick_answer(s),
            {
                "quick_answer": END,
                "process_attachments": "process_attachments",
                "full_research": "tier_detection",
            }
        )

        # Tier detection routing - fast path or full research subgraph
        workflow.add_conditional_edges(
            "tier_detection",
            lambda state: state.get("research_tier", "standard"),
            {
                "fast": "fast_path",
                "standard": "research_subgraph",
                "web": "research_subgraph",
            }
        )

        # Fast path goes directly to query type detection then synthesis
        workflow.add_edge("fast_path", "detect_query_type")
        
        # After processing attachments, go to final synthesis (attachments provide direct answer)
        workflow.add_edge("process_attachments", "final_synthesis")
        
        # Research subgraph routing - check if cache hit or if we need more research
        workflow.add_conditional_edges(
            "research_subgraph",
            lambda s: route_from_research_subgraph(s),
            {
                "use_cache": "detect_query_type",  # Cache hit - go straight to synthesis
                "sufficient": "full_doc_analysis_decision",  # Research sufficient - check if full docs needed
                "needs_round2": "full_doc_analysis_decision"  # Need Round 2 - check if full docs needed first
            }
        )
        
        # Full document analysis decision routing
        workflow.add_conditional_edges(
            "full_doc_analysis_decision",
            lambda s: route_from_full_doc_decision(s),
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
            lambda s: route_from_gap_analysis_check(s),
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
            lambda s: route_from_synthesis(s),
            {
                "post_process": "post_process",
                "complete": END
            }
        )
        
        # Post-processing node goes to end
        workflow.add_edge("post_process", END)
        
        return workflow.compile(checkpointer=checkpointer)
    
    async def _full_document_analysis_decision_node(self, state: ResearchState) -> Dict[str, Any]:
        from .research_core_nodes import full_document_analysis_decision_node
        return await full_document_analysis_decision_node(self, state)

    async def _gap_analysis_check_node(self, state: ResearchState) -> Dict[str, Any]:
        from .research_core_nodes import gap_analysis_check_node
        return await gap_analysis_check_node(self, state)

    async def _call_full_document_analysis_subgraph_node(self, state: ResearchState) -> Dict[str, Any]:
        from .research_core_nodes import call_full_document_analysis_subgraph_node
        return await call_full_document_analysis_subgraph_node(self, state)

    async def _quick_answer_check_node(self, state: ResearchState) -> Dict[str, Any]:
        from .research_fast_path_nodes import quick_answer_check_node
        return await quick_answer_check_node(self, state)

    async def _tier_detection_node(self, state: ResearchState) -> Dict[str, Any]:
        from .research_fast_path_nodes import tier_detection_node
        return await tier_detection_node(self, state)

    async def _fast_path_node(self, state: ResearchState) -> Dict[str, Any]:
        from .research_fast_path_nodes import fast_path_node
        return await fast_path_node(self, state)

    # ===== Workflow Nodes =====
    # Note: The following nodes are now handled by the research workflow subgraph:
    # - cache_check -> research_subgraph
    # - query_expansion -> research_subgraph  
    # - round1_local_search -> research_subgraph (local + entity + images only)
    # - assess_local_results -> research_subgraph (NEW: assess local before web)
    # - round1_web_search -> research_subgraph (CONDITIONAL: only if local insufficient)
    # - assess_combined_round1 -> research_subgraph (only after web search)
    # - gap_analysis -> research_subgraph (via universal gap_analysis_subgraph)
    
    async def _detect_attachments_node(self, state: ResearchState) -> Dict[str, Any]:
        """Detect if user has attached images; delegates to research_attachment_nodes."""
        from orchestrator.agents.research.research_attachment_nodes import detect_attachments_node
        return await detect_attachments_node(self, state)

    async def _attachment_analysis_node(self, state: ResearchState) -> Dict[str, Any]:
        """Run attachment analysis subgraph; delegates to research_attachment_nodes."""
        from orchestrator.agents.research.research_attachment_nodes import attachment_analysis_node
        return await attachment_analysis_node(self, state)

    async def _process_attachments_node(self, state: ResearchState) -> Dict[str, Any]:
        """Process attached images for face identification; delegates to research_attachment_nodes."""
        from orchestrator.agents.research.research_attachment_nodes import process_attachments_node
        return await process_attachments_node(self, state)

    async def _round2_parallel_node(self, state: ResearchState) -> Dict[str, Any]:
        from .research_core_nodes import round2_parallel_node
        return await round2_parallel_node(self, state)

    async def _detect_query_type_node(self, state: ResearchState) -> Dict[str, Any]:
        from .research_synthesis_nodes import detect_query_type_node
        return await detect_query_type_node(self, state)

    async def _final_synthesis_node(self, state: ResearchState) -> Dict[str, Any]:
        from .research_synthesis_nodes import final_synthesis_node
        return await final_synthesis_node(self, state)

    async def _post_process_results_node(self, state: ResearchState) -> Dict[str, Any]:
        from .research_synthesis_nodes import post_process_results_node
        return await post_process_results_node(self, state)

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
            
            query_preview = query[:100] + "..." if len(query) > 100 else query
            logger.info(f"📥 RESEARCH PROCESS: Starting research agent - query: {query_preview}")
            logger.debug(f"Research Agent processing: {query[:80]}...")

            # Load checkpoint shared_memory before fast path so collection_search can exclude already-shown results
            workflow = await self._get_workflow()
            config = self._get_checkpoint_config(metadata)
            checkpoint_state = await workflow.aget_state(config)
            if checkpoint_state and checkpoint_state.values:
                checkpoint_sm = checkpoint_state.values.get("shared_memory", {}) or {}
                sm = dict(metadata.get("shared_memory", {}) or {})
                if "shown_document_ids" in checkpoint_sm:
                    sm["shown_document_ids"] = checkpoint_sm["shown_document_ids"]
                metadata = {**metadata, "shared_memory": sm}

            # Try fast path (collection_search or factual_query) via LLM classification only for default research skill
            skill_name = (metadata or {}).get("skill_name", "research")
            fast_result = None
            if skill_name == "research":
                shared_memory_for_classification = metadata.get("shared_memory", {}) or {}
                if "user_chat_model" in shared_memory_for_classification and "user_chat_model" not in metadata:
                    metadata = {**metadata, "user_chat_model": shared_memory_for_classification["user_chat_model"]}
                from orchestrator.subgraphs.research_workflow_subgraph import intelligent_research_with_classification
                fast_result = await intelligent_research_with_classification(
                    query=query,
                    user_id=user_id,
                    metadata=metadata,
                    messages=messages,
                )
            if fast_result is not None:
                newly_shown = fast_result.pop("_shown_document_ids", [])
                sm = dict(metadata.get("shared_memory", {}) or {})
                workflow = await self._get_workflow()
                config = self._get_checkpoint_config(metadata)
                if newly_shown:
                    try:
                        checkpoint_state = await workflow.aget_state(config)
                        if checkpoint_state and checkpoint_state.values:
                            sm = dict(checkpoint_state.values.get("shared_memory", {}))
                        existing = sm.get("shown_document_ids", [])
                        sm["shown_document_ids"] = existing + newly_shown
                        await workflow.aupdate_state(
                            config, {"shared_memory": sm}, as_node="detect_attachments"
                        )
                    except Exception as e:
                        logger.debug("Could not persist shown_document_ids to checkpoint: %s", e)
                try:
                    from langchain_core.messages import HumanMessage, AIMessage
                    save_messages = list(messages or [])
                    save_messages.append(HumanMessage(content=query))
                    response_text = fast_result.get("response", "")
                    if response_text:
                        save_messages.append(AIMessage(content=response_text[:1000]))
                    save_state = {
                        "messages": save_messages,
                        "shared_memory": sm,
                        "query": query,
                    }
                    await workflow.aupdate_state(
                        config, save_state, as_node="detect_attachments"
                    )
                except Exception as e:
                    logger.debug("Could not persist fast path state to checkpoint: %s", e)
                return {
                    **fast_result,
                    "quick_answer_provided": False,
                }
            
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
            
            # Merge shared_memory: NEW data from metadata overwrites OLD checkpoint data
            # This ensures fresh active_editor content overwrites stale cached content
            shared_memory = metadata.get("shared_memory", {}) or {}
            shared_memory_merged = existing_shared_memory.copy()
            shared_memory_merged.update(shared_memory)
            
            # Ensure user_chat_model is in metadata if it's in shared_memory (for subgraph nodes)
            if "user_chat_model" in shared_memory_merged and "user_chat_model" not in metadata:
                metadata["user_chat_model"] = shared_memory_merged["user_chat_model"]
            
            # Store user_id in shared_memory so it's available in state
            if user_id and user_id != "system":
                shared_memory_merged["user_id"] = user_id
            
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
                shared_memory=shared_memory_merged,
                messages=conversation_messages,
                metadata=metadata  # Pass metadata to preserve user_chat_model
            )
            
            # Format response in standard agent format using AgentResponse contract
            final_response = result.get("final_response", "")
            research_complete = result.get("research_complete", False)
            structured_images = result.get("structured_images")
            visualization_results = result.get("visualization_results")
            static_visualization_data = result.get("static_visualization_data")
            static_format = result.get("static_format")
            citations = result.get("citations", [])
            sources_used = result.get("sources_used", [])
            
            # Build standard response using AgentResponse contract  
            # CRITICAL: Don't exclude empty lists - only exclude None values
            standard_response = AgentResponse(
                response=final_response,
                task_status="complete" if research_complete else "incomplete",
                agent_type="research_agent",
                timestamp=datetime.now().isoformat(),
                images=structured_images,  # Keep even if empty list
                static_visualization_data=static_visualization_data,
                static_format=static_format,
                chart_result=visualization_results if visualization_results else None,
                citations=citations if citations else None,
                sources=sources_used if sources_used else None
            )
            
            return {
                **standard_response.dict(exclude_none=True),
                "quick_answer_provided": result.get("quick_answer_provided", False)
            }
            
        except Exception as e:
            logger.error(f"Research Agent process failed: {e}")
            # Return standard error response
            error_response = AgentResponse(
                response=f"Research failed: {str(e)}",
                task_status="error",
                agent_type="research_agent",
                timestamp=datetime.now().isoformat(),
                error=str(e)
            )
            logger.info(f"📤 RESEARCH PROCESS: Returning error response (standard format) after exception")
            return error_response.dict(exclude_none=True)
    
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
            tool_analysis = await analyze_tool_needs_for_research(query, conversation_context)
            
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

            # Skip quick answer when user is asking about local/owned content or when image search is needed.
            # Otherwise we short-circuit with "general knowledge" and never run local/image search.
            if not skip_quick_answer:
                if is_local_search_intent(query):
                    skip_quick_answer = True
                    logger.info("Skipping quick answer: query implies local search (our photos/documents)")
                elif "image_search" in tool_analysis.get("categories", []):
                    skip_quick_answer = True
                    logger.info("Skipping quick answer: image search needed for query")

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

            # Skill config: gates node behavior for content_analysis, knowledge_builder, etc.
            skill_name = state_metadata.get("skill_name", "research")
            skill_config = get_research_skill_config(skill_name)
            skip_quick_answer = skip_quick_answer or skill_config.get("skip_quick_answer", False)

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
                "skill_config": skill_config,
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
