"""
Writing Assistant Agent

Unified agent for all writing document types (outline, character, rules, style, series, fiction).
Routes to appropriate subgraphs based on active_editor.frontmatter.type.

Phase 1: Outline subgraph integration
Phase 2: Rules subgraph integration
Phase 3: Style subgraph integration
Phase 4: Character subgraph integration
Future phases: Series subgraph
"""

import logging
from typing import Dict, Any, TypedDict
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from orchestrator.agents.base_agent import BaseAgent
from orchestrator.models.agent_response_contract import AgentResponse, TaskStatus
from orchestrator.utils.writing_subgraph_utilities import preserve_critical_state

logger = logging.getLogger(__name__)


# ============================================
# State Definition
# ============================================

class WritingAssistantState(TypedDict, total=False):
    """State for Writing Assistant agent LangGraph workflow"""
    query: str
    user_id: str
    metadata: Dict[str, Any]
    messages: list
    shared_memory: Dict[str, Any]
    response: Dict[str, Any]
    task_status: str
    error: str
    editor_operations: list  # Editor operations from subgraphs
    manuscript_edit: Dict[str, Any]  # Manuscript edit metadata from subgraphs


# ============================================
# Writing Assistant Agent
# ============================================

class WritingAssistantAgent(BaseAgent):
    """
    Writing Assistant Agent for all writing document types
    
    Routes to appropriate subgraphs based on active_editor.frontmatter.type:
    - type: outline → outline_editing_subgraph
    - type: nfoutline → nonfiction_outline_subgraph
    - type: rules → rules_editing_subgraph
    - type: style → style_editing_subgraph
    - type: character → character_development_subgraph
    - type: article, substack, blog → article_writing_subgraph
    - type: fiction → fiction_editing_subgraph
    - type: series → series_editing_subgraph (future)
    
    Uses LangGraph workflow for explicit state management
    """
    
    def __init__(self):
        super().__init__("writing_assistant_agent")
        logger.debug("Writing Assistant Agent ready!")
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """Build LangGraph workflow for writing assistant agent"""
        workflow = StateGraph(WritingAssistantState)
        
        # Build subgraphs (Phase 1: outline, Phase 1.5: nonfiction outline, Phase 2: rules, Phase 3: style, Phase 4: character)
        from orchestrator.subgraphs.outline_editing_subgraph import build_outline_editing_subgraph
        from orchestrator.subgraphs.nonfiction_outline_subgraph import build_nonfiction_outline_subgraph
        from orchestrator.subgraphs.rules_editing_subgraph import build_rules_editing_subgraph
        from orchestrator.subgraphs.style_editing_subgraph import build_style_editing_subgraph
        from orchestrator.subgraphs.character_development_subgraph import build_character_development_subgraph
        from orchestrator.subgraphs.article_writing_subgraph import build_article_writing_subgraph
        from orchestrator.subgraphs.fiction_editing_subgraph import build_fiction_editing_subgraph

        outline_subgraph = build_outline_editing_subgraph(
            checkpointer,
            llm_factory=self._get_llm,
            get_datetime_context=self._get_datetime_context
        )
        
        rules_subgraph = build_rules_editing_subgraph(
            checkpointer,
            llm_factory=self._get_llm,
            get_datetime_context=self._get_datetime_context
        )
        
        style_subgraph = build_style_editing_subgraph(
            checkpointer,
            llm_factory=self._get_llm,
            get_datetime_context=self._get_datetime_context
        )
        
        character_subgraph = build_character_development_subgraph(
            checkpointer,
            llm_factory=self._get_llm,
            get_datetime_context=self._get_datetime_context
        )
        
        nonfiction_outline_subgraph = build_nonfiction_outline_subgraph(
            checkpointer,
            llm_factory=self._get_llm,
            get_datetime_context=self._get_datetime_context
        )
        
        article_subgraph = build_article_writing_subgraph(
            checkpointer,
            llm_factory=self._get_llm,
            get_datetime_context=self._get_datetime_context
        )
        fiction_subgraph = build_fiction_editing_subgraph(
            checkpointer,
            llm_factory=self._get_llm,
            get_datetime_context=self._get_datetime_context
        )
        
        # Add nodes
        workflow.add_node("prepare_context", self._prepare_context_node)
        workflow.add_node("route_by_document_type", self._route_by_document_type_node)
        workflow.add_node("outline_subgraph", outline_subgraph)
        workflow.add_node("nonfiction_outline_subgraph", nonfiction_outline_subgraph)
        workflow.add_node("rules_subgraph", rules_subgraph)
        workflow.add_node("style_subgraph", style_subgraph)
        workflow.add_node("character_subgraph", character_subgraph)
        workflow.add_node("article_subgraph", article_subgraph)
        workflow.add_node("fiction_subgraph", fiction_subgraph)
        workflow.add_node("format_response", self._format_response_node)
        
        # Entry point
        workflow.set_entry_point("prepare_context")
        
        # Flow: prepare_context -> route_by_document_type -> (subgraph) -> format_response
        workflow.add_edge("prepare_context", "route_by_document_type")
        
        # Route to appropriate subgraph based on document type
        workflow.add_conditional_edges(
            "route_by_document_type",
            self._route_after_type_check,
            {
                "outline": "outline_subgraph",
                "nfoutline": "nonfiction_outline_subgraph",
                "rules": "rules_subgraph",
                "style": "style_subgraph",
                "character": "character_subgraph",
                "article": "article_subgraph",
                "fiction": "fiction_subgraph",
                "error": "format_response"
            }
        )
        
        # All subgraphs flow to format_response
        workflow.add_edge("outline_subgraph", "format_response")
        workflow.add_edge("nonfiction_outline_subgraph", "format_response")
        workflow.add_edge("rules_subgraph", "format_response")
        workflow.add_edge("style_subgraph", "format_response")
        workflow.add_edge("character_subgraph", "format_response")
        workflow.add_edge("article_subgraph", "format_response")
        workflow.add_edge("fiction_subgraph", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow.compile(checkpointer=checkpointer)
    
    async def _prepare_context_node(self, state: WritingAssistantState) -> Dict[str, Any]:
        """Prepare context from active editor"""
        try:
            logger.info("📋 Preparing context for Writing Assistant...")
            
            shared_memory = state.get("shared_memory", {}) or {}
            active_editor = shared_memory.get("active_editor", {}) or {}
            
            if not active_editor:
                return {
                    "error": "No active editor found. Writing Assistant requires an active editor with a writing document type.",
                    "task_status": "error",
                    "response": {
                        "response": "No active editor found. Writing Assistant requires an active editor with a writing document type.",
                        "task_status": "error",
                        "agent_type": "writing_assistant_agent",
                        "timestamp": datetime.now().isoformat()
                    },
                    **preserve_critical_state(state)
                }
            
            frontmatter = active_editor.get("frontmatter", {}) or {}
            doc_type = str(frontmatter.get("type", "")).strip().lower()
            
            logger.info(f"📝 Writing Assistant: Document type detected: '{doc_type}'")
            
            return {
                **preserve_critical_state(state)
            }
            
        except Exception as e:
            logger.error(f"❌ WRITING ASSISTANT PREPARE: Exception in _prepare_context_node: {e}")
            error_response = AgentResponse(
                response=f"Error preparing context: {str(e)}",
                task_status=TaskStatus.ERROR,
                agent_type="writing_assistant_agent",
                timestamp=datetime.now().isoformat(),
                error=str(e)
            )
            return {
                "response": error_response.dict(exclude_none=True),
                "task_status": "error",
                **preserve_critical_state(state)
            }
    
    async def _route_by_document_type_node(self, state: WritingAssistantState) -> Dict[str, Any]:
        """Route to appropriate subgraph based on document type"""
        try:
            shared_memory = state.get("shared_memory", {}) or {}
            active_editor = shared_memory.get("active_editor", {}) or {}
            frontmatter = active_editor.get("frontmatter", {}) or {}
            doc_type = str(frontmatter.get("type", "")).strip().lower()
            
            logger.info(f"🎯 Writing Assistant: Routing document type '{doc_type}'")
            
            # Phase 1: outline, Phase 2: rules, Phase 3: style, Phase 4: character
            if doc_type == "outline":
                logger.info("✅ Writing Assistant: Routing to outline_editing_subgraph")
                return {
                    **preserve_critical_state(state)
                }
            elif doc_type == "rules":
                logger.info("✅ Writing Assistant: Routing to rules_editing_subgraph")
                return {
                    **preserve_critical_state(state)
                }
            elif doc_type == "style":
                logger.info("✅ Writing Assistant: Routing to style_editing_subgraph")
                return {
                    **preserve_critical_state(state)
                }
            elif doc_type == "character":
                logger.info("✅ Writing Assistant: Routing to character_development_subgraph")
                return {
                    **preserve_critical_state(state)
                }
            elif doc_type == "nfoutline":
                logger.info("✅ Writing Assistant: Routing to nonfiction_outline_subgraph")
                return {
                    **preserve_critical_state(state)
                }
            elif doc_type in ["article", "substack", "blog"]:
                logger.info("✅ Writing Assistant: Routing to article_writing_subgraph")
                return {
                    **preserve_critical_state(state)
                }
            elif doc_type == "fiction":
                logger.info("✅ Writing Assistant: Routing to fiction_editing_subgraph")
                return {
                    **preserve_critical_state(state)
                }
            
            # Unsupported document type
            error_msg = f"Writing Assistant does not yet support document type '{doc_type}'. Supported types: outline, nfoutline, rules, style, character, article, substack, blog, fiction."
            logger.warning(f"⚠️ {error_msg}")
            error_response = AgentResponse(
                response=error_msg,
                task_status=TaskStatus.ERROR,
                agent_type="writing_assistant_agent",
                timestamp=datetime.now().isoformat(),
                error=f"unsupported_document_type: {doc_type}"
            )
            return {
                "response": error_response.dict(exclude_none=True),
                "task_status": "error",
                **preserve_critical_state(state)
            }
            
        except Exception as e:
            logger.error(f"❌ WRITING ASSISTANT ROUTE: Exception in _route_by_document_type_node: {e}")
            error_response = AgentResponse(
                response=f"Error routing by document type: {str(e)}",
                task_status=TaskStatus.ERROR,
                agent_type="writing_assistant_agent",
                timestamp=datetime.now().isoformat(),
                error=str(e)
            )
            return {
                "response": error_response.dict(exclude_none=True),
                "task_status": "error",
                **preserve_critical_state(state)
            }
    
    def _route_after_type_check(self, state: WritingAssistantState) -> str:
        """Route after type check - determine which subgraph to use"""
        task_status = state.get("task_status", "")
        if task_status == "error":
            return "error"
        
        shared_memory = state.get("shared_memory", {}) or {}
        active_editor = shared_memory.get("active_editor", {}) or {}
        frontmatter = active_editor.get("frontmatter", {}) or {}
        doc_type = str(frontmatter.get("type", "")).strip().lower()
        
        if doc_type == "outline":
            return "outline"
        elif doc_type == "nfoutline":
            return "nfoutline"
        elif doc_type == "rules":
            return "rules"
        elif doc_type == "style":
            return "style"
        elif doc_type == "character":
            return "character"
        elif doc_type in ["article", "substack", "blog"]:
            return "article"
        elif doc_type == "fiction":
            return "fiction"
        
        return "error"
    
    async def _format_response_node(self, state: WritingAssistantState) -> Dict[str, Any]:
        """Format final response - subgraph should have already formatted it"""
        try:
            logger.info("📝 Writing Assistant: Formatting response...")
            
            # DEBUG: Log all state keys to see what we have
            logger.info(f"📊 WRITING ASSISTANT FORMAT: State keys available: {list(state.keys())}")
            
            # Subgraph should have already set response in state
            response = state.get("response", {})
            task_status = state.get("task_status", "")
            
            # Normalize task_status
            if not task_status or task_status not in ["complete", "incomplete", "permission_required", "error"]:
                task_status = "complete"
                logger.info(f"📊 WRITING ASSISTANT FORMAT: Normalized task_status to 'complete'")
            
            # If response is already a dict (standard format), use it
            if isinstance(response, dict):
                # Ensure it has required fields
                if "task_status" not in response:
                    response["task_status"] = task_status
                if "agent_type" not in response:
                    response["agent_type"] = "writing_assistant_agent"
                if "timestamp" not in response:
                    response["timestamp"] = datetime.now().isoformat()
                
                logger.info(f"✅ WRITING ASSISTANT FORMAT: Response already formatted (keys: {list(response.keys())})")
                
                # Extract editor_operations and manuscript_edit from STATE level (standardized location)
                # All subgraphs now return operations at state level only
                editor_operations = state.get("editor_operations")
                manuscript_edit = state.get("manuscript_edit")
                
                logger.info(f"📊 WRITING ASSISTANT FORMAT: Extracted from state - editor_operations: {len(editor_operations) if editor_operations else 0} operation(s), manuscript_edit: {'present' if manuscript_edit else 'missing'}")
                
                # DEBUG: Log the actual values
                if editor_operations:
                    logger.info(f"📊 WRITING ASSISTANT FORMAT: editor_operations type: {type(editor_operations)}, length: {len(editor_operations)}")
                if manuscript_edit:
                    logger.info(f"📊 WRITING ASSISTANT FORMAT: manuscript_edit type: {type(manuscript_edit)}, keys: {list(manuscript_edit.keys()) if isinstance(manuscript_edit, dict) else 'not a dict'}")
                
                # Add to response dict for gRPC extraction compatibility
                if editor_operations is not None or manuscript_edit is not None:
                    if not isinstance(response, dict):
                        response = {}
                    if editor_operations is not None:
                        response["editor_operations"] = editor_operations
                        logger.info(f"✅ WRITING ASSISTANT FORMAT: Added {len(editor_operations)} editor_operations to response dict")
                    if manuscript_edit is not None:
                        response["manuscript_edit"] = manuscript_edit
                        logger.info(f"✅ WRITING ASSISTANT FORMAT: Added manuscript_edit to response dict")
                
                from orchestrator.utils.writing_subgraph_utilities import preserve_critical_state
                
                # Build return dict - preserve editor_operations and manuscript_edit at state level
                # so process() can extract them, AND add them to response dict for gRPC extraction
                # CRITICAL: Extract editor_operations and manuscript_edit BEFORE preserve_critical_state
                # to ensure they're not lost during state preservation
                return_dict = {
                    "response": response,
                    "task_status": task_status,
                    **preserve_critical_state(state)
                }
                
                # CRITICAL: Always preserve editor_operations and manuscript_edit at state level
                # even if they're None (so they don't get lost), but only add to response dict if not None
                # This ensures process() can extract them from result_state
                if "editor_operations" in state:
                    return_dict["editor_operations"] = state.get("editor_operations")
                    if editor_operations:
                        logger.info(f"✅ WRITING ASSISTANT FORMAT: Preserved {len(editor_operations)} editor_operations at state level")
                    else:
                        logger.info(f"📊 WRITING ASSISTANT FORMAT: Preserved editor_operations=None at state level (was in state)")
                elif editor_operations is not None:
                    # Fallback: if not in state but we extracted it, preserve it
                    return_dict["editor_operations"] = editor_operations
                    logger.info(f"✅ WRITING ASSISTANT FORMAT: Preserved {len(editor_operations)} editor_operations at state level (fallback)")
                
                if "manuscript_edit" in state:
                    return_dict["manuscript_edit"] = state.get("manuscript_edit")
                    if manuscript_edit:
                        logger.info(f"✅ WRITING ASSISTANT FORMAT: Preserved manuscript_edit at state level")
                    else:
                        logger.info(f"📊 WRITING ASSISTANT FORMAT: Preserved manuscript_edit=None at state level (was in state)")
                elif manuscript_edit is not None:
                    # Fallback: if not in state but we extracted it, preserve it
                    return_dict["manuscript_edit"] = manuscript_edit
                    logger.info(f"✅ WRITING ASSISTANT FORMAT: Preserved manuscript_edit at state level (fallback)")
                
                return return_dict
            
            # Fallback: create response from state
            logger.warning("⚠️ WRITING ASSISTANT FORMAT: Response not in standard format, creating fallback")
            fallback_response = AgentResponse(
                response=str(response) if response else "Writing Assistant processing complete",
                task_status=TaskStatus.COMPLETE if task_status != "error" else TaskStatus.ERROR,
                agent_type="writing_assistant_agent",
                timestamp=datetime.now().isoformat()
            )
            
            return {
                "response": fallback_response.dict(exclude_none=True),
                "task_status": task_status or "complete",
                    **preserve_critical_state(state)
                }
            
        except Exception as e:
            logger.error(f"❌ WRITING ASSISTANT FORMAT: Exception in _format_response_node: {e}")
            error_response = AgentResponse(
                response=f"Error formatting response: {str(e)}",
                task_status=TaskStatus.ERROR,
                agent_type="writing_assistant_agent",
                timestamp=datetime.now().isoformat(),
                error=str(e)
            )
            return {
                "response": error_response.dict(exclude_none=True),
                "task_status": "error",
                **preserve_critical_state(state)
            }
    
    async def process(self, query: str, metadata: Dict[str, Any] = None, messages: list = None) -> Dict[str, Any]:
        """
        Process writing assistant request
        
        Routes to appropriate subgraph based on active_editor.frontmatter.type
        """
        try:
            logger.info(f"🚀 WRITING ASSISTANT PROCESS: Starting workflow (query length: {len(query) if query else 0})")
            
            # Get workflow (lazy initialization with checkpointer)
            workflow = await self._get_workflow()
            
            # Extract user_id from metadata
            metadata = metadata or {}
            user_id = metadata.get("user_id", "system")
            
            # Get checkpoint config (handles thread_id from conversation_id/user_id)
            config = self._get_checkpoint_config(metadata)
            
            # Prepare new messages (current query)
            new_messages = self._prepare_messages_with_query(messages, query)
            
            # Load and merge checkpointed messages to preserve conversation history
            conversation_messages = await self._load_and_merge_checkpoint_messages(
                workflow, config, new_messages
            )
            
            # Load shared_memory from checkpoint if available
            checkpoint_state = await workflow.aget_state(config)
            existing_shared_memory = {}
            if checkpoint_state and checkpoint_state.values:
                existing_shared_memory = checkpoint_state.values.get("shared_memory", {})
            
            # ⚠️ CRITICAL: Merge shared_memory in CORRECT ORDER
            # Start with checkpoint (old), then update with NEW data (from metadata)
            # This ensures fresh active_editor content (current file state) overwrites stale checkpoint data
            # 
            # ✅ CORRECT:
            #   shared_memory_merged = existing_shared_memory.copy()  # Start with old
            #   shared_memory_merged.update(metadata.get("shared_memory", {}))  # New overwrites old
            # 
            # ❌ WRONG (causes stale content bug):
            #   shared_memory = metadata.get("shared_memory", {})  # Start with new
            #   shared_memory.update(existing_shared_memory)  # Old overwrites new!
            #
            # See dev-notes/AGENT_INTEGRATION_GUIDE.md for details
            shared_memory_merged = existing_shared_memory.copy()
            shared_memory_merged.update(metadata.get("shared_memory", {}) or {})  # New data (including updated active_editor) takes precedence
            
            # Initialize state for LangGraph workflow
            initial_state: WritingAssistantState = {
                "query": query,
                "user_id": user_id,
                "metadata": metadata,
                "messages": conversation_messages,
                "shared_memory": shared_memory_merged,
                "response": {},
                "task_status": "complete",
                "error": ""
            }
            
            logger.info(f"📊 WRITING ASSISTANT PROCESS: Invoking workflow with state keys: {list(initial_state.keys())}")
            
            # Run LangGraph workflow with checkpointing
            result_state = await workflow.ainvoke(initial_state, config=config)
            
            logger.info(f"✅ WRITING ASSISTANT PROCESS: Workflow completed - extracting response from state")
            
            # Extract final response
            response = result_state.get("response", {})
            task_status = result_state.get("task_status", "complete")
            
            # Normalize task_status
            if not task_status or task_status not in ["complete", "incomplete", "permission_required", "error"]:
                task_status = "complete"
                logger.info(f"📊 WRITING ASSISTANT PROCESS: Normalized task_status to 'complete'")
            
            if task_status == "error":
                error_msg = result_state.get("error", "Unknown error")
                logger.error(f"❌ WRITING ASSISTANT PROCESS: Agent failed with error: {error_msg}")
                return self._create_error_response(error_msg)
            
            # Check if response is already in standard format
            if isinstance(response, dict) and all(key in response for key in ["response", "task_status", "agent_type", "timestamp"]):
                logger.info(f"📊 WRITING ASSISTANT PROCESS: Response is in standard format (keys: {list(response.keys())})")
                
                # DEBUG: Log result_state keys to see what we have
                logger.info(f"📊 WRITING ASSISTANT PROCESS: result_state keys: {list(result_state.keys())}")
                
                # Preserve editor_operations and manuscript_edit from result_state
                # Check both state level (rules subgraph pattern) and inside response dict (outline subgraph pattern)
                editor_operations = result_state.get("editor_operations")
                manuscript_edit = result_state.get("manuscript_edit")
                
                logger.info(f"📊 WRITING ASSISTANT PROCESS: Extracted from result_state - editor_operations: {len(editor_operations) if editor_operations else 0} ops, manuscript_edit: {'present' if manuscript_edit else 'missing'}")
                
                # If not at state level, check inside response dict (outline subgraph pattern)
                if editor_operations is None and isinstance(response, dict):
                    editor_operations = response.get("editor_operations")
                    logger.info(f"📊 WRITING ASSISTANT PROCESS: Checked response dict - editor_operations: {len(editor_operations) if editor_operations else 0} ops")
                if manuscript_edit is None and isinstance(response, dict):
                    manuscript_edit = response.get("manuscript_edit")
                    logger.info(f"📊 WRITING ASSISTANT PROCESS: Checked response dict - manuscript_edit: {'present' if manuscript_edit else 'missing'}")
                
                # Add to response if found and not already present
                if editor_operations is not None and "editor_operations" not in response:
                    response["editor_operations"] = editor_operations
                    logger.info(f"✅ WRITING ASSISTANT PROCESS: Added {len(editor_operations)} editor_operations from state/response to response dict")
                elif editor_operations is not None:
                    logger.info(f"📊 WRITING ASSISTANT PROCESS: editor_operations already in response dict ({len(editor_operations)} ops)")
                else:
                    logger.warning(f"⚠️ WRITING ASSISTANT PROCESS: editor_operations is None - not adding to response")
                    
                if manuscript_edit is not None and "manuscript_edit" not in response:
                    response["manuscript_edit"] = manuscript_edit
                    logger.info(f"✅ WRITING ASSISTANT PROCESS: Added manuscript_edit from state/response to response dict")
                elif manuscript_edit is not None:
                    logger.info(f"📊 WRITING ASSISTANT PROCESS: manuscript_edit already in response dict")
                else:
                    logger.info(f"📊 WRITING ASSISTANT PROCESS: manuscript_edit is None - not adding to response")
                
                logger.info(f"📊 WRITING ASSISTANT PROCESS: Final response dict keys: {list(response.keys())}")
                logger.info(f"📊 WRITING ASSISTANT PROCESS: Response dict has 'response' key: True, has 'editor_operations': {'editor_operations' in response}, has 'manuscript_edit': {'manuscript_edit' in response}")
                if "editor_operations" in response:
                    logger.info(f"📊 WRITING ASSISTANT PROCESS: editor_operations in response: type={type(response['editor_operations'])}, length={len(response['editor_operations']) if isinstance(response['editor_operations'], list) else 'not a list'}")
                return response
            
            # Reconstruct from legacy format
            logger.warning("⚠️ WRITING ASSISTANT PROCESS: Response not in standard format, reconstructing")
            reconstructed = AgentResponse(
                response=str(response) if response else "Writing Assistant processing complete",
                task_status=TaskStatus.COMPLETE if task_status != "error" else TaskStatus.ERROR,
                agent_type="writing_assistant_agent",
                timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"📤 WRITING ASSISTANT PROCESS: Returning reconstructed response (standard format)")
            return reconstructed.dict(exclude_none=True)
            
        except Exception as e:
            logger.error(f"❌ WRITING ASSISTANT PROCESS: Agent failed: {e}")
            return self._create_error_response(str(e))


# ============================================
# Factory Function
# ============================================

_writing_assistant_agent_instance = None

def get_writing_assistant_agent() -> WritingAssistantAgent:
    """Get or create Writing Assistant agent instance (singleton)"""
    global _writing_assistant_agent_instance
    if _writing_assistant_agent_instance is None:
        _writing_assistant_agent_instance = WritingAssistantAgent()
    return _writing_assistant_agent_instance