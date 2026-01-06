"""
Proofreading Agent - LangGraph Implementation
Scopes active editor content (chapter or whole doc < 7,500 words),
loads referenced style guide, and returns structured corrections.
Supports modes: clarity, compliance (style guide), and accuracy.
"""

import logging
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict, Tuple

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import ValidationError

from .base_agent import BaseAgent, TaskStatus
from orchestrator.models.editor_models import EditorOperation, ManuscriptEdit
from orchestrator.utils.editor_operation_resolver import resolve_editor_operation
from orchestrator.subgraphs import build_context_preparation_subgraph, build_proofreading_subgraph
from orchestrator.utils.frontmatter_utils import strip_frontmatter_block

logger = logging.getLogger(__name__)


# ============================================
# Utility Functions
# ============================================

# Chapter detection utilities are now in context subgraph
# Import them for backward compatibility
from orchestrator.subgraphs.fiction_context_subgraph import (
    find_chapter_ranges,
    locate_chapter_index,
    ChapterRange
)


def paragraph_bounds(text: str, cursor_offset: int) -> Tuple[int, int]:
    """Find paragraph boundaries around cursor."""
    if not text:
        return 0, 0
    cursor = max(0, min(len(text), cursor_offset))
    left = text.rfind("\n\n", 0, cursor)
    start = left + 2 if left != -1 else 0
    right = text.find("\n\n", cursor)
    end = right if right != -1 else len(text)
    return start, end


# ============================================
# State Definition
# ============================================

class ProofreadingState(TypedDict):
    """State for proofreading agent LangGraph workflow"""
    query: str
    user_id: str
    metadata: Dict[str, Any]
    messages: List[Any]
    shared_memory: Dict[str, Any]
    active_editor: Optional[Dict[str, Any]]
    manuscript_content: str
    style_guide_content: Optional[str]
    scope_text: str
    mode: str  # clarity, compliance, accuracy
    proofreading_result: Dict[str, Any]
    structured_edit: Optional[Dict[str, Any]]
    editor_operations: List[Dict[str, Any]]
    failed_operations: List[Dict[str, Any]]
    response: Dict[str, Any]
    task_status: str
    error: str


# ============================================
# Proofreading Agent
# ============================================

class ProofreadingAgent(BaseAgent):
    """
    Agent for proofreading fiction manuscripts from active editor.
    
    Handles:
    - Grammar, style, and consistency corrections
    - Style guide compliance
    - Chapter or paragraph scoping
    - Editor operations for HITL
    Uses LangGraph workflow for explicit state management
    """
    
    def __init__(self):
        super().__init__("proofreading_agent")
        logger.info("Proofreading Agent initialized")
    
    async def process(self, query: str, metadata: Dict[str, Any] = None, messages: List[Any] = None) -> Dict[str, Any]:
        """Process proofreading request with LangGraph workflow"""
        try:
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
            
            # Merge with any shared_memory from metadata
            shared_memory = metadata.get("shared_memory", {}) or {}
            shared_memory.update(existing_shared_memory)
            
            # Initialize state for LangGraph workflow
            initial_state: ProofreadingState = {
                "query": query,
                "user_id": user_id,
                "metadata": metadata,
                "messages": conversation_messages,
                "shared_memory": shared_memory,
                "active_editor": None,
                "manuscript_content": "",
                "style_guide_content": None,
                "scope_text": "",
                "mode": "clarity",
                "proofreading_result": {},
                "structured_edit": None,
                "editor_operations": [],
                "failed_operations": [],
                "response": {},
                "task_status": "",
                "error": ""
            }
            
            # Run LangGraph workflow with checkpointing
            result_state = await workflow.ainvoke(initial_state, config=config)
            
            # Extract final response
            response = result_state.get("response", {})
            task_status = result_state.get("task_status", "complete")
            
            if task_status == "error":
                error_msg = result_state.get("error", "Unknown error")
                logger.error(f"❌ {self.agent_type} failed: {error_msg}")
                return self._create_error_response(error_msg)
            
            # Add assistant response to messages for checkpoint persistence
            if response.get("response"):
                self._add_assistant_response_to_messages(result_state, response["response"])
            
            return response
            
        except Exception as e:
            logger.error(f"❌ {self.agent_type} failed: {e}")
            return self._create_error_response(str(e))

    def _build_workflow(self, checkpointer) -> StateGraph:
        """Build LangGraph workflow for proofreading agent"""
        workflow = StateGraph(ProofreadingState)
        
        # Build context preparation subgraph (reusable)
        context_subgraph = build_context_preparation_subgraph(checkpointer)
        
        # Build proofreading subgraph (reusable)
        proofreading_subgraph = build_proofreading_subgraph(
            checkpointer,
            llm_factory=self._get_llm,
            get_datetime_context=self._get_datetime_context
        )
        
        # Add nodes
        workflow.add_node("context_preparation", context_subgraph)
        workflow.add_node("proofreading", proofreading_subgraph)
        workflow.add_node("format_response", self._format_response_node)
        
        # Entry point
        workflow.set_entry_point("context_preparation")
        
        # Flow: context_preparation -> proofreading -> format_response -> END
        workflow.add_edge("context_preparation", "proofreading")
        workflow.add_edge("proofreading", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow.compile(checkpointer=checkpointer)
    
    # Context preparation and proofreading are now handled by subgraphs
    # These methods are kept for backward compatibility but not used in workflow
    
    async def _format_response_node(self, state: ProofreadingState) -> Dict[str, Any]:
        """Format final response"""
        try:
            logger.info("Formatting proofreading response...")
            
            structured_edit_dict = state.get("structured_edit", {})
            editor_operations = state.get("editor_operations", [])
            failed_operations = state.get("failed_operations", [])
            mode = state.get("mode", "clarity")
            active_editor = state.get("active_editor", {}) or {}
            filename = active_editor.get("filename") or "document.md"
            
            summary = structured_edit_dict.get("summary", "Proofreading completed.")
            
            # Format user message
            response_text = f"## Proofreading ({mode})\n\n{summary}\n\n"
            
            if editor_operations:
                response_text += f"**{len(editor_operations)} corrections** ready to apply.\n"
            
            if failed_operations:
                response_text += f"\n**⚠️ {len(failed_operations)} corrections could not be placed automatically.**\n"
            
            # Create manuscript edit for HITL
            manuscript_edit = None
            if editor_operations:
                manuscript_edit = {
                    "target_filename": filename,
                    "operations": editor_operations,
                    "scope": structured_edit_dict.get("scope", "paragraph"),
                    "summary": f"Proofreading corrections ({len(editor_operations)} operations)",
                    "safety": structured_edit_dict.get("safety", "low")
                }
            
            return {
                "response": {
                    "task_status": TaskStatus.COMPLETE.value,
                    "response": response_text,
                    "editor_operations": editor_operations,
                    "failed_operations": failed_operations,
                    "manuscript_edit": manuscript_edit,
                    "mode": mode
                },
                "task_status": "complete",
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            return self._handle_node_error(e, state, "Response formatting")
    
    def _infer_mode_from_request(self, state: ProofreadingState) -> str:
        """Infer proofreading mode from user request"""
        try:
            latest = ""
            messages = state.get("messages", [])
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    latest = (msg.get("content") or "").lower()
                    break
                if hasattr(msg, "type") and msg.type == "human":
                    latest = (msg.content or "").lower()
                    break
            if "style" in latest or "compliance" in latest:
                return "compliance"
            if "accuracy" in latest or "fact" in latest or "fact-check" in latest or "fact check" in latest:
                return "accuracy"
            return "clarity"
        except Exception:
            return "clarity"


# ============================================
# Factory Functions
# ============================================

_proofreading_agent_instance: Optional[ProofreadingAgent] = None


def get_proofreading_agent() -> ProofreadingAgent:
    """Get or create singleton proofreading agent instance"""
    global _proofreading_agent_instance
    if _proofreading_agent_instance is None:
        _proofreading_agent_instance = ProofreadingAgent()
    return _proofreading_agent_instance

