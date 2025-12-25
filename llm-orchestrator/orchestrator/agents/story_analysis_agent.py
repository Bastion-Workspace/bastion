"""
Story Analysis Agent - LangGraph Implementation
Provides focused analysis for fiction manuscripts in the active editor.
Consumes active_editor and returns analysis with structured payload.
"""

import logging
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from .base_agent import BaseAgent, TaskStatus

logger = logging.getLogger(__name__)


class StoryAnalysisState(TypedDict):
    """State for story analysis agent LangGraph workflow"""
    query: str
    user_id: str
    metadata: Dict[str, Any]
    messages: List[Any]
    shared_memory: Dict[str, Any]
    active_editor: Optional[Dict[str, Any]]
    manuscript_content: str
    analysis_result: str
    response: Dict[str, Any]
    task_status: str
    error: str


class StoryAnalysisAgent(BaseAgent):
    """
    Agent for analyzing fiction manuscripts from active editor.
    
    Handles:
    - Chapter-specific analysis
    - Character-focused analysis
    - General manuscript review
    - Scope-aware analysis (full manuscript vs specific sections)
    Uses LangGraph workflow for explicit state management
    """
    
    def __init__(self):
        super().__init__("story_analysis_agent")
        logger.info("Story Analysis Agent initialized")
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """Build LangGraph workflow for story analysis agent"""
        workflow = StateGraph(StoryAnalysisState)
        
        # Add nodes
        workflow.add_node("prepare_context", self._prepare_context_node)
        workflow.add_node("analyze_content", self._analyze_content_node)
        workflow.add_node("format_response", self._format_response_node)
        
        # Entry point
        workflow.set_entry_point("prepare_context")
        
        # Linear flow: prepare_context -> analyze_content -> format_response -> END
        workflow.add_edge("prepare_context", "analyze_content")
        workflow.add_edge("analyze_content", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow.compile(checkpointer=checkpointer)
    
    async def _prepare_context_node(self, state: StoryAnalysisState) -> Dict[str, Any]:
        """Prepare context from active editor"""
        try:
            logger.info("Preparing story analysis context...")
            
            shared_memory = state.get("shared_memory", {}) or {}
            active_editor = shared_memory.get("active_editor", {}) or {}
            
            manuscript = active_editor.get("content", "") or ""
            filename = active_editor.get("filename") or "document.md"
            frontmatter = active_editor.get("frontmatter", {}) or {}
            
            doc_type = str((frontmatter.get("type") or "")).strip().lower()
            if doc_type != "fiction":
                return {
                    "response": {
                        "task_status": TaskStatus.ERROR.value,
                        "response": "Story analysis is only available for fiction manuscripts.",
                        "error": "not_fiction_document"
                    },
                    "task_status": "error",
                    "error": "not_fiction_document"
                }
            
            title = str(frontmatter.get("title") or "").strip()
            
            # Extract user query from messages
            query = state.get("query", "")
            messages = state.get("messages", [])
            if not query and messages:
                for msg in reversed(messages):
                    if isinstance(msg, HumanMessage):
                        query = msg.content
                        break
                    elif hasattr(msg, "type") and msg.type == "human":
                        query = msg.content
                        break
            
            return {
                "active_editor": active_editor,
                "manuscript_content": manuscript,
                "query": query or "Please provide a comprehensive analysis of this manuscript.",
                # Preserve critical state keys
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", [])
            }
            
        except Exception as e:
            logger.error(f"Failed to prepare context: {e}")
            return {
                "error": str(e),
                "task_status": "error",
                "response": {
                    "task_status": TaskStatus.ERROR.value,
                    "response": f"Failed to prepare context: {str(e)}"
                },
                # Preserve critical state keys even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", [])
            }
    
    async def _analyze_content_node(self, state: StoryAnalysisState) -> Dict[str, Any]:
        """Analyze manuscript content using LLM"""
        try:
            logger.info("Analyzing manuscript content...")
            
            manuscript = state.get("manuscript_content", "")
            query = state.get("query", "")
            active_editor = state.get("active_editor", {}) or {}
            frontmatter = active_editor.get("frontmatter", {}) or {}
            title = str(frontmatter.get("title") or "").strip()
            filename = active_editor.get("filename") or "document.md"
            
            if not manuscript:
                return {
                    "analysis_result": "",
                    "error": "No manuscript content available",
                    "task_status": "error"
                }
            
            # Build system prompt
            system_prompt = (
                "You are a professional story analysis expert. You will receive the FULL manuscript "
                "for context, but you must FOCUS YOUR ANALYSIS on exactly what the user requested.\n\n"
                "**CRITICAL SCOPE RULES:**\n"
                "- If user asks about a SPECIFIC CHAPTER (e.g., 'Review Chapter 2'), analyze ONLY that chapter\n"
                "- If user asks a SPECIFIC QUESTION (e.g., 'Does Vivian have enough screen time?'), answer that question using evidence from the full manuscript\n"
                "- If user asks about MULTIPLE CHAPTERS (e.g., 'Analyze chapters 3-5'), focus on those chapters\n"
                "- If user asks for GENERAL REVIEW (e.g., 'Review the manuscript', 'Is this ready for publication?'), provide comprehensive analysis of the whole work\n\n"
                "**ALWAYS:**\n"
                "- Start by acknowledging what you're analyzing (e.g., 'Analysis of Chapter 2:' or 'Assessment of Vivian's presence throughout the manuscript:')\n"
                "- Provide specific, actionable recommendations\n"
                "- Reference specific passages when helpful\n"
                "- Keep tone supportive and direct\n"
                "- Use the full manuscript context to provide deeper insights, but stay focused on the user's specific request"
            )
            
            # Build user prompt with manuscript and query
            header_lines = []
            
            if title:
                header_lines.append(f"STORY TITLE: {title}")
                header_lines.append("")
            
            header_lines.append("=== USER REQUEST ===")
            header_lines.append(f'"{query}"')
            header_lines.append("")
            header_lines.append("INSTRUCTIONS:")
            header_lines.append("- The FULL MANUSCRIPT is provided below for context")
            header_lines.append("- FOCUS your analysis on exactly what the user requested above")
            header_lines.append("- If analyzing a specific chapter, START with that chapter's heading in your response")
            header_lines.append("- If answering a specific question, START with a direct answer")
            header_lines.append("- Be specific and constructive, with actionable recommendations")
            header_lines.append("- Reference specific passages when relevant")
            header_lines.append("")
            header_lines.append("=== FULL MANUSCRIPT (FOR CONTEXT) ===")
            header_lines.append(manuscript)
            
            user_prompt = "\n".join(header_lines)
            
            # Get LLM instance
            llm = self._get_llm(temperature=0.3, state=state)
            
            # Generate analysis
            datetime_context = self._get_datetime_context()
            messages = [
                SystemMessage(content=system_prompt),
                SystemMessage(content=datetime_context),
                HumanMessage(content=user_prompt)
            ]
            
            response = await llm.ainvoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Unwrap JSON if present
            content = self._unwrap_json_response(content)
            
            return {
                "analysis_result": content,
                # Preserve critical state keys
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "active_editor": state.get("active_editor", {}),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze content: {e}")
            return {
                "analysis_result": "",
                "error": str(e),
                "task_status": "error",
                # Preserve critical state keys even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "active_editor": state.get("active_editor", {}),
                "query": state.get("query", "")
            }
    
    async def _format_response_node(self, state: StoryAnalysisState) -> Dict[str, Any]:
        """Format final response"""
        try:
            logger.info("Formatting story analysis response...")
            
            analysis_result = state.get("analysis_result", "")
            active_editor = state.get("active_editor", {}) or {}
            filename = active_editor.get("filename") or "document.md"
            
            if not analysis_result:
                error = state.get("error", "Unknown error")
                return {
                    "response": {
                        "task_status": TaskStatus.ERROR.value,
                        "response": f"Story analysis failed: {error}",
                        "error": error
                    },
                    "task_status": "error"
                }
            
            # Build structured response
            response_dict = {
                "task_status": TaskStatus.COMPLETE.value,
                "response": analysis_result,
                "structured_response": {
                    "task_status": "complete",
                    "analysis_text": analysis_result,
                    "mode": "story_analysis",
                    "filename": filename,
                },
                "timestamp": datetime.now().isoformat(),
                "mode": "story_analysis"
            }
            
            # Add assistant message to state for checkpointing
            updated_state = self._add_assistant_response_to_messages(state, analysis_result)
            
            return {
                "response": response_dict,
                "task_status": "complete",
                # Preserve critical state keys
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": updated_state.get("messages", state.get("messages", [])),
                "active_editor": state.get("active_editor", {}),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.error(f"Failed to format response: {e}")
            return {
                "response": {
                    "task_status": TaskStatus.ERROR.value,
                    "response": f"Failed to format response: {str(e)}"
                },
                "task_status": "error",
                "error": str(e),
                # Preserve critical state keys even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "active_editor": state.get("active_editor", {}),
                "query": state.get("query", "")
            }
    
    def _unwrap_json_response(self, content: str) -> str:
        """Unwrap accidental JSON/code-fence envelopes and return plain text."""
        try:
            txt = content.strip()
            if '```json' in txt:
                m = re.search(r'```json\s*\n([\s\S]*?)\n```', txt)
                if m:
                    txt = m.group(1).strip()
            elif '```' in txt:
                m = re.search(r'```\s*\n([\s\S]*?)\n```', txt)
                if m:
                    txt = m.group(1).strip()
            # If JSON envelope like {"message": "..."}
            if txt.startswith('{') and txt.endswith('}'):
                obj = json.loads(txt)
                if isinstance(obj, dict):
                    return obj.get('message') or obj.get('text') or content
            return content
        except Exception:
            return content
    
    async def process(
        self,
        query: str = None,
        metadata: Dict[str, Any] = None,
        messages: List[Any] = None
    ) -> Dict[str, Any]:
        """
        Process story analysis request using LangGraph workflow
        
        Args:
            query: User query
            metadata: Optional metadata dictionary (persona, editor context, etc.)
            messages: Optional conversation history
            
        Returns:
            Dict with structured response and task status
        """
        try:
            metadata = metadata or {}
            messages = messages or []
            
            # Extract query from messages if not provided
            if not query and messages:
                for msg in reversed(messages):
                    if isinstance(msg, HumanMessage):
                        query = msg.content
                        break
                    elif hasattr(msg, "type") and msg.type == "human":
                        query = msg.content
                        break
            
            if not query:
                query = "Please provide a comprehensive analysis of this manuscript."
            
            user_id = metadata.get("user_id", "system")
            shared_memory = metadata.get("shared_memory", {}) or {}
            
            logger.info(f"Story Analysis Agent processing: {query[:80]}...")
            
            # Build initial state for LangGraph workflow
            initial_state: StoryAnalysisState = {
                "query": query,
                "user_id": user_id,
                "metadata": metadata,
                "messages": messages,
                "shared_memory": shared_memory,
                "active_editor": None,
                "manuscript_content": "",
                "analysis_result": "",
                "response": {},
                "task_status": "",
                "error": ""
            }
            
            # Get workflow (lazy initialization with checkpointer)
            workflow = await self._get_workflow()
            
            # Get checkpoint config
            config = self._get_checkpoint_config(metadata)
            
            # Load checkpointed messages and merge with new messages
            merged_messages = await self._load_and_merge_checkpoint_messages(
                workflow, config, messages
            )
            initial_state["messages"] = merged_messages
            
            # Invoke LangGraph workflow with checkpointing
            final_state = await workflow.ainvoke(initial_state, config=config)
            
            # Return response from final state
            return final_state.get("response", {
                "task_status": TaskStatus.ERROR.value,
                "response": "Story analysis failed"
            })
            
        except Exception as e:
            logger.error(f"Story Analysis Agent failed: {e}")
            return {
                "task_status": TaskStatus.ERROR.value,
                "response": f"Story analysis failed: {str(e)}"
            }


# Singleton instance
_story_analysis_agent_instance = None


def get_story_analysis_agent() -> StoryAnalysisAgent:
    """Get global story analysis agent instance"""
    global _story_analysis_agent_instance
    if _story_analysis_agent_instance is None:
        _story_analysis_agent_instance = StoryAnalysisAgent()
    return _story_analysis_agent_instance

