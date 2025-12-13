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

from .base_agent import BaseAgent, TaskStatus
from orchestrator.subgraphs import build_context_preparation_subgraph
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
    editor_operations: List[Dict[str, Any]]
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
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """Build LangGraph workflow for proofreading agent"""
        workflow = StateGraph(ProofreadingState)
        
        # Build context preparation subgraph (reusable)
        context_subgraph = build_context_preparation_subgraph(checkpointer)
        
        # Add nodes
        workflow.add_node("context_preparation", context_subgraph)
        workflow.add_node("infer_mode", self._infer_and_set_mode_node)
        workflow.add_node("load_style_guide", self._load_style_guide_node)
        workflow.add_node("determine_scope", self._determine_scope_node)
        workflow.add_node("proofread_content", self._proofread_content_node)
        workflow.add_node("generate_operations", self._generate_operations_node)
        workflow.add_node("format_response", self._format_response_node)
        
        # Entry point
        workflow.set_entry_point("context_preparation")
        
        # Flow: context_preparation -> infer_mode -> load_style_guide -> determine_scope -> proofread_content -> generate_operations -> format_response -> END
        workflow.add_edge("context_preparation", "infer_mode")
        workflow.add_edge("infer_mode", "load_style_guide")
        workflow.add_edge("load_style_guide", "determine_scope")
        workflow.add_edge("determine_scope", "proofread_content")
        workflow.add_edge("proofread_content", "generate_operations")
        workflow.add_edge("generate_operations", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow.compile(checkpointer=checkpointer)
    
    # Context preparation is now handled by context subgraph
    # This method is kept for backward compatibility but not used in workflow
    async def _prepare_context_node(self, state: ProofreadingState) -> Dict[str, Any]:
        """Prepare context from active editor (DEPRECATED - use context subgraph)"""
        # Context subgraph handles this now
        # This is a no-op wrapper for backward compatibility
        return {}
    
    async def _load_style_guide_node(self, state: ProofreadingState) -> Dict[str, Any]:
        """Load referenced style guide from frontmatter"""
        try:
            logger.info("Loading style guide...")
            
            active_editor = state.get("active_editor", {}) or {}
            frontmatter = active_editor.get("frontmatter", {}) or {}
            canonical_path = active_editor.get("canonical_path")
            filename = active_editor.get("filename") or "document.md"
            
            style_text = None
            
            # Load style guide using reference file loader
            try:
                from orchestrator.tools.reference_file_loader import load_referenced_files
                
                user_id = state.get("user_id", "system")
                reference_config = {
                    "style": ["style"]
                }
                
                result = await load_referenced_files(
                    active_editor=active_editor,
                    user_id=user_id,
                    reference_config=reference_config,
                    doc_type_filter=None  # Load for any document type
                )
                
                loaded_files = result.get("loaded_files", {})
                if loaded_files.get("style") and len(loaded_files["style"]) > 0:
                    style_text = loaded_files["style"][0].get("content", "")
                    if style_text:
                        style_text = strip_frontmatter_block(style_text)
                        logger.info("Style guide loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load style guide: {e}")
                style_text = None
            
            return {
                "style_guide_content": style_text
            }
            
        except Exception as e:
            logger.error(f"Error loading style guide: {e}")
            return {
                "style_guide_content": None
            }
    
    async def _determine_scope_node(self, state: ProofreadingState) -> Dict[str, Any]:
        """Determine content scope: chapter around cursor, or entire doc if < 7,500 words"""
        try:
            logger.info("Determining proofreading scope...")
            
            # Get manuscript from state (set by context subgraph)
            manuscript = state.get("manuscript_content") or state.get("manuscript", "")
            active_editor = state.get("active_editor", {}) or {}
            cursor_offset = state.get("cursor_offset", -1)
            
            # Get chapter ranges from context subgraph (if available)
            chapter_ranges = state.get("chapter_ranges", [])
            
            body = strip_frontmatter_block(manuscript)
            word_count = len((body or "").split())
            scope_text = body
            
            if word_count >= 7500:
                # Scope to chapter or paragraph
                try:
                    # Use chapter ranges from context subgraph, or find them if not available
                    if not chapter_ranges:
                        chapter_ranges = find_chapter_ranges(manuscript)
                    
                    if cursor_offset >= 0 and chapter_ranges:
                        idx = locate_chapter_index(chapter_ranges, cursor_offset)
                        if idx != -1:
                            rng = chapter_ranges[idx]
                            scope_text = strip_frontmatter_block(manuscript[rng.start:rng.end])
                        else:
                            # Fallback to paragraph
                            p0, p1 = paragraph_bounds(manuscript, max(cursor_offset, 0))
                            scope_text = strip_frontmatter_block(manuscript[p0:p1])
                    else:
                        # Fallback to paragraph around 0
                        p0, p1 = paragraph_bounds(manuscript, 0)
                        scope_text = strip_frontmatter_block(manuscript[p0:p1])
                except Exception:
                    p0, p1 = paragraph_bounds(manuscript, max(cursor_offset, 0))
                    scope_text = strip_frontmatter_block(manuscript[p0:p1])
            
            logger.info(f"Proofreading scope: {len(scope_text)} characters, {len(scope_text.split())} words")
            
            return {
                "scope_text": scope_text
            }
            
        except Exception as e:
            logger.error(f"Error determining scope: {e}")
            # Fallback to full document
            manuscript = state.get("manuscript_content") or state.get("manuscript", "")
            body = strip_frontmatter_block(manuscript)
            return {
                "scope_text": body
            }
    
    async def _proofread_content_node(self, state: ProofreadingState) -> Dict[str, Any]:
        """Execute proofreading analysis using LLM"""
        try:
            logger.info("Executing proofreading analysis...")
            
            scope_text = state.get("scope_text", "")
            style_guide = state.get("style_guide_content")
            mode = state.get("mode", "clarity")
            active_editor = state.get("active_editor", {}) or {}
            filename = active_editor.get("filename") or "document.md"
            frontmatter = active_editor.get("frontmatter", {}) or {}
            doc_type = str((frontmatter.get("type") or "")).strip().lower()
            
            # Build system prompt
            system_prompt = (
                "You are a MASTER COPY EDITOR and PROFESSIONAL PROOFREADER with expertise in grammar, style, "
                "consistency, and technical accuracy. Identify and correct errors while maintaining the author's voice.\n\n"
                "=== RESPONSE FORMAT ===\n"
                "Return ONLY valid JSON matching the provided schema. For each specific correction, produce entries with:\n"
                "- original_text: exact, verbatim span from the source including surrounding punctuation\n"
                "- changed_to: your corrected version\n"
                "- explanation: short reason only when unclear\n\n"
                "CRITICAL SCOPE ANALYSIS: When a change affects surrounding grammar, select the full sentence or clause."
            )
            
            style_block = (
                "=== AUTHOR'S STYLE GUIDE ===\n"
                "The following style guide defines the author's preferred conventions.\n"
                "CRITICAL: When style conflicts with general grammar, FOLLOW THE STYLE GUIDE.\n\n"
                f"{style_guide}\n"
            ) if style_guide else ""
            
            required_schema_hint = (
                "STRUCTURED OUTPUT REQUIRED: Respond ONLY with valid JSON for ProofreadingResult. Fields: "
                "task_status, mode, summary, corrections[] (original_text, changed_to, explanation, scope), "
                "style_guide_used, consistency_checks, permission_request, metadata. "
                "SCOPE must be one of: 'word', 'phrase', 'clause', 'sentence', 'paragraph', 'duplicate'. "
                "Do not use any other scope values. "
                "CRITICAL: Provide exact, verbatim original_text that can be found in the source document. "
                "The system will automatically generate editor operations from your corrections."
            )
            
            user_prompt = (
                "=== DOCUMENT METADATA ===\n"
                f"Filename: {filename}\nType: {doc_type}\n\n"
                + (style_block if style_block else "")
                + "=== SCOPE TEXT (frontmatter stripped) ===\n"
                f"{scope_text}\n\n"
                "=== RESPONSE DIRECTIVES ===\n"
                "- Respect intentional choices and genre conventions.\n"
                "- Focus on actual errors; maintain voice.\n"
                "- Use complete, verbatim original_text with natural boundaries.\n"
                f"- Mode: {mode} (clarity|compliance|accuracy).\n\n"
                + required_schema_hint
            )
            
            # Call LLM using centralized mechanism
            datetime_context = self._get_datetime_context()
            messages = [
                SystemMessage(content=system_prompt),
                SystemMessage(content=datetime_context),
                HumanMessage(content=user_prompt)
            ]
            
            llm = self._get_llm(temperature=0.2, state=state)
            llm_response = await llm.ainvoke(messages)
            content = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            
            # Parse structured response
            proofreading_result = self._parse_proofreading_result(content)
            
            return {
                "proofreading_result": proofreading_result
            }
            
        except Exception as e:
            logger.error(f"Error during proofreading: {e}")
            return {
                "proofreading_result": {
                    "task_status": "error",
                    "mode": state.get("mode", "clarity"),
                    "summary": f"Proofreading failed: {str(e)}",
                    "corrections": [],
                    "metadata": {"error": str(e)}
                },
                "error": str(e)
            }
    
    async def _generate_operations_node(self, state: ProofreadingState) -> Dict[str, Any]:
        """Generate editor operations from proofreading corrections"""
        try:
            logger.info("Generating editor operations...")
            
            proofreading_result = state.get("proofreading_result", {})
            scope_text = state.get("scope_text", "")
            active_editor = state.get("active_editor", {}) or {}
            filename = active_editor.get("filename") or "document.md"
            
            corrections = proofreading_result.get("corrections", [])
            operations = []
            
            # Simple hash function for pre_hash (matches frontend implementation)
            def simple_hash(text: str) -> str:
                h = 0
                for char in text:
                    h = (h * 31 + ord(char)) & 0xFFFFFFFF
                return hex(h)[2:]
            
            for correction in corrections:
                try:
                    original_text = correction.get("original_text", "") if isinstance(correction, dict) else getattr(correction, "original_text", "")
                    changed_to = correction.get("changed_to", "") if isinstance(correction, dict) else getattr(correction, "changed_to", "")
                    explanation = correction.get("explanation", "") if isinstance(correction, dict) else getattr(correction, "explanation", "")
                    
                    # Skip if no actual change
                    if original_text == changed_to:
                        continue
                    
                    # Find the original text in the scope
                    start_pos = scope_text.find(original_text)
                    if start_pos == -1:
                        # Try to find a partial match or similar text
                        continue
                    
                    end_pos = start_pos + len(original_text)
                    
                    # Create editor operation
                    operation = {
                        "op_type": "replace_range",
                        "start": start_pos,
                        "end": end_pos,
                        "text": changed_to,
                        "pre_hash": simple_hash(original_text),
                        "note": f"Proofreading: {explanation}" if explanation else "Proofreading correction"
                    }
                    
                    operations.append(operation)
                    
                except Exception as e:
                    logger.warning(f"Failed to generate editor operation for correction: {e}")
                    continue
            
            logger.info(f"Generated {len(operations)} editor operations")
            
            return {
                "editor_operations": operations
            }
            
        except Exception as e:
            logger.error(f"Error generating operations: {e}")
            return {
                "editor_operations": []
            }
    
    async def _format_response_node(self, state: ProofreadingState) -> Dict[str, Any]:
        """Format final response"""
        try:
            logger.info("Formatting proofreading response...")
            
            proofreading_result = state.get("proofreading_result", {})
            editor_operations = state.get("editor_operations", [])
            mode = state.get("mode", "clarity")
            active_editor = state.get("active_editor", {}) or {}
            filename = active_editor.get("filename") or "document.md"
            
            corrections = proofreading_result.get("corrections", [])
            style_guide_used = proofreading_result.get("style_guide_used")
            
            # Format user message
            header = f"## Proofreading ({mode})\n"
            if style_guide_used:
                header += f"Using style guide: {style_guide_used}\n\n"
            if not corrections:
                response_text = header + "No corrections suggested."
            else:
                lines = [header, f"{len(corrections)} correction(s) suggested:\n"]
                for idx, c in enumerate(corrections, 1):
                    original = c.get("original_text", "") if isinstance(c, dict) else getattr(c, "original_text", "")
                    changed = c.get("changed_to", "") if isinstance(c, dict) else getattr(c, "changed_to", "")
                    explanation = c.get("explanation", "") if isinstance(c, dict) else getattr(c, "explanation", "")
                    
                    lines.append(f"### {idx}.")
                    lines.append("Original text:")
                    lines.append("```")
                    lines.append(original)
                    lines.append("```")
                    lines.append("")
                    lines.append("Changed to:")
                    lines.append("```")
                    lines.append(changed)
                    lines.append("```")
                    if explanation:
                        lines.append("")
                        lines.append(f"Reason: {explanation}")
                    lines.append("")
                response_text = "\n".join(lines)
            
            # Add editor operations to proofreading result
            proofreading_result["editor_operations"] = editor_operations
            
            # Create manuscript edit for HITL
            manuscript_edit = None
            if editor_operations:
                manuscript_edit = {
                    "target_filename": filename,
                    "operations": editor_operations,
                    "scope": "paragraph",
                    "summary": f"Proofreading corrections ({len(editor_operations)} operations)",
                    "safety": "low"
                }
            
            return {
                "response": {
                    "task_status": TaskStatus.COMPLETE.value,
                    "response": response_text,
                    "proofreading_result": proofreading_result,
                    "editor_operations": editor_operations,
                    "manuscript_edit": manuscript_edit
                },
                "task_status": "complete"
            }
            
        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            return {
                "response": {
                    "task_status": TaskStatus.ERROR.value,
                    "response": f"Error formatting response: {str(e)}",
                    "error": str(e)
                },
                "task_status": "error",
                "error": str(e)
            }
    
    def _parse_proofreading_result(self, content: str) -> Dict[str, Any]:
        """Parse structured proofreading result from LLM response"""
        try:
            text = content.strip()
            
            # Handle various JSON formats
            if '```json' in text:
                m = re.search(r'```json\s*\n([\s\S]*?)\n```', text)
                if m:
                    text = m.group(1).strip()
            elif '```' in text:
                m = re.search(r'```\s*\n([\s\S]*?)\n```', text)
                if m:
                    text = m.group(1).strip()
            
            # Find JSON object if not at start
            if not text.startswith('{'):
                m = re.search(r'\{[\s\S]*\}', text)
                if m:
                    text = m.group(0)
            
            # Parse JSON
            data = json.loads(text)
            
            # Normalize fields
            data.setdefault("task_status", "complete")
            data.setdefault("mode", "clarity")
            data.setdefault("corrections", [])
            data.setdefault("summary", "Proofreading completed")
            data.setdefault("style_guide_used", False)
            data.setdefault("metadata", {})
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to parse proofreading JSON: {e}")
            # Fallback
            return {
                "task_status": "complete",
                "mode": "clarity",
                "summary": "Proofreading completed with some formatting issues",
                "corrections": [{
                    "original_text": "[Formatting issue]",
                    "changed_to": "Please review the text manually for corrections",
                    "explanation": "LLM response format needs adjustment",
                    "scope": "sentence"
                }],
                "metadata": {"parse_error": str(e), "raw_content_length": len(content)},
            }
    
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

