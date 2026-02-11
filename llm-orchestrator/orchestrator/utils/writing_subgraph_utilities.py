"""
Writing Subgraph Utilities

Shared utilities for outline, rules, and style editing subgraphs.
Consolidates duplicate code and provides consistent patterns.
"""

import hashlib
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable

from orchestrator.models.agent_response_contract import AgentResponse, ManuscriptEditMetadata
from orchestrator.tools.reference_file_loader import load_referenced_files

logger = logging.getLogger(__name__)


# ============================================
# Core Utilities
# ============================================

def preserve_critical_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract the 5 critical state keys that MUST be preserved across nodes.
    
    These keys are essential for:
    - metadata: User model preference, timezone, etc.
    - user_id: Database queries and access control
    - shared_memory: Cross-agent and cross-turn context
    - messages: Conversation history
    - query: Original user query
    
    Args:
        state: Current state dict
        
    Returns:
        Dict with the 5 critical keys
    """
    return {
        "metadata": state.get("metadata", {}),
        "user_id": state.get("user_id", "system"),
        "shared_memory": state.get("shared_memory", {}),
        "messages": state.get("messages", []),
        "query": state.get("query", "")
    }


def preserve_fiction_state(state: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preserve all fiction-specific state while applying updates.
    Used by flat fiction_editing_subgraph nodes to avoid state dropping.
    """
    preserved = {
        "metadata": state.get("metadata", {}),
        "user_id": state.get("user_id", "system"),
        "shared_memory": state.get("shared_memory", {}),
        "messages": state.get("messages", []),
        "query": state.get("query", ""),
        "active_editor": state.get("active_editor", {}),
        "manuscript": state.get("manuscript", ""),
        "manuscript_content": state.get("manuscript_content", ""),
        "filename": state.get("filename", ""),
        "frontmatter": state.get("frontmatter", {}),
        "current_chapter_text": state.get("current_chapter_text", ""),
        "current_chapter_number": state.get("current_chapter_number"),
        "prev_chapter_text": state.get("prev_chapter_text"),
        "prev_chapter_number": state.get("prev_chapter_number"),
        "next_chapter_text": state.get("next_chapter_text"),
        "next_chapter_number": state.get("next_chapter_number"),
        "chapter_ranges": state.get("chapter_ranges", []),
        "explicit_primary_chapter": state.get("explicit_primary_chapter"),
        "requested_chapter_number": state.get("requested_chapter_number"),
        "reference_chapter_numbers": state.get("reference_chapter_numbers", []),
        "reference_chapter_texts": state.get("reference_chapter_texts", {}),
        "outline_body": state.get("outline_body"),
        "rules_body": state.get("rules_body"),
        "style_body": state.get("style_body"),
        "characters_bodies": state.get("characters_bodies", []),
        "series_body": state.get("series_body"),
        "outline_current_chapter_text": state.get("outline_current_chapter_text"),
        "outline_prev_chapter_text": state.get("outline_prev_chapter_text"),
        "story_overview": state.get("story_overview"),
        "book_map": state.get("book_map"),
        "system_prompt": state.get("system_prompt", ""),
        "datetime_context": state.get("datetime_context", ""),
        "current_request": state.get("current_request", ""),
        "selection_start": state.get("selection_start", -1),
        "selection_end": state.get("selection_end", -1),
        "cursor_offset": state.get("cursor_offset", -1),
        "mode_guidance": state.get("mode_guidance", ""),
        "reference_guidance": state.get("reference_guidance", ""),
        "reference_quality": state.get("reference_quality", {}),
        "reference_warnings": state.get("reference_warnings", []),
        "has_references": state.get("has_references", False),
        "creative_freedom_requested": state.get("creative_freedom_requested", False),
        "request_type": state.get("request_type", "edit_request"),
        "outline_sync_analysis": state.get("outline_sync_analysis"),
        "consistency_warnings": state.get("consistency_warnings", []),
        "structured_edit": state.get("structured_edit"),
        "editor_operations": state.get("editor_operations", []),
        "failed_operations": state.get("failed_operations", []),
        "response": state.get("response"),
        "task_status": state.get("task_status", ""),
        "error": state.get("error", ""),
    }
    preserved.update(updates)
    return preserved


def create_writing_error_response(
    error_msg: str,
    agent_type: str,
    state: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create standardized error response for writing subgraphs.
    
    Args:
        error_msg: Error message
        agent_type: Agent type identifier
        state: Current state for preserving critical keys
        
    Returns:
        Standardized error response dict
    """
    error_response = AgentResponse(
        response=f"{agent_type} failed: {error_msg}",
        task_status="error",
        agent_type=agent_type,
        timestamp=datetime.now().isoformat(),
        error=error_msg
    )
    
    return {
        "response": error_response.dict(exclude_none=True),
        "task_status": "error",
        **preserve_critical_state(state)
    }


def extract_user_request(state: Dict[str, Any]) -> str:
    """
    Extract current user request from messages or query.
    
    Args:
        state: Current state
        
    Returns:
        User request string
    """
    messages = state.get("messages", [])
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, 'content'):
            return last_message.content if isinstance(last_message.content, str) else str(last_message.content)
    
    return state.get("query", "")


def paragraph_bounds(text: str, cursor_offset: int) -> Tuple[int, int]:
    """
    Find paragraph boundaries around cursor.
    
    Args:
        text: Document text
        cursor_offset: Cursor position
        
    Returns:
        Tuple of (start, end) indices
    """
    if not text:
        return 0, 0
    cursor = max(0, min(len(text), cursor_offset))
    left = text.rfind("\n\n", 0, cursor)
    start = left + 2 if left != -1 else 0
    right = text.find("\n\n", cursor)
    end = right if right != -1 else len(text)
    return start, end


def strip_frontmatter_block(text: str) -> str:
    """
    Strip YAML frontmatter from text.
    
    Args:
        text: Text with potential frontmatter
        
    Returns:
        Text without frontmatter
    """
    try:
        return re.sub(r'^---\s*\n[\s\S]*?\n---\s*\n', '', text, flags=re.MULTILINE)
    except Exception:
        return text


def slice_hash(text: str) -> str:
    """
    Match frontend sliceHash: 32-bit rolling hash to hex string.
    
    Args:
        text: Text to hash
        
    Returns:
        Hex string hash
    """
    try:
        h = 0
        for ch in text:
            h = (h * 31 + ord(ch)) & 0xFFFFFFFF
        return format(h, 'x')
    except Exception:
        return ""


# ============================================
# Format Response Utilities
# ============================================

def build_response_text_for_question(
    structured_edit: Optional[Dict[str, Any]],
    editor_operations: List[Dict],
    fallback: str = "Analysis complete."
) -> str:
    """
    Build response text for question requests.
    
    Prioritizes summary (answer/analysis) over operation text.
    
    Args:
        structured_edit: Structured edit dict with summary
        editor_operations: List of editor operations
        fallback: Fallback text if no summary
        
    Returns:
        Response text for question requests
    """
    if structured_edit:
        summary = structured_edit.get("summary", "").strip()
        if summary and len(summary) > 20:
            # Question with answer - use summary as response text
            if editor_operations:
                logger.info(f"Question request with {len(editor_operations)} operations - using summary as conversational response")
            return summary
        elif editor_operations:
            # Question with operations but no summary
            return f"Analysis complete. Made {len(editor_operations)} edit(s) based on your question."
    
    return fallback


def build_response_text_for_edit(
    structured_edit: Optional[Dict[str, Any]],
    editor_operations: List[Dict],
    fallback: str = "Edit plan ready."
) -> str:
    """
    Build response text for edit requests.
    
    Uses summary if available, otherwise preview from operations.
    
    Args:
        structured_edit: Structured edit dict with summary
        editor_operations: List of editor operations
        fallback: Fallback text
        
    Returns:
        Response text for edit requests
    """
    summary = structured_edit.get("summary", "").strip() if structured_edit else ""
    
    # Build preview from operations
    preview = "\n\n".join([
        op.get("text", "").strip()
        for op in editor_operations
        if op.get("text", "").strip()
    ]).strip()
    
    # Prioritize summary if available and meaningful
    if summary and len(summary) > 20:
        if preview:
            return f"{summary}\n\n---\n\n{preview}"
        return summary
    elif preview:
        return preview
    
    return fallback


def build_failed_operations_section(
    failed_operations: List[Dict],
    doc_type: str
) -> str:
    """
    Build formatted section for unresolved operations.
    
    Args:
        failed_operations: List of failed operations
        doc_type: Document type for context
        
    Returns:
        Formatted markdown section
    """
    if not failed_operations:
        return ""
    
    failed_section = "\n\n**⚠️ UNRESOLVED EDITS (Manual Action Required)**\n"
    failed_section += f"The following generated content could not be automatically placed in the {doc_type}. You can copy and paste these sections manually:\n\n"
    
    for i, op in enumerate(failed_operations, 1):
        op_type = op.get("op_type", "edit")
        error = op.get("error", "Anchor text not found")
        text = op.get("text", "")
        anchor = op.get("anchor_text") or op.get("original_text")
        
        failed_section += f"#### Unresolved Edit {i} ({op_type})\n"
        failed_section += f"- **Reason**: {error}\n"
        if anchor:
            failed_section += f"- **Intended near**:\n> {anchor[:200]}...\n"
        
        failed_section += "\n**Generated Content** (Scroll-safe):\n"
        failed_section += f"{text}\n\n"
        failed_section += "---\n"
    
    return failed_section


def create_manuscript_edit_metadata(
    structured_edit: Optional[Dict[str, Any]],
    editor_operations: List[Dict]
) -> Optional[ManuscriptEditMetadata]:
    """
    Build ManuscriptEditMetadata from structured_edit.
    
    Args:
        structured_edit: Structured edit dict
        editor_operations: List of editor operations
        
    Returns:
        ManuscriptEditMetadata or None
    """
    if not structured_edit or not editor_operations:
        return None
    
    return ManuscriptEditMetadata(
        target_filename=structured_edit.get("target_filename"),
        scope=structured_edit.get("scope"),
        summary=structured_edit.get("summary"),
        chapter_index=structured_edit.get("chapter_index"),
        safety=structured_edit.get("safety"),
        operations=editor_operations
    )


# ============================================
# Context Preparation Utilities
# ============================================

async def prepare_writing_context(
    state: Dict[str, Any],
    doc_type: str,
    default_filename: str,
    content_key: str = "content",
    validate_type: bool = False,
    clarification_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Unified context preparation for writing subgraphs.
    
    Args:
        state: Current state
        doc_type: Expected document type ('outline', 'rules', 'style', etc.)
        default_filename: Fallback filename
        content_key: Key to use for content in return dict ('outline', 'rules', 'style', etc.)
        validate_type: If True, gate on frontmatter.type == doc_type
        clarification_key: Optional key for clarification context in shared_memory
        
    Returns:
        State dict with extracted context + preserved critical state
    """
    try:
        shared_memory = state.get("shared_memory", {}) or {}
        active_editor = shared_memory.get("active_editor", {})
        
        if not active_editor:
            logger.warning(f"⚠️ No active editor found for {doc_type} editing")
            return {
                "error": f"No active editor found for {doc_type} editing",
                "task_status": "error",
                **preserve_critical_state(state)
            }
        
        filename = active_editor.get("filename", default_filename)
        content = active_editor.get("content", "")
        frontmatter = active_editor.get("frontmatter", {})
        cursor_offset = active_editor.get("cursor_offset", -1)
        selection_start = active_editor.get("selection_start", -1)
        selection_end = active_editor.get("selection_end", -1)
        
        # Type validation if requested
        if validate_type:
            doc_type_from_frontmatter = frontmatter.get("type", "").lower()
            if doc_type_from_frontmatter != doc_type:
                logger.warning(f"⚠️ Active editor is not a {doc_type} file (type: {doc_type_from_frontmatter})")
                return {
                    "error": f"Active editor is not a {doc_type} file; skipping.",
                    "task_status": "error",
                    **preserve_critical_state(state)
                }
        
        # Strip frontmatter for body
        body_only = strip_frontmatter_block(content)
        
        # Extract paragraph bounds if cursor is set
        para_start, para_end = 0, len(body_only)
        if cursor_offset >= 0:
            para_start, para_end = paragraph_bounds(body_only, cursor_offset)
        
        # Build result dict
        result = {
            "active_editor": active_editor,
            "filename": filename,
            "frontmatter": frontmatter,
            "cursor_offset": cursor_offset,
            "selection_start": selection_start,
            "selection_end": selection_end,
            "body_only": body_only,
            "para_start": para_start,
            "para_end": para_end,
            **preserve_critical_state(state)
        }
        
        # Add content with appropriate key
        result[content_key] = content
        
        # Add clarification context if requested
        if clarification_key:
            clarification_context = shared_memory.get(clarification_key)
            if clarification_context:
                result["clarification_context"] = clarification_context
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to prepare {doc_type} context: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return {
            "error": str(e),
            "task_status": "error",
            **preserve_critical_state(state)
        }


# ============================================
# Reference Loading Utilities
# ============================================

async def load_writing_references(
    state: Dict[str, Any],
    reference_config: Dict[str, List[str]],
    cascade_config: Optional[Dict[str, Dict[str, List[str]]]] = None,
    doc_type_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load referenced files for writing subgraphs.
    
    Supports both direct references (rules, style) and cascaded references (fiction via outline).
    
    Args:
        state: Current state with active_editor and user_id
        reference_config: Direct references to load from frontmatter
            Format: {"key": ["frontmatter_key1", "frontmatter_key2"]}
        cascade_config: Optional cascading config (for fiction subgraphs)
            Format: {"source_key": {"target_key": ["frontmatter_keys"]}}
        doc_type_filter: Optional type filter for logging
        
    Returns:
        State dict with loaded reference bodies + preserved critical state
    """
    try:
        shared_memory = state.get("shared_memory", {}) or {}
        active_editor = shared_memory.get("active_editor", {})
        user_id = state.get("user_id", "system")
        
        if not active_editor:
            logger.warning(f"⚠️ No active editor found for reference loading")
            return {
                "error": "No active editor found",
                "task_status": "error",
                **preserve_critical_state(state)
            }
        
        # Load references using the unified loader
        result = await load_referenced_files(
            active_editor=active_editor,
            user_id=user_id,
            reference_config=reference_config,
            cascade_config=cascade_config
        )
        
        loaded_files = result.get("loaded_files", {})
        
        # Extract content bodies for each reference type
        # loaded_files is a dict: {category: [file_dict, ...], ...}
        result_dict = {
            **preserve_critical_state(state)
        }
        
        # Add loaded content with appropriate keys
        # For each category in reference_config, extract content from loaded_files
        for category in reference_config.keys():
            if category in loaded_files and loaded_files[category]:
                # Get first file's content (most subgraphs use first file only)
                content = loaded_files[category][0].get("content", "")
                if content:
                    # Strip frontmatter for body
                    body = strip_frontmatter_block(content)
                    result_dict[f"{category}_body"] = body
                else:
                    result_dict[f"{category}_body"] = None
            else:
                result_dict[f"{category}_body"] = None
        
        # Handle cascaded references - they're already loaded by load_referenced_files
        # Just extract them from loaded_files
        if cascade_config:
            for source_key, target_config in cascade_config.items():
                # Cascaded references are already in loaded_files by their category names
                for cascade_category in target_config.keys():
                    if cascade_category in loaded_files and loaded_files[cascade_category]:
                        # Get first file's content
                        content = loaded_files[cascade_category][0].get("content", "")
                        if content:
                            body = strip_frontmatter_block(content)
                            result_dict[f"{cascade_category}_body"] = body
                        else:
                            result_dict[f"{cascade_category}_body"] = None
                    elif cascade_category not in result_dict:
                        result_dict[f"{cascade_category}_body"] = None
        
        # For characters, handle list of bodies (multiple character files)
        if "characters" in reference_config or (cascade_config and any("characters" in cfg.values() for cfg in cascade_config.values())):
            characters_bodies = []
            if "characters" in loaded_files:
                for char_file in loaded_files["characters"]:
                    content = char_file.get("content", "")
                    if content:
                        body = strip_frontmatter_block(content)
                        characters_bodies.append(body)
            result_dict["characters_bodies"] = characters_bodies
        
        if doc_type_filter:
            logger.info(f"📄 Loaded references for {doc_type_filter}: {list(result_dict.keys())}")
        
        return result_dict
        
    except Exception as e:
        logger.error(f"❌ Failed to load references: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return {
            "error": str(e),
            "task_status": "error",
            **preserve_critical_state(state)
        }


# ============================================
# Request Type Detection Utilities
# ============================================

async def detect_writing_request_type(
    state: Dict[str, Any],
    llm_factory: Callable,
    doc_type: str,
    body_context: str
) -> Dict[str, Any]:
    """
    Detect if request is a question or edit request using LLM.
    
    Args:
        state: Current state
        llm_factory: Factory function for LLM instances
        doc_type: Document type for context ('outline', 'rules', 'style')
        body_context: Document body for context
        
    Returns:
        State dict with request_type + preserved critical state
    """
    try:
        current_request = extract_user_request(state)
        
        if not current_request:
            logger.warning("No current request found - defaulting to edit_request")
            return {
                "request_type": "edit_request",
                "current_request": "",
                **preserve_critical_state(state)
            }
        
        # Build detection prompt
        detection_prompt = f"""You are analyzing a user request for {doc_type} editing.

Document context (first 500 chars):
{body_context[:500]}

User request:
{current_request}

Determine if this is:
1. A QUESTION/ANALYSIS request - user wants information, analysis, or clarification about the {doc_type}
2. An EDIT request - user wants to modify, add, or change content in the {doc_type}

Respond with ONLY a JSON object:
{{
    "request_type": "question" or "edit_request",
    "reasoning": "brief explanation"
}}"""

        # Get LLM
        llm = llm_factory(temperature=0.1, state=state)
        
        # Call LLM
        from langchain_core.messages import HumanMessage
        response = await llm.ainvoke([HumanMessage(content=detection_prompt)])
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Parse response
        import json
        try:
            # Try direct JSON parse
            parsed = json.loads(content)
        except json.JSONDecodeError:
            # Try extracting JSON from markdown code blocks
            import re
            json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
            else:
                # Fallback: use keyword detection
                request_lower = current_request.lower()
                if any(kw in request_lower for kw in ["what", "how", "why", "when", "where", "explain", "analyze", "tell me", "describe"]):
                    parsed = {"request_type": "question"}
                else:
                    parsed = {"request_type": "edit_request"}
        
        request_type = parsed.get("request_type", "edit_request")
        if request_type not in ["question", "edit_request"]:
            request_type = "edit_request"
        
        logger.info(f"📊 Detected request type: {request_type}")
        
        return {
            "request_type": request_type,
            "current_request": current_request,
            **preserve_critical_state(state)
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to detect request type: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        # Fallback to edit_request
        return {
            "request_type": "edit_request",
            "current_request": extract_user_request(state),
            **preserve_critical_state(state)
        }


# ============================================
# Editor Operation Prompt Components
# ============================================

def build_editor_safety_rules() -> str:
    """
    Core editor operation safety rules - universal across all editing agents.
    
    Teaches LLMs to use minimal matches and avoid stomping unrelated content.
    This is the most critical guidance to prevent operations from replacing
    more content than intended.
    
    Returns:
        Formatted safety rules string for inclusion in system prompts
    """
    return (
        "**CRITICAL: ALL OPERATIONS MUST HAVE ANCHORS**\n\n"
        "**FOR replace_range/delete_range:**\n"
        "- **MANDATORY**: You MUST provide 'original_text' with EXACT, VERBATIM text from the file\n"
        "- **CRITICAL**: If you don't provide 'original_text', the operation will FAIL completely\n"
        "- **NEVER** create replace_range or delete_range without 'original_text' - it will fail or stomp wrong content!\n"
        "- **USE MINIMAL MATCHES**: Only include the portion you're actually changing in 'original_text'\n"
        "  - BAD: original_text=\"John walked down the street\" + text=\"ran\" → Result: entire phrase becomes just \"ran\" (loses \"John\" and \"down the street\")\n"
        "  - GOOD: original_text=\"walked\" + text=\"ran\" → Result: \"John walked down the street\" becomes \"John ran down the street\"\n"
        "- **REPLACEMENT TEXT IS COMPLETE**: Everything in 'original_text' will be replaced with ONLY what's in 'text'\n"
        "  - Example: original_text=\"caught Walter's eye and smiled\" + text=\"caught his eye\"\n"
        "  - Result: \"caught Walter's eye and smiled\" → \"caught his eye\" (loses \"Walter's\" and \"smiled\" unless included in text)\n"
        "- **Golden Rule**: If your original_text contains words/content that should NOT be deleted, make it smaller!\n\n"
        "**FOR insert_after_heading/insert_after:**\n"
        "- **MANDATORY**: You MUST provide 'anchor_text' with EXACT text from the file to insert after\n"
        "- **CRITICAL**: If you don't provide 'anchor_text', the operation will FAIL to find insertion point\n"
        "- For insert_after_heading: anchor_text must be exact section header (e.g., '### Transformation & Physiology')\n"
        "  - **ONLY use for COMPLETELY EMPTY sections** (no bullets, no text below the header)\n"
        "  - If section has ANY content, use replace_range instead!\n"
        "- For insert_after: anchor_text must be last few words before insertion point (minimum 10-20 words)\n"
        "  - **ABSOLUTE PROHIBITION**: NEVER use insert_after for adding bullets to bullet lists!\n"
        "  - **ALWAYS use replace_range** to expand bullet lists (replace ONLY THE LAST BULLET with expanded version)\n"
        "  - insert_after is ONLY for continuing narrative paragraphs or prose, NOT bullet lists\n"
        "  - WHY: insert_after with bullet anchor will INSERT IN THE MIDDLE if you pick the wrong bullet!\n"
        "  - Using replace_range guarantees clean placement - surgical replacement of last bullet, no middle-insertion\n"
        "  - ⚠️ BE SURGICAL: original_text should be 1-2 bullets max (the last one), NOT 10+ bullets!\n\n"
        "**WHAT IS VALID ANCHOR TEXT:**\n"
        "✅ CORRECT: Text that ACTUALLY EXISTS in the file content shown above (between START/END markers)\n"
        "✅ CORRECT: Section headers for insert_after_heading (e.g., '## Magic Systems') - must exist in file\n"
        "✅ CORRECT: Exact sentences/paragraphs/bullets from the file for replace_range - must exist in file\n"
        "✅ CORRECT: Text copied VERBATIM from the current file content (shown in context above)\n"
        "❌ WRONG: Paraphrased or summarized text from the file\n"
        "❌ WRONG: Text you THINK should be in the file but isn't shown in the content above\n"
        "❌ WRONG: Invented text that 'sounds right' but isn't actually in the file\n"
        "❌ WRONG: Text from reference documents (style guides, character docs) - use ONLY from the ACTUAL file being edited!\n"
        "❌ WRONG: Text from outlines, summaries, or other sections - use ONLY from the file content section!\n\n"
        "**CRITICAL: NEVER INCLUDE SECTION HEADERS IN original_text**:\n"
        "- Section headers (###, ##, #) should NEVER appear in 'original_text' for replace_range/delete_range\n"
        "- Use insert_after_heading to add content below headers instead\n"
        "- Including headers in original_text will DELETE the headers!\n\n"
    )


def build_editor_verification_checklist() -> str:
    """
    Pre-operation verification checklist - ensures LLM validates operations before generating.
    
    This checklist prompts the LLM to verify each operation before generating JSON,
    catching common mistakes before they cause problems.
    
    Returns:
        Formatted verification checklist string
    """
    return (
        "**VERIFICATION CHECKLIST (READ BEFORE GENERATING OPERATIONS):**\n"
        "Before you generate ANY operations, verify:\n"
        "☐ **CRITICAL**: I copied text EXACTLY as written in the file content shown above (VERBATIM, no paraphrasing)\n"
        "☐ **CRITICAL**: For replace_range/delete_range: I included 'original_text' and it EXISTS between START/END markers\n"
        "☐ **CRITICAL**: For insert_after_heading/insert_after: I included 'anchor_text' and it EXISTS between START/END markers\n"
        "☐ I did a mental Ctrl+F search in the content above and confirmed I can FIND my anchor_text/original_text\n"
        "☐ **CRITICAL**: My anchor_text/original_text is NOT from reference docs - it's from the ACTUAL file being edited\n"
        "☐ I did NOT invent, paraphrase, or generate anchor text - I COPIED it EXACTLY from what's shown\n"
        "☐ For replace_range: My 'original_text' includes ONLY the minimal text I'm changing (not adjacent unrelated content)\n"
        "☐ For replace_range: If changing bullet #2 in a list, my original_text is ONLY bullet #2, not #1 or #3\n"
        "☐ For replace_range: I did NOT include section headers in 'original_text' (headers should never be in original_text)\n"
        "☐ For delete_range: My 'original_text' matches EXACT text to delete (minimum 10-20 words for reliable matching)\n"
        "☐ For delete_range: I did NOT include section headers in 'original_text' (use dedicated header operations instead)\n"
        "☐ For insert_after_heading: The section is COMPLETELY EMPTY below header - if content exists, I must use replace_range\n"
        "☐ For insert_after_heading: My 'anchor_text' is the EXACT section header (e.g., '### Magic Systems', not 'magic section')\n"
        "☐ **FOR BULLET LISTS**: If adding bullets, my anchor_text is from the LAST bullet in the section (not a middle bullet!)\n"
        "☐ **FOR MULTI-LINE BULLETS**: I verified my anchor is the LAST LINE of the LAST BULLET (not just end of any bullet)\n"
        "☐ For insert_after with bullet lists: I scanned the ENTIRE section and confirmed NO MORE BULLETS exist after my anchor\n"
        "☐ For insert_after with bullet lists: I checked that the next content after my anchor is a section header (###) or end of file\n"
        "☐ My 'text' field is COMPLETE - everything in original_text will be replaced with ONLY what's in 'text'\n"
        "☐ I verified that no unrelated content will be accidentally deleted by my original_text match\n"
        "☐ **FINAL CHECK**: My operation will NOT fail due to missing anchors - I provided original_text or anchor_text\n\n"
    )


def build_editor_common_mistakes() -> str:
    """
    Visual examples of common mistakes - prevents repeated errors.
    
    Shows concrete BAD vs GOOD examples with explicit results to help LLMs
    understand the consequences of incorrect operations.
    
    Returns:
        Formatted common mistakes string with examples
    """
    return (
        "**COMMON MISTAKES TO AVOID:**\n\n"
        "**MISTAKE #1: Missing text anchors (causes complete failure)**\n"
        "❌ BAD: {\"op_type\": \"delete_range\", \"start\": 632, \"end\": 1056, \"original_text\": null}\n"
        "   Result: Operation FAILS or uses unreliable index-based fallback, may stomp wrong content!\n"
        "✅ GOOD: {\"op_type\": \"delete_range\", \"original_text\": \"- The transformation triggers...superiority.\"}\n"
        "   Result: Operation finds and deletes ONLY the specified text, reliably!\n\n"
        "**MISTAKE #2: Too broad original_text (loses unrelated content)**\n"
        "❌ BAD: original_text=\"John walked down the street\" + text=\"ran\"\n"
        "   Result: \"John walked down the street\" → \"ran\" (loses \"John\" and \"down the street\"!)\n"
        "✅ GOOD: original_text=\"walked\" + text=\"ran\"\n"
        "   Result: \"John walked down the street\" → \"John ran down the street\"\n\n"
        "❌ BAD: original_text contains 3 bullet points when you only want to change 1\n"
        "   Result: All 3 bullets deleted, only your new bullet remains!\n"
        "✅ GOOD: original_text contains only the single bullet being changed\n"
        "   Result: Only that bullet changes, others preserved\n\n"
        "**MISTAKE #3: Including section headers in original_text (deletes headers)**\n"
        "❌ BAD: original_text=\"### Magic Systems\\n- Magic requires components\"\n"
        "   Result: Header gets DELETED! Section structure broken!\n"
        "✅ GOOD: For empty section: Use insert_after_heading with anchor_text=\"### Magic Systems\"\n"
        "✅ GOOD: For existing content: original_text=\"- Magic requires components\" (no header)\n"
        "   Result: Header preserved, only content updated\n\n"
        "**MISTAKE #4: Using insert_after_heading when content exists (splits section)**\n"
        "❌ BAD: Section has \"### Magic\\n- Existing rule\" → insert_after_heading with anchor=\"### Magic\"\n"
        "   Result: \"### Magic\\n[NEW CONTENT INSERTED HERE]\\n- Existing rule\" (splits the section!)\n"
        "✅ GOOD: Use replace_range with original_text=\"- Existing rule\" to update existing content\n"
        "   Result: Existing content cleanly replaced, section structure intact\n\n"
        "**MISTAKE #5: Paraphrased or invented anchor text (operation fails)**\n"
        "❌ BAD: File says \"Magic requires physical components\" but you use original_text=\"Magic needs components\"\n"
        "   Result: Text not found! Operation FAILS!\n"
        "✅ GOOD: Copy EXACTLY: original_text=\"Magic requires physical components\"\n"
        "   Result: Text found and operation succeeds!\n\n"
        "**MISTAKE #6: Anchor text from wrong section (operation fails or places edit at beginning)**\n"
        "❌ BAD: Using anchor_text from STYLE GUIDE or CHARACTER DOC instead of from the ACTUAL FILE CONTENT\n"
        "   Result: Text not found in file! Operation defaults to beginning or FAILS!\n"
        "❌ BAD: Using text from OUTLINE or SUMMARY sections when editing the manuscript\n"
        "   Result: Text not found! Operation places edit in wrong location!\n"
        "✅ GOOD: Copy anchor_text ONLY from the file content section (between START/END markers)\n"
        "   Result: Text found in correct location, operation succeeds!\n\n"
        "**MISTAKE #7: Using insert_after for bullet lists (FORBIDDEN - causes middle-insertion!)**\n"
        "This is the #1 cause of fragmented bullet lists in rules documents!\n\n"
        "❌ ABSOLUTELY FORBIDDEN: Using insert_after to add bullets to a bullet list\n"
        "   WHY IT FAILS: insert_after uses an anchor from the document. If you pick ANY bullet\n"
        "   other than the ABSOLUTE LAST bullet, new content gets inserted IN THE MIDDLE!\n"
        "   Example Problem:\n"
        "   * Section has: A, B, C, D (4 bullets)\n"
        "   * You use insert_after with anchor=\"last line of bullet B\"\n"
        "   * Result: A, B, [NEW CONTENT HERE], C, D ← New bullets split the list!\n"
        "\n"
        "❌ FORBIDDEN: insert_after with multi-line bullet anchor\n"
        "   Example: Section has:\n"
        "   ```\n"
        "   - Vampires transform at will.\n"
        "   - Human form requires\n"
        "     continuous effort.\n"
        "   - Another rule here.\n"
        "   ```\n"
        "   * You use anchor=\"continuous effort.\" (last line of middle bullet!)\n"
        "   * Result: New bullets inserted BETWEEN \"Human form...\" and \"Another rule\"\n"
        "   * This splits the list and creates fragmented sections!\n\n"
        "✅ CORRECT APPROACH: Use replace_range to replace ONLY THE LAST BULLET with expanded version\n"
        "   1. Find the LAST bullet in the target section (just the final one)\n"
        "   2. Copy ONLY that last bullet into original_text: \"- Bullet D\"\n"
        "   3. Create expanded version in text field: \"- Bullet D\\n- NEW Bullet E\"\n"
        "   4. Result: Surgical replacement - only last bullet touched, bullets A, B, C untouched!\n"
        "   ⚠️ NEVER include 10+ bullets in original_text - that's too broad and deletes too much!\n\n"
        "✅ WHEN TO USE insert_after (NARROW USE CASE - NOT FOR BULLET LISTS!):\n"
        "   - **ONLY** for continuing narrative paragraphs or prose\n"
        "   - Example: Adding to the end of a multi-sentence explanation\n"
        "   - Example: Continuing a narrative description\n"
        "   - **NEVER** for bullet lists - use replace_range instead!\n\n"
    )


def build_chapter_continuation_guidance() -> str:
    """
    Chapter-specific guidance for fiction agents.
    
    Teaches fiction agents how to properly continue existing chapters vs adding new ones.
    This is critical for fiction generation to avoid duplicating chapter headers.
    
    Returns:
        Formatted chapter continuation guidance string
    """
    return (
        "**FOR NEW CHAPTERS AT END OF MANUSCRIPT:**\n"
        "- Look for 'LAST LINE OF CHAPTER X' marker in the context\n"
        "- Use that EXACT line as anchor_text with op_type='insert_after_heading'\n"
        "- Your 'text' field MUST start with '## Chapter N' followed by two newlines, then your chapter content\n"
        "- **MANDATORY**: All new chapter content must include the chapter header - do not omit it!\n\n"
        "**FOR CONTINUING/EXTENDING EXISTING CHAPTERS:**\n"
        "- **CRITICAL**: When user asks to 'continue chapter X' or 'finish chapter X', they want to ADD to existing content!\n"
        "- **DO NOT** use 'insert_after_heading' with '## Chapter X' - this inserts AFTER the heading, causing duplication!\n"
        "- **CORRECT APPROACH**: Use 'insert_after' (NOT insert_after_heading) with the LAST FEW WORDS of the existing chapter text\n"
        "- Find the LAST SENTENCE/PARAGRAPH of the chapter in the MANUSCRIPT TEXT section\n"
        "- Copy the LAST 15-30 WORDS of that chapter (the ending words before the next chapter or end of file)\n"
        "- Use op_type='insert_after' with those words as anchor_text\n"
        "- **DO NOT** include '## Chapter X' heading in your generated text - the chapter already exists!\n"
        "- Your generated text should flow seamlessly from where the chapter ended\n"
        "- Example: Chapter 3 ends with '...and walked out the door.'\n"
        "  - CORRECT: {\"op_type\": \"insert_after\", \"anchor_text\": \"walked out the door.\", \"text\": \"\\n\\nThe morning air was crisp...\"}\n"
        "  - WRONG: {\"op_type\": \"insert_after_heading\", \"anchor_text\": \"## Chapter 3\", \"text\": \"## Chapter 3\\n\\nThe morning air...\"}\n\n"
        "**DECISION RULE:**\n"
        "- Editing EXISTING text? → Use 'replace_range' with 'original_text' (MANDATORY)\n"
        "- Adding NEW chapter? → Use 'insert_after_heading' with last line of previous chapter as anchor (MANDATORY)\n"
        "- Continuing EXISTING chapter? → Use 'insert_after' with last words of that chapter as anchor (MANDATORY)\n"
        "- **NEVER** use 'replace_range' without 'original_text' - it will fail!\n"
        "- **NEVER** use 'insert_after_heading' or 'insert_after' without 'anchor_text' - it will fail!\n"
        "- **NEVER** use 'insert_after_heading' to continue existing chapters - use 'insert_after' instead!\n\n"
    )


def build_rules_document_guidance() -> str:
    """
    Domain-specific guidance for rules documents.
    
    Provides semantic section mapping and automatic placement logic that the old
    rules agent had. This enables LLMs to understand where content belongs based
    on keywords and document structure.
    
    Returns:
        Formatted rules document guidance string
    """
    return (
        "**RULES DOCUMENT STRUCTURE AWARENESS:**\n"
        "This is a worldbuilding rules document with specific sections:\n"
        "- ## Background - World history and context\n"
        "- ## Universe Constraints - Physical/magical/technological laws\n"
        "  - ### Transformation & Physiology - Biological processes, transformations\n"
        "  - ### Sustenance & Biological Constraints - Blood, feeding, survival needs\n"
        "- ## Systems - Magic, technology, economy\n"
        "  - ### Magic or Technology Systems - How magic/tech works\n"
        "  - ### Resource & Economy Constraints - Economic rules, resource limits\n"
        "- ## Social Structures & Culture\n"
        "  - ### Institutions & Power Dynamics ← Family structures, hierarchies, governance, bride systems\n"
        "- ## Geography & Environment - World geography, climate, locations\n"
        "- ## Religion & Philosophy - Belief systems, philosophical frameworks\n"
        "- ## Timeline & Continuity - Chronology and no-retcon rules\n"
        "  - ### Chronology (canonical) - Timeline of events\n"
        "  - ### Continuity Rules (no-retcon constraints) - What cannot be changed\n"
        "- ## Series Synopsis - Book-by-book summaries\n"
        "  - ### Book 1, ### Book 2, etc.\n"
        "- ## Character References - Cast integration\n"
        "  - ### Cast Integration & Constraints - Character rules and limits\n\n"
        "**SEMANTIC SECTION MAPPING (AUTOMATIC PLACEMENT):**\n"
        "When the user provides content, identify which section it belongs in semantically:\n"
        "- Family/bride/hierarchy/governance rules → ### Institutions & Power Dynamics (under ## Social Structures & Culture)\n"
        "- Magic/technology system rules → ### Magic or Technology Systems (under ## Systems)\n"
        "- Economic/resource rules → ### Resource & Economy Constraints (under ## Systems)\n"
        "- Transformation/biological process rules → ### Transformation & Physiology (under ## Universe Constraints)\n"
        "- Blood/feeding/survival rules → ### Sustenance & Biological Constraints (under ## Universe Constraints)\n"
        "- Geography/location/climate rules → ## Geography & Environment\n"
        "- Timeline events/chronology → ## Timeline & Continuity or ### Chronology (canonical)\n"
        "- Character constraints → ## Character References or ### Cast Integration & Constraints\n"
        "- Series/book summaries → ## Series Synopsis with appropriate ### Book N subsection\n\n"
        "**CRITICAL: NEVER USE insert_after FOR BULLET LISTS (ABSOLUTE RULE):**\n"
        "When adding content to sections with bullet lists, you MUST use replace_range, NOT insert_after!\n\n"
        "**WHY THIS RULE EXISTS:**\n"
        "- insert_after uses an anchor from the document\n"
        "- If you use the last line of a bullet as your anchor, but there are MORE bullets after it, the new content gets inserted IN THE MIDDLE\n"
        "- Example PROBLEM:\n"
        "  * Section has bullets A, B, C\n"
        "  * You use insert_after with anchor=\"second line of bullet B\"\n"
        "  * Result: A, B, [NEW CONTENT INSERTED HERE], C ← New content splits the list!\n"
        "- This creates fragmented, disjointed sections that break document flow\n\n"
        "**SURGICAL APPROACH FOR ADDING TO BULLET LISTS (BE PRECISE!):**\n"
        "Use replace_range to replace ONLY THE LAST BULLET with an expanded version:\n"
        "1. **Find the target section** - Identify which section the new content belongs in (use semantic mapping above)\n"
        "2. **Find the LAST bullet** - Identify ONLY the final bullet point in that section\n"
        "3. **Use replace_range with original_text = ONLY THE LAST BULLET** - Match ONLY the last bullet (typically 1-2 lines)\n"
        "4. **Set text = Last bullet + new bullets** - Include the existing last bullet PLUS your new bullets\n"
        "5. **Result**: Surgical addition at the end - NO middle-insertion, NO deleting unrelated content!\n\n"
        "⚠️ **CRITICAL WARNING - KEEP IT SURGICAL:**\n"
        "- **WRONG**: original_text with 10+ bullets, 500+ chars → deletes too much!\n"
        "- **RIGHT**: original_text with 1-2 bullets max (just the last one or last two)\n"
        "- **Only include what you need to modify** - don't grab extra bullets!\n\n"
        "**EXAMPLE - CORRECT APPROACH:**\n"
        "Section currently has:\n"
        "```\n"
        "### Transformation & Physiology\n"
        "- Vampires can transform at will.\n"
        "- The transformation is instantaneous.\n"
        "- Human appearance requires continuous effort.\n"
        "```\n"
        "\n"
        "User wants to add: \"Clothing is absorbed during transformation\"\n"
        "\n"
        "✅ CORRECT Operation (SURGICAL - only last bullet):\n"
        "```json\n"
        "{\n"
        '  "op_type": "replace_range",\n'
        '  "original_text": "- Human appearance requires continuous effort.",\n'
        '  "text": "- Human appearance requires continuous effort.\\n- Clothing is absorbed during transformation."\n'
        "}\n"
        "```\n"
        "Result: Surgical replacement - ONLY last bullet modified, first two bullets untouched!\n\n"
        "❌ WRONG Operation (DO NOT DO THIS!):\n"
        "```json\n"
        "{\n"
        '  "op_type": "insert_after",\n'
        '  "anchor_text": "Human appearance requires continuous effort.",\n'
        '  "text": "\\n- Clothing is absorbed during transformation."\n'
        "}\n"
        "```\n"
        "Problem: If there are more bullets after \"Human appearance requires continuous effort\", the new bullet gets inserted IN THE MIDDLE!\n\n"
        "**WHEN TO USE insert_after (NARROW USE CASE):**\n"
        "- **ONLY** for continuing narrative paragraphs or prose (not for bullet lists!)\n"
        "- Example: Continuing a multi-sentence explanation or description\n"
        "- Example: Adding to the end of a narrative paragraph\n"
        "- **NEVER** for adding bullet points to bullet lists (use replace_range instead!)\n\n"
        "**FOR EMPTY SECTIONS:**\n"
        "- Use insert_after_heading when section is COMPLETELY empty (no bullets, no text)\n"
        "- anchor_text = exact section header (e.g., \"### Transformation & Physiology\")\n"
        "- This is safe because there's no content to split\n\n"
        "**FOR REPLACING SPECIFIC BULLETS:**\n"
        "- Use replace_range with original_text = ONLY the bullet(s) being changed\n"
        "- Example: Replacing bullet #2 in a 5-bullet list:\n"
        "  * original_text = \"- Second bullet text here\" (ONLY bullet #2)\n"
        "  * text = \"- Updated second bullet text\" (replacement)\n"
        "  * Result: Bullets 1, 3, 4, 5 unchanged, only bullet #2 updated\n\n"
        "**FOR DELETING SPECIFIC BULLETS:**\n"
        "- Use replace_range with original_text = bullet to delete, text = \"\" (empty string)\n"
        "- OR use delete_range with original_text = bullet to delete\n"
        "- Example: Removing bullet #3 from a 5-bullet list:\n"
        "  * original_text = \"- Third bullet text here\" (ONLY bullet #3)\n"
        "  * text = \"\" (delete it)\n"
        "  * Result: Bullets 1, 2, 4, 5 remain, bullet #3 removed cleanly\n\n"
    )


def build_style_document_guidance() -> str:
    """
    Domain-specific guidance for style guide documents.
    
    Provides style section awareness and cross-referencing patterns that help
    LLMs understand how style guidelines relate to each other.
    
    Returns:
        Formatted style document guidance string
    """
    return (
        "**STYLE DOCUMENT STRUCTURE AWARENESS:**\n"
        "This is a style guide document with specific sections for narrative technique:\n"
        "- ## Narrative Voice - Tone, personality, voice characteristics\n"
        "- ## Point of View - POV type and technique (first-person, third-person limited, omniscient, etc.)\n"
        "- ## Tense - Tense usage and patterns (past, present, mixed)\n"
        "- ## Pacing - Pacing characteristics and techniques\n"
        "- ## Sensory Detail Level - Level of sensory description (minimal, moderate, rich)\n"
        "- ## Sentence Structure - Sentence length patterns, rhythm, flow\n"
        "- ## Descriptive Techniques - Show-don't-tell approach, metaphor usage, imagery style\n"
        "- ## Rhythm and Flow - Paragraph structure, pacing techniques, transitions\n"
        "- ## Notes - Additional style considerations, techniques, or guidelines\n\n"
        "**SECTION RELATIONSHIPS (CROSS-REFERENCING):**\n"
        "When adding or updating style guidelines, related sections may need updates:\n"
        "- **POV guidelines** → May affect both ## Point of View AND ## Narrative Voice sections\n"
        "  - If adding POV rules, check if Narrative Voice section references POV and update both\n"
        "- **Pacing techniques** → May affect both ## Pacing AND ## Rhythm and Flow sections\n"
        "  - If adding pacing rules, check if Rhythm and Flow section overlaps and update both\n"
        "- **Descriptive techniques** → May affect both ## Descriptive Techniques AND ## Sentence Structure sections\n"
        "  - If adding descriptive rules, check if Sentence Structure section relates and update both\n"
        "- **Sensory detail** → May affect ## Sensory Detail Level AND ## Descriptive Techniques\n"
        "  - If adding sensory rules, check if Descriptive Techniques section references sensory detail\n\n"
        "**CRITICAL: CROSS-REFERENCE RELATED SECTIONS**\n"
        "Before generating operations:\n"
        "1. **SCAN THE ENTIRE DOCUMENT** - Read through ALL sections to identify related style guidelines\n"
        "2. **IDENTIFY ALL AFFECTED SECTIONS** - When adding/updating a style guideline, find ALL places it should appear\n"
        "3. **GENERATE MULTIPLE OPERATIONS** - If a guideline affects multiple sections, create operations for EACH affected section\n"
        "4. **ENSURE CONSISTENCY** - Related sections must be updated together to maintain style guide coherence\n"
        "\n"
        "**STYLE GUIDE SCOPE (HOW vs WHAT):**\n"
        "- Style guides document HOW to write, not WHAT happens\n"
        "- Focus on narrative technique, voice, tone, pacing, structure\n"
        "- **DO NOT include**:\n"
        "  - Dialogue style (that's character-specific, belongs in character profiles)\n"
        "  - Plot elements or story content\n"
        "  - Character-specific voice patterns\n"
        "- **DO include**:\n"
        "  - Narrative voice and tone (formal, casual, poetic, gritty, etc.)\n"
        "  - POV technique and boundaries\n"
        "  - Tense usage patterns\n"
        "  - Pacing techniques and transitions\n"
        "  - Sensory detail level and approach\n"
        "  - Sentence structure patterns\n"
        "  - Descriptive techniques (show-don't-tell, metaphor, imagery)\n"
        "  - Paragraph structure and rhythm\n\n"
        "**SECTION-SPECIFIC PLACEMENT:**\n"
        "- When adding to a section, find the LAST bullet/guideline in that section\n"
        "- Use insert_after with LAST LINE of LAST bullet as anchor_text\n"
        "- Verify no more content exists after your anchor (next content should be section header or end of file)\n"
        "- For empty sections, use insert_after_heading with section header as anchor_text\n\n"
    )


def build_outline_document_guidance() -> str:
    """
    Domain-specific guidance for outline documents.
    
    Provides chapter structure awareness, beat limit enforcement, and dialogue
    restrictions that are critical for outline editing.
    
    Returns:
        Formatted outline document guidance string
    """
    return (
        "**OUTLINE DOCUMENT STRUCTURE AWARENESS:**\n"
        "This is a story outline document with a specific hierarchical structure:\n"
        "- # Overall Synopsis - High-level story summary (major elements only)\n"
        "- # Notes - Rules, themes, worldbuilding notes\n"
        "- # Characters - BRIEF list only (name + role like 'Protagonist: John', 'Antagonist: Sarah')\n"
        "  **CRITICAL**: Character DETAILS belong in character profile files, NOT in the outline!\n"
        "- ## Chapter N - Chapter headings (EXACTLY '## Chapter N' with no titles!)\n"
        "  - ### Status (optional, AT BEGINNING of chapter, after ## Chapter N)\n"
        "    * State of tracked items AT THE BEGINNING of the chapter (from previous chapter's events)\n"
        "  - ### Pacing (optional, AFTER Status, BEFORE Summary)\n"
        "    * FROM: [Previous chapter's emotional state]\n"
        "    * TO: [Current chapter's target state]\n"
        "    * TECHNIQUE: [How to achieve the transition]\n"
        "  - ### Summary - BRIEF 3-5 sentence overview (NOT detailed synopsis!)\n"
        "  - ### Beats - Detailed bullet point events (MAXIMUM 100 beats per chapter!)\n"
        "    * Each beat MUST start with '- ' (dash space) - NO numbering (1., 2., 3., etc.)\n"
        "    * **NEVER number beats** - Use bullet points only, never numbered lists\n\n"
        "**CRITICAL - CHAPTER HEADING FORMAT**:\n"
        "- Chapter headings MUST be EXACTLY \"## Chapter N\" where N is the chapter number\n"
        "- **NEVER add titles, names, or descriptions to chapter headings**\n"
        "- **CORRECT**: \"## Chapter 1\", \"## Chapter 5\", \"## Chapter 12\"\n"
        "- **WRONG**: \"## Chapter 1: The Beginning\", \"## Chapter 5 - The Confrontation\"\n"
        "- Chapter headings are ONLY for numbering - no additional text after the number\n\n"
        "**100-BEAT LIMIT ENFORCEMENT (CRITICAL):**\n"
        "- Each chapter MUST have a MAXIMUM of 100 beats\n"
        "- When adding new beats to a chapter that already has beats:\n"
        "  1. **Count existing beats first** - Count all lines starting with '- ' in the ### Beats section\n"
        "  2. **Check if adding would exceed 100** - If (existing beats + new beats) > 100, you MUST prune\n"
        "  3. **Prune less important beats** - Remove redundant or less essential beats to make room\n"
        "  4. **Prioritize plot-critical beats** - Keep important plot events, remove minor details\n"
        "  5. **Example**: Chapter has 48 beats, user wants to add 5 new beats → Remove 3 least important existing beats\n"
        "- Beat count metadata is shown in context: \"(METADATA: Beat count: {count}/100)\"\n"
        "- Warnings appear at 95+ beats, hard limit at 100\n\n"
        "**STATUS BLOCK PLACEMENT (CRITICAL):**\n"
        "- Status blocks appear IMMEDIATELY after the chapter heading (## Chapter N), BEFORE summary\n"
        "- Status represents state AT THE BEGINNING of the chapter (from previous chapter's events)\n"
        "- Format: ### Status followed by bullets with tracked items\n"
        "- Include tracked items that are: (1) relevant, (2) changed, or (3) should be carried forward\n"
        "- **CARRY FORWARD**: If item was in previous chapter's status and still active, include it even if unchanged\n"
        "- Chapter 1 may have status (optional) for series continuity\n\n"
        "**PACING BLOCK PLACEMENT (CRITICAL):**\n"
        "- Pacing blocks appear AFTER Status (if present), BEFORE Summary\n"
        "- Pacing guides transition FROM previous chapter's emotional state TO current chapter's target state\n"
        "- Format: ### Pacing with FROM/TO/TECHNIQUE structure\n"
        "- FROM state for Chapter N should match TO state from Chapter N-1 (ensures continuity)\n"
        "- Optional for Chapter 1, recommended for Chapter 2+\n\n"
        "**DIALOGUE RESTRICTIONS (ABSOLUTE PROHIBITION):**\n"
        "- **NEVER include actual dialogue** (quoted speech) in outline beats\n"
        "- Dialogue belongs in fiction manuscript, NOT outline\n"
        "- **CAN mention talking as event**: \"- Character discusses plan with ally\"\n"
        "- **CAN describe what is discussed**: \"- Character reveals secret during conversation\"\n"
        "- **CANNOT include**: \"- Character says 'I have a secret'\"\n"
        "- Beats are plot events, not prose\n\n"
        "**SUMMARY REQUIREMENTS (CRITICAL):**\n"
        "- Each chapter MUST have a '### Summary' header followed by a BRIEF, HIGH-LEVEL summary paragraph\n"
        "- Summary should be 3-5 sentences MAXIMUM\n"
        "- Think of it as a \"back of the book\" description for this chapter\n"
        "- DO NOT write lengthy, detailed synopses - keep summaries concise and focused\n"
        "- The summary captures the ESSENCE of the chapter, not every plot detail (details go in beats)\n\n"
        "**BEAT PLACEMENT STRATEGY:**\n"
        "When adding beats to an existing chapter:\n"
        "1. **Find the chapter** - Locate ## Chapter N heading\n"
        "2. **Find ### Beats section** - Locate the beats section within that chapter\n"
        "3. **Count existing beats** - Count all lines starting with '- ' in the beats section\n"
        "4. **Check 100-beat limit** - If adding would exceed 100, prune less important beats first\n"
        "5. **Find LAST beat** - Scan to find the LAST bullet point in the ### Beats section\n"
        "6. **Use LAST LINE of LAST BEAT as anchor** - For multi-line beats, use the LAST LINE\n"
        "7. **Verify no more beats exist** - Confirm next content is ### Summary, ### Pacing, ### Status, ## Chapter N+1, or end of file\n"
        "8. **Use insert_after with anchor_text** - Set anchor_text to LAST LINE of LAST BEAT\n"
        "\n"
        "**CRITICAL FOR BEAT PLACEMENT:**\n"
        "- **NEVER use a middle beat as anchor** - This will INSERT IN THE MIDDLE, splitting the beat list!\n"
        "- **ALWAYS scan the ENTIRE ### Beats section** - Don't assume the first beat you see is the last one\n"
        "- **For multi-line beats**: The LAST LINE of the LAST BEAT is your anchor\n"
        "- **Verify placement**: After finding your anchor, check that the next content is a section header (###) or next chapter (##)\n\n"
    )


def build_editor_operation_instructions(
    document_type: str = "document",
    include_chapter_guidance: bool = False,
    custom_rules: Optional[List[str]] = None,
    include_domain_guidance: bool = True
) -> str:
    """
    Build comprehensive editor operation instructions with domain-specific customization.
    
    This is the main function for assembling complete editor operation guidance.
    It combines universal safety rules with optional domain-specific components.
    
    Args:
        document_type: Type of document being edited (e.g., "rules", "outline", "style", "fiction manuscript")
        include_chapter_guidance: If True, includes chapter continuation guidance (for fiction agents)
        custom_rules: Optional list of domain-specific rules to append
        include_domain_guidance: If True, includes domain-specific document structure guidance (default: True)
        
    Returns:
        Complete editor operation instruction string ready for inclusion in prompts
        
    Example:
        # For rules agent
        instructions = build_editor_operation_instructions(
            document_type="rules",
            include_chapter_guidance=False,
            custom_rules=[
                "Prefer bullet points over paragraphs for rules",
                "Cross-reference related rules across sections"
            ]
        )
        
        # For fiction agent
        instructions = build_editor_operation_instructions(
            document_type="fiction manuscript",
            include_chapter_guidance=True,
            custom_rules=[
                "Copy EXACT text from MANUSCRIPT sections, not OUTLINE"
            ]
        )
    """
    instructions = []
    
    # Core safety rules (universal)
    instructions.append(build_editor_safety_rules())
    
    # Domain-specific guidance (NEW)
    if include_domain_guidance:
        if document_type == "rules":
            instructions.append(build_rules_document_guidance())
        elif document_type == "style":
            instructions.append(build_style_document_guidance())
        elif document_type == "outline":
            instructions.append(build_outline_document_guidance())
    
    # Operation type guidance
    instructions.append(
        "**OPERATION TYPES (ALL REQUIRE TEXT ANCHORS - OPERATIONS FAIL WITHOUT THEM!):**\n"
        "- **replace_range**: Replace existing text (PRIMARY operation for bullet lists!)\n"
        "  * **MANDATORY**: 'original_text' with EXACT, VERBATIM text from file\n"
        "  * Without original_text: Operation will FAIL or stomp wrong content\n"
        "  * **USE THIS for adding to bullet lists**: Replace ONLY THE LAST BULLET with expanded version (surgical!)\n"
        "  * **USE THIS for updating specific bullets**: original_text = bullet to change, text = updated bullet\n"
        "  * **USE THIS for replacing any existing content**: Guarantees clean edits, no middle-insertion issues\n"
        "- **delete_range**: Remove text\n"
        "  * **MANDATORY**: 'original_text' with EXACT, VERBATIM text to delete\n"
        "  * Without original_text: Operation will FAIL or delete wrong content\n"
        "  * NEVER use start/end indices alone!\n"
        "  * Can also use replace_range with text=\"\" (empty string) to delete\n"
        "- **insert_after_heading**: Add content after a section header (ONLY for COMPLETELY empty sections!)\n"
        "  * **MANDATORY**: 'anchor_text' = exact section header (e.g., '### Magic Systems')\n"
        "  * Without anchor_text: Operation will FAIL to find insertion point\n"
        "  * **CRITICAL**: ONLY use when section is COMPLETELY EMPTY (no bullets, no text)\n"
        "  * If section has ANY content (even a single bullet), use replace_range instead!\n"
        "  * WARNING: Using this on non-empty section will split the section and create duplicates!\n"
        "- **insert_after**: Continue narrative paragraphs or prose (FORBIDDEN for bullet lists!)\n"
        "  * **MANDATORY**: 'anchor_text' = last few words before insertion point (minimum 10-20 words)\n"
        "  * Without anchor_text: Operation will FAIL to find insertion point\n"
        "  * **ABSOLUTELY FORBIDDEN**: NEVER use insert_after for adding bullets to bullet lists!\n"
        "  * **ONLY USE FOR**: Continuing narrative paragraphs, prose descriptions, multi-sentence explanations\n"
        "  * **NEVER USE FOR**: Adding bullet points, expanding bullet lists, adding to sections with bullets\n"
        "  * **WHY FORBIDDEN**: insert_after with bullet anchor will INSERT IN THE MIDDLE if you pick the wrong bullet!\n"
        "  * **FOR BULLET LISTS**: Use replace_range instead (replace ONLY LAST BULLET with expanded version - be surgical!)\n\n"
        "**CRITICAL - HOW OPERATIONS WORK**:\n"
        "- Start/end indices are APPROXIMATE guidance only (used to narrow search window)\n"
        "- Text anchors ('original_text' or 'anchor_text') are the PRIMARY mechanism for finding content\n"
        "- Operations without text anchors will FAIL or use unreliable index-based fallback\n"
        "- Index-based fallback has low confidence and may stomp wrong content!\n\n"
        "**DECISION TREE FOR RULES DOCUMENT EDITING (FOLLOW THIS EXACTLY!):**\n\n"
        "STEP 1: Identify the target section where content should go (use semantic mapping)\n\n"
        "STEP 2: Check if section has existing bullet points\n"
        "- Read the section from header to next header\n"
        "- Count bullet points (lines starting with '- ')\n\n"
        "STEP 3: Choose operation based on section state:\n\n"
        "**SCENARIO A: Adding new bullet(s) to section with existing bullets**\n"
        "→ **ALWAYS use replace_range with ONLY THE LAST BULLET** (NEVER use insert_after!)\n"
        "   • Find the LAST bullet in that section (just the final one)\n"
        "   • Copy ONLY the last bullet into original_text: \"- Bullet C\" (typically 1-2 lines)\n"
        "   • Create expanded version in text: \"- Bullet C\\n- NEW Bullet D\"\n"
        "   • Result: ✅ Surgical replacement - Bullets A, B untouched, only C→C+D modified\n"
        "   • ⚠️ CRITICAL: Do NOT include 5, 10, or more bullets in original_text - too broad!\n"
        "   • ⚠️ Only include what you need to modify (last bullet) - keeps it surgical and safe\n\n"
        "**SCENARIO B: Section is COMPLETELY EMPTY (no bullets, no text)**\n"
        "→ Use insert_after_heading\n"
        "   • anchor_text = \"### Section Header\" (exact header)\n"
        "   • text = \"- First bullet\\n- Second bullet\"\n"
        "   • Result: ✅ Bullets added below empty header\n\n"
        "**SCENARIO C: Updating specific bullet(s) within a list**\n"
        "→ Use replace_range with ONLY the bullet(s) being changed\n"
        "   • original_text = \"- Old bullet text\" (ONLY the bullet being changed)\n"
        "   • text = \"- New bullet text\" (replacement)\n"
        "   • Result: ✅ Only targeted bullet changes, rest unchanged\n\n"
        "**SCENARIO D: Deleting specific bullet(s)**\n"
        "→ Use replace_range with text=\"\" (empty string)\n"
        "   • original_text = \"- Bullet to delete\" (ONLY the bullet being deleted)\n"
        "   • text = \"\" (delete it)\n"
        "   • Result: ✅ Targeted bullet removed cleanly\n\n"
        "**SCENARIO E: Continuing a narrative paragraph (NOT a bullet list!)**\n"
        "→ Use insert_after (ONLY for prose paragraphs, NOT bullets!)\n"
        "   • anchor_text = last 10-20 words of the paragraph\n"
        "   • text = continuation text\n"
        "   • Result: ✅ Prose continues smoothly\n\n"
        "**IRON-CLAD RULE: If section has bullet points → Use replace_range, NEVER insert_after!**\n\n"
    )
    
    # Chapter-specific guidance for fiction
    if include_chapter_guidance:
        instructions.append(build_chapter_continuation_guidance())
    
    # Verification checklist
    instructions.append(build_editor_verification_checklist())
    
    # Common mistakes
    instructions.append(build_editor_common_mistakes())
    
    # Custom domain rules
    if custom_rules:
        instructions.append(f"**DOMAIN-SPECIFIC RULES ({document_type.upper()}):**\n")
        for rule in custom_rules:
            instructions.append(f"- {rule}\n")
        instructions.append("\n")
    
    return "".join(instructions)
