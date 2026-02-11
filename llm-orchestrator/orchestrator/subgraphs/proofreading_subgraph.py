"""
Proofreading Subgraph

Reusable subgraph for proofreading workflows. Standalone proofreading_agent removed;
this subgraph is the canonical implementation, integrated into:
- Fiction Editing Agent
- Article Writing (subgraph)
- Podcast Script Agent

Handles:
- Mode inference (clarity/compliance/accuracy)
- Style guide loading (fiction only - loads from frontmatter "style" reference)
- Content scoping (chapter for fiction, paragraph/segment for non-fiction)
- LLM proofreading analysis
- Operation resolution

FOCUSED SCOPE:
- Fiction: Style guide + current chapter only (focused, no outline/rules/characters)
- Non-fiction: Current segment/paragraph only (no style guide, no other references)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Callable

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from orchestrator.models.editor_models import ManuscriptEdit
from orchestrator.utils.editor_operation_resolver import resolve_editor_operation
from orchestrator.utils.frontmatter_utils import strip_frontmatter_block
from orchestrator.subgraphs.fiction_context_subgraph import (
    find_chapter_ranges,
    locate_chapter_index,
    ChapterRange
)

logger = logging.getLogger(__name__)


# ============================================
# Utility Functions
# ============================================

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


def infer_mode_from_request(state: Dict[str, Any]) -> str:
    """Infer proofreading mode from user request"""
    try:
        latest = ""
        messages = state.get("messages", [])
        query = state.get("query", "")
        
        # Try to get latest user message
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                latest = (msg.get("content") or "").lower()
                break
            if hasattr(msg, "type") and msg.type == "human":
                latest = (msg.content or "").lower()
                break
        
        # Fallback to query
        if not latest:
            latest = query.lower()
        
        if "style" in latest or "compliance" in latest:
            return "compliance"
        if "accuracy" in latest or "fact" in latest or "fact-check" in latest or "fact check" in latest:
            return "accuracy"
        return "clarity"
    except Exception:
        return "clarity"


# ============================================
# Node Functions
# ============================================

async def infer_mode_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Infer proofreading mode from user request and set in state"""
    mode = infer_mode_from_request(state)
    logger.info(f"Inferred proofreading mode: {mode}")
    return {
        "mode": mode,
            # ✅ CRITICAL: Preserve all critical state keys
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # ✅ CRITICAL: Preserve manuscript content for scope determination
            "manuscript": state.get("manuscript", ""),
            "manuscript_content": state.get("manuscript_content", ""),
            "active_editor": state.get("active_editor", {}),
            "cursor_offset": state.get("cursor_offset", -1),
            "chapter_ranges": state.get("chapter_ranges", [])
        }


async def load_style_guide_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load referenced style guide from frontmatter.
    
    FOCUSED: Only loads style guide (no outline, rules, characters, etc.)
    - Fiction documents: Loads style guide if frontmatter has "style" reference
    - Non-fiction documents: No style guide available (returns None)
    """
    try:
        logger.info("Loading style guide...")
        
        active_editor = state.get("active_editor", {}) or {}
        shared_memory = state.get("shared_memory", {}) or {}
        
        # Get active_editor from shared_memory if not in state
        if not active_editor:
            active_editor = shared_memory.get("active_editor", {}) or {}
        
        style_text = None
        
        # Load style guide using reference file loader
        # FOCUSED: Only loads "style" reference, nothing else
        try:
            from orchestrator.tools.reference_file_loader import load_referenced_files
            
            user_id = state.get("user_id", "system")
            reference_config = {
                "style": ["style"]  # Only style guide, no other references
            }
            
            result = await load_referenced_files(
                active_editor=active_editor,
                user_id=user_id,
                reference_config=reference_config,
                doc_type_filter=None  # Load for any document type (fiction will have style, non-fiction won't)
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
            "style_guide_content": style_text,
            # ✅ CRITICAL: Preserve all critical state keys
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # ✅ CRITICAL: Preserve manuscript content for scope determination
            "manuscript": state.get("manuscript", ""),
            "manuscript_content": state.get("manuscript_content", ""),
            "active_editor": state.get("active_editor", {}),
            "cursor_offset": state.get("cursor_offset", -1),
            "chapter_ranges": state.get("chapter_ranges", [])
        }
        
    except Exception as e:
        logger.error(f"Error loading style guide: {e}")
        return {
            "style_guide_content": None,
            # ✅ CRITICAL: Preserve all critical state keys
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # ✅ CRITICAL: Preserve manuscript content
            "manuscript": state.get("manuscript", ""),
            "manuscript_content": state.get("manuscript_content", ""),
            "active_editor": state.get("active_editor", {}),
            "cursor_offset": state.get("cursor_offset", -1),
            "chapter_ranges": state.get("chapter_ranges", [])
        }


async def determine_scope_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Determine content scope for proofreading.
    
    FOCUSED SCOPING:
    - Fiction: Current chapter around cursor (if >= 7500 words), or full doc if < 7500 words
    - Non-fiction: Current paragraph/segment around cursor, or full doc if < 7500 words
    - No outline, rules, or character references - just the focused text to proofread
    """
    try:
        logger.info("Determining proofreading scope...")
        
        # Get manuscript from state (set by context subgraph or parent agent)
        # Try multiple sources: manuscript_content, manuscript, editor_content, or active_editor
        manuscript = state.get("manuscript_content") or state.get("manuscript", "")
        
        # If not found, try editor_content (used by article/podcast agents)
        if not manuscript:
            manuscript = state.get("editor_content", "")
        
        # If still not found, try active_editor from state or shared_memory
        if not manuscript:
            active_editor = state.get("active_editor", {}) or {}
            if not active_editor:
                shared_memory = state.get("shared_memory", {}) or {}
                active_editor = shared_memory.get("active_editor", {}) or {}
            manuscript = active_editor.get("content", "")
        
        # Log for debugging and validate manuscript
        if not manuscript:
            logger.warning("⚠️ No manuscript found in state - checking all sources")
            logger.warning(f"  manuscript_content: {len(state.get('manuscript_content', ''))} chars")
            logger.warning(f"  manuscript: {len(state.get('manuscript', ''))} chars")
            logger.warning(f"  editor_content: {len(state.get('editor_content', ''))} chars")
            active_editor_debug = state.get("active_editor", {}) or {}
            if not active_editor_debug:
                shared_memory = state.get("shared_memory", {}) or {}
                active_editor_debug = shared_memory.get("active_editor", {}) or {}
            logger.warning(f"  active_editor content: {len(active_editor_debug.get('content', ''))} chars")
            logger.error("❌ No manuscript content found for proofreading scope determination!")
            return {
                "scope_text": "",
                "error": "No manuscript content available",
                "task_status": "error",
                # ✅ CRITICAL: Preserve all critical state keys
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "manuscript": "",
                "manuscript_content": "",
                "active_editor": state.get("active_editor", {}),
                "cursor_offset": state.get("cursor_offset", -1),
                "chapter_ranges": state.get("chapter_ranges", [])
            }
        else:
            logger.info(f"✅ Found manuscript: {len(manuscript)} characters")
        
        cursor_offset = state.get("cursor_offset", -1)
        
        # Get chapter ranges from context subgraph (if available)
        chapter_ranges = state.get("chapter_ranges", [])
        
        body = strip_frontmatter_block(manuscript)
        word_count = len((body or "").split())
        scope_text = body
        
        logger.info(f"Manuscript body after frontmatter strip: {len(body)} characters, {word_count} words")
        
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
            "scope_text": scope_text,
            # ✅ CRITICAL: Preserve all critical state keys
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # ✅ CRITICAL: Preserve manuscript content
            "manuscript": state.get("manuscript", ""),
            "manuscript_content": state.get("manuscript_content", ""),
            "active_editor": state.get("active_editor", {}),
            "cursor_offset": state.get("cursor_offset", -1),
            "chapter_ranges": state.get("chapter_ranges", [])
        }
        
    except Exception as e:
        logger.error(f"Error determining scope: {e}")
        # Fallback to full document
        manuscript = state.get("manuscript_content") or state.get("manuscript", "")
        if not manuscript:
            active_editor = state.get("active_editor", {}) or {}
            if not active_editor:
                shared_memory = state.get("shared_memory", {}) or {}
                active_editor = shared_memory.get("active_editor", {}) or {}
            manuscript = active_editor.get("content", "")
        body = strip_frontmatter_block(manuscript)
        return {
            "scope_text": body,
            # ✅ CRITICAL: Preserve all critical state keys
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # ✅ CRITICAL: Preserve manuscript content
            "manuscript": state.get("manuscript", ""),
            "manuscript_content": state.get("manuscript_content", ""),
            "active_editor": state.get("active_editor", {}),
            "cursor_offset": state.get("cursor_offset", -1),
            "chapter_ranges": state.get("chapter_ranges", [])
        }


async def proofread_content_node(
    state: Dict[str, Any],
    llm_factory: Callable,
    get_datetime_context: Callable
) -> Dict[str, Any]:
    """Execute proofreading analysis using LLM"""
    try:
        logger.info("Executing proofreading analysis...")
        
        scope_text = state.get("scope_text", "")
        style_guide = state.get("style_guide_content")
        mode = state.get("mode", "clarity")
        
        active_editor = state.get("active_editor", {}) or {}
        shared_memory = state.get("shared_memory", {}) or {}
        
        # Get active_editor from shared_memory if not in state
        if not active_editor:
            active_editor = shared_memory.get("active_editor", {}) or {}
        
        filename = active_editor.get("filename") or "document.md"
        frontmatter = active_editor.get("frontmatter", {}) or {}
        doc_type = str((frontmatter.get("type") or "")).strip().lower()
        
        # Build system prompt
        system_prompt = (
            "You are a MASTER COPY EDITOR and PROFESSIONAL PROOFREADER. Persona disabled.\n\n"
            "=== PROOFREADING DIRECTIVES ===\n"
            f"MODE: {mode} (clarity|compliance|accuracy)\n"
            "- clarity: Focus on grammar, flow, and readability.\n"
            "- compliance: Focus on author's style guide and consistency.\n"
            "- accuracy: Focus on technical precision and fact-checking.\n\n"
            "- Identify and correct errors while maintaining the author's voice.\n"
            "- Respect intentional choices and genre conventions.\n"
            "- Focus on actual errors; maintain voice.\n"
            "- Use complete, verbatim original_text with natural boundaries.\n\n"
            "=== CRITICAL TEXT PRECISION REQUIREMENTS ===\n"
            "For 'original_text' and 'anchor_text' fields:\n"
            "- Must be EXACT, COMPLETE, and VERBATIM from the current manuscript text.\n"
            "- Include ALL whitespace, line breaks, and formatting exactly as written.\n"
            "- Include complete sentences or natural text boundaries (periods, paragraph breaks).\n"
            "- NEVER paraphrase, summarize, or reformat the text.\n"
            "- Minimum 10-20 words for unique identification.\n"
            "- NEVER include chapter headers (##) in original_text for replace_range!\n\n"
            "=== GRANULAR EDIT PRINCIPLES ===\n"
            "1. **Minimize 'original_text' size**: Use the SMALLEST unique text match possible.\n"
            "2. **Preserve surrounding text**: Only change what needs changing.\n"
            "3. **Break large edits into multiple operations**: One operation per sentence/phrase.\n\n"
            "STRUCTURED OUTPUT REQUIRED: Respond ONLY with valid JSON matching ManuscriptEdit schema.\n"
        )
        
        style_block = (
            "=== AUTHOR'S STYLE GUIDE ===\n"
            "The following style guide defines the author's preferred conventions.\n"
            "CRITICAL: When style conflicts with general grammar, FOLLOW THE STYLE GUIDE.\n\n"
            f"{style_guide}\n"
        ) if style_guide else ""
        
        required_schema_hint = (
            "REQUIRED JSON STRUCTURE:\n"
            "{\n"
            f'  "target_filename": "{filename}",\n'
            '  "scope": "paragraph",\n'
            '  "summary": "Summary of proofreading findings and mode applied",\n'
            '  "safety": "low",\n'
            '  "operations": [\n'
            '    {\n'
            '      "op_type": "replace_range",\n'
            '      "original_text": "EXACT sentence or phrase to correct",\n'
            '      "text": "Corrected version of the text",\n'
            '      "note": "Short explanation for this specific correction"\n'
            '    }\n'
            '  ]\n'
            "}\n"
        )
        
        user_prompt = (
            "=== DOCUMENT METADATA ===\n"
            f"Filename: {filename}\nType: {doc_type}\n\n"
            + (style_block if style_block else "")
            + "=== SCOPE TEXT (frontmatter stripped) ===\n"
            f"{scope_text}\n\n"
            + required_schema_hint
        )
        
        # Call LLM using factory
        datetime_context = get_datetime_context()
        messages = [
            SystemMessage(content=system_prompt),
            SystemMessage(content=datetime_context),
            HumanMessage(content=user_prompt)
        ]
        
        llm = llm_factory(temperature=0.1, state=state)  # Lower temperature for proofreading
        llm_response = await llm.ainvoke(messages)
        
        # Extract content with better error handling
        content = ""
        if hasattr(llm_response, 'content'):
            content = llm_response.content or ""
        elif hasattr(llm_response, 'text'):
            content = llm_response.text or ""
        else:
            content = str(llm_response) if llm_response else ""
        
        # Log response details for debugging
        logger.info(f"LLM response type: {type(llm_response)}, has content: {hasattr(llm_response, 'content')}, content length: {len(content)}")
        if not content or len(content.strip()) == 0:
            logger.error(f"⚠️ LLM returned empty response! Response object: {llm_response}")
            # Create error ManuscriptEdit immediately
            error_edit = {
                "target_filename": filename,
                "scope": "paragraph",
                "summary": "Proofreading analysis failed: LLM returned an empty response. This may indicate a model error or timeout.",
                "safety": "low",
                "operations": []
            }
            return {
                "structured_edit": error_edit,
                "error": "Empty LLM response",
                "task_status": "error",
                # ✅ CRITICAL: Preserve all critical state keys
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                # ✅ CRITICAL: Preserve manuscript content
                "manuscript": state.get("manuscript", ""),
                "manuscript_content": state.get("manuscript_content", ""),
                "active_editor": state.get("active_editor", {}),
                "cursor_offset": state.get("cursor_offset", -1),
                "chapter_ranges": state.get("chapter_ranges", [])
            }
        
        # Parse structured response
        import json
        import re
        
        # Remove markdown code blocks if present
        json_text = content.strip()
        if '```json' in json_text:
            match = re.search(r'```json\s*\n(.*?)\n```', json_text, re.DOTALL)
            if match:
                json_text = match.group(1).strip()
        elif '```' in json_text:
            match = re.search(r'```\s*\n(.*?)\n```', json_text, re.DOTALL)
            if match:
                json_text = match.group(1).strip()
        
        try:
            structured_edit_dict = json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            logger.warning(f"Raw LLM response (first 500 chars): {repr(content[:500])}")
            logger.warning(f"Full content length: {len(content)}, stripped length: {len(json_text)}")
            # Create a valid ManuscriptEdit with empty operations on error
            error_summary = f"Proofreading analysis failed: Could not parse LLM response as JSON. Error: {str(e)}"
            if len(content) == 0:
                error_summary = "Proofreading analysis failed: LLM returned an empty response."
            elif len(json_text) == 0:
                error_summary = f"Proofreading analysis failed: LLM response was empty after removing markdown. Original length: {len(content)}"
            structured_edit_dict = {
                "target_filename": filename,
                "scope": "paragraph",
                "summary": error_summary,
                "safety": "low",
                "operations": []
            }
        
        # Validate as ManuscriptEdit
        try:
            validated_edit = ManuscriptEdit(**structured_edit_dict)
            logger.info("Successfully validated LLM response as ManuscriptEdit")
            structured_edit_dict = validated_edit.model_dump()
        except ValidationError as e:
            logger.warning(f"LLM response failed ManuscriptEdit validation: {e}")
            logger.warning(f"Invalid structured_edit_dict keys: {list(structured_edit_dict.keys()) if isinstance(structured_edit_dict, dict) else 'not a dict'}")
            # Create a valid ManuscriptEdit with empty operations on validation error
            structured_edit_dict = {
                "target_filename": filename,
                "scope": "paragraph",
                "summary": f"Proofreading analysis failed: LLM response did not match required schema. Error: {str(e)}",
                "safety": "low",
                "operations": []
            }
        
        return {
            "structured_edit": structured_edit_dict,
            # ✅ CRITICAL: Preserve all critical state keys
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # ✅ CRITICAL: Preserve manuscript content for operation resolution
            "manuscript": state.get("manuscript", ""),
            "manuscript_content": state.get("manuscript_content", ""),
            "active_editor": state.get("active_editor", {}),
            "cursor_offset": state.get("cursor_offset", -1),
            "chapter_ranges": state.get("chapter_ranges", [])
        }
        
    except Exception as e:
        logger.error(f"Error during proofreading: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Get filename for error ManuscriptEdit
        active_editor = state.get("active_editor", {}) or {}
        shared_memory = state.get("shared_memory", {}) or {}
        if not active_editor:
            active_editor = shared_memory.get("active_editor", {}) or {}
        filename = active_editor.get("filename") or "document.md"
        
        # Create a valid ManuscriptEdit with empty operations on error
        error_edit = {
            "target_filename": filename,
            "scope": "paragraph",
            "summary": f"Proofreading analysis failed due to an error: {str(e)}",
            "safety": "low",
            "operations": []
        }
        
        return {
            "structured_edit": error_edit,
            "error": str(e),
            "task_status": "error",
            # ✅ CRITICAL: Preserve all critical state keys
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # ✅ CRITICAL: Preserve manuscript content
            "manuscript": state.get("manuscript", ""),
            "manuscript_content": state.get("manuscript_content", ""),
            "active_editor": state.get("active_editor", {}),
            "cursor_offset": state.get("cursor_offset", -1),
            "chapter_ranges": state.get("chapter_ranges", [])
        }


async def generate_operations_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate editor operations from proofreading corrections using robust resolver"""
    try:
        logger.info("Generating editor operations with robust resolver...")
        
        structured_edit_dict = state.get("structured_edit", {})
        if not structured_edit_dict:
            return {
                "editor_operations": [],
                "failed_operations": [],
                # ✅ CRITICAL: Preserve all critical state keys
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
        
        manuscript = state.get("manuscript_content") or state.get("manuscript", "")
        cursor_offset = state.get("cursor_offset", -1)
        
        # Extract operations from structured edit
        llm_operations = structured_edit_dict.get("operations", [])
        resolved_operations = []
        failed_operations = []
        
        for i, op_dict in enumerate(llm_operations):
            try:
                logger.info(f"Resolving proofreading operation {i+1}/{len(llm_operations)}")
                
                # Use centralized resolver
                start, end, text, confidence = resolve_editor_operation(
                    content=manuscript,
                    op_dict=op_dict,
                    cursor_offset=cursor_offset if cursor_offset >= 0 else None
                )
                
                if start != -1 and end != -1:
                    # Success - build operation with pre_hash
                    # Simple hash function for pre_hash
                    def simple_hash(text_slice: str) -> str:
                        h = 0
                        for char in text_slice:
                            h = (h * 31 + ord(char)) & 0xFFFFFFFF
                        return hex(h)[2:]
                    
                    pre_slice = manuscript[start:end]
                    resolved_op = {
                        "op_type": op_dict.get("op_type", "replace_range"),
                        "start": start,
                        "end": end,
                        "text": text,
                        "pre_hash": simple_hash(pre_slice),
                        "note": op_dict.get("note", "Proofreading correction"),
                        "confidence": confidence,
                        "original_text": op_dict.get("original_text")
                    }
                    resolved_operations.append(resolved_op)
                    logger.info(f"✅ Resolved operation {i+1} at [{start}:{end}] with confidence {confidence}")
                else:
                    # Failure
                    logger.warning(f"❌ Failed to resolve operation {i+1}: {op_dict.get('original_text', 'No original text')[:50]}...")
                    failed_operations.append({
                        "op": op_dict,
                        "error": "Could not find original_text in manuscript"
                    })
                    
            except Exception as e:
                logger.error(f"Error resolving operation {i+1}: {e}")
                failed_operations.append({
                    "op": op_dict,
                    "error": str(e)
                })
        
        logger.info(f"Generated {len(resolved_operations)} resolved operations, {len(failed_operations)} failed")
        
        return {
            "editor_operations": resolved_operations,
            "failed_operations": failed_operations,
            # ✅ CRITICAL: Preserve all critical state keys
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # ✅ CRITICAL: Preserve manuscript and structured_edit for parent agent
            "manuscript": state.get("manuscript", ""),
            "manuscript_content": state.get("manuscript_content", ""),
            "structured_edit": structured_edit_dict,
            "active_editor": state.get("active_editor", {}),
            "cursor_offset": state.get("cursor_offset", -1),
            "chapter_ranges": state.get("chapter_ranges", [])
        }
        
    except Exception as e:
        logger.error(f"Error generating operations: {e}")
        return {
            "editor_operations": [],
            "failed_operations": [{"error": str(e)}],
            # ✅ CRITICAL: Preserve all critical state keys
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # ✅ CRITICAL: Preserve manuscript content
            "manuscript": state.get("manuscript", ""),
            "manuscript_content": state.get("manuscript_content", ""),
            "active_editor": state.get("active_editor", {}),
            "cursor_offset": state.get("cursor_offset", -1),
            "chapter_ranges": state.get("chapter_ranges", [])
        }


# ============================================
# Subgraph Factory
# ============================================

def build_proofreading_subgraph(
    checkpointer,
    llm_factory: Callable,
    get_datetime_context: Callable
):
    """
    Build proofreading subgraph for integration into parent agents.
    
    Args:
        checkpointer: LangGraph checkpointer for state persistence
        llm_factory: Function that creates LLM instances (signature: llm_factory(temperature, state) -> LLM)
        get_datetime_context: Function that returns datetime context string
    
    Returns:
        Compiled StateGraph ready for integration
    """
    workflow = StateGraph(dict)  # Use flexible dict state
    
    # Add nodes with bound functions
    workflow.add_node("infer_mode", infer_mode_node)
    workflow.add_node("load_style_guide", load_style_guide_node)
    workflow.add_node("determine_scope", determine_scope_node)
    
    # Bind LLM factory and datetime context to proofread_content_node
    async def proofread_node(state: Dict[str, Any]) -> Dict[str, Any]:
        return await proofread_content_node(state, llm_factory, get_datetime_context)
    
    workflow.add_node("proofread_content", proofread_node)
    workflow.add_node("generate_operations", generate_operations_node)
    
    # Entry point
    workflow.set_entry_point("infer_mode")
    
    # Flow: infer_mode -> load_style_guide -> determine_scope -> proofread_content -> generate_operations -> END
    workflow.add_edge("infer_mode", "load_style_guide")
    workflow.add_edge("load_style_guide", "determine_scope")
    workflow.add_edge("determine_scope", "proofread_content")
    workflow.add_edge("proofread_content", "generate_operations")
    workflow.add_edge("generate_operations", END)
    
    return workflow.compile(checkpointer=checkpointer)
