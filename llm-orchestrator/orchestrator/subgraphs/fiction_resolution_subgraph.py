"""
Resolution Subgraph for Fiction Editing

Reusable subgraph that handles:
- Context preparation for operation resolution
- Individual operation resolution with progressive search
- Validation of resolved operations
- Finalization of operations list

Used by fiction_editing_agent for resolving editor operations.
"""

import logging
import re
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from orchestrator.models.editor_models import ManuscriptEdit
from orchestrator.utils.editor_operation_resolver import resolve_editor_operation
from orchestrator.utils.fiction_utilities import (
    frontmatter_end_index as _frontmatter_end_index,
    strip_frontmatter_block as _strip_frontmatter_block,
    slice_hash as _slice_hash,
    ensure_chapter_heading as _ensure_chapter_heading,
    detect_chapter_heading_at_position as _detect_chapter_heading_at_position,
)

logger = logging.getLogger(__name__)


# ============================================
# State Schema
# ============================================

# Use Dict[str, Any] for compatibility with main agent state
FictionResolutionState = Dict[str, Any]


# ============================================
# Helper Functions
# ============================================

def _get_structured_edit(state: Dict[str, Any]) -> Optional[ManuscriptEdit]:
    """Safely extract and validate structured_edit from state"""
    edit_dict = state.get("structured_edit")
    if not edit_dict:
        return None
    
    if isinstance(edit_dict, ManuscriptEdit):
        return edit_dict
    
    if not isinstance(edit_dict, dict):
        return None
    
    try:
        return ManuscriptEdit(**edit_dict)
    except Exception:
        return None


# ============================================
# Node Functions
# ============================================

async def prepare_resolution_context_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare resolution context: extract manuscript, chapters, selection, structured_edit"""
    try:
        logger.info("Preparing resolution context...")
        
        manuscript = state.get("manuscript", "")
        structured_edit = _get_structured_edit(state)
        selection_start = state.get("selection_start", -1)
        selection_end = state.get("selection_end", -1)
        cursor_offset = state.get("cursor_offset", -1)
        current_chapter_number = state.get("current_chapter_number")
        requested_chapter_number = state.get("requested_chapter_number")
        chapter_ranges = state.get("chapter_ranges", [])
        
        if not structured_edit:
            # Check if this is a question with no edits needed
            request_type = state.get("request_type", "")
            if request_type == "question":
                logger.info("Question request with no structured_edit - returning empty operations (analysis in summary)")
                return {
                    "editor_operations": [],
                    "task_status": "complete",
                    "resolution_complete": True,
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", ""),
                    "manuscript": state.get("manuscript", ""),
                    "filename": state.get("filename", ""),
                    "current_chapter_text": state.get("current_chapter_text", ""),
                    "current_chapter_number": state.get("current_chapter_number"),
                    "chapter_ranges": state.get("chapter_ranges", []),
                    "current_request": state.get("current_request", ""),
                    "active_editor": state.get("active_editor", {}),
                    "frontmatter": state.get("frontmatter", {}),
                    "structured_edit": state.get("structured_edit"),
                }
            return {
                "editor_operations": [],
                "error": "No operations to resolve",
                "task_status": "error",
                "resolution_complete": True,
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "manuscript": state.get("manuscript", ""),
                "filename": state.get("filename", ""),
                "current_chapter_text": state.get("current_chapter_text", ""),
                "current_chapter_number": state.get("current_chapter_number"),
                "chapter_ranges": state.get("chapter_ranges", []),
                "current_request": state.get("current_request", ""),
                "active_editor": state.get("active_editor", {}),
                "frontmatter": state.get("frontmatter", {}),
                "structured_edit": state.get("structured_edit"),
            }
        
        operations = structured_edit.operations
        if not isinstance(operations, list):
            return {
                "editor_operations": [],
                "error": "No operations to resolve",
                "task_status": "error",
                "resolution_complete": True,
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "manuscript": state.get("manuscript", ""),
                "filename": state.get("filename", ""),
                "current_chapter_text": state.get("current_chapter_text", ""),
                "current_chapter_number": state.get("current_chapter_number"),
                "chapter_ranges": state.get("chapter_ranges", []),
                "current_request": state.get("current_request", ""),
                "active_editor": state.get("active_editor", {}),
                "frontmatter": state.get("frontmatter", {}),
                "structured_edit": state.get("structured_edit"),
            }
        
        # Check if this is a question with no edits needed (empty operations array)
        request_type = state.get("request_type", "")
        if request_type == "question" and len(operations) == 0:
            logger.info("Question request with empty operations array - this is valid (analysis-only, no edits needed)")
            return {
                "editor_operations": [],
                "task_status": "complete",
                "resolution_complete": True,
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "manuscript": state.get("manuscript", ""),
                "filename": state.get("filename", ""),
                "current_chapter_text": state.get("current_chapter_text", ""),
                "current_chapter_number": state.get("current_chapter_number"),
                "chapter_ranges": state.get("chapter_ranges", []),
                "current_request": state.get("current_request", ""),
                "active_editor": state.get("active_editor", {}),
                "frontmatter": state.get("frontmatter", {}),
                "structured_edit": state.get("structured_edit"),
            }
        
        fm_end_idx = _frontmatter_end_index(manuscript)
        selection = {"start": selection_start, "end": selection_end} if selection_start >= 0 and selection_end >= 0 else None
        
        # Check if file is empty (only frontmatter)
        body_only = _strip_frontmatter_block(manuscript)
        is_empty_file = not body_only.strip()
        
        # Determine desired chapter number
        # Priority: 1) User requested chapter, 2) LLM chapter_index (0-indexed, convert to 1-indexed), 3) Current chapter, 4) Next chapter
        if requested_chapter_number is not None:
            desired_ch_num = requested_chapter_number
        else:
            llm_chapter_index = structured_edit.chapter_index
            if llm_chapter_index is not None:
                desired_ch_num = int(llm_chapter_index) + 1
            elif current_chapter_number:
                desired_ch_num = current_chapter_number
            else:
                max_num = max([r.chapter_number for r in chapter_ranges if r.chapter_number is not None], default=0)
                desired_ch_num = (max_num or 0) + 1
        
        return {
            "resolution_manuscript": manuscript,
            "resolution_structured_edit": structured_edit,
            "resolution_operations": operations,
            "resolution_selection": selection,
            "resolution_cursor_offset": cursor_offset if cursor_offset >= 0 else None,
            "resolution_fm_end_idx": fm_end_idx,
            "resolution_is_empty_file": is_empty_file,
            "resolution_desired_ch_num": desired_ch_num,
            "resolution_chapter_ranges": chapter_ranges,
            "resolution_current_chapter_number": current_chapter_number,
            "resolution_requested_chapter_number": requested_chapter_number,
            "resolution_complete": False,
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "manuscript": state.get("manuscript", ""),
            "filename": state.get("filename", ""),
            "current_chapter_text": state.get("current_chapter_text", ""),
            "current_chapter_number": state.get("current_chapter_number"),
            "chapter_ranges": state.get("chapter_ranges", []),
            "current_request": state.get("current_request", ""),
            "active_editor": state.get("active_editor", {}),
            "frontmatter": state.get("frontmatter", {}),
            "structured_edit": state.get("structured_edit"),
        }
        
    except Exception as e:
        logger.error(f"Failed to prepare resolution context: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "editor_operations": [],
            "error": str(e),
            "task_status": "error",
            "resolution_complete": True,
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "manuscript": state.get("manuscript", ""),
            "filename": state.get("filename", ""),
            "current_chapter_text": state.get("current_chapter_text", ""),
            "current_chapter_number": state.get("current_chapter_number"),
            "chapter_ranges": state.get("chapter_ranges", []),
            "current_request": state.get("current_request", ""),
            "active_editor": state.get("active_editor", {}),
            "frontmatter": state.get("frontmatter", {}),
            "structured_edit": state.get("structured_edit"),
        }


async def resolve_individual_operations_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve individual operations with progressive search"""
    try:
        logger.info("Resolving individual operations...")
        
        manuscript = state.get("resolution_manuscript", "")
        structured_edit = state.get("resolution_structured_edit")
        operations = state.get("resolution_operations", [])
        selection = state.get("resolution_selection")
        cursor_offset = state.get("resolution_cursor_offset")
        fm_end_idx = state.get("resolution_fm_end_idx", 0)
        is_empty_file = state.get("resolution_is_empty_file", False)
        desired_ch_num = state.get("resolution_desired_ch_num", 1)
        chapter_ranges = state.get("resolution_chapter_ranges", [])
        current_chapter_number = state.get("resolution_current_chapter_number")
        requested_chapter_number = state.get("resolution_requested_chapter_number")
        
        if not operations:
            return {
                "resolved_operations": [],
                "resolution_complete": True,
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "manuscript": state.get("manuscript", ""),
                "filename": state.get("filename", ""),
                "current_chapter_text": state.get("current_chapter_text", ""),
                "current_chapter_number": state.get("current_chapter_number"),
                "chapter_ranges": state.get("chapter_ranges", []),
                "current_request": state.get("current_request", ""),
                "active_editor": state.get("active_editor", {}),
                "frontmatter": state.get("frontmatter", {}),
                "structured_edit": state.get("structured_edit"),
                "resolution_manuscript": state.get("resolution_manuscript", ""),
                "resolution_structured_edit": state.get("resolution_structured_edit"),
                "resolution_operations": state.get("resolution_operations", []),
            }
        
        editor_operations = []
        
        for i, op in enumerate(operations):
            # Resolve operation
            try:
                # Convert Pydantic model to dict for resolve_editor_operation
                op_dict = op.model_dump() if hasattr(op, 'model_dump') else op
                
                # Log operation details for debugging
                op_type = op_dict.get("op_type", "unknown")
                original_text = op_dict.get("original_text")
                anchor_text = op_dict.get("anchor_text")
                op_start = op_dict.get("start", -1)
                op_end = op_dict.get("end", -1)
                
                logger.info(f"🔍 Resolving operation {i+1}/{len(operations)}: type={op_type}, has_original_text={bool(original_text)}, has_anchor_text={bool(anchor_text)}, approximate_pos=[{op_start}:{op_end}]")
                
                # CRITICAL VALIDATION: Operations MUST have proper anchors
                if op_type in ("replace_range", "delete_range"):
                    if not original_text or not original_text.strip():
                        logger.error(f"❌ Operation {i+1} ({op_type}) is MISSING 'original_text' - this operation will fail!")
                        logger.error(f"   ⚠️ For {op_type} operations, you MUST provide 'original_text' with EXACT text from the manuscript")
                        logger.error(f"   💡 If this is NEW text to insert, use 'insert_after_heading' with 'anchor_text' instead")
                        # Skip this operation - it can't be resolved
                        continue
                elif op_type in ("insert_after_heading", "insert_after"):
                    # Check if file is empty - if so, anchor_text is optional
                    body_only = _strip_frontmatter_block(manuscript)
                    is_empty_file_check = not body_only.strip()
                    
                    if not anchor_text or not anchor_text.strip():
                        if is_empty_file_check:
                            logger.info(f"✅ Operation {i+1} ({op_type}) for empty file - anchor_text not required, will insert after frontmatter")
                        else:
                            logger.error(f"❌ Operation {i+1} ({op_type}) is MISSING 'anchor_text' - this operation will fail!")
                            logger.error(f"   ⚠️ For {op_type} operations in non-empty files, you MUST provide 'anchor_text' with EXACT text to insert after")
                            # Skip this operation - it can't be resolved
                            continue
                
                if original_text:
                    logger.debug(f"   original_text preview: {original_text[:100]}...")
                
                # Use centralized resolver
                resolved_start, resolved_end, resolved_text, resolved_confidence = resolve_editor_operation(
                    content=manuscript,
                    op_dict=op_dict,
                    selection=selection,
                    frontmatter_end=fm_end_idx,
                    cursor_offset=cursor_offset
                )
                
                # Check if resolution failed (returned -1, -1)
                if resolved_start == -1 and resolved_end == -1:
                    logger.error(f"❌ Operation {i+1} resolution FAILED - original_text could not be found")
                    logger.error(f"   original_text: {repr(original_text[:200]) if original_text else 'None'}")
                    logger.error(f"   This operation will be SKIPPED")
                    continue
                
                # Log resolution result - check if it resolved to cursor position when original_text should have been found
                cursor_pos = selection["start"] if selection and selection.get("start", -1) >= 0 and selection.get("start") == selection.get("end") else -1
                if resolved_start == resolved_end and resolved_start == cursor_pos and cursor_pos >= 0 and original_text:
                    logger.warning(f"⚠️ Operation {i+1} resolved to cursor position [{resolved_start}:{resolved_end}] - original_text matching failed!")
                    # Check if original_text exists in manuscript
                    if original_text not in manuscript:
                        logger.warning(f"   ❌ original_text NOT FOUND in manuscript - searching for partial matches...")
                        # Try to find similar text with various strategies
                        # Strategy 1: First 30 characters
                        similar = manuscript.find(original_text[:30]) if len(original_text) > 30 else -1
                        if similar >= 0:
                            logger.info(f"   💡 Found similar text at position {similar} (first 30 chars match) - using this position")
                            # Use this position instead
                            resolved_start = similar
                            resolved_end = similar + len(original_text)
                            resolved_confidence = 0.65
                        else:
                            # Strategy 2: Normalize whitespace and try again
                            normalized_original = ' '.join(original_text.split())
                            # Search in chunks to avoid memory issues
                            for chunk_start in range(0, len(manuscript), 10000):
                                chunk = manuscript[chunk_start:chunk_start + 10000]
                                normalized_chunk = ' '.join(chunk.split())
                                similar_in_chunk = normalized_chunk.find(normalized_original[:30]) if len(normalized_original) > 30 else -1
                                if similar_in_chunk >= 0:
                                    logger.info(f"   💡 Found similar text after whitespace normalization at position {chunk_start + similar_in_chunk}")
                                    # Approximate position in original manuscript
                                    resolved_start = chunk_start + similar_in_chunk
                                    resolved_end = resolved_start + len(original_text)
                                    resolved_confidence = 0.6
                                    break
                            else:
                                logger.error(f"   ❌ Could not find original_text even with partial matching - operation will fail!")
                    else:
                        logger.warning(f"   ⚠️ original_text EXISTS in manuscript but matching failed - check occurrence_index or text formatting")
                        # Try to find it manually (maybe occurrence_index is wrong)
                        found_pos = manuscript.find(original_text)
                        if found_pos >= 0:
                            logger.info(f"   💡 Found original_text at position {found_pos} (occurrence_index may be wrong) - using this position")
                            resolved_start = found_pos
                            resolved_end = found_pos + len(original_text)
                            resolved_confidence = 0.9
                
                # Special handling for empty files: ensure operations insert after frontmatter
                if is_empty_file and resolved_start < fm_end_idx:
                    resolved_start = fm_end_idx
                    resolved_end = fm_end_idx
                    resolved_confidence = 0.7
                    logger.info(f"Empty manuscript detected - adjusting operation to insert after frontmatter at {fm_end_idx}")
                
                # If resolution failed (0:0) and this is a new chapter, find the last chapter
                is_chapter_scope = (str(structured_edit.scope or "").lower() == "chapter")
                is_new_chapter = (resolved_start == resolved_end == 0) and is_chapter_scope
                
                if is_new_chapter and chapter_ranges:
                    # Find the last existing chapter and insert after it
                    last_chapter_range = chapter_ranges[-1]
                    resolved_start = last_chapter_range.end
                    resolved_end = last_chapter_range.end
                    resolved_confidence = 0.8
                    logger.info(f"New chapter detected - inserting after last chapter (Chapter {last_chapter_range.chapter_number}) at position {resolved_start}")
                
                # Detect if we're creating a new chapter (not just editing an existing one)
                # Check if requested chapter number doesn't exist in manuscript
                is_creating_new_chapter = False
                if requested_chapter_number is not None:
                    existing_chapter_numbers = {r.chapter_number for r in chapter_ranges if r.chapter_number is not None}
                    is_creating_new_chapter = requested_chapter_number not in existing_chapter_numbers
                elif is_chapter_scope and op_type == "insert_after_heading":
                    # If it's chapter scope and insert_after_heading, likely creating new chapter
                    is_creating_new_chapter = True
                
                logger.info(f"Resolved {op_dict.get('op_type')} [{resolved_start}:{resolved_end}] confidence={resolved_confidence:.2f}")
                
                # Ensure chapter heading for new chapters - check ALL new chapter insertions
                if is_creating_new_chapter or (is_chapter_scope and is_new_chapter):
                    # Check if text already has a chapter header
                    has_header = bool(re.match(r'^\s*#{1,6}\s*Chapter\s+\d+\b', resolved_text.strip(), re.IGNORECASE))
                    if not has_header:
                        chapter_num = desired_ch_num or requested_chapter_number or current_chapter_number or 1
                        resolved_text = _ensure_chapter_heading(resolved_text, int(chapter_num))
                        logger.info(f"Added chapter header '## Chapter {chapter_num}' to new chapter content")
                
                # Fix duplicate chapter heading issue: if editing at start of existing chapter
                # and replacement includes heading, ensure we replace the existing heading too
                if op_dict.get("op_type") == "replace_range":
                    # Check if replacement text includes a chapter heading
                    replacement_has_heading = bool(re.match(r'^\s*#{1,6}\s+Chapter\s+\d+\b', resolved_text.strip(), re.IGNORECASE))
                    
                    if replacement_has_heading:
                        # Check if there's an existing chapter heading at or near the start position
                        # This indicates we're editing at the start of an existing chapter
                        heading_info = _detect_chapter_heading_at_position(manuscript, resolved_start)
                        
                        if heading_info:
                            heading_start, heading_end, heading_text = heading_info
                            # Check if the original text slice includes the heading
                            original_slice = manuscript[resolved_start:resolved_end]
                            
                            if heading_text not in original_slice:
                                # Original doesn't include heading, but replacement does
                                # This would create a duplicate heading - expand the start position to include the existing heading
                                logger.info(f"⚠️ Duplicate heading detected: expanding operation to include existing heading at {heading_start}")
                                resolved_start = heading_start
                                # Recalculate resolved_end to maintain the same relative position
                                resolved_confidence = min(resolved_confidence, 0.9)  # Slightly reduce confidence
                
                # Calculate pre_hash
                pre_slice = manuscript[resolved_start:resolved_end]
                pre_hash = _slice_hash(pre_slice)
                
                # Build operation dict (use op_dict since op might be Pydantic model)
                resolved_op = {
                    "op_type": op_dict.get("op_type", "replace_range"),
                    "start": resolved_start,
                    "end": resolved_end,
                    "text": resolved_text,
                    "pre_hash": pre_hash,
                    "original_text": op_dict.get("original_text"),
                    "anchor_text": op_dict.get("anchor_text"),
                    "left_context": op_dict.get("left_context"),
                    "right_context": op_dict.get("right_context"),
                    "occurrence_index": op_dict.get("occurrence_index", 0),
                    "confidence": resolved_confidence
                }
                
                editor_operations.append(resolved_op)
                
            except Exception as e:
                logger.error(f"❌ Operation {i+1} resolution failed: {e}")
                # Ensure op_dict is a dict for error logging
                if not isinstance(op, dict):
                    op_dict = op.model_dump() if hasattr(op, 'model_dump') else (op.dict() if hasattr(op, 'dict') else {})
                else:
                    op_dict = op
                op_type = op_dict.get('op_type', 'unknown') if isinstance(op_dict, dict) else 'unknown'
                original_text = op_dict.get('original_text') if isinstance(op_dict, dict) else None
                logger.error(f"   Operation type: {op_type}, original_text: {repr(original_text[:100]) if original_text else 'None'}")
                logger.error(f"   This operation will be SKIPPED - cannot safely place it without proper resolution")
                # Skip this operation - don't use fallback positioning as it's unreliable
                continue
        
        return {
            "resolved_operations": editor_operations,
            "resolution_complete": False,
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "manuscript": state.get("manuscript", ""),
            "filename": state.get("filename", ""),
            "current_chapter_text": state.get("current_chapter_text", ""),
            "current_chapter_number": state.get("current_chapter_number"),
            "chapter_ranges": state.get("chapter_ranges", []),
            "current_request": state.get("current_request", ""),
            "active_editor": state.get("active_editor", {}),
            "frontmatter": state.get("frontmatter", {}),
            "structured_edit": state.get("structured_edit"),
            "resolution_manuscript": state.get("resolution_manuscript", ""),
            "resolution_structured_edit": state.get("resolution_structured_edit"),
            "resolution_operations": state.get("resolution_operations", []),
        }
        
    except Exception as e:
        logger.error(f"Failed to resolve individual operations: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "resolved_operations": [],
            "error": str(e),
            "task_status": "error",
            "resolution_complete": True,
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "manuscript": state.get("manuscript", ""),
            "filename": state.get("filename", ""),
            "current_chapter_text": state.get("current_chapter_text", ""),
            "current_chapter_number": state.get("current_chapter_number"),
            "chapter_ranges": state.get("chapter_ranges", []),
            "current_request": state.get("current_request", ""),
            "active_editor": state.get("active_editor", {}),
            "frontmatter": state.get("frontmatter", {}),
            "structured_edit": state.get("structured_edit"),
            "resolution_manuscript": state.get("resolution_manuscript", ""),
            "resolution_structured_edit": state.get("resolution_structured_edit"),
            "resolution_operations": state.get("resolution_operations", []),
        }


async def validate_resolved_operations_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Validate resolved operations: check for conflicts, overlaps"""
    try:
        logger.info("Validating resolved operations...")
        
        resolved_operations = state.get("resolved_operations", [])
        
        # Basic validation: check for overlapping operations
        # Sort by start position
        sorted_ops = sorted(resolved_operations, key=lambda op: op.get("start", 0))
        
        for i in range(len(sorted_ops) - 1):
            current_op = sorted_ops[i]
            next_op = sorted_ops[i + 1]
            
            current_end = current_op.get("end", 0)
            next_start = next_op.get("start", 0)
            
            if current_end > next_start:
                logger.warning(f"⚠️ Overlapping operations detected: op {i} ends at {current_end}, op {i+1} starts at {next_start}")
                # This is a warning, not an error - operations might be intentionally overlapping
        
        # All operations validated
        return {
            "validated_operations": resolved_operations,
            "resolution_complete": False,
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "manuscript": state.get("manuscript", ""),
            "filename": state.get("filename", ""),
            "current_chapter_text": state.get("current_chapter_text", ""),
            "current_chapter_number": state.get("current_chapter_number"),
            "chapter_ranges": state.get("chapter_ranges", []),
            "current_request": state.get("current_request", ""),
            "active_editor": state.get("active_editor", {}),
            "frontmatter": state.get("frontmatter", {}),
            "structured_edit": state.get("structured_edit"),
            "resolution_manuscript": state.get("resolution_manuscript", ""),
            "resolution_structured_edit": state.get("resolution_structured_edit"),
            "resolution_operations": state.get("resolution_operations", []),
        }
        
    except Exception as e:
        logger.error(f"Failed to validate resolved operations: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "validated_operations": state.get("resolved_operations", []),
            "error": str(e),
            "task_status": "error",
            "resolution_complete": True,
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "manuscript": state.get("manuscript", ""),
            "filename": state.get("filename", ""),
            "current_chapter_text": state.get("current_chapter_text", ""),
            "current_chapter_number": state.get("current_chapter_number"),
            "chapter_ranges": state.get("chapter_ranges", []),
            "current_request": state.get("current_request", ""),
            "active_editor": state.get("active_editor", {}),
            "frontmatter": state.get("frontmatter", {}),
            "structured_edit": state.get("structured_edit"),
            "resolution_manuscript": state.get("resolution_manuscript", ""),
            "resolution_structured_edit": state.get("resolution_structured_edit"),
            "resolution_operations": state.get("resolution_operations", []),
        }


async def finalize_operations_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Finalize operations: prepare final operations list"""
    try:
        logger.info("Finalizing operations...")
        
        validated_operations = state.get("validated_operations", [])
        
        return {
            "editor_operations": validated_operations,
            "resolution_complete": True,
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "manuscript": state.get("manuscript", ""),
            "filename": state.get("filename", ""),
            "current_chapter_text": state.get("current_chapter_text", ""),
            "current_chapter_number": state.get("current_chapter_number"),
            "chapter_ranges": state.get("chapter_ranges", []),
            "current_request": state.get("current_request", ""),
            "active_editor": state.get("active_editor", {}),
            "frontmatter": state.get("frontmatter", {}),
            "structured_edit": state.get("structured_edit"),
            "resolution_manuscript": state.get("resolution_manuscript", ""),
            "resolution_structured_edit": state.get("resolution_structured_edit"),
            "resolution_operations": state.get("resolution_operations", []),
        }
        
    except Exception as e:
        logger.error(f"Failed to finalize operations: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "editor_operations": [],
            "error": str(e),
            "task_status": "error",
            "resolution_complete": True,
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "manuscript": state.get("manuscript", ""),
            "filename": state.get("filename", ""),
            "current_chapter_text": state.get("current_chapter_text", ""),
            "current_chapter_number": state.get("current_chapter_number"),
            "chapter_ranges": state.get("chapter_ranges", []),
            "current_request": state.get("current_request", ""),
            "active_editor": state.get("active_editor", {}),
            "frontmatter": state.get("frontmatter", {}),
            "structured_edit": state.get("structured_edit"),
            "resolution_manuscript": state.get("resolution_manuscript", ""),
            "resolution_structured_edit": state.get("resolution_structured_edit"),
            "resolution_operations": state.get("resolution_operations", []),
        }


# ============================================
# Subgraph Builder
# ============================================

def build_resolution_subgraph(checkpointer):
    """
    Build the resolution subgraph for fiction editing.
    
    Args:
        checkpointer: LangGraph checkpointer for state persistence
    
    Returns:
        Compiled StateGraph subgraph
    """
    workflow = StateGraph(FictionResolutionState)
    
    # Add nodes
    workflow.add_node("prepare_context", prepare_resolution_context_node)
    workflow.add_node("resolve_operations", resolve_individual_operations_node)
    workflow.add_node("validate_operations", validate_resolved_operations_node)
    workflow.add_node("finalize_operations", finalize_operations_node)
    
    # Set entry point
    workflow.set_entry_point("prepare_context")
    
    # Define edges with conditional routing
    def route_after_prepare(state: Dict[str, Any]) -> str:
        if state.get("resolution_complete", False):
            return "finalize_operations"
        return "resolve_operations"
    
    workflow.add_conditional_edges(
        "prepare_context",
        route_after_prepare,
        {
            "finalize_operations": "finalize_operations",
            "resolve_operations": "resolve_operations"
        }
    )
    
    workflow.add_edge("resolve_operations", "validate_operations")
    workflow.add_edge("validate_operations", "finalize_operations")
    workflow.add_edge("finalize_operations", END)
    
    return workflow.compile(checkpointer=checkpointer)

