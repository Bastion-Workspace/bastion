"""
Context Preparation Subgraph for Fiction Agents

Reusable subgraph that handles:
- Context extraction from active editor
- Chapter detection and scope analysis
- Reference loading (outline, style, rules, characters)
- Reference quality assessment

Can be used by fiction_editing_agent, proofreading_agent, and other fiction-aware agents.
"""

import logging
import re
from typing import Any, Dict, List, Optional, TypedDict, Tuple

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage

from orchestrator.utils.frontmatter_utils import strip_frontmatter_block
from orchestrator.utils.fiction_utilities import (
    ChapterRange,
    CHAPTER_PATTERN,
    find_chapter_ranges,
    locate_chapter_index,
    get_adjacent_chapters,
)

logger = logging.getLogger(__name__)


# ============================================
# State Schema
# ============================================

# Use Dict[str, Any] for compatibility with main agent state
FictionContextState = Dict[str, Any]


# ============================================
# Node Functions
# ============================================

async def prepare_context_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare context: extract active editor, validate fiction type"""
    try:
        logger.info("Preparing context for fiction editing...")
        
        active_editor = state.get("active_editor", {}) or {}
        
        manuscript = active_editor.get("content", "") or ""
        filename = active_editor.get("filename") or "manuscript.md"
        frontmatter = active_editor.get("frontmatter", {}) or {}
        
        cursor_offset = int(active_editor.get("cursor_offset", -1))
        selection_start = int(active_editor.get("selection_start", -1))
        selection_end = int(active_editor.get("selection_end", -1))
        
        # Note: Type checking is handled by the calling agent (fiction_editing_agent gates on fiction)
        # This subgraph is reusable and doesn't gate on document type
        
        # Extract user request
        metadata = state.get("metadata", {}) or {}
        messages = metadata.get("messages", [])
        
        if not messages:
            # Try getting from shared_memory
            shared_memory = state.get("shared_memory", {}) or {}
            messages = shared_memory.get("messages", [])
        
        try:
            if messages:
                latest_message = messages[-1]
                current_request = latest_message.content if hasattr(latest_message, 'content') else str(latest_message)
            else:
                current_request = state.get("query", "")
        except Exception as e:
            logger.error(f"Exception extracting current_request: {e}")
            current_request = ""
        
        current_request = current_request.strip()
        
        return {
            "active_editor": active_editor,
            "manuscript": manuscript,  # Use 'manuscript' for compatibility with main agent
            "manuscript_content": manuscript,  # Also set for subgraph internal use
            "filename": filename,
            "frontmatter": frontmatter,
            "cursor_offset": cursor_offset,
            "selection_start": selection_start,
            "selection_end": selection_end,
            "current_request": current_request,
            "user_id": state.get("user_id", "system"),  # CRITICAL: Preserve user_id from parent state
            # ✅ CRITICAL: Preserve all critical state keys
            "metadata": state.get("metadata", {}),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }
        
    except Exception as e:
        logger.error(f"Failed to prepare context: {e}")
        return {
            "error": str(e),
            "task_status": "error",
            "user_id": state.get("user_id", "system"),  # CRITICAL: Preserve user_id even on error
            # ✅ CRITICAL: Preserve all critical state keys even on error
            "metadata": state.get("metadata", {}),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }


async def detect_chapter_mentions_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Parse user query for explicit chapter mentions"""
    try:
        logger.info("Detecting chapter mentions in user query...")
        
        current_request = state.get("current_request", "")
        if not current_request:
            logger.info("No current request - skipping chapter detection")
            return {
                "explicit_primary_chapter": None,
                "explicit_secondary_chapters": [],
                # ✅ CRITICAL: Preserve all critical state keys
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
        
        # Regex patterns for chapter detection
        CHAPTER_PATTERNS = [
            # Action verbs + Chapter
            r'\b(?:Look over|Review|Check|Edit|Update|Revise|Modify|Address|Fix|Change)\s+[Cc]hapter\s+(\d+)\b',
            # Preposition + Chapter
            r'\b(?:in|at|for|to)\s+[Cc]hapter\s+(\d+)\b',
            # Chapter + verb
            r'\b[Cc]hapter\s+(\d+)\s+(?:needs|has|shows|contains|should|must|is|requires)',
            # Verb + in/at + Chapter
            r'\b(?:address|fix|change|edit|update|revise|modify)\s+(?:in|at)\s+[Cc]hapter\s+(\d+)\b',
            # Chapter + punctuation + relative clause
            r'\b[Cc]hapter\s+(\d+)[:,]?\s+(?:where|when|that|which)',
        ]
        
        all_mentions = []
        for pattern in CHAPTER_PATTERNS:
            matches = re.finditer(pattern, current_request)
            for match in matches:
                chapter_num = int(match.group(1))
                all_mentions.append({
                    "chapter": chapter_num,
                    "position": match.start(),
                    "text": match.group(0)
                })
        
        # Remove duplicates while preserving order
        seen = set()
        unique_mentions = []
        for mention in all_mentions:
            if mention["chapter"] not in seen:
                seen.add(mention["chapter"])
                unique_mentions.append(mention)
        
        # Sort by position in query (first mention = primary)
        unique_mentions.sort(key=lambda x: x["position"])
        
        primary_chapter = None
        secondary_chapters = []
        
        if unique_mentions:
            primary_chapter = unique_mentions[0]["chapter"]
            secondary_chapters = [m["chapter"] for m in unique_mentions[1:]]
        
        return {
            "explicit_primary_chapter": primary_chapter,
            "explicit_secondary_chapters": secondary_chapters,
            "current_request": state.get("current_request", ""),  # Preserve from prepare_context
            "cursor_offset": state.get("cursor_offset", -1),  # CRITICAL: Preserve cursor
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "active_editor": state.get("active_editor", {}),  # CRITICAL: Preserve active_editor
            "manuscript": state.get("manuscript", ""),  # CRITICAL: Preserve manuscript
            "filename": state.get("filename", ""),  # CRITICAL: Preserve filename
            "frontmatter": state.get("frontmatter", {}),  # CRITICAL: Preserve frontmatter
            "user_id": state.get("user_id", "system"),  # CRITICAL: Preserve user_id for DB queries
            # ✅ CRITICAL: Preserve all critical state keys
            "metadata": state.get("metadata", {}),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }
        
    except Exception as e:
        logger.error(f"Failed to detect chapter mentions: {e}")
        return {
            "explicit_primary_chapter": None,
            "explicit_secondary_chapters": [],
            "current_request": state.get("current_request", ""),  # Preserve even on error
            "cursor_offset": state.get("cursor_offset", -1),  # CRITICAL: Preserve cursor
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "active_editor": state.get("active_editor", {}),  # CRITICAL: Preserve active_editor
            "manuscript": state.get("manuscript", ""),
            "filename": state.get("filename", ""),
            "frontmatter": state.get("frontmatter", {}),
            "user_id": state.get("user_id", "system"),  # CRITICAL: Preserve user_id for DB queries
            # ✅ CRITICAL: Preserve all critical state keys even on error
            "metadata": state.get("metadata", {}),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }


async def analyze_scope_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze chapter scope: find chapters, determine current/prev/next"""
    try:
        logger.info("Analyzing chapter scope...")
        
        # Support both manuscript_content (from subgraph) and manuscript (from main agent)
        manuscript = state.get("manuscript_content") or state.get("manuscript", "")
        cursor_offset = state.get("cursor_offset", -1)
        
        # If cursor is -1, treat it as end of document
        if cursor_offset == -1 and len(manuscript) > 0:
            cursor_offset = len(manuscript)
        
        # Get explicit chapter mentions from query
        explicit_primary_chapter = state.get("explicit_primary_chapter")
        explicit_secondary_chapters = state.get("explicit_secondary_chapters", [])
        
        # Find chapter ranges
        chapter_ranges = find_chapter_ranges(manuscript)
        
        # DIAGNOSTIC: Log first 500 chars of manuscript to debug chapter detection
        if len(chapter_ranges) == 0 and len(manuscript) > 0:
            logger.warning(f"CHAPTER DETECTION FAILED - Manuscript preview (first 500 chars): {repr(manuscript[:500])}")
        
        # Priority system: explicit chapter > cursor position > default
        active_idx = -1
        current_chapter_number: Optional[int] = None
        detection_method = "unknown"
        
        # 1. Explicit chapter from query (highest priority)
        if explicit_primary_chapter is not None:
            for i, ch_range in enumerate(chapter_ranges):
                if ch_range.chapter_number == explicit_primary_chapter:
                    active_idx = i
                    current_chapter_number = explicit_primary_chapter
                    detection_method = "explicit_query"
                    break
            
            if active_idx == -1:
                logger.warning(f"Explicit chapter {explicit_primary_chapter} not found in manuscript - falling back to cursor/default")
        
        # 2. Cursor position (if no explicit chapter and cursor is valid)
        if active_idx == -1 and cursor_offset >= 0:
            active_idx = locate_chapter_index(chapter_ranges, cursor_offset)
            if active_idx != -1:
                current_chapter_number = chapter_ranges[active_idx].chapter_number
                detection_method = "cursor_position"
        
        # 3. Default: entire manuscript (if no explicit chapter and invalid cursor)
        if active_idx == -1:
            detection_method = "default_entire_manuscript"
        
        prev_c, next_c = (None, None)
        current_chapter_text = manuscript
        
        if active_idx != -1:
            current = chapter_ranges[active_idx]
            prev_c, next_c = get_adjacent_chapters(chapter_ranges, active_idx)
            current_chapter_text = manuscript[current.start:current.end]
            current_chapter_number = current.chapter_number
            if active_idx == len(chapter_ranges) - 1:
                if current.end != len(manuscript):
                    logger.warning(f"Last chapter end ({current.end}) doesn't match manuscript end ({len(manuscript)}) - potential truncation issue!")
            
            if next_c and next_c.start == current.start:
                logger.warning(f"Next chapter has same start as current chapter - likely last chapter bug. Setting next_c to None.")
                next_c = None
        
        # Get adjacent chapter text
        prev_chapter_text = None
        next_chapter_text = None
        prev_chapter_number = None
        next_chapter_number = None
        
        if prev_c:
            if active_idx != -1 and prev_c.start == chapter_ranges[active_idx].start:
                logger.warning(f"Previous chapter has same start as current chapter - skipping prev chapter.")
            else:
                prev_chapter_text = strip_frontmatter_block(manuscript[prev_c.start:prev_c.end])
                prev_chapter_number = prev_c.chapter_number
        
        if next_c:
            if active_idx != -1 and next_c.start == chapter_ranges[active_idx].start:
                logger.warning(f"Next chapter has same start as current chapter - likely last chapter. Setting next_chapter_text to None.")
                next_chapter_text = None
            else:
                next_chapter_text = strip_frontmatter_block(manuscript[next_c.start:next_c.end])
                next_chapter_number = next_c.chapter_number
        
        # Strip YAML frontmatter only when we're operating on the full document.
        # When we've already sliced a specific chapter, stripping is unsafe because
        # chapter text may legitimately contain Markdown horizontal rules ("---").
        if active_idx != -1:
            context_current_chapter_text = current_chapter_text
        else:
            context_current_chapter_text = strip_frontmatter_block(current_chapter_text)
        
        return {
            "chapter_ranges": chapter_ranges,
            "active_chapter_idx": active_idx,  # Use 'active_chapter_idx' for compatibility
            "working_chapter_index": active_idx,  # Also set for subgraph internal use
            "current_chapter_text": context_current_chapter_text,
            "current_chapter_number": current_chapter_number,
            "prev_chapter_text": prev_chapter_text,
            "prev_chapter_number": prev_chapter_number,
            "next_chapter_text": next_chapter_text,
            "next_chapter_number": next_chapter_number,
            "current_request": state.get("current_request", ""),  # Preserve from earlier nodes
            "cursor_offset": state.get("cursor_offset", -1),  # CRITICAL: Preserve cursor
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "active_editor": state.get("active_editor", {}),  # CRITICAL: Preserve active_editor for next nodes
            "manuscript": state.get("manuscript", ""),  # CRITICAL: Preserve manuscript
            "filename": state.get("filename", ""),  # CRITICAL: Preserve filename
            "frontmatter": state.get("frontmatter", {}),  # CRITICAL: Preserve frontmatter
            "user_id": state.get("user_id", "system"),  # CRITICAL: Preserve user_id for DB queries
            # ✅ CRITICAL: Preserve all critical state keys
            "metadata": state.get("metadata", {}),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }
        
    except Exception as e:
        logger.error(f"Failed to analyze scope: {e}")
        return {
            "error": str(e),
            "task_status": "error",
            "current_request": state.get("current_request", ""),  # Preserve even on error
            "cursor_offset": state.get("cursor_offset", -1),  # CRITICAL: Preserve cursor
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "active_editor": state.get("active_editor", {}),  # CRITICAL: Preserve active_editor
            "manuscript": state.get("manuscript", ""),
            "filename": state.get("filename", ""),
            "frontmatter": state.get("frontmatter", {}),
            "user_id": state.get("user_id", "system"),  # CRITICAL: Preserve user_id for DB queries
            # ✅ CRITICAL: Preserve all critical state keys even on error
            "metadata": state.get("metadata", {}),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }


async def load_references_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Load referenced context files (outline, rules, style, characters)"""
    try:
        logger.info("Loading referenced context files...")
        
        from orchestrator.tools.reference_file_loader import load_referenced_files
        
        active_editor = state.get("active_editor", {})
        user_id = state.get("user_id", "system")
        
        # Fiction reference configuration
        reference_config = {
            "outline": ["outline"]
        }
        
        # Cascading: outline frontmatter has rules, style, characters
        cascade_config = {
            "outline": {
                "rules": ["rules"],
                "style": ["style"],
                "characters": ["characters", "character_*"]
            }
        }
        
        # Use unified loader with cascading
        result = await load_referenced_files(
            active_editor=active_editor,
            user_id=user_id,
            reference_config=reference_config,
            doc_type_filter="fiction",
            cascade_config=cascade_config
        )
        
        loaded_files = result.get("loaded_files", {})
        
        # Extract content from loaded files
        outline_body = None
        if loaded_files.get("outline") and len(loaded_files["outline"]) > 0:
            outline_body = loaded_files["outline"][0].get("content")
        
        rules_body = None
        if loaded_files.get("rules") and len(loaded_files["rules"]) > 0:
            rules_body = loaded_files["rules"][0].get("content")
        
        style_body = None
        if loaded_files.get("style") and len(loaded_files["style"]) > 0:
            style_body = loaded_files["style"][0].get("content")
        
        characters_bodies = []
        if loaded_files.get("characters"):
            characters_bodies = [char_file.get("content", "") for char_file in loaded_files["characters"] if char_file.get("content")]
        
        # Extract current chapter outline if we have chapter number
        outline_current_chapter_text = None
        current_chapter_number = state.get("current_chapter_number")
        if outline_body and current_chapter_number:
            chapter_pattern = rf"(?i)(?:^|\n)##?\s*(?:Chapter\s+)?{current_chapter_number}[:\s]*(.*?)(?=\n##?\s*(?:Chapter\s+)?\d+|\Z)"
            match = re.search(chapter_pattern, outline_body, re.DOTALL)
            if match:
                outline_current_chapter_text = match.group(1).strip()
        
        has_references_value = bool(outline_body)
        
        return {
            "outline_body": outline_body,
            "rules_body": rules_body,
            "style_body": style_body,
            "characters_bodies": characters_bodies,
            "outline_current_chapter_text": outline_current_chapter_text,
            "loaded_references": loaded_files,
            "has_references": has_references_value,
            "current_request": state.get("current_request", ""),  # Preserve from earlier nodes
            "cursor_offset": state.get("cursor_offset", -1),  # CRITICAL: Preserve cursor
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "user_id": state.get("user_id", "system"),  # CRITICAL: Preserve user_id
            "active_editor": state.get("active_editor", {}),  # CRITICAL: Preserve for subsequent nodes
            "manuscript": state.get("manuscript", ""),
            "filename": state.get("filename", ""),
            "frontmatter": state.get("frontmatter", {}),
            # ✅ CRITICAL: Preserve chapter context from analyze_scope_node
            "chapter_ranges": state.get("chapter_ranges", []),
            "active_chapter_idx": state.get("active_chapter_idx", -1),
            "working_chapter_index": state.get("working_chapter_index", -1),
            "current_chapter_text": state.get("current_chapter_text", ""),
            "current_chapter_number": state.get("current_chapter_number"),
            "prev_chapter_text": state.get("prev_chapter_text"),
            "prev_chapter_number": state.get("prev_chapter_number"),
            "next_chapter_text": state.get("next_chapter_text"),
            "next_chapter_number": state.get("next_chapter_number"),
            "explicit_primary_chapter": state.get("explicit_primary_chapter"),
            "explicit_secondary_chapters": state.get("explicit_secondary_chapters", []),
            # ✅ CRITICAL: Preserve critical 5 keys for downstream nodes / parent agent
            "metadata": state.get("metadata", {}),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }
        
    except Exception as e:
        logger.error(f"Failed to load references: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "outline_body": None,
            "rules_body": None,
            "style_body": None,
            "characters_bodies": [],
            "outline_current_chapter_text": None,
            "loaded_references": {},
            "has_references": False,
            "error": str(e),
            "current_request": state.get("current_request", ""),  # Preserve even on error
            "cursor_offset": state.get("cursor_offset", -1),  # CRITICAL: Preserve cursor
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "user_id": state.get("user_id", "system"),  # CRITICAL: Preserve user_id
            "active_editor": state.get("active_editor", {}),  # CRITICAL: Preserve for subsequent nodes
            "manuscript": state.get("manuscript", ""),
            "filename": state.get("filename", ""),
            "frontmatter": state.get("frontmatter", {}),
            # ✅ CRITICAL: Preserve chapter context even on error
            "chapter_ranges": state.get("chapter_ranges", []),
            "active_chapter_idx": state.get("active_chapter_idx", -1),
            "working_chapter_index": state.get("working_chapter_index", -1),
            "current_chapter_text": state.get("current_chapter_text", ""),
            "current_chapter_number": state.get("current_chapter_number"),
            "prev_chapter_text": state.get("prev_chapter_text"),
            "prev_chapter_number": state.get("prev_chapter_number"),
            "next_chapter_text": state.get("next_chapter_text"),
            "next_chapter_number": state.get("next_chapter_number"),
            "explicit_primary_chapter": state.get("explicit_primary_chapter"),
            "explicit_secondary_chapters": state.get("explicit_secondary_chapters", []),
            # ✅ CRITICAL: Preserve critical 5 keys
            "metadata": state.get("metadata", {}),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }


async def assess_reference_quality_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Assess completeness of reference materials and provide guidance"""
    try:
        logger.info("Assessing reference quality...")
        
        outline_body = state.get("outline_body")
        rules_body = state.get("rules_body")
        style_body = state.get("style_body")
        characters_bodies = state.get("characters_bodies", [])
        # Get generation_mode from state (may be set later in workflow)
        generation_mode = state.get("generation_mode", "editing")
        
        reference_quality = {
            "has_outline": bool(outline_body),
            "has_rules": bool(rules_body),
            "has_style": bool(style_body),
            "has_characters": bool(characters_bodies),
            "completeness_score": 0.0
        }
        
        # Calculate completeness
        components = [outline_body, rules_body, style_body, characters_bodies]
        reference_quality["completeness_score"] = sum(1 for c in components if c) / len(components)
        
        warnings = []
        guidance_additions = []
        
        # Only warn for generation mode - editing can work without references
        if generation_mode == "generation":
            if not outline_body:
                warnings.append("⚠️ No outline found - generating without story structure guidance")
                guidance_additions.append(
                    "\n**NOTE:** No outline available. Generate content that continues "
                    "naturally from existing manuscript context and maintains consistency."
                )
            
            if not style_body:
                warnings.append("⚠️ No style guide found - using general fiction style")
                guidance_additions.append(
                    "\n**NOTE:** No style guide available. Infer narrative style from "
                    "existing manuscript and maintain consistency."
                )
            
            if not rules_body:
                warnings.append("⚠️ No universe rules found - no explicit worldbuilding constraints")
                guidance_additions.append(
                    "\n**NOTE:** No universe rules document. Infer world constraints from "
                    "existing manuscript and maintain internal consistency."
                )
            
            if not characters_bodies:
                warnings.append("⚠️ No character profiles found - inferring behavior from context")
                guidance_additions.append(
                    "\n**NOTE:** No character profiles available. Infer character traits "
                    "from existing manuscript and maintain behavioral consistency."
                )
        
        # Build additional guidance to add to LLM context
        reference_guidance = "".join(guidance_additions) if guidance_additions else ""
        
        has_references_final = state.get("has_references", False)
        
        # Preserve current_request and other important state from earlier nodes
        return {
            "reference_quality": reference_quality,
            "reference_warnings": warnings,
            "reference_guidance": reference_guidance,
            "current_request": state.get("current_request", ""),  # Preserve from prepare_context_node
            "has_references": has_references_final,  # Preserve from load_references_node
            "cursor_offset": state.get("cursor_offset", -1),  # CRITICAL: Preserve cursor
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "user_id": state.get("user_id", "system"),  # CRITICAL: Preserve user_id
            "active_editor": state.get("active_editor", {}),  # CRITICAL: Preserve for parent
            "manuscript": state.get("manuscript", ""),
            "filename": state.get("filename", ""),
            "frontmatter": state.get("frontmatter", {}),
            "outline_body": state.get("outline_body"),  # CRITICAL: Pass outline to parent!
            "rules_body": state.get("rules_body"),  # CRITICAL: Pass rules to parent!
            "style_body": state.get("style_body"),  # CRITICAL: Pass style to parent!
            "characters_bodies": state.get("characters_bodies", []),  # CRITICAL: Pass characters to parent!
            "outline_current_chapter_text": state.get("outline_current_chapter_text"),
            "loaded_references": state.get("loaded_references", {}),
            # ✅ CRITICAL: Preserve chapter context for parent agent
            "chapter_ranges": state.get("chapter_ranges", []),
            "active_chapter_idx": state.get("active_chapter_idx", -1),
            "working_chapter_index": state.get("working_chapter_index", -1),
            "current_chapter_text": state.get("current_chapter_text", ""),
            "current_chapter_number": state.get("current_chapter_number"),
            "prev_chapter_text": state.get("prev_chapter_text"),
            "prev_chapter_number": state.get("prev_chapter_number"),
            "next_chapter_text": state.get("next_chapter_text"),
            "next_chapter_number": state.get("next_chapter_number"),
            "explicit_primary_chapter": state.get("explicit_primary_chapter"),
            "explicit_secondary_chapters": state.get("explicit_secondary_chapters", []),
            # ✅ CRITICAL: Preserve critical 5 keys
            "metadata": state.get("metadata", {}),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }
        
    except Exception as e:
        logger.error(f"Reference assessment failed: {e}")
        return {
            "reference_quality": {"completeness_score": 0.0},
            "reference_warnings": [],
            "reference_guidance": "",
            "current_request": state.get("current_request", ""),  # Preserve even on error
            "has_references": state.get("has_references", False),
            "cursor_offset": state.get("cursor_offset", -1),  # CRITICAL: Preserve cursor
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "user_id": state.get("user_id", "system"),  # CRITICAL: Preserve user_id
            "active_editor": state.get("active_editor", {}),  # CRITICAL: Preserve for parent
            "manuscript": state.get("manuscript", ""),
            "filename": state.get("filename", ""),
            "frontmatter": state.get("frontmatter", {}),
            "outline_body": state.get("outline_body"),  # CRITICAL: Pass outline to parent!
            "rules_body": state.get("rules_body"),  # CRITICAL: Pass rules to parent!
            "style_body": state.get("style_body"),  # CRITICAL: Pass style to parent!
            "characters_bodies": state.get("characters_bodies", []),  # CRITICAL: Pass characters to parent!
            "outline_current_chapter_text": state.get("outline_current_chapter_text"),
            "loaded_references": state.get("loaded_references", {}),
            # ✅ CRITICAL: Preserve chapter context even on error
            "chapter_ranges": state.get("chapter_ranges", []),
            "active_chapter_idx": state.get("active_chapter_idx", -1),
            "working_chapter_index": state.get("working_chapter_index", -1),
            "current_chapter_text": state.get("current_chapter_text", ""),
            "current_chapter_number": state.get("current_chapter_number"),
            "prev_chapter_text": state.get("prev_chapter_text"),
            "prev_chapter_number": state.get("prev_chapter_number"),
            "next_chapter_text": state.get("next_chapter_text"),
            "next_chapter_number": state.get("next_chapter_number"),
            "explicit_primary_chapter": state.get("explicit_primary_chapter"),
            "explicit_secondary_chapters": state.get("explicit_secondary_chapters", []),
            # ✅ CRITICAL: Preserve critical 5 keys
            "metadata": state.get("metadata", {}),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }


# ============================================
# Subgraph Builder
# ============================================

def build_context_preparation_subgraph(checkpointer) -> StateGraph:
    """Build context preparation subgraph for fiction agents"""
    # Use Dict[str, Any] for state compatibility
    from typing import Dict, Any
    subgraph = StateGraph(Dict[str, Any])
    
    # Add nodes
    subgraph.add_node("prepare_context", prepare_context_node)
    subgraph.add_node("detect_chapter_mentions", detect_chapter_mentions_node)
    subgraph.add_node("analyze_scope", analyze_scope_node)
    subgraph.add_node("load_references", load_references_node)
    subgraph.add_node("assess_references", assess_reference_quality_node)
    
    # Set entry point
    subgraph.set_entry_point("prepare_context")
    
    # Flow
    subgraph.add_edge("prepare_context", "detect_chapter_mentions")
    subgraph.add_edge("detect_chapter_mentions", "analyze_scope")
    subgraph.add_edge("analyze_scope", "load_references")
    subgraph.add_edge("load_references", "assess_references")
    subgraph.add_edge("assess_references", END)
    
    return subgraph.compile(checkpointer=checkpointer)

