"""
Context Preparation Subgraph for Fiction Agents

Reusable subgraph that handles:
- Context extraction from active editor
- Chapter detection and scope analysis
- Reference loading (outline, style, rules, characters)
- Reference quality assessment

Can be used by fiction_editing_agent and other fiction-aware agents (proofreading path uses proofreading_subgraph).
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
    extract_chapter_outline,
    extract_story_overview,
    extract_book_map,
)
from orchestrator.utils.writing_subgraph_utilities import preserve_fiction_state

logger = logging.getLogger(__name__)


# ============================================
# State Schema
# ============================================

# Use Dict[str, Any] for compatibility with main agent state
FictionContextState = Dict[str, Any]


# ============================================
# Node Functions
# ============================================

def _get_active_editor_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Get active_editor from top-level state or shared_memory (nested subgraph may not have inject run)."""
    ae = state.get("active_editor") or {}
    if ae:
        return ae if isinstance(ae, dict) else {}
    sm = state.get("shared_memory") or {}
    ae = sm.get("active_editor") if isinstance(sm, dict) else None
    return (ae if isinstance(ae, dict) else {}) or {}


async def prepare_context_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare context: extract active editor, validate fiction type"""
    try:
        logger.info("Preparing context for fiction editing...")
        
        active_editor = _get_active_editor_from_state(state)
        
        manuscript = active_editor.get("content", "") or ""
        filename = active_editor.get("filename") or "manuscript.md"
        frontmatter = active_editor.get("frontmatter", {}) or {}
        
        cursor_offset = int(active_editor.get("cursor_offset", -1))
        selection_start = int(active_editor.get("selection_start", -1))
        selection_end = int(active_editor.get("selection_end", -1))
        
        # Note: Type checking is handled by the calling agent (fiction_editing_agent gates on fiction)
        # This subgraph is reusable and doesn't gate on document type
        
        # Extract user request: prefer state["query"] (current turn from writing assistant)
        # over messages[-1] so we always have the latest user input.
        metadata = state.get("metadata", {}) or {}
        messages = metadata.get("messages", [])
        if not messages:
            shared_memory = state.get("shared_memory", {}) or {}
            messages = shared_memory.get("messages", [])
        
        # DEBUG: Log what we're receiving
        query_from_state = state.get("query", "")
        logger.info(f"🔍 PREPARE CONTEXT: state['query'] = {repr(query_from_state)}")
        logger.info(f"🔍 PREPARE CONTEXT: len(messages) = {len(messages)}")
        
        current_request = (state.get("query") or "").strip()
        if not current_request and messages:
            try:
                latest_message = messages[-1]
                current_request = latest_message.content if hasattr(latest_message, 'content') else str(latest_message)
                current_request = (current_request or "").strip()
                logger.info(f"🔍 PREPARE CONTEXT: Used messages[-1], current_request = {repr(current_request)}")
            except Exception as e:
                logger.error(f"Exception extracting current_request from messages: {e}")
                current_request = ""
        else:
            logger.info(f"🔍 PREPARE CONTEXT: Used state['query'], current_request = {repr(current_request)}")
        
        # Early whole-story intent detection: skip chapter resolution when user wants full-manuscript analysis.
        # Negative guards override: edit/revise or "this chapter/scene" => keep chapter scope.
        analysis_scope = "chapter"
        if current_request:
            q = current_request.lower().strip()
            # Negative guards: any match forces chapter scope (user is editing/revising a specific part).
            chapter_scope_phrases = [
                "this chapter", "this scene", "this paragraph", "this section",
                "edit this", "revise this", "edit the chapter", "revise the chapter",
                "edit this chapter", "revise this chapter",
            ]
            if any(p in q for p in chapter_scope_phrases):
                analysis_scope = "chapter"
            elif re.search(r"\b(edit|revise|rewrite)\b", q):
                # Whole-word edit/revise/rewrite => actionable edit, not whole-story analysis.
                analysis_scope = "chapter"
            elif any(p in q for p in (
                "the outline", "in the outline", "in our outline", "our outline",
                "the style guide", "style guide", "in the style",
                "the rules", "our rules", "in the rules",
                "character sheet", "character notes", "in the character",
            )):
                # Reference-document query: user wants outline/rules/style/characters. Manuscript scope excludes these.
                analysis_scope = "chapter"
            else:
                # Only explicit whole-story / analysis phrases (no generic words like "discrepancies").
                whole_story_phrases = [
                    "whole story", "entire story", "full story", "our story", "the story so far",
                    "whole manuscript", "entire manuscript", "full manuscript",
                    "whole book", "entire book", "full book", "the book",
                    "analyze the story", "analyze our story", "analyze the book", "analyze the manuscript",
                    "across the book", "across chapters", "across the manuscript",
                    "character arcs", "narrative arc", "thematic consistency", "pacing across",
                    "review the manuscript", "review the story", "review the book",
                ]
                if any(p in q for p in whole_story_phrases):
                    analysis_scope = "manuscript"
                    logger.info("Whole-story intent detected early: analysis_scope=manuscript (skipping chapter resolution)")
        
        return preserve_fiction_state(state, {
            "active_editor": active_editor,
            "manuscript": manuscript,
            "manuscript_content": manuscript,
            "filename": filename,
            "frontmatter": frontmatter,
            "cursor_offset": cursor_offset,
            "selection_start": selection_start,
            "selection_end": selection_end,
            "current_request": current_request,
            "analysis_scope": analysis_scope,
        })
        
    except Exception as e:
        logger.error(f"Failed to prepare context: {e}")
        return preserve_fiction_state(state, {
            "error": str(e),
            "task_status": "error",
        })


async def detect_chapter_mentions_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Parse user query for explicit chapter mentions with intent-aware prioritization"""
    try:
        # Skip chapter detection when whole-manuscript analysis was detected early
        if state.get("analysis_scope") == "manuscript":
            logger.info("Skipping chapter detection (analysis_scope=manuscript)")
            return preserve_fiction_state(state, {
                "explicit_primary_chapter": None,
                "mentioned_chapters": [],
                "wants_next_chapter": False,
            })
        logger.info("Detecting chapter mentions in user query...")
        
        # DEBUG: Log what we received from prepare_context
        current_request = state.get("current_request", "")
        query_from_state = state.get("query", "")
        logger.info(f"🔍 DETECT CHAPTERS: state['current_request'] = {repr(current_request)}")
        logger.info(f"🔍 DETECT CHAPTERS: state['query'] = {repr(query_from_state)}")
        
        if not current_request:
            logger.info("No current_request - skipping chapter detection")
            return preserve_fiction_state(state, {
                "explicit_primary_chapter": None,
                "mentioned_chapters": [],
                "wants_next_chapter": False,
            })
        
        # Regex patterns for chapter detection with PRIORITY LEVELS
        # Priority 1: Explicit action verbs (edit intent) - HIGHEST
        # Priority 2: Generation verbs (create intent)
        # Priority 3: Context mentions (no direct action)
        CHAPTER_PATTERNS = [
            # PRIORITY 1: Editing/Action verbs + Chapter (STRONGEST INTENT)
            {
                "pattern": r'\b(?:Look over|Review|Check|Edit|Update|Revise|Modify|Address|Fix|Change|Rewrite|Polish|Proofread)\s+[Cc]hapter\s+(\d+)\b',
                "priority": 1,
                "description": "Action verb (editing intent)"
            },
            # PRIORITY 2: Generation verbs + Chapter (CREATE INTENT)
            {
                "pattern": r'\b(?:Generate|Craft|Write|Create|Draft|Compose|Produce|Add)\s+[Cc]hapter\s+(\d+)\b',
                "priority": 2,
                "description": "Generation verb (creation intent)"
            },
            # PRIORITY 2: Verb + in/at + Chapter (specific location action)
            {
                "pattern": r'\b(?:address|fix|change|edit|update|revise|modify|generate|craft|write|create)\s+(?:in|at)\s+[Cc]hapter\s+(\d+)\b',
                "priority": 2,
                "description": "Action verb with location preposition"
            },
            # PRIORITY 3: Preposition + Chapter (location reference only)
            {
                "pattern": r'\b(?:in|at|for|to)\s+[Cc]hapter\s+(\d+)\b',
                "priority": 3,
                "description": "Preposition (location reference)"
            },
            # PRIORITY 3: Chapter + modal/state verb (context description)
            {
                "pattern": r'\b[Cc]hapter\s+(\d+)\s+(?:needs|has|shows|contains|should|must|is|requires|will|would|can)',
                "priority": 3,
                "description": "Chapter + modal/state verb (context)"
            },
            # PRIORITY 3: Chapter + punctuation + relative clause (explanatory context)
            {
                "pattern": r'\b[Cc]hapter\s+(\d+)[:,]?\s+(?:where|when|that|which)',
                "priority": 3,
                "description": "Chapter + relative clause (explanation)"
            },
        ]
        
        all_mentions = []
        logger.info(f"🔍 CHAPTER DETECTION: Analyzing query '{current_request}'")
        
        for pattern_def in CHAPTER_PATTERNS:
            pattern = pattern_def["pattern"]
            priority = pattern_def["priority"]
            description = pattern_def["description"]
            
            matches = re.finditer(pattern, current_request, re.IGNORECASE)
            for match in matches:
                chapter_num = int(match.group(1))
                match_text = match.group(0)
                all_mentions.append({
                    "chapter": chapter_num,
                    "position": match.start(),
                    "text": match_text,
                    "priority": priority,
                    "description": description
                })
                logger.info(f"   📍 Found: '{match_text}' → Chapter {chapter_num} (priority {priority}: {description})")
        
        # Remove duplicates: keep HIGHEST priority match for each chapter
        chapter_best_match = {}
        for mention in all_mentions:
            chapter_num = mention["chapter"]
            if chapter_num not in chapter_best_match:
                chapter_best_match[chapter_num] = mention
            else:
                # Keep the higher priority match (lower priority number = higher priority)
                existing = chapter_best_match[chapter_num]
                if mention["priority"] < existing["priority"]:
                    logger.info(f"   🔄 Chapter {chapter_num}: Upgrading from priority {existing['priority']} ({existing['description']}) to priority {mention['priority']} ({mention['description']})")
                    chapter_best_match[chapter_num] = mention
        
        unique_mentions = list(chapter_best_match.values())
        
        # Sort by PRIORITY FIRST, then position (earlier mentions break ties)
        unique_mentions.sort(key=lambda x: (x["priority"], x["position"]))
        
        primary_chapter = None
        all_mentioned_chapters = []
        
        if unique_mentions:
            primary_match = unique_mentions[0]
            primary_chapter = primary_match["chapter"]
            all_mentioned_chapters = [m["chapter"] for m in unique_mentions]
            
            logger.info(f"📖 CHAPTER DETECTION RESULT:")
            logger.info(f"   🎯 PRIMARY TARGET: Chapter {primary_chapter}")
            logger.info(f"   📝 Reason: {primary_match['description']} (priority {primary_match['priority']})")
            logger.info(f"   💬 Match text: '{primary_match['text']}'")
            logger.info(f"   📍 Position in query: {primary_match['position']}")
            
            if len(all_mentioned_chapters) > 1:
                other_chapters = [ch for ch in all_mentioned_chapters if ch != primary_chapter]
                logger.info(f"   📚 OTHER MENTIONS: Chapters {other_chapters} (context references)")
                for mention in unique_mentions[1:]:
                    logger.info(f"      - Chapter {mention['chapter']}: '{mention['text']}' ({mention['description']})")
        else:
            logger.info(f"📖 NO CHAPTER MENTIONS DETECTED in '{current_request}'")
        
        # "Next chapter" (no number): resolve later from cursor + manuscript in analyze_scope
        wants_next_chapter = False
        next_chapter_pattern = re.compile(
            r'\b(?:Generate|Craft|Write|Create|Draft|Compose|Produce|Add)\s+(?:the\s+)?next\s+chapter\b',
            re.IGNORECASE
        )
        if next_chapter_pattern.search(current_request):
            wants_next_chapter = True
            logger.info(f"   📍 Found: 'craft/write/... (the) next chapter' → will resolve from cursor position")
        
        return preserve_fiction_state(state, {
            "explicit_primary_chapter": primary_chapter,
            "mentioned_chapters": all_mentioned_chapters,
            "wants_next_chapter": wants_next_chapter,
        })
        
    except Exception as e:
        logger.error(f"Failed to detect chapter mentions: {e}")
        return preserve_fiction_state(state, {
            "explicit_primary_chapter": None,
            "mentioned_chapters": [],
            "wants_next_chapter": False,
            "error": str(e),
        })


async def analyze_scope_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze chapter scope: find chapters, determine current/prev/next"""
    try:
        # Whole-manuscript analysis: skip cursor/chapter resolution, only compute chapter_ranges
        if state.get("analysis_scope") == "manuscript":
            manuscript = state.get("manuscript_content") or state.get("manuscript", "")
            chapter_ranges = find_chapter_ranges(manuscript)
            logger.info(
                "Skipping chapter resolution (analysis_scope=manuscript); chapter_ranges=%d chapters",
                len(chapter_ranges),
            )
            return preserve_fiction_state(state, {
                "chapter_ranges": chapter_ranges,
                "active_chapter_idx": -1,
                "working_chapter_index": -1,
                "current_chapter_text": "",
                "current_chapter_number": None,
                "requested_chapter_number": None,
                "prev_chapter_text": None,
                "prev_chapter_number": None,
                "next_chapter_text": None,
                "next_chapter_number": None,
                "reference_chapter_numbers": [],
                "reference_chapter_texts": {},
            })
        logger.info("Analyzing chapter scope...")
        
        # Support both manuscript_content (from subgraph) and manuscript (from main agent)
        manuscript = state.get("manuscript_content") or state.get("manuscript", "")
        cursor_offset = state.get("cursor_offset", -1)
        
        # If cursor is -1, treat it as end of document
        if cursor_offset == -1 and len(manuscript) > 0:
            cursor_offset = len(manuscript)
        
        # Get explicit chapter mentions from query
        explicit_primary_chapter = state.get("explicit_primary_chapter")
        mentioned_chapters = state.get("mentioned_chapters", [])  # All chapters mentioned in query
        wants_next_chapter = state.get("wants_next_chapter", False)
        
        # Find chapter ranges
        chapter_ranges = find_chapter_ranges(manuscript)
        
        # Resolve "the next chapter" to a number using cursor + manuscript (only when no numeric chapter in query)
        if wants_next_chapter and explicit_primary_chapter is None:
            if chapter_ranges:
                cursor_chapter_idx = locate_chapter_index(chapter_ranges, cursor_offset)
                if cursor_chapter_idx != -1:
                    next_num = chapter_ranges[cursor_chapter_idx].chapter_number + 1
                    logger.info(f"   📖 Resolved 'next chapter' to Chapter {next_num} (cursor is in Chapter {chapter_ranges[cursor_chapter_idx].chapter_number})")
                else:
                    next_num = chapter_ranges[-1].chapter_number + 1
                    logger.info(f"   📖 Resolved 'next chapter' to Chapter {next_num} (cursor after last chapter)")
                explicit_primary_chapter = next_num
            else:
                explicit_primary_chapter = 1
                logger.info(f"   📖 Resolved 'next chapter' to Chapter 1 (empty manuscript)")
        
        # DIAGNOSTIC: Log first 500 chars of manuscript to debug chapter detection
        if len(chapter_ranges) == 0 and len(manuscript) > 0:
            logger.warning(f"CHAPTER DETECTION FAILED - Manuscript preview (first 500 chars): {repr(manuscript[:500])}")
        
        # 🎯 NEW PRIORITY SYSTEM: Cursor position ALWAYS determines editable chapter
        # Mentioned chapters become READ-ONLY reference chapters
        active_idx = -1
        current_chapter_number: Optional[int] = None
        detection_method = "unknown"
        requested_chapter_number: Optional[int] = None  # For new chapter generation
        
        logger.info(f"🔍 CHAPTER TARGET SELECTION: Starting priority resolution...")
        logger.info(f"   📍 Cursor offset: {cursor_offset}")
        logger.info(f"   💬 Explicit primary from query: {explicit_primary_chapter}")
        logger.info(f"   📚 All mentioned chapters: {mentioned_chapters}")
        logger.info(f"   📖 Existing chapters in manuscript: {[r.chapter_number for r in chapter_ranges] if chapter_ranges else 'None'}")
        
        # 1. Cursor position vs explicit "craft/generate Chapter N" for a chapter that doesn't exist
        if cursor_offset >= 0:
            active_idx = locate_chapter_index(chapter_ranges, cursor_offset)
            if active_idx != -1:
                # Cursor is WITHIN a chapter - but if user explicitly asked to craft/generate a
                # different chapter that doesn't exist yet, honor that over cursor position
                explicit_chapter_exists = (
                    explicit_primary_chapter is not None
                    and any(r.chapter_number == explicit_primary_chapter for r in chapter_ranges)
                )
                if (
                    explicit_primary_chapter is not None
                    and not explicit_chapter_exists
                ):
                    # User said e.g. "Craft Chapter 2" and Chapter 2 doesn't exist - generate it
                    cursor_was_in_chapter = chapter_ranges[active_idx].chapter_number
                    current_chapter_number = explicit_primary_chapter
                    requested_chapter_number = explicit_primary_chapter
                    active_idx = -1  # No existing chapter for target; prev_c will be set from last chapter below
                    detection_method = "explicit_new_chapter_overrides_cursor"
                    logger.info(f"✅ PRIORITY 1 WIN: Explicit craft/generate (overrides cursor)")
                    logger.info(f"   🎯 Target: Chapter {explicit_primary_chapter} (NEW)")
                    logger.info(f"   💬 Reason: User asked to craft/generate Chapter {explicit_primary_chapter}; it does not exist yet")
                    logger.info(f"   📍 Note: Cursor was in Chapter {cursor_was_in_chapter}, but explicit request wins")
                else:
                    # Cursor is within a chapter and no conflicting explicit new-chapter request
                    current_chapter_number = chapter_ranges[active_idx].chapter_number
                    detection_method = "cursor_position"
                    logger.info(f"✅ PRIORITY 1 WIN: Cursor position")
                    logger.info(f"   🎯 Selected Chapter {current_chapter_number}")
                    logger.info(f"   📍 Reason: Cursor at offset {cursor_offset} is inside Chapter {current_chapter_number}")
                    if explicit_primary_chapter and explicit_primary_chapter != current_chapter_number:
                        logger.info(f"   ℹ️ Note: Ignoring query mention of Chapter {explicit_primary_chapter} (cursor position overrides)")
            elif len(chapter_ranges) > 0 and cursor_offset >= chapter_ranges[-1].end:
                # Cursor is AFTER all chapters - check if explicit action verb should win
                if explicit_primary_chapter is not None:
                    # User has explicit action verb (e.g., "Revise Chapter 1") - honor it!
                    # Check if the explicit chapter exists
                    explicit_chapter_exists = any(r.chapter_number == explicit_primary_chapter for r in chapter_ranges)
                    
                    if explicit_chapter_exists:
                        # Use the explicit chapter from query
                        for i, ch_range in enumerate(chapter_ranges):
                            if ch_range.chapter_number == explicit_primary_chapter:
                                active_idx = i
                                current_chapter_number = explicit_primary_chapter
                                detection_method = "explicit_action_verb_overrides_eof_cursor"
                                logger.info(f"✅ PRIORITY 1 WIN: Explicit action verb (overrides EOF cursor)")
                                logger.info(f"   🎯 Selected Chapter {current_chapter_number}")
                                logger.info(f"   💬 Reason: User said 'Revise Chapter {explicit_primary_chapter}' (explicit action verb)")
                                logger.info(f"   📍 Note: Cursor is at EOF, but explicit action verb takes precedence")
                                break
                    else:
                        # Explicit chapter doesn't exist - new chapter generation
                        current_chapter_number = explicit_primary_chapter
                        requested_chapter_number = explicit_primary_chapter
                        detection_method = "explicit_new_chapter_at_eof"
                        logger.info(f"✅ PRIORITY 1 WIN: New chapter generation")
                        logger.info(f"   🎯 Target: Chapter {explicit_primary_chapter} (NEW)")
                        logger.info(f"   💬 Reason: User requested Chapter {explicit_primary_chapter} which doesn't exist yet")
                        logger.info(f"   📍 Cursor at EOF is appropriate for new chapter generation")
                
                # If no explicit chapter or we didn't handle it above, fall back to last chapter
                if active_idx == -1 and current_chapter_number is None:
                    last_chapter = chapter_ranges[-1]
                    current_chapter_number = last_chapter.chapter_number
                    detection_method = "cursor_after_all_chapters"
                    logger.info(f"✅ PRIORITY 1 WIN: Cursor after all chapters (fallback)")
                    logger.info(f"   🎯 Selected Chapter {current_chapter_number} (last existing)")
                    logger.info(f"   📍 Reason: Cursor at offset {cursor_offset} is after last chapter (ends at {last_chapter.end})")
                    logger.info(f"   ℹ️ Note: No explicit action verb found, using last chapter as safe default")
        
        # 2. If no cursor-based chapter found, check explicit chapter (for new chapter generation)
        if active_idx == -1 and current_chapter_number is None:
            if explicit_primary_chapter is not None:
                # Check if explicit chapter exists
                for i, ch_range in enumerate(chapter_ranges):
                    if ch_range.chapter_number == explicit_primary_chapter:
                        active_idx = i
                        current_chapter_number = explicit_primary_chapter
                        detection_method = "explicit_query_fallback"
                        logger.info(f"✅ PRIORITY 2 WIN: Explicit query mention")
                        logger.info(f"   🎯 Selected Chapter {current_chapter_number}")
                        logger.info(f"   💬 Reason: User explicitly mentioned Chapter {explicit_primary_chapter} in query")
                        logger.info(f"   📍 Note: No cursor position available, using query intent")
                        break
                
                if active_idx == -1:
                    # Explicit chapter doesn't exist yet - this is a generation request
                    current_chapter_number = explicit_primary_chapter
                    requested_chapter_number = explicit_primary_chapter
                    detection_method = "explicit_new_chapter"
                    logger.info(f"✅ PRIORITY 2 WIN: New chapter generation")
                    logger.info(f"   🎯 Target: Chapter {explicit_primary_chapter} (NEW)")
                    logger.info(f"   💬 Reason: User requested Chapter {explicit_primary_chapter} which doesn't exist yet")
                    logger.info(f"   📝 Mode: Generation (will create new chapter)")
        
        # 3. Smart default for empty or ambiguous files
        if active_idx == -1 and current_chapter_number is None:
            if len(chapter_ranges) == 0:
                # Empty file or no chapters detected - default to Chapter 1
                current_chapter_number = 1
                requested_chapter_number = 1
                detection_method = "empty_file_default"
                logger.info(f"✅ PRIORITY 3 WIN: Default to Chapter 1")
                logger.info(f"   🎯 Selected Chapter 1 (default)")
                logger.info(f"   📄 Reason: Empty manuscript or no chapters detected")
            else:
                # Fallback to entire manuscript
                detection_method = "default_entire_manuscript"
                logger.info(f"⚠️ FALLBACK: Using entire manuscript")
                logger.info(f"   📄 Reason: Could not determine specific chapter target")
        
        # 🎯 NEW: Identify reference chapters (mentioned but not current/adjacent)
        reference_chapter_numbers = []
        reference_chapter_texts = {}  # Map chapter_number -> text
        
        if mentioned_chapters and current_chapter_number is not None:
            # Get adjacent chapter numbers
            prev_chapter_num = None
            next_chapter_num = None
            if active_idx != -1:
                prev_c, next_c = get_adjacent_chapters(chapter_ranges, active_idx)
                if prev_c:
                    prev_chapter_num = prev_c.chapter_number
                if next_c:
                    next_chapter_num = next_c.chapter_number
            
            # Filter mentioned chapters: include only if NOT current or adjacent
            for mentioned_ch in mentioned_chapters:
                if (mentioned_ch != current_chapter_number and 
                    mentioned_ch != prev_chapter_num and 
                    mentioned_ch != next_chapter_num):
                    reference_chapter_numbers.append(mentioned_ch)
                    logger.info(f"📖 Adding Chapter {mentioned_ch} as READ-ONLY reference (mentioned but not current/adjacent)")
            
            # Extract text for reference chapters
            for ref_ch_num in reference_chapter_numbers:
                for ch_range in chapter_ranges:
                    if ch_range.chapter_number == ref_ch_num:
                        ref_text = strip_frontmatter_block(manuscript[ch_range.start:ch_range.end])
                        reference_chapter_texts[ref_ch_num] = ref_text
                        logger.info(f"📖 Extracted {len(ref_text)} chars for reference Chapter {ref_ch_num}")
                        break
        
        prev_c, next_c = (None, None)
        # 🎯 ROOSEVELT FIX: Default to empty string for current chapter text if not found
        # This prevents the "whole manuscript fallback" that blows up token counts
        current_chapter_text = ""
        
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
        else:
            # Smart scouting for new chapter: get last chapter as prev context
            if len(chapter_ranges) > 0:
                prev_c = chapter_ranges[-1]
                logger.info(f"New chapter generation detected - using last Chapter {prev_c.chapter_number} as continuity context")
        
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
        elif not current_chapter_text:
            # Generation mode - empty current chapter is correct
            context_current_chapter_text = ""
        else:
            context_current_chapter_text = strip_frontmatter_block(current_chapter_text)
        
        # Log comprehensive chapter detection summary
        logger.info(f"=" * 80)
        logger.info(f"📖 FINAL CHAPTER DETECTION SUMMARY:")
        logger.info(f"   🎯 TARGET CHAPTER: {current_chapter_number}")
        logger.info(f"   🔧 DETECTION METHOD: {detection_method}")
        logger.info(f"   📍 ACTIVE INDEX: {active_idx} (in manuscript chapter list)")
        
        if requested_chapter_number:
            logger.info(f"   ✨ GENERATION MODE: Will create new Chapter {requested_chapter_number}")
        else:
            logger.info(f"   ✏️ EDIT MODE: Will edit existing Chapter {current_chapter_number}")
        
        logger.info(f"")
        logger.info(f"   📊 CONTEXT:")
        if chapter_ranges:
            logger.info(f"      - Manuscript contains chapters: {[r.chapter_number for r in chapter_ranges]}")
        else:
            logger.info(f"      - Manuscript contains NO chapters")
        logger.info(f"      - Cursor position: {cursor_offset}")
        logger.info(f"      - Query mentioned Chapter {explicit_primary_chapter}" if explicit_primary_chapter else "      - No explicit chapter in query")
        if mentioned_chapters and len(mentioned_chapters) > 1:
            other_mentions = [ch for ch in mentioned_chapters if ch != explicit_primary_chapter]
            logger.info(f"      - Other chapters mentioned (context): {other_mentions}")
        
        logger.info(f"")
        logger.info(f"   🔍 DECISION RATIONALE:")
        if detection_method == "cursor_position":
            logger.info(f"      ✅ Cursor is positioned in Chapter {current_chapter_number}")
            logger.info(f"      ✅ Cursor position ALWAYS wins (highest priority)")
            if explicit_primary_chapter and explicit_primary_chapter != current_chapter_number:
                logger.info(f"      ℹ️ Query mentioned Chapter {explicit_primary_chapter}, but cursor overrides")
        elif detection_method == "explicit_action_verb_overrides_eof_cursor":
            logger.info(f"      ✅ User explicitly requested editing Chapter {current_chapter_number}")
            logger.info(f"      ✅ Action verb ('Revise Chapter {current_chapter_number}') overrides EOF cursor")
            logger.info(f"      ℹ️ Cursor is at end-of-file, but explicit intent is clear")
        elif detection_method == "explicit_new_chapter_at_eof":
            logger.info(f"      ✅ User requested non-existent Chapter {explicit_primary_chapter}")
            logger.info(f"      ✅ Cursor at EOF is appropriate for new chapter generation")
        elif detection_method == "explicit_new_chapter_overrides_cursor":
            logger.info(f"      ✅ User asked to craft/generate Chapter {explicit_primary_chapter} (does not exist yet)")
            logger.info(f"      ✅ Explicit chapter request overrides cursor position")
        elif detection_method == "cursor_after_all_chapters":
            logger.info(f"      ✅ Cursor is after all chapters, using last chapter")
            logger.info(f"      ✅ Prevents accidental edits to wrong chapter")
            logger.info(f"      ℹ️ No explicit action verb found, safe default applied")
        elif detection_method == "explicit_query_fallback":
            logger.info(f"      ✅ No cursor position, using explicit mention from query")
            logger.info(f"      ✅ User action verb targeted Chapter {current_chapter_number}")
        elif detection_method == "explicit_new_chapter":
            logger.info(f"      ✅ User requested non-existent Chapter {explicit_primary_chapter}")
            logger.info(f"      ✅ Will generate new chapter content")
        elif detection_method == "empty_file_default":
            logger.info(f"      ✅ Empty manuscript, defaulting to Chapter 1")
        else:
            logger.info(f"      ⚠️ Fallback method used")
        
        logger.info(f"=" * 80)
        
        return preserve_fiction_state(state, {
            "chapter_ranges": chapter_ranges,
            "active_chapter_idx": active_idx,
            "working_chapter_index": active_idx,
            "current_chapter_text": context_current_chapter_text,
            "current_chapter_number": current_chapter_number,
            "requested_chapter_number": requested_chapter_number,
            "prev_chapter_text": prev_chapter_text,
            "prev_chapter_number": prev_chapter_number,
            "next_chapter_text": next_chapter_text,
            "next_chapter_number": next_chapter_number,
            "reference_chapter_numbers": reference_chapter_numbers,
            "reference_chapter_texts": reference_chapter_texts,
        })
        
    except Exception as e:
        logger.error(f"Failed to analyze scope: {e}")
        return preserve_fiction_state(state, {
            "error": str(e),
            "task_status": "error",
            "reference_chapter_numbers": [],
            "reference_chapter_texts": {},
        })


async def load_references_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Load referenced context files (outline, rules, style, characters)"""
    try:
        logger.info("Loading referenced context files...")
        
        from orchestrator.tools.reference_file_loader import load_referenced_files
        
        active_editor = _get_active_editor_from_state(state)
        user_id = state.get("user_id", "system")
        
        # Fiction reference configuration
        reference_config = {
            "outline": ["outline"],
            "series": ["series"]
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
        
        series_body = None
        if loaded_files.get("series") and len(loaded_files["series"]) > 0:
            series_body = loaded_files["series"][0].get("content")
        
        # Extract story overview and book map from outline
        story_overview = None
        book_map = None
        if outline_body:
            story_overview = extract_story_overview(outline_body)
            if story_overview:
                logger.info(f"Sending: outline.md -> Story Overview (Global narrative context)")
            
            book_map = extract_book_map(outline_body)
            if book_map:
                logger.info(f"Sending: outline.md -> Book Structure Map ({len(book_map)} chapters)")
        
        # Extract current chapter outline if we have chapter number
        outline_current_chapter_text = None
        current_chapter_number = state.get("current_chapter_number")
        if outline_body and current_chapter_number:
            outline_current_chapter_text = extract_chapter_outline(outline_body, current_chapter_number)
            if outline_current_chapter_text:
                logger.info(f"Sending: outline.md -> Chapter {current_chapter_number} (DETECTED)")
                logger.info(f"Extracted outline for Chapter {current_chapter_number} ({len(outline_current_chapter_text)} chars)")
            else:
                logger.info(f"Sending: outline.md -> Chapter {current_chapter_number} (NOT FOUND)")
                logger.warning(f"Failed to extract outline for Chapter {current_chapter_number} - regex pattern did not match")
                logger.warning(f"   This may indicate the outline format doesn't match expected pattern")
                logger.warning(f"   Outline preview: {outline_body[:200]}...")
                # DO NOT fall back to full outline - this would leak later chapters into earlier ones!
        
        # Extract previous chapter outline if we have previous chapter number
        outline_prev_chapter_text = None
        prev_chapter_number = state.get("prev_chapter_number")
        if outline_body and prev_chapter_number:
            outline_prev_chapter_text = extract_chapter_outline(outline_body, prev_chapter_number)
            if outline_prev_chapter_text:
                logger.info(f"Sending: outline.md -> Chapter {prev_chapter_number} (PREVIOUS)")
                logger.info(f"Extracted outline for Chapter {prev_chapter_number} ({len(outline_prev_chapter_text)} chars)")
        
        has_references_value = bool(outline_body)
        
        return preserve_fiction_state(state, {
            "outline_body": outline_body,
            "rules_body": rules_body,
            "style_body": style_body,
            "characters_bodies": characters_bodies,
            "series_body": series_body,
            "story_overview": story_overview,
            "book_map": book_map,
            "outline_current_chapter_text": outline_current_chapter_text,
            "outline_prev_chapter_text": outline_prev_chapter_text,
            "loaded_references": loaded_files,
            "has_references": has_references_value,
        })
        
    except Exception as e:
        logger.error(f"Failed to load references: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return preserve_fiction_state(state, {
            "outline_body": None,
            "rules_body": None,
            "style_body": None,
            "characters_bodies": [],
            "series_body": None,
            "story_overview": None,
            "book_map": None,
            "outline_current_chapter_text": None,
            "outline_prev_chapter_text": None,
            "loaded_references": {},
            "has_references": False,
            "error": str(e),
        })


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
        return preserve_fiction_state(state, {
            "reference_quality": reference_quality,
            "reference_warnings": warnings,
            "reference_guidance": reference_guidance,
        })
        
    except Exception as e:
        logger.error(f"Reference assessment failed: {e}")
        return preserve_fiction_state(state, {
            "reference_quality": {"completeness_score": 0.0},
            "reference_warnings": [],
            "reference_guidance": "",
            "error": str(e),
        })


# build_context_preparation_subgraph removed; fiction only via Writing Assistant → fiction_editing_subgraph (flat nodes from this module)

