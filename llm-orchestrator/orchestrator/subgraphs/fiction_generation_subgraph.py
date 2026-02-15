"""
Generation Subgraph for Fiction Editing

Reusable subgraph that handles:
- Context assembly (manuscript chapters, references, outlines)
- Prompt construction (system prompt + user message)
- LLM generation calls with structured output
- Output validation (outline copying detection, Pydantic validation)

Used by fiction_editing_agent for edit plan generation.
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from orchestrator.models.editor_models import ManuscriptEdit
from orchestrator.utils.fiction_utilities import (
    strip_frontmatter_block as _strip_frontmatter_block,
    unwrap_json_response as _unwrap_json_response,
    looks_like_outline_copied as _looks_like_outline_copied,
    extract_character_name as _extract_character_name,
    extract_chapter_outline,
    extract_story_overview,
    extract_book_map,
    extract_pacing_block,
)
from orchestrator.utils.anchor_correction import (
    auto_correct_operation_anchor,
    batch_correct_operations,
)
from orchestrator.utils.paragraph_numbering import number_paragraphs
from orchestrator.utils.writing_subgraph_utilities import sanitize_ai_response_for_history

logger = logging.getLogger(__name__)


# ============================================
# State Schema
# ============================================

# Use Dict[str, Any] for compatibility with main agent state
FictionGenerationState = Dict[str, Any]


# ============================================
# Node Functions
# ============================================

async def build_generation_context_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Build generation context: assemble manuscript chapters, references, outlines"""
    try:
        logger.debug("Building generation context...")
        
        manuscript = state.get("manuscript", "")
        filename = state.get("filename", "manuscript.md")
        frontmatter = state.get("frontmatter", {})
        current_request = state.get("current_request", "")
        
        current_chapter_text = state.get("current_chapter_text", "")
        current_chapter_number = state.get("current_chapter_number")
        prev_chapter_text = state.get("prev_chapter_text")
        next_chapter_text = state.get("next_chapter_text")
        reference_chapter_numbers = state.get("reference_chapter_numbers", [])
        reference_chapter_texts = state.get("reference_chapter_texts", {})
        
        outline_body = state.get("outline_body")
        rules_body = state.get("rules_body")
        style_body = state.get("style_body")
        characters_bodies = state.get("characters_bodies", [])
        outline_current_chapter_text = state.get("outline_current_chapter_text")
        
        selection_start = state.get("selection_start", -1)
        selection_end = state.get("selection_end", -1)
        cursor_offset = state.get("cursor_offset", -1)
        requested_chapter_number = state.get("requested_chapter_number")
        explicit_primary_chapter = state.get("explicit_primary_chapter")
        
        # DIAGNOSTIC: Log what we received
        logger.debug(f"📖 [CONTEXT] explicit_primary_chapter={explicit_primary_chapter}, requested_chapter_number={requested_chapter_number}, current_chapter_number={current_chapter_number}")
        
        # Initialize is_fresh_request to avoid UnboundLocalError
        is_fresh_request = False
        
        # CRITICAL: When user explicitly mentions a chapter, prioritize it over cursor position
        # This ensures "In Chapter 2..." edits Chapter 2, not the chapter where cursor is
        if explicit_primary_chapter is not None:
            # User explicitly mentioned a chapter - use it regardless of cursor position
            target_chapter_number = explicit_primary_chapter
            is_fresh_request = True  # Explicit chapter mention is always fresh
            logger.debug(f"📖 Using EXPLICIT chapter {explicit_primary_chapter} (user mentioned it in query, overriding cursor-based detection)")
        elif requested_chapter_number is not None:
            # Check if requested_chapter_number is "fresh" (matches explicit mention)
            is_fresh_request = (
                explicit_primary_chapter is not None and 
                requested_chapter_number == explicit_primary_chapter
            )
            if is_fresh_request:
                # User explicitly requested this chapter in the current query - use it
                target_chapter_number = requested_chapter_number
                logger.debug(f"📖 Using FRESH requested chapter {requested_chapter_number} (explicitly mentioned in current query)")
            else:
                # Stale requested_chapter_number - ignore it, use cursor-based detection
                target_chapter_number = current_chapter_number
                logger.debug(f"📖 Ignoring STALE requested_chapter_number={requested_chapter_number} (not in current query), using cursor-based current_chapter_number={current_chapter_number}")
        else:
            # Use cursor-based detection (current_chapter_number from cursor position)
            target_chapter_number = current_chapter_number
            logger.debug(f"📖 Using cursor-based current_chapter_number={current_chapter_number}")
        
        # Determine chapter labels with actual chapter numbers
        prev_chapter_number = state.get("prev_chapter_number")
        next_chapter_number = state.get("next_chapter_number")
        
        # Use target_chapter_number (which respects freshness check) for label
        if target_chapter_number is not None:
            current_chapter_label = f"Chapter {target_chapter_number}"
        else:
            current_chapter_label = "Current Chapter"
        
        prev_chapter_label = f"Chapter {prev_chapter_number}" if prev_chapter_number else None
        next_chapter_label = f"Chapter {next_chapter_number}" if next_chapter_number else None
        
        # Check if file is empty (only frontmatter)
        body_only = _strip_frontmatter_block(manuscript)
        is_empty_file = not body_only.strip()
        
        # Build context message
        context_parts = [
            "=== MANUSCRIPT CONTEXT ===\n",
            f"Primary file: {filename}\n",
            f"Working area: {current_chapter_label}\n",
        ]
        
        if is_empty_file:
            context_parts.append("EMPTY FILE DETECTED: This file contains only frontmatter (no chapters yet)\n")
            context_parts.append("If creating the first chapter, use 'insert_after_heading' WITHOUT 'anchor_text' - it will insert after frontmatter.\n")
            context_parts.append("Example: {\"op_type\": \"insert_after_heading\", \"text\": \"## Chapter 1\\n\\n[your chapter content]\"}\n\n")
        else:
            context_parts.append(f"Cursor position: paragraph shown below\n\n")
        
        # Debug logging: Track context structure
        context_structure = {
            "sections": [],
            "total_length": 0
        }
        
        # Include previously generated chapters for continuity (multi-chapter mode)
        generated_chapters = state.get("generated_chapters", {})
        is_multi_chapter = state.get("is_multi_chapter", False)
        
        if is_multi_chapter and generated_chapters:
            # Add all previously generated chapters for continuity
            context_parts.append("=== PREVIOUSLY GENERATED CHAPTERS (FOR CONTINUITY - DO NOT EDIT) ===\n")
            for ch_num in sorted(generated_chapters.keys()):
                if ch_num < target_chapter_number:
                    context_parts.append(f"=== Chapter {ch_num} (Previously Generated) ===\n{generated_chapters[ch_num]}\n\n")
            context_parts.append("CRITICAL: Maintain continuity with these previously generated chapters!\n")
            context_parts.append("Ensure character states, plot threads, and story flow connect seamlessly.\n\n")
        
        # 🎯 ROOSEVELT'S CONTEXT PRUNING: Only include immediately adjacent chapters if the file is large
        # This prevents the 150k token context bloat and reduces "log leakage"
        manuscript_len = len(manuscript)
        context_parts.append(f"Manuscript Status: Large document ({manuscript_len:,} chars). Providing local context only.\n\n")
        
        # Extract last line of previous chapter for new chapter insertion guidance
        prev_chapter_last_line = None
        if prev_chapter_text and is_fresh_request:
            # This is a new chapter generation - extract the last line for anchor guidance
            prev_lines = prev_chapter_text.strip().split('\n')
            # Get last non-empty line
            for line in reversed(prev_lines):
                if line.strip():
                    prev_chapter_last_line = line.strip()
                    break
        
        if prev_chapter_text:
            section_header = f"=== MANUSCRIPT TEXT: {prev_chapter_label.upper()} (PREVIOUS - READ ONLY, FOR CONTINUITY AND ANCHORS) ===\n"
            context_parts.append(section_header)
            context_parts.append("⚠️ READ ONLY - DO NOT EDIT THIS CHAPTER ⚠️\n")
            context_parts.append("This chapter is provided for context and continuity reference ONLY.\n")
            context_parts.append("**EDITING RESTRICTIONS:**\n")
            context_parts.append("- **YOU MUST NOT EDIT, MODIFY, OR CHANGE ANY TEXT IN THIS CHAPTER.**\n")
            context_parts.append("- You may use text from this chapter as 'anchor_text' for operations in other chapters.\n")
            context_parts.append("- **ANALYSIS ALLOWED**: You CAN and SHOULD analyze this chapter and report any issues you find to the user.\n")
            context_parts.append("- If you spot inconsistencies, errors, or problems in this READ ONLY chapter, mention them in your 'summary' field.\n")
            context_parts.append("- **DO NOT create operations to fix issues in this chapter** - only report them to the user.\n\n")
            
            # For new chapter generation, emphasize continuity assessment
            if is_fresh_request:
                context_parts.append("**CRITICAL FOR NEW CHAPTER GENERATION:**\n")
                context_parts.append(f"READ THIS CHAPTER CAREFULLY to understand where the story left off!\n")
                context_parts.append(f"Your new Chapter {target_chapter_number} must pick up the narrative thread naturally.\n")
                context_parts.append(f"- What was the last scene/location/emotional state?\n")
                context_parts.append(f"- Where are the characters physically and emotionally?\n")
                context_parts.append(f"- What narrative momentum exists to build on?\n")
                context_parts.append(f"- DO NOT repeat the same scene - continue forward!\n\n")
                
                if prev_chapter_last_line:
                    context_parts.append(f"**LAST LINE OF {prev_chapter_label.upper()}:**\n")
                    context_parts.append(f'"{prev_chapter_last_line}"\n\n')
                    context_parts.append(f"⚠️ CRITICAL: Use this EXACT line as your 'anchor_text' for insertion!\n")
                    context_parts.append(f"DO NOT paraphrase, summarize, or invent anchor text - copy this line VERBATIM!\n")
                    context_parts.append(f"Your new chapter will be inserted immediately after this line.\n\n")
            
            context_parts.append(f"{prev_chapter_text}\n\n")
            context_structure["sections"].append({
                "type": "manuscript_prev_chapter",
                "heading": section_header.strip(),
                "content_length": len(prev_chapter_text),
                "chapter_number": prev_chapter_number,
                "last_line": prev_chapter_last_line
            })
        
        section_header = f"=== MANUSCRIPT TEXT: {current_chapter_label.upper()} (CURRENT - EDITABLE, USE FOR ANCHORS) ===\n"
        context_parts.append(section_header)
        # Number paragraphs for two-phase editing; store map for paragraph-edit path
        paragraph_map: List[Any] = []
        if current_chapter_text.strip():
            numbered_current, paragraph_map = number_paragraphs(current_chapter_text)
            context_parts.append(f"{numbered_current}\n\n")
        else:
            context_parts.append(f"{current_chapter_text}\n\n")
        context_structure["sections"].append({
            "type": "manuscript_current_chapter",
            "heading": section_header.strip(),
            "content_length": len(current_chapter_text),
            "chapter_number": current_chapter_number
        })
        
        if next_chapter_text:
            section_header = f"=== MANUSCRIPT TEXT: {next_chapter_label.upper()} (NEXT - READ ONLY, FOR CONTEXT ONLY, USE FOR ANCHORS IF NEEDED) ===\n"
            context_parts.append(section_header)
            context_parts.append("⚠️ READ ONLY - DO NOT EDIT THIS CHAPTER ⚠️\n")
            context_parts.append("This chapter is provided for context and transition awareness ONLY.\n")
            context_parts.append("**EDITING RESTRICTIONS:**\n")
            context_parts.append("- **YOU MUST NOT EDIT, MODIFY, OR CHANGE ANY TEXT IN THIS CHAPTER.**\n")
            context_parts.append("- You may use text from this chapter as 'anchor_text' for operations in other chapters.\n")
            context_parts.append("- **ANALYSIS ALLOWED**: You CAN and SHOULD analyze this chapter and report any issues you find to the user.\n")
            context_parts.append("- If you spot inconsistencies, errors, or problems in this READ ONLY chapter, mention them in your 'summary' field.\n")
            context_parts.append("- **DO NOT create operations to fix issues in this chapter** - only report them to the user.\n\n")
            context_parts.append(f"{next_chapter_text}\n\n")
            context_structure["sections"].append({
                "type": "manuscript_next_chapter",
                "heading": section_header.strip(),
                "content_length": len(next_chapter_text),
                "chapter_number": next_chapter_number
            })
        
        # 🎯 NEW: Include reference chapters (mentioned but not current/adjacent) as READ-ONLY
        reference_chapter_numbers = state.get("reference_chapter_numbers", [])
        reference_chapter_texts = state.get("reference_chapter_texts", {})
        
        if reference_chapter_numbers:
            for ref_ch_num in sorted(reference_chapter_numbers):
                ref_text = reference_chapter_texts.get(ref_ch_num)
                if ref_text:
                    ref_label = f"Chapter {ref_ch_num}"
                    section_header = f"=== MANUSCRIPT TEXT: {ref_label.upper()} (REFERENCE - READ ONLY, MENTIONED IN QUERY) ===\n"
                    context_parts.append(section_header)
                    context_parts.append("⚠️ READ ONLY - DO NOT EDIT THIS CHAPTER ⚠️\n")
                    context_parts.append(f"This chapter was mentioned in your query but is not the current editable chapter.\n")
                    context_parts.append("It is provided for reference and context ONLY.\n")
                    context_parts.append("**EDITING RESTRICTIONS:**\n")
                    context_parts.append("- **YOU MUST NOT EDIT, MODIFY, OR CHANGE ANY TEXT IN THIS CHAPTER.**\n")
                    context_parts.append("- You may use text from this chapter as 'anchor_text' for operations in other chapters.\n")
                    context_parts.append("- **ANALYSIS ALLOWED**: You CAN and SHOULD analyze this chapter and report any issues you find to the user.\n")
                    context_parts.append("- If you spot inconsistencies, errors, or problems in this READ ONLY chapter, mention them in your 'summary' field.\n")
                    context_parts.append("- **DO NOT create operations to fix issues in this chapter** - only report them to the user.\n\n")
                    context_parts.append(f"{ref_text}\n\n")
                    context_structure["sections"].append({
                        "type": "manuscript_reference_chapter",
                        "heading": section_header.strip(),
                        "content_length": len(ref_text),
                        "chapter_number": ref_ch_num
                    })
                    logger.debug(f"📖 Added reference Chapter {ref_ch_num} as READ-ONLY ({len(ref_text)} chars)")
        
        # Close manuscript section explicitly
        context_parts.append("=== END OF MANUSCRIPT CONTEXT ===\n")
        context_parts.append("="*80 + "\n")
        context_parts.append("⚠️ CRITICAL BOUNDARY: MANUSCRIPT TEXT ENDS HERE ⚠️\n")
        context_parts.append("="*80 + "\n")
        context_parts.append("All text ABOVE this line is MANUSCRIPT TEXT (use for anchors and text matching)\n")
        context_parts.append("All text BELOW this line is REFERENCE DOCUMENTS (use for story context, NOT for text matching)\n\n")
        context_parts.append("**CHAPTER EDITING RESTRICTIONS:**\n")
        context_parts.append(f"✅ **ONLY EDIT**: Chapter {target_chapter_number if target_chapter_number else 'CURRENT'} (marked as 'CURRENT - EDITABLE')\n")
        context_parts.append("❌ **READ ONLY**: PREVIOUS, NEXT, and REFERENCE chapters (marked as 'READ ONLY') - DO NOT EDIT THESE!\n")
        context_parts.append("⚠️ **CRITICAL**: If PREVIOUS, NEXT, or REFERENCE chapters are shown above, they are for context reference ONLY.\n")
        context_parts.append("   - You may use their text as 'anchor_text' for operations in the current chapter.\n")
        context_parts.append("   - You MUST NOT create operations that modify READ ONLY chapters.\n")
        context_parts.append("   - **BUT**: You CAN and SHOULD analyze READ ONLY chapters and report issues to the user in your 'summary'.\n\n")
        context_parts.append("**REMINDER**: If you need to use 'original_text' or 'anchor_text' in your operations:\n")
        context_parts.append("✅ Copy from MANUSCRIPT TEXT sections above (marked with chapter numbers)\n")
        context_parts.append("✅ **ONLY use text from the CURRENT/EDITABLE chapter** for 'original_text' in your operations\n")
        context_parts.append("❌ DO NOT copy from OUTLINE, STORY OVERVIEW, or CHARACTER PROFILES below\n")
        context_parts.append("❌ Reference documents below do NOT exist in the manuscript file!\n\n")
        
        # Log manuscript section boundary
        manuscript_end_marker = "=== END OF MANUSCRIPT CONTEXT ==="
        context_structure["manuscript_end_marker"] = manuscript_end_marker
        
        # References (if present, add usage guidance)
        has_any_refs = bool(style_body or rules_body or characters_bodies or outline_body)
        if has_any_refs:
            context_parts.append("=== REFERENCE DOCUMENTS (NOT MANUSCRIPT TEXT — DO NOT USE FOR ANCHORS) ===\n")
            context_parts.append("="*80 + "\n")
            context_parts.append("⚠️ THESE ARE PLANNING DOCUMENTS — NOT MANUSCRIPT TEXT ⚠️\n")
            context_parts.append("="*80 + "\n")
            context_parts.append("**DO NOT** copy text from these sections into 'original_text' or 'anchor_text' fields!\n")
            context_parts.append("These documents DO NOT exist in the manuscript file.\n\n")
            context_parts.append("**Reference Hierarchy (in order of creative authority):**\n")
            context_parts.append("1. STYLE GUIDE — Your voice. Every word choice and sentence must conform to it.\n")
            context_parts.append("2. OUTLINE — Your story knowledge. What happens, not how to write it.\n")
            context_parts.append("3. CHARACTER PROFILES — Authentic character details for dialogue and behavior.\n")
            context_parts.append("4. UNIVERSE RULES — World-building constraints that must not be violated.\n\n")
        
        if style_body:
            context_parts.append("="*80 + "\n")
            context_parts.append("★ STYLE GUIDE — PRIMARY CREATIVE AUTHORITY ★\n")
            context_parts.append("="*80 + "\n")
            context_parts.append("READ THIS FIRST. This defines how you write — your voice, tone, pacing,\n")
            context_parts.append("sentence length, word choices, what to avoid, and narrative technique.\n")
            context_parts.append("If a writing sample is included, study it: match its rhythms, density,\n")
            context_parts.append("and feel. Your prose must read as though it came from the same author.\n")
            context_parts.append("When the Style Guide conflicts with your instincts, the Style Guide wins.\n\n")
            context_parts.append(f"{style_body}\n\n")
            context_parts.append("="*80 + "\n")
            context_parts.append("END OF STYLE GUIDE — Every sentence you write must conform to the above.\n")
            context_parts.append("="*80 + "\n\n")
        
        if rules_body:
            context_parts.append("=== RULES (universe constraints) ===\n")
            context_parts.append("Use these rules to ensure world-building consistency in narrative prose:\n")
            context_parts.append(f"{rules_body}\n\n")
        
        if characters_bodies:
            context_parts.append("=== CHARACTER PROFILES ===\n")
            context_parts.append("Use these profiles when writing character appearances, actions, dialogue, and internal thoughts:\n")
            context_parts.append("**CRITICAL**: Each character profile below is for a DIFFERENT character. Pay careful attention to which dialogue patterns, traits, and behaviors belong to which character.\n\n")
            
            for i, char_body in enumerate(characters_bodies, 1):
                char_name = _extract_character_name(char_body)
                context_parts.append("="*60 + "\n")
                context_parts.append(f"CHARACTER PROFILE {i}: {char_name}\n")
                context_parts.append("="*60 + "\n")
                context_parts.append(f"{char_body}\n")
                context_parts.append("="*60 + "\n")
                context_parts.append(f"END OF PROFILE FOR {char_name}\n\n")
            
            context_parts.append("**REMINDER**: Each character has distinct dialogue patterns, traits, and behaviors. When writing dialogue or character actions, ensure you match the correct character's profile.\n\n")
        
        # Include outline for story context with full continuity support
        # Show previous, current (emphasized), and next chapter outlines for better continuity
        if outline_body and target_chapter_number:
            logger.debug(f"Drafting outline.md content for Chapter {target_chapter_number} and surroundings")
            
            # Extract and include story overview (synopsis before first chapter)
            story_overview = state.get("story_overview")
            if story_overview is None:
                story_overview = extract_story_overview(outline_body)
            
            if story_overview:
                logger.debug(f"Sending: outline.md -> Story Overview (Global narrative context)")
                context_parts.append("="*80 + "\n")
                context_parts.append("STORY OVERVIEW AND NARRATIVE THEMES\n")
                context_parts.append("="*80 + "\n")
                context_parts.append("=== STORY OVERVIEW (GLOBAL CONTEXT - READ FIRST) ===\n\n")
                context_parts.append("⚠️ THIS IS A PLANNING DOCUMENT - NOT MANUSCRIPT TEXT ⚠️\n")
                context_parts.append("DO NOT copy text from this section into 'original_text' or 'anchor_text' fields!\n\n")
                context_parts.append("**CRITICAL**: This is the high-level overview of the entire story.\n")
                context_parts.append("**PURPOSE**: Understand the overarching narrative, themes, and story goals BEFORE writing any chapter.\n")
                context_parts.append("**USE THIS TO**: Ensure every chapter you write serves the larger story arc and maintains thematic consistency.\n")
                context_parts.append("**DO NOT** lose sight of these themes and goals when writing individual chapters!\n\n")
                context_parts.append(f"{story_overview}\n\n")
                context_parts.append("=== END OF STORY OVERVIEW ===\n")
                context_parts.append("="*80 + "\n\n")
            
            # Extract and include book structure map
            book_map = state.get("book_map")
            if book_map is None:
                book_map = extract_book_map(outline_body)
            
            if book_map:
                logger.debug(f"Sending: outline.md -> Book Structure Map ({len(book_map)} sections)")
                context_parts.append("=== BOOK STRUCTURE MAP ===\n\n")
                context_parts.append("This shows the complete structure of the book. Use this to understand where the current chapter fits in the larger narrative arc.\n\n")
                for section_id, header_text in book_map:
                    if isinstance(section_id, str):
                        # Special section (Introduction, Prologue, Epilogue)
                        marker = " <-- YOU ARE HERE" if section_id.lower() == str(target_chapter_number).lower() else ""
                        context_parts.append(f"  {section_id}: {header_text}{marker}\n")
                    else:
                        # Numbered chapter
                        marker = " <-- YOU ARE HERE" if section_id == target_chapter_number else ""
                        context_parts.append(f"  Chapter {section_id}: {header_text}{marker}\n")
                context_parts.append("\n=== END OF BOOK STRUCTURE MAP ===\n\n")
            
            warning_banner = "\n" + "="*80 + "\n" + "STORY OUTLINE — YOUR STORY KNOWLEDGE (NOT MANUSCRIPT TEXT)\n" + "="*80 + "\n"
            context_parts.append(warning_banner)
            context_parts.append("=== STORY OUTLINE (WHAT HAPPENS — NOT FOR TEXT MATCHING) ===\n\n")
            context_parts.append("The outline below is your story knowledge. Read it to understand what\n")
            context_parts.append("happens, then write the chapter as original prose in the Style Guide's voice.\n\n")
            context_parts.append("**CRITICAL REMINDERS:**\n")
            context_parts.append("- Outline text does NOT exist in the manuscript — never use for anchors or text matching\n")
            context_parts.append("- Never copy or paraphrase outline language into your prose\n")
            context_parts.append("- These are story ingredients — craft them into scenes using the Style Guide's voice\n")
            context_parts.append("- For text matching, ONLY use text from MANUSCRIPT TEXT sections above\n\n")
            
            # Include previous chapter outline (extracted by context subgraph)
            outline_prev_chapter_text = state.get("outline_prev_chapter_text")
            prev_chapter_number = state.get("prev_chapter_number")
            if outline_prev_chapter_text and prev_chapter_number:
                logger.debug(f"Sending: outline.md -> Chapter {prev_chapter_number} (PREVIOUS)")
                context_parts.append(f"=== OUTLINE: CHAPTER {prev_chapter_number} (PREVIOUS - FOR CONTINUITY CONTEXT) ===\n")
                context_parts.append(f"This shows what happened in the previous chapter for continuity reference.\n")
                context_parts.append(f"Use this to ensure smooth transitions and character state consistency.\n\n")
                
                # Extract pacing from previous chapter to understand where it ended emotionally
                prev_pacing = extract_pacing_block(outline_prev_chapter_text)
                if prev_pacing and prev_pacing.get("to_state"):
                    context_parts.append(f"**PREVIOUS CHAPTER EMOTIONAL STATE**:\n")
                    context_parts.append(f"The previous chapter (Chapter {prev_chapter_number}) ended with this emotional/tonal state:\n")
                    context_parts.append(f"  {prev_pacing['to_state']}\n")
                    context_parts.append(f"This is where your new chapter should BEGIN emotionally.\n\n")
                
                context_parts.append(f"{outline_prev_chapter_text}\n\n")
                context_parts.append(f"=== END OF CHAPTER {prev_chapter_number} OUTLINE ===\n\n")
            
            # Extract and emphasize CURRENT chapter outline
            if outline_current_chapter_text:
                logger.debug(f"Sending: outline.md -> Chapter {target_chapter_number} (CURRENT TARGET)")
                context_parts.append("="*80 + "\n")
                context_parts.append(f"CURRENT CHAPTER OUTLINE - PRIMARY FOCUS\n")
                context_parts.append("="*80 + "\n")
                context_parts.append(f"=== OUTLINE: CHAPTER {target_chapter_number} (CURRENT - THIS IS YOUR PRIMARY FOCUS) ===\n\n")
                context_parts.append(f"⚠️ THIS IS OUTLINE TEXT - NOT MANUSCRIPT TEXT - DO NOT USE FOR ANCHORS ⚠️\n\n")
                context_parts.append(f"**CRITICAL**: You are generating Chapter {target_chapter_number} RIGHT NOW.\n")
                context_parts.append(f"**MANDATORY CHAPTER HEADER**: Your generated text MUST start with '## Chapter {target_chapter_number}' (NOT Chapter {target_chapter_number - 1}, NOT Chapter {target_chapter_number + 1}, ONLY Chapter {target_chapter_number}!)\n")
                context_parts.append(f"**PRIMARY TASK**: Write Chapter {target_chapter_number} as a cohesive narrative in the Style Guide's voice.\n")
                context_parts.append(f"The outline below tells you what happens. The Style Guide (above) tells you how to write it.\n")
                context_parts.append(f"**DO NOT** include events, characters, or plot points from LATER chapters in this chapter!\n")
                context_parts.append(f"**DO NOT** use a different chapter number in your header - it MUST be Chapter {target_chapter_number}!\n")
                context_parts.append(f"**DO NOT** copy text from this outline into 'original_text' or 'anchor_text' fields!\n\n")
                context_parts.append(f"**OUTLINE STRUCTURE NOTE**: This outline uses structured headers:\n")
                context_parts.append(f"  - '### Status' (if present): State of tracked items at chapter beginning\n")
                context_parts.append(f"  - '### Pacing' (if present): Emotional/tonal transition guidance\n")
                context_parts.append(f"  - '### Summary': High-level overview of the chapter's narrative arc\n")
                context_parts.append(f"  - '### Beats': The events that occur in this chapter\n")
                context_parts.append(f"  These are story INGREDIENTS — raw material for you to craft into scenes.\n")
                context_parts.append(f"  Combine, reorder, interleave, or restructure beats into cohesive scenes.\n")
                context_parts.append(f"  Not every beat needs its own scene or paragraph. Let the Style Guide's\n")
                context_parts.append(f"  pacing determine how much space each moment gets.\n\n")
                
                # Extract and emphasize pacing guidance if present
                current_pacing = extract_pacing_block(outline_current_chapter_text)
                if current_pacing:
                    context_parts.append(f"**PACING TRANSITION GUIDANCE** (CRITICAL FOR SMOOTH TRANSITIONS):\n")
                    if current_pacing.get("from_state"):
                        context_parts.append(f"  FROM: {current_pacing['from_state']}\n")
                        context_parts.append(f"  This tells you where the previous chapter left off emotionally.\n")
                        context_parts.append(f"  Your chapter should BEGIN in this emotional/tonal state.\n\n")
                    if current_pacing.get("to_state"):
                        context_parts.append(f"  TO: {current_pacing['to_state']}\n")
                        context_parts.append(f"  This tells you where this chapter should END emotionally.\n")
                        context_parts.append(f"  Your chapter should build toward this emotional/tonal state.\n\n")
                    if current_pacing.get("technique"):
                        context_parts.append(f"  TECHNIQUE: {current_pacing['technique']}\n")
                        context_parts.append(f"  This tells you HOW to transition between the FROM and TO states.\n")
                        context_parts.append(f"  Use this as guidance for pacing, scene structure, and emotional development.\n\n")
                    elif current_pacing.get("full_text"):
                        context_parts.append(f"  PACING GUIDANCE: {current_pacing['full_text']}\n")
                        context_parts.append(f"  Use this to understand the emotional arc and transition approach for this chapter.\n\n")
                    context_parts.append(f"**REMEMBER**: Pacing guidance helps create smooth transitions and prevents jarring tonal shifts.\n")
                    context_parts.append(f"Interpret this creatively while achieving the transition goals - it's guidance, not a script.\n\n")
                
                context_parts.append(f"{outline_current_chapter_text}\n\n")
                context_parts.append(f"=== END OF CHAPTER {target_chapter_number} OUTLINE (CURRENT) ===\n")
                context_parts.append("="*80 + "\n")
                context_parts.append(f"REMEMBER: You are generating Chapter {target_chapter_number} - your header MUST be '## Chapter {target_chapter_number}'!\n")
                context_parts.append("="*80 + "\n\n")
                
                context_structure["sections"].append({
                    "type": "reference_outline_current_chapter",
                    "heading": f"OUTLINE: CHAPTER {target_chapter_number} (CURRENT)",
                    "content_length": len(outline_current_chapter_text),
                    "has_warning_banner": True,
                    "warning": "OUTLINE TEXT - NOT MANUSCRIPT TEXT",
                    "chapter_number": target_chapter_number,
                    "is_emphasized": True
                })
            else:
                logger.warning(f"Could not extract outline for current Chapter {target_chapter_number}")
            
            # Extract next chapter outline (for transition planning)
            # SMART SCUTING: If state doesn't have next_chapter_number, assume target_chapter_number + 1
            # to provide forward-looking context from the outline even for new manuscripts.
            next_chapter_num_to_extract = state.get("next_chapter_number")
            if next_chapter_num_to_extract is None and target_chapter_number is not None:
                next_chapter_num_to_extract = target_chapter_number + 1
                
            if next_chapter_num_to_extract:
                next_outline = extract_chapter_outline(outline_body, next_chapter_num_to_extract)
                if next_outline:
                    logger.debug(f"Sending: outline.md -> Chapter {next_chapter_num_to_extract} (NEXT - for transition awareness)")
                    context_parts.append(f"=== OUTLINE: CHAPTER {next_chapter_num_to_extract} (NEXT - FOR TRANSITION PLANNING ONLY) ===\n")
                    context_parts.append(f"This shows what will happen in the next chapter for transition planning.\n")
                    context_parts.append(f"**CRITICAL**: Use this ONLY to plan smooth transitions and set up future events.\n")
                    context_parts.append(f"**DO NOT** include events from Chapter {next_chapter_num_to_extract} in Chapter {target_chapter_number}!\n")
                    context_parts.append(f"**DO** use this to ensure your chapter ending sets up the next chapter naturally.\n\n")
                    context_parts.append(f"{next_outline}\n\n")
                    context_parts.append(f"=== END OF CHAPTER {next_chapter_num_to_extract} OUTLINE ===\n\n")
                else:
                    logger.debug(f"No outline found for Chapter {next_chapter_num_to_extract} (skipping NEXT context)")
            
            context_parts.append("=== END OF STORY OUTLINE ===\n\n")
            context_parts.append(f"**NOW WRITE**: You know the story (outline above) and you know the voice (Style Guide above).\n")
            context_parts.append(f"Write Chapter {target_chapter_number} as cohesive narrative prose — scenes, not beat summaries.\n")
            context_parts.append(f"Do NOT include events from later chapters. Stay within Chapter {target_chapter_number}'s scope.\n\n")
        elif outline_body:
            # FALLBACK: If we have outline but couldn't extract current chapter, show warning
            logger.debug("Sending: outline.md -> No matching chapter found (Safe fallback - skipping outline)")
            logger.error(f"CRITICAL: Failed to extract chapter-specific outline for Chapter {target_chapter_number}")
            logger.error(f"   Falling back to NO outline (safe) rather than full outline (dangerous - would leak later chapters)")
            logger.error(f"   This means the LLM will generate without chapter-specific outline guidance")
            context_parts.append("=== OUTLINE UNAVAILABLE ===\n")
            context_parts.append(f"WARNING: Could not extract outline for Chapter {target_chapter_number}.\n")
            context_parts.append("Generating without chapter-specific outline guidance.\n")
            context_parts.append(f"Please ensure the outline has a properly formatted chapter header: '## Chapter {target_chapter_number}' or '## {target_chapter_number}'\n\n")
        
        # Add outline sync analysis if present
        outline_sync_analysis = state.get("outline_sync_analysis")
        if outline_sync_analysis and isinstance(outline_sync_analysis, dict):
            needs_sync = outline_sync_analysis.get("needs_sync", False)
            if needs_sync:
                context_parts.append("=== OUTLINE SYNC ANALYSIS (ADVISORY) ===\n")
                context_parts.append("The outline comparison found some discrepancies with the manuscript.\n")
                context_parts.append("These are ADVISORY - you may choose to address them if significant, but you are NOT required to match the outline exactly.\n")
                context_parts.append("The outline is a guide - preserve good prose and Style Guide voice, and use your judgment.\n\n")
                
                discrepancies = outline_sync_analysis.get("discrepancies", [])
                if discrepancies:
                    context_parts.append("Discrepancies found:\n")
                    for i, disc in enumerate(discrepancies, 1):
                        if isinstance(disc, dict):
                            disc_type = disc.get("type", "unknown")
                            outline_exp = disc.get("outline_expectation", "")
                            manuscript_curr = disc.get("manuscript_current", "")
                            severity = disc.get("severity", "medium")
                            suggestion = disc.get("suggestion", "")
                            context_parts.append(
                                f"{i}. [{severity.upper()}] {disc_type}:\n"
                                f"   Outline expects: {outline_exp}\n"
                                f"   Manuscript has: {manuscript_curr}\n"
                                f"   Suggestion: {suggestion}\n\n"
                            )
                    context_parts.append(
                        "**ADVISORY NOTE**: The outline comparison found some discrepancies.\n"
                        "You may choose to address these if they are significant, but you are NOT required to match the outline exactly.\n"
                        "The outline is a guide - preserve good prose and Style Guide voice, and use your judgment about what needs updating.\n"
                        "Only make changes if the discrepancies are truly important to the story.\n\n"
                    )
        
        # Add mode guidance
        mode_guidance = state.get("mode_guidance", "")
        if mode_guidance:
            context_parts.append(mode_guidance)
        
        # Add reference quality guidance
        reference_guidance = state.get("reference_guidance", "")
        if reference_guidance:
            context_parts.append(reference_guidance)
        
        # Add creative freedom indicator
        creative_freedom = state.get("creative_freedom_requested", False)
        if creative_freedom:
            context_parts.append(
                "\nCREATIVE FREEDOM GRANTED: User has requested enhancements/additions. "
                "You may add story elements beyond the outline, but MUST validate all additions "
                "against Style Guide, Universe Rules, Character profiles, and manuscript continuity.\n\n"
            )
        
        # Add question request guidance (all questions route here - can analyze and optionally edit)
        request_type = state.get("request_type", "edit_request")
        current_request = state.get("current_request", "")
        if request_type == "question":
            # Check for explicit "don't revise" / "just report" instructions
            request_lower = current_request.lower()
            explicit_no_edit = any(phrase in request_lower for phrase in [
                "don't revise", "dont revise", "do not revise", "no revisions", "no edits",
                "just report", "just tell me", "just answer", "only report", "only tell",
                "don't change", "dont change", "do not change", "no changes", "no modifications"
            ])
            
            if explicit_no_edit:
                context_parts.append(
                    "\n=== QUESTION REQUEST: ANALYSIS ONLY (NO EDITS) ===\n"
                    "The user has asked a question and EXPLICITLY requested NO revisions or edits.\n\n"
                    "**YOUR TASK**:\n"
                    "1. **ANALYZE AND ANSWER**: Answer the user's question by evaluating the current content\n"
                    "   - Provide a clear, helpful answer based on the manuscript and references\n"
                    "   - Explain what you observe, identify, or verify\n"
                    "   - Be specific and reference the actual content when relevant\n"
                    "2. **NO EDITS**: Do NOT provide any editor operations - user explicitly requested analysis only\n"
                    "   - Return empty operations array: []\n"
                    "   - Put your complete answer in the 'summary' field\n\n"
                    "**RESPONSE FORMAT**:\n"
                    "- In the 'summary' field: Complete answer to the question with analysis\n"
                    "- In the 'operations' array: MUST be empty [] (user requested no edits)\n\n"
                )
            else:
                context_parts.append(
                    "\n=== QUESTION REQUEST: ANALYZE AND OPTIONALLY EDIT ===\n"
                    "The user has asked a question about the manuscript.\n\n"
                    "**YOUR TASK**:\n"
                    "1. **ANALYZE FIRST**: Answer the user's question by evaluating the current content\n"
                    "   - Pure questions: 'How old is Tom here?' → Find and report Tom's age\n"
                    "   - Evaluation questions: 'Are we using enough description?' → Evaluate description level\n"
                    "   - Verification questions: 'Does this follow the style guide?' → Check style guide compliance\n"
                    "   - Conditional questions: 'Is Tom 23? We want him to be 24' → Check age, then edit if needed\n"
                    "   - Questions with edit hints: 'How does our chapter look? Let me know if there are revisions needed' → Analyze, then edit if issues found\n"
                    "2. **THEN EDIT IF NEEDED**: Based on your analysis, make edits if necessary\n"
                    "   - If question implies a desired state ('We want him to be 24') → Provide editor operations\n"
                    "   - If question asks for evaluation ('Are we using enough?') → Edit if answer is 'no'\n"
                    "   - If question hints at revisions ('Let me know if revisions needed') → Edit if issues found\n"
                    "   - If question is pure information ('How old is Tom?') → No edits needed, just answer\n"
                    "   - Include your analysis in the 'summary' field of your response\n\n"
                    "**RESPONSE FORMAT**:\n"
                    "- In the 'summary' field: Answer the question clearly and explain your analysis\n"
                    "- In the 'operations' array: Provide editor operations ONLY if edits are needed\n"
                    "- If no edits needed: Return empty operations array, but answer the question in summary\n"
                    "- If edits needed: Provide operations AND explain what you found in summary\n\n"
                    "**EXAMPLES**:\n"
                    "- 'How old is Tom here?' → Summary: 'Tom is 23 years old in this chapter.' Operations: []\n"
                "- 'Is Tom 23? We want him to be 24' → Summary: 'Tom is currently 23. Updating to 24.' Operations: [replace_range with age change]\n"
                "- 'Are we using enough description? Revise if necessary' → Summary: 'Description level is low. Adding sensory details.' Operations: [replace_range with enhanced description]\n\n"
            )
        elif outline_current_chapter_text:
            context_parts.append(
                "=== STYLE-FIRST NARRATIVE APPROACH ===\n"
                "You have a Style Guide that defines your voice and an Outline that defines your story.\n"
                "Your job: write the story of what happened, in the style provided.\n\n"
                "**HOW TO WRITE THIS CHAPTER:**\n"
                "1. The Style Guide is your voice — every sentence must sound like it belongs to that voice\n"
                "2. The Outline is your story knowledge — you know what happens, now WRITE it as literature\n"
                "3. Think in SCENES, not beats — combine related events into cohesive narrative moments\n"
                "4. Let the Style Guide's pacing determine how much space each moment gets\n"
                "5. Write original prose — never copy, paraphrase, or echo the outline's language\n\n"
            )
        
        # Add operation guidance
        context_parts.append(
            "=== OPERATION GUIDANCE ===\n"
            "**PREFER GRANULAR, PRECISE EDITS:**\n"
            "- Make the SMALLEST possible edit that achieves the user's goal\n"
            "- Use MINIMAL 'original_text' matches (10-20 words when possible, only larger if needed for uniqueness)\n"
            "- If editing multiple locations, use MULTIPLE operations (one per location)\n"
            "- Only use large paragraph-level edits when absolutely necessary\n"
            "- Preserve all surrounding text exactly - only change what needs changing\n\n"
            "- **USE GRANULAR EDITS** for precision: word/phrase changes, small corrections\n"
            "- **USE LARGE EDITS** for scope: complete block removal/replacement, major rewrites\n"
            "**KEY PRINCIPLE**: Granular for precision, large for scope. Match the edit size to the user's request.\n"
            "If your revision requires continuity fixes in multiple locations, include all necessary operations.\n"
            "Example: If changing a character name, provide operations to update all references in the current chapter.\n\n"
        )
        
        # 🎯 ROOSEVELT DEBUG: Log context_parts size before returning
        total_context_size = sum(len(part) for part in context_parts)
        logger.debug(f"📊 CONTEXT_PARTS DEBUG: {len(context_parts)} parts, total size: {total_context_size:,} chars")
        logger.debug(f"📊 CONTEXT_PARTS type check: {type(context_parts)}, is list: {isinstance(context_parts, list)}")
        
        return {
            "generation_context_parts": context_parts,
            "generation_context_structure": context_structure,
            "paragraph_map": paragraph_map,
            "is_empty_file": is_empty_file,
            "target_chapter_number": target_chapter_number,
            "current_chapter_label": current_chapter_label,
            "prev_chapter_last_line": prev_chapter_last_line,  # NEW: For anchor guidance
            # CRITICAL: Preserve state for subsequent nodes
            "system_prompt": state.get("system_prompt", ""),  # PRESERVE system_prompt!
            "datetime_context": state.get("datetime_context", ""),  # PRESERVE datetime_context!
            "metadata": state.get("metadata", {}),  # Contains user_chat_model!
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # PRESERVE manuscript context for next node!
            "manuscript": state.get("manuscript", ""),
            "filename": state.get("filename", ""),
            "frontmatter": state.get("frontmatter", {}),  # CRITICAL: Preserve frontmatter for temperature access
            "current_chapter_text": state.get("current_chapter_text", ""),
            "current_chapter_number": state.get("current_chapter_number"),
            "chapter_ranges": state.get("chapter_ranges", []),
            "current_request": state.get("current_request", ""),
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "cursor_offset": state.get("cursor_offset", -1),
            "requested_chapter_number": state.get("requested_chapter_number"),
            "explicit_primary_chapter": state.get("explicit_primary_chapter"),  # CRITICAL: For validating requested_chapter_number freshness
            # PRESERVE reference chapters
            "reference_chapter_numbers": state.get("reference_chapter_numbers", []),
            "reference_chapter_texts": state.get("reference_chapter_texts", {}),
            # PRESERVE outline context
            "outline_body": state.get("outline_body"),
            "story_overview": story_overview if 'story_overview' in locals() else state.get("story_overview"),
            "book_map": book_map if 'book_map' in locals() else state.get("book_map"),
        }
        
    except Exception as e:
        logger.error(f"Failed to build generation context: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "generation_context_parts": [],
            "paragraph_map": [],
            "error": str(e),
            "task_status": "error",
            "prev_chapter_last_line": None,
            # CRITICAL: Preserve state even on error
            "system_prompt": state.get("system_prompt", ""),  # PRESERVE system_prompt!
            "frontmatter": state.get("frontmatter", {}),  # CRITICAL: Preserve frontmatter for temperature access
            "datetime_context": state.get("datetime_context", ""),  # PRESERVE datetime_context!
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # PRESERVE manuscript context even on error!
            "manuscript": state.get("manuscript", ""),
            "filename": state.get("filename", ""),
            "current_chapter_text": state.get("current_chapter_text", ""),
            "current_chapter_number": state.get("current_chapter_number"),
            "chapter_ranges": state.get("chapter_ranges", []),
            "current_request": state.get("current_request", ""),
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "cursor_offset": state.get("cursor_offset", -1),
            "requested_chapter_number": state.get("requested_chapter_number"),
            "explicit_primary_chapter": state.get("explicit_primary_chapter"),
            "reference_chapter_numbers": state.get("reference_chapter_numbers", []),
            "reference_chapter_texts": state.get("reference_chapter_texts", {}),
        }


async def build_generation_prompt_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Build generation prompt: construct system prompt and user message"""
    try:
        logger.debug("Building generation prompt...")
        
        # Get system prompt from state (built by main agent)
        system_prompt = state.get("system_prompt", "")
        logger.debug(f"📊 SYSTEM PROMPT DEBUG: length = {len(system_prompt):,} chars (~{len(system_prompt) // 4:,} tokens)")
        logger.debug(f"📊 SYSTEM PROMPT first 500 chars: {system_prompt[:500]}")
        logger.debug(f"📊 SYSTEM PROMPT last 500 chars: {system_prompt[-500:]}")
        
        # CRITICAL DEBUG: Check if references are INSIDE the system prompt
        if "===  OUTLINE" in system_prompt or "outline_body" in system_prompt or len(system_prompt) > 100000:
            logger.error(f"🚨 SYSTEM PROMPT CONTAINS REFERENCES! This should NEVER happen!")
            logger.error(f"🚨 Searching for reference markers in system_prompt:")
            if "OUTLINE" in system_prompt:
                outline_pos = system_prompt.find("OUTLINE")
                logger.error(f"  - Found 'OUTLINE' at position {outline_pos}")
                logger.error(f"  - Context: ...{system_prompt[max(0, outline_pos-100):outline_pos+200]}...")
            if len(system_prompt) > 100000:
                # Show the middle section where references might be
                mid_point = len(system_prompt) // 2
                logger.error(f"📊 SYSTEM PROMPT middle section ({mid_point-500}:{mid_point+500}): {system_prompt[mid_point-500:mid_point+500]}")
        
        if not system_prompt:
            # Fallback: build it if not provided
            # This should normally come from the main agent's _build_system_prompt()
            logger.warning("No system_prompt in state - using fallback")
            system_prompt = "You are a MASTER NOVELIST editor/generator."
        
        generation_context_parts = state.get("generation_context_parts", [])
        current_request = state.get("current_request", "")
        manuscript = state.get("manuscript", "")
        filename = state.get("filename", "manuscript.md")
        selection_start = state.get("selection_start", -1)
        selection_end = state.get("selection_end", -1)
        cursor_offset = state.get("cursor_offset", -1)
        requested_chapter_number = state.get("requested_chapter_number")
        explicit_primary_chapter = state.get("explicit_primary_chapter")
        current_chapter_number = state.get("current_chapter_number")
        target_chapter_number = state.get("target_chapter_number")
        
        # DIAGNOSTIC: Log what we received
        logger.debug(f"📖 [PROMPT] explicit_primary_chapter={explicit_primary_chapter}, requested_chapter_number={requested_chapter_number}, current_chapter_number={current_chapter_number}, target_chapter_number={target_chapter_number}")
        
        if target_chapter_number is None:
            # CRITICAL: When user explicitly mentions a chapter, prioritize it over cursor position
            # This ensures "In Chapter 2..." edits Chapter 2, not the chapter where cursor is
            if explicit_primary_chapter is not None:
                # User explicitly mentioned a chapter - use it regardless of cursor position
                target_chapter_number = explicit_primary_chapter
                logger.debug(f"📖 [PROMPT] Using EXPLICIT chapter {explicit_primary_chapter} (user mentioned it in query)")
            elif requested_chapter_number is not None:
                # Check if requested_chapter_number is "fresh" (matches explicit mention)
                is_fresh_request = (
                    explicit_primary_chapter is not None and 
                    requested_chapter_number == explicit_primary_chapter
                )
                if is_fresh_request:
                    target_chapter_number = requested_chapter_number
                else:
                    target_chapter_number = current_chapter_number
            else:
                target_chapter_number = current_chapter_number
        chapter_ranges = state.get("chapter_ranges", [])
        
        # datetime_context should be provided by main agent via state
        datetime_context = state.get("datetime_context", "")
        
        # CRITICAL: Include conversation history for context
        # Extract previous user queries and assistant responses (not the references/context)
        from langchain_core.messages import AIMessage
        
        conversation_messages = state.get("messages", [])
        conversation_history = []
        
        if conversation_messages:
            # Keep only lightweight conversational turns (user queries + assistant summaries)
            # Skip the massive reference dumps that were in previous turns
            for msg in conversation_messages[:-1]:  # Exclude current turn (it's being built now)
                if isinstance(msg, HumanMessage):
                    # Extract just the user's query (not the full context)
                    # User queries are typically short
                    user_query = msg.content
                    if len(user_query) < 500:  # User queries are short
                        conversation_history.append(HumanMessage(content=user_query))
                elif isinstance(msg, AIMessage):
                    # Strip ManuscriptEdit JSON so LLM does not recycle stale original_text anchors
                    assistant_response = sanitize_ai_response_for_history(msg.content)
                    if assistant_response and len(assistant_response) < 5000:  # Summaries are concise
                        conversation_history.append(AIMessage(content=assistant_response))
        
        # Build messages: system prompt first, then conversation history, then current turn
        messages = [
            SystemMessage(content=system_prompt),
            SystemMessage(content=datetime_context) if datetime_context else None,
        ]
        # Add conversation history after system prompts
        messages.extend(conversation_history)
        # Add current turn context
        messages.append(HumanMessage(content="".join(generation_context_parts)))
        
        # Remove None entries
        messages = [m for m in messages if m is not None]
        
        if conversation_history:
            logger.debug(f"📜 Included {len(conversation_history)} previous conversational messages for context")
        
        # Add selection/cursor context
        selection_context = ""
        if selection_start >= 0 and selection_end > selection_start:
            selected_text = manuscript[selection_start:selection_end]
            selection_context = (
                f"\n\n=== USER HAS SELECTED TEXT ===\n"
                f"Selected text (characters {selection_start}-{selection_end}):\n"
                f'"""{selected_text[:500]}{"..." if len(selected_text) > 500 else ""}"""\n\n'
                "User selected this specific text! Use it as your anchor:\n"
                "- For edits within selection: Use 'original_text' matching the selected text (or portion of it)\n"
                "- System will automatically constrain your edit to the selection\n"
            )
        elif cursor_offset >= 0:
            selection_context = (
                f"\n\n=== CURSOR POSITION ===\n"
                f"Cursor is in the paragraph shown above (character offset {cursor_offset}).\n"
                "If editing this paragraph, provide EXACT text from it as 'original_text'.\n"
            )
        
        if current_request:
            # Detect granular correction patterns
            granular_correction_hints = ""
            request_lower = current_request.lower()
            granular_patterns = ["not ", "should be ", "instead of ", "change ", " to "]
            is_granular = any(pattern in request_lower for pattern in granular_patterns) and any(
                word in request_lower for word in ["not", "instead", "change", "should be"]
            )
            
            # 🎯 ROOSEVELT DEBUG: Check what's in these hint variables
            logger.debug(f"📊 HINTS DEBUG: is_granular={is_granular}")
            
            if is_granular:
                granular_correction_hints = (
                    "\n=== GRANULAR CORRECTION DETECTED ===\n"
                    "User is requesting a specific word/phrase change (e.g., 'boat not canoe').\n\n"
                    "**CRITICAL INSTRUCTIONS FOR GRANULAR CORRECTIONS:**\n"
                    "1. Read the CURRENT CHAPTER or PARAGRAPH AROUND CURSOR above to find the exact text containing the word/phrase\n"
                    "2. Find the MINIMAL unique context (10-20 words) that contains the word/phrase - NOT the entire paragraph\n"
                    "3. Set 'original_text' to the MINIMAL unique match (just enough to uniquely identify the location)\n"
                    "4. Set 'text' to the same text with ONLY the specific word/phrase changed\n"
                    "5. **PRESERVE ALL OTHER TEXT EXACTLY** - do not rewrite or regenerate anything\n"
                    "6. **DO NOT replace entire paragraphs** - only change the specific word/phrase\n"
                    "7. **USE THE SMALLEST POSSIBLE MATCH** - if 10 words uniquely identifies it, use 10 words, not 40\n\n"
                    "Example: If user says 'boat not canoe' and manuscript has:\n"
                    "  'He paddled the canoe across the river, feeling the current pull against him.'\n"
                    "Then (MINIMAL match):\n"
                    "  original_text: 'He paddled the canoe across the river' (15 words - sufficient for uniqueness)\n"
                    "  text: 'He paddled the boat across the river' (ONLY the word 'canoe' changed to 'boat')\n\n"
                    "NOT (too large):\n"
                    "  original_text: 'He paddled the canoe across the river, feeling the current pull against him.' (entire sentence when smaller match works)\n\n"
                )
            else:
                # Even for non-granular requests, emphasize precision
                granular_correction_hints = (
                    "\n=== EDIT PRECISION GUIDANCE ===\n"
                    "**PREFER GRANULAR, PRECISE EDITS:**\n"
                    "- Make the SMALLEST possible edit that achieves the user's goal\n"
                    "- Use MINIMAL 'original_text' matches (10-20 words when possible, only larger if needed for uniqueness)\n"
                    "- If editing multiple locations, use MULTIPLE operations (one per location)\n"
                    "- Only use large paragraph-level edits when absolutely necessary\n"
                    "- Preserve all surrounding text exactly - only change what needs changing\n\n"
                )
            
            # Check if creating a new chapter
            # Only treat as new chapter if target_chapter_number is explicitly requested (fresh) and current chapter is empty
            explicit_primary_chapter = state.get("explicit_primary_chapter")
            is_fresh_new_chapter_request = (
                explicit_primary_chapter is not None and 
                target_chapter_number is not None and
                target_chapter_number == explicit_primary_chapter and
                state.get("current_chapter_text", "").strip() == "" and
                any(keyword in current_request.lower() for keyword in ["create", "craft", "write", "generate", "chapter"])
            )
            is_creating_new_chapter = is_fresh_new_chapter_request
            
            new_chapter_hints = ""
            if is_creating_new_chapter:
                # Get the last line from state (extracted in build_generation_context_node)
                prev_chapter_last_line = state.get("prev_chapter_last_line")
                prev_chapter_number = state.get("prev_chapter_number")
                
                # Find the last chapter in the manuscript
                if chapter_ranges and prev_chapter_last_line:
                    last_chapter_range = chapter_ranges[-1]
                    last_chapter_num = last_chapter_range.chapter_number
                    
                    new_chapter_hints = (
                        f"\n=== CREATING NEW CHAPTER {target_chapter_number} ===\n"
                        f"The chapter doesn't exist yet - you need to insert it after the last existing chapter.\n\n"
                        f"**CONTINUITY ASSESSMENT (CRITICAL):**\n"
                        f"SCROLL UP and READ the CHAPTER {prev_chapter_number} MANUSCRIPT TEXT section carefully!\n"
                        f"- Where did the last chapter END (location, emotional state, action)?\n"
                        f"- What narrative momentum exists to build on?\n"
                        f"- DO NOT repeat the last scene - pick up AFTER it!\n"
                        f"- If Chapter {last_chapter_num} ended with characters in a location, Chapter {target_chapter_number} should continue FROM that location (not re-enter it)\n"
                        f"- Maintain emotional continuity from where Chapter {last_chapter_num} left off\n\n"
                        f"**INSERTION MECHANICS:**\n"
                        f"Last existing chapter: Chapter {last_chapter_num}\n"
                        f"**THE LAST LINE OF CHAPTER {last_chapter_num} IS:**\n"
                        f'"{prev_chapter_last_line}"\n\n'
                        f"**CRITICAL**: Use 'insert_after_heading' with anchor_text set to this EXACT line!\n"
                        f"Copy it VERBATIM - this is where your new chapter will be inserted.\n\n"
                        f"Required JSON structure:\n"
                        f"{{\n"
                        f'  "target_filename": "manuscript.md",\n'
                        f'  "scope": "chapter",\n'
                        f'  "summary": "Generated Chapter {target_chapter_number}",\n'
                        f'  "safety": "medium",\n'
                        f'  "operations": [{{\n'
                        f'    "op_type": "insert_after_heading",\n'
                        f'    "anchor_text": "{prev_chapter_last_line}",\n'
                        f'    "text": "## Chapter {target_chapter_number}\\n\\nYour chapter content here...",\n'
                        f'    "start": 0,\n'
                        f'    "end": 0\n'
                        f"  }}]\n"
                        f"}}\n\n"
                        f"**MANDATORY**: Your 'text' field MUST start with '## Chapter {target_chapter_number}' followed by two newlines, then your chapter content.\n"
                        f"**MANDATORY**: Use the exact last line shown above as your anchor_text!\n"
                        f"**DO NOT** use '## Chapter {target_chapter_number}' as anchor_text - it doesn't exist yet!\n"
                        f"**DO NOT** insert at the beginning of the file - insert after the last line shown above!\n"
                        f"**DO NOT** forget the chapter header - it is REQUIRED for all new chapters!\n\n"
                    )
                elif chapter_ranges:
                    # Fallback if we couldn't extract last line
                    last_chapter_range = chapter_ranges[-1]
                    last_chapter_num = last_chapter_range.chapter_number
                    
                    new_chapter_hints = (
                        f"\n=== CREATING NEW CHAPTER {target_chapter_number} ===\n"
                        f"The chapter doesn't exist yet - you need to insert it after the last existing chapter.\n"
                        f"Last existing chapter: Chapter {last_chapter_num}\n"
                        f"**CRITICAL**: Find the LAST LINE of Chapter {last_chapter_num} in the manuscript above and use it as anchor_text.\n"
                        f"**CONTINUITY**: READ Chapter {last_chapter_num} carefully to understand where the story left off!\n"
                        f"Your new chapter should pick up the narrative thread naturally, not repeat the previous scene.\n\n"
                    )
            
            # Build chapter clarification if there's a discrepancy
            chapter_clarification = ""
            # Only show clarification if target_chapter_number was explicitly requested (fresh)
            logger.debug(f"📖 [CLARIFICATION CHECK] explicit_primary_chapter={explicit_primary_chapter}, target_chapter_number={target_chapter_number}, will_show_clarification={(explicit_primary_chapter is not None and target_chapter_number == explicit_primary_chapter)}")
            if explicit_primary_chapter is not None and target_chapter_number == explicit_primary_chapter:
                # Check if this is a new chapter generation or editing an existing chapter
                current_chapter_text = state.get("current_chapter_text", "")
                is_existing_chapter = current_chapter_text.strip() != ""
                
                if is_existing_chapter:
                    # User explicitly mentioned an existing chapter - CRITICAL: ONLY edit that chapter!
                    chapter_clarification = (
                        f"\n{'='*80}\n"
                        f"⚠️ CRITICAL: USER EXPLICITLY MENTIONED CHAPTER {target_chapter_number} ⚠️\n"
                        f"{'='*80}\n"
                        f"**YOU MUST ONLY EDIT CHAPTER {target_chapter_number}**\n\n"
                        f"**MANDATORY RESTRICTIONS:**\n"
                        f"1. **ONLY search for text in the 'MANUSCRIPT TEXT: CHAPTER {target_chapter_number} (CURRENT - EDITABLE)' section above**\n"
                        f"2. **PREVIOUS and NEXT chapters are MARKED AS READ ONLY** - they are for context reference ONLY!\n"
                        f"3. **DO NOT edit Chapter {target_chapter_number - 1 if target_chapter_number > 1 else 'previous'} or Chapter {target_chapter_number + 1}** - they are READ ONLY!\n"
                        f"4. **If you find similar text in PREVIOUS or NEXT chapters, IGNORE IT for editing** - only edit Chapter {target_chapter_number}!\n"
                        f"5. **Your 'original_text' MUST come from Chapter {target_chapter_number} ONLY**\n"
                        f"6. **Any operations you create MUST target Chapter {target_chapter_number} text only**\n"
                        f"7. **ANALYSIS ALLOWED**: You CAN analyze READ ONLY chapters and report issues in your 'summary' field\n"
                        f"8. **DO NOT create operations for READ ONLY chapters** - only report issues, don't fix them\n\n"
                        f"**WHY THIS MATTERS:**\n"
                        f"The user specifically said 'In Chapter {target_chapter_number}...' - they want changes in THAT chapter, not others.\n"
                        f"PREVIOUS and NEXT chapters are provided for continuity context but are READ ONLY - do not modify them.\n"
                        f"However, if you spot problems in READ ONLY chapters, you SHOULD mention them to the user in your summary.\n"
                        f"Even if similar text exists in other chapters, you must ONLY edit Chapter {target_chapter_number}.\n"
                        f"{'='*80}\n\n"
                    )
                else:
                    # Clear generation instruction matching detected chapter
                    chapter_clarification = (
                        f"\n{'='*80}\n"
                        f"CHAPTER GENERATION TARGET: CHAPTER {target_chapter_number}\n"
                        f"{'='*80}\n"
                        f"**YOU ARE GENERATING CHAPTER {target_chapter_number}**\n"
                        f"**MANDATORY**: Your chapter header MUST be '## Chapter {target_chapter_number}' (NOT any other number!)\n"
                        f"**CRITICAL**: Use the outline for Chapter {target_chapter_number} (marked as CURRENT above) as your guide\n"
                        f"{'='*80}\n\n"
                    )
            
            # 🎯 ROOSEVELT DEBUG: Log component sizes BEFORE concatenation
            logger.debug(f"📊 PRE-CONCAT DEBUG:")
            logger.debug(f"   chapter_clarification type: {type(chapter_clarification)}, len: {len(chapter_clarification)}")
            logger.debug(f"   chapter_clarification preview: {repr(chapter_clarification[:200])}")
            logger.debug(f"   selection_context type: {type(selection_context)}, len: {len(selection_context)}")
            logger.debug(f"   selection_context preview: {repr(selection_context[:200])}")
            logger.debug(f"   granular_correction_hints type: {type(granular_correction_hints)}, len: {len(granular_correction_hints)}")
            logger.debug(f"   granular_correction_hints preview: {repr(granular_correction_hints[:200])}")
            logger.debug(f"   new_chapter_hints type: {type(new_chapter_hints)}, len: {len(new_chapter_hints)}")
            logger.debug(f"   new_chapter_hints preview: {repr(new_chapter_hints[:200])}")
            logger.debug(f"   current_request type: {type(current_request)}, len: {len(current_request)}")
            
            # Build the message content as a single string ONCE
            # 🎯 ROOSEVELT DEBUG: Build incrementally to find where duplication occurs
            # Check request type to determine if operations are required
            request_type = state.get("request_type", "edit_request")
            is_question = request_type == "question"
            
            # Build operations requirement message based on request type
            if is_question:
                operations_requirement = (
                    "\n" + ("="*80) + "\n" +
                    "QUESTION REQUEST: OPERATIONS ARE OPTIONAL\n" +
                    ("="*80) + "\n\n" +
                    "This is a QUESTION request. You may return empty operations if you are only analyzing/answering.\n"
                    "- If the question requires edits: Provide operations to make the changes\n"
                    "- If the question is informational only: Return empty operations array and answer in the 'summary' field\n"
                    "- The summary field MUST contain your complete answer to the question\n\n"
                )
            else:
                operations_requirement = (
                    "\n" + ("="*80) + "\n" +
                    "⚠️ CRITICAL: OPERATIONS ARE MANDATORY ⚠️\n" +
                    ("="*80) + "\n\n" +
                    "This is a GENERATION/EDITING request. You MUST provide at least one operation in the operations array.\n"
                    "- **EMPTY OPERATIONS ARE NOT ALLOWED** for generation/editing requests\n"
                    "- You MUST generate actual content changes (replace_range, insert_after_heading, or delete_range)\n"
                    "- If you cannot find the exact text to edit, use the best available anchor and proceed\n"
                    "- If you have concerns, include them in the summary but STILL provide operations\n"
                    "- **DO NOT return empty operations** - the user expects actual changes to be made\n\n"
                )
            
            # Scene-thinking creative framing (only for generation, not questions)
            scene_framing = ""
            if not is_question:
                scene_framing = (
                    "\n" + ("="*80) + "\n" +
                    "CREATIVE APPROACH — WRITE SCENES, NOT EXPANDED BEATS\n" +
                    ("="*80) + "\n\n" +
                    "Before you write, internalize this approach:\n\n"
                    "1. **You are a novelist, not a transcriber.** You have absorbed the Style Guide's voice\n"
                    "   and you know what happens in this chapter from the outline. Now forget the outline's\n"
                    "   words entirely and write the chapter as original literature.\n\n"
                    "2. **Think in scenes.** Group related beats into unified scenes with a setting, action,\n"
                    "   and emotional arc. A scene may encompass several beats, or a single beat may span\n"
                    "   multiple paragraphs — let the Style Guide's pacing determine the proportion.\n\n"
                    "3. **The Style Guide controls everything about HOW you write:**\n"
                    "   - Sentence length and rhythm\n"
                    "   - Words and phrasing to use or avoid\n"
                    "   - Level of sensory detail and interiority\n"
                    "   - Dialogue style and attribution patterns\n"
                    "   - Tone, register, and narrative distance\n"
                    "   - If a writing sample was provided, your prose must feel like it was written by\n"
                    "     the same author — match its cadence, density, and style choices\n\n"
                    "4. **The reader should never feel they are reading an expanded outline.**\n"
                    "   If your output reads like 'First X happened, then Y happened, then Z happened,'\n"
                    "   you are summarizing, not writing fiction. Immerse the reader in the scene.\n\n"
                )
            
            base_template = (
                scene_framing +
                "\n" + ("="*80) + "\n" +
                "OUTPUT FORMAT REQUIREMENTS\n" +
                ("="*80) + "\n\n" +
                "YOU MUST RESPOND WITH **JSON ONLY**\n\n"
                "DO NOT use XML tags like <operation> or <op_type>\n"
                "DO NOT use YAML format (key: value without braces)\n"
                "DO NOT use any format other than JSON\n"
                "ONLY return valid JSON with curly braces { }\n"
                "ONLY return a JSON object matching ManuscriptEdit structure\n\n"
                f"USER REQUEST: {current_request}\n\n" +
                operations_requirement
            )
            logger.debug(f"📊 STEP 1 - base_template: {len(base_template):,} chars")
            
            step2 = base_template + chapter_clarification
            logger.debug(f"📊 STEP 2 - after chapter_clarification: {len(step2):,} chars")
            
            step3 = step2 + selection_context
            logger.debug(f"📊 STEP 3 - after selection_context: {len(step3):,} chars")
            
            step4 = step3 + granular_correction_hints
            logger.debug(f"📊 STEP 4 - after granular_correction_hints: {len(step4):,} chars")
            
            step5 = step4 + new_chapter_hints
            logger.debug(f"📊 STEP 5 - after new_chapter_hints: {len(step5):,} chars")
            
            rest_of_template = (
                "\n" + ("="*80) + "\n" +
                "BEFORE YOU GENERATE YOUR JSON RESPONSE\n" +
                ("="*80) + "\n\n" +
                "**STEP 1: SCROLL UP TO FIND MANUSCRIPT TEXT**\n"
                "Look for sections labeled 'MANUSCRIPT TEXT: CHAPTER N'.\n"
                "These sections contain the ACTUAL text from the manuscript file.\n\n"
                "**STEP 2: FIND THE EXACT TEXT TO EDIT IN THE MANUSCRIPT**\n"
                "Find the sentence or paragraph in the MANUSCRIPT TEXT sections that matches the user's request.\n"
                "Copy 10-20 words of EXACT, VERBATIM text (minimum for uniqueness; only more if needed).\n\n"
                "**STEP 3: DO NOT USE OUTLINE TEXT**\n"
                "The OUTLINE sections (below '=== END OF MANUSCRIPT CONTEXT ===') contain story beats.\n"
                "These words DO NOT EXIST in the manuscript file!\n"
                "If you copy outline text into 'original_text', the system will fail to find it!\n\n"
                "**STEP 4: CROSS-CHAPTER EDITS FORBIDDEN**\n"
                "original_text and anchor_text must NEVER span a chapter boundary. Each operation targets ONE chapter only (the CURRENT - EDITABLE chapter).\n"
                "If editing near the end of a chapter, do NOT include the next chapter's heading (## Chapter N) or any text from the next chapter in original_text.\n"
                "If editing near the start of a chapter, do NOT include the previous chapter's last line. Stay strictly within the chapter you are editing.\n\n"
                "**CRITICAL: ALL OPERATIONS MUST HAVE ANCHORS**\n\n"
                "**FOR replace_range/delete_range:**\n"
                "- **MANDATORY**: You MUST provide 'original_text' with EXACT text from the manuscript\n"
                "- **CRITICAL**: If you don't provide 'original_text', the operation will FAIL completely\n"
                "- **NEVER** create a replace_range operation without 'original_text' - it will fail!\n"
                "- **USE MINIMAL MATCHES (10-20 words when possible)**: Only include in 'original_text' the exact span you are changing. Smaller = more precise.\n"
                "  - GOOD: original_text=\"walked\" + text=\"ran\" → \"John Smith walked\" becomes \"John Smith ran\"\n"
                "  - BAD: original_text=\"John Smith walked\" + text=\"walked\" → entire phrase becomes just \"walked\"\n"
                "- **REPLACEMENT MUST BE SYNACTICALLY COMPLETE**: The 'text' field replaces the whole 'original_text' span. So 'text' must be a complete phrase/sentence if the change would otherwise leave broken grammar (e.g. if you replace a clause, the new clause must read correctly in context). Do NOT use large original_text to \"be safe\"—use the smallest unique match.\n\n"
                "**FOR insert_after_heading (NEW TEXT):**\n"
                "- Use this when adding NEW content that doesn't exist in the manuscript\n"
                "- **MANDATORY**: You MUST provide 'anchor_text' with EXACT text from the manuscript to insert after\n"
                "- **CRITICAL**: If you don't provide 'anchor_text', the operation will FAIL to find the insertion point and your content will be lost\n\n"
                "**WHAT IS VALID ANCHOR TEXT:**\n"
                "✅ CORRECT: Text that ACTUALLY EXISTS in the MANUSCRIPT TEXT sections above\n"
                "✅ CORRECT: The last line of the previous chapter (if provided as 'LAST LINE OF CHAPTER X')\n"
                "✅ CORRECT: A chapter heading like '## Chapter 5' if inserting after that chapter\n"
                "❌ WRONG: Text from the outline (it doesn't exist in the manuscript!)\n"
                "❌ WRONG: Text you THINK should be in the manuscript but isn't shown above\n"
                "❌ WRONG: Paraphrased or summarized text from the manuscript\n"
                "❌ WRONG: Text you are generating yourself\n"
                "❌ WRONG: Invented text that 'sounds right' but isn't in the manuscript\n\n"
                "**HOW TO FIND VALID ANCHOR TEXT:**\n"
                "1. Scroll UP to find MANUSCRIPT TEXT sections (marked with chapter numbers)\n"
                "2. Locate the EXACT sentence/paragraph where new text should be inserted after\n"
                "3. Copy that text VERBATIM (word-for-word, character-for-character) from the manuscript\n"
                "4. Use Ctrl+F/Cmd+F to verify the text exists in the MANUSCRIPT TEXT sections above\n"
                "5. **NEVER** use text from OUTLINE, STORY OVERVIEW, or CHARACTER PROFILES\n\n"
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
                "**VERIFICATION CHECKLIST BEFORE GENERATING JSON:**\n"
                "☐ I found the text in a 'MANUSCRIPT TEXT: CHAPTER N' section\n"
                "☐ I copied it EXACTLY as written in that section (VERBATIM, no changes)\n"
                "☐ I did NOT copy text from any OUTLINE section\n"
                "☐ The text I copied is BEFORE the '=== END OF MANUSCRIPT CONTEXT ===' marker\n"
                "☐ For replace_range/delete_range: I verified my 'original_text' exists in the MANUSCRIPT TEXT section above\n"
                "☐ For insert_after_heading: I verified my 'anchor_text' exists in the MANUSCRIPT TEXT section above\n"
                "☐ I did a mental Ctrl+F search and confirmed I can find my anchor_text/original_text in the manuscript sections above\n"
                "☐ I did NOT invent, paraphrase, or generate anchor text - I COPIED it EXACTLY from what's shown\n"
                "☐ I did NOT use outline text, chapter headers (except '## Chapter N'), or any text that doesn't appear in the manuscript\n"
                "☐ My original_text/anchor_text does NOT span across a chapter boundary (stays within one chapter only)\n\n" +
                ("="*80) + "\n" +
                "NOW GENERATE YOUR JSON RESPONSE:\n" +
                ("="*80) + "\n\n" +
                "CRITICAL: REQUIRED JSON STRUCTURE\n\n"
                "**OUTPUT FORMAT: JSON ONLY - NO XML, NO YAML, ONLY JSON!**\n\n"
                "Your response MUST be a complete ManuscriptEdit JSON object with this EXACT structure:\n\n"
                "{\n"
                '  "target_filename": "manuscript.md",\n'
                '  "scope": "chapter",\n'
                '  "summary": "Brief description of what you did",\n'
                '  "safety": "medium",\n'
                '  "operations": [\n'
                "    {\n"
                '      "op_type": "insert_after_heading",\n'
                '      "anchor_text": "EXACT text from manuscript to insert after",\n'
                '      "text": "## Chapter 1: Title\\n\\nYour generated prose here...",\n'
                '      "start": 0,\n'
                '      "end": 0\n'
                "    }\n"
                "  ]\n"
                "}\n\n"
                "DO NOT use XML tags: <operation>, <op_type>, <text>, etc.\n"
                "DO NOT use YAML format: op_type: value\n"
                "DO NOT return a single operation object!\n"
                "DO NOT use 'operation' (singular) - must be 'operations' (array)!\n"
                "DO NOT nest op_type at top level - it goes INSIDE operations array!\n"
                "ONLY return JSON with curly braces { } and square brackets [ ]\n"
                "ONLY use the exact field names shown above\n\n"
                "For REPLACE/DELETE operations in prose (no headers), you MUST provide robust anchors:\n\n"
                "**MANDATORY REQUIREMENT**: 'original_text' is REQUIRED for all replace_range/delete_range operations\n"
                "- **NO EXCEPTIONS**: Every replace_range/delete_range operation MUST have 'original_text'\n"
                "- **CRITICAL**: Copy EXACT, VERBATIM text from the MANUSCRIPT TEXT sections above\n"
                "- If you don't provide 'original_text', the operation will FAIL completely\n\n"
                "**PREFER GRANULAR EDITS: Use the SMALLEST possible 'original_text' match**\n"
                "- MINIMIZE the text you replace - only replace what actually needs to change\n"
                "- If changing one word/phrase, only include that word/phrase in 'original_text'\n"
                "- If you include extra context in 'original_text', you MUST re-include it in 'text'\n"
                "- Example editing \"caught Walter's eye and smiled, though tension showed\" → \"caught Walter's eye and smiled, slight shadows showing\":\n"
                "  - GOOD: original_text=\"though tension showed\" + text=\"slight shadows showing\" (minimal match)\n"
                "  - BAD: original_text=\"caught Walter's eye and smiled, though tension showed\" + text=\"caught Walter's eye and smiled, slight shadows showing\" (too broad)\n"
                "  - VERY BAD: original_text=\"caught Walter's eye and smiled, though tension showed\" + text=\"slight shadows showing\" (loses context!)\n\n"
                "- For word-level changes: 10-15 words of context (minimal, unique match)\n"
                "- For phrase changes: 15-20 words of context (minimal, unique match)\n"
                "- For sentence changes: Just the sentence(s) that need changing (15-30 words)\n"
                "- Only use 30-40 words when smaller matches aren't unique enough\n"
                "- **SMALLER IS BETTER** - precise alignment is more important than large context\n\n"
                + ("="*80) + "\n"
                "⚠️ CRITICAL: OUTLINE TEXT CANNOT BE USED AS ANCHOR TEXT ⚠️\n"
                + ("="*80) + "\n"
                "The outline is a PLANNING DOCUMENT ONLY - it does NOT exist in the manuscript file!\n"
                "If you use outline text as 'original_text' or 'anchor_text', the operation will FAIL.\n\n"
                "**WHERE TO COPY TEXT FROM:**\n"
                "✅ CORRECT: Copy from 'MANUSCRIPT TEXT' sections (labeled with chapter numbers)\n"
                "✅ CORRECT: Copy from user's selected text if provided\n"
                "✅ CORRECT: Copy from 'Previous Chapter End' if provided\n"
                "❌ WRONG: Copy from 'Outline' sections\n"
                "❌ WRONG: Copy from 'Story Overview' sections\n"
                "❌ WRONG: Invent text that doesn't exist in manuscript\n\n"
                "**OUTLINE IS FOR GUIDANCE ONLY:**\n"
                "- Use outline to understand story direction and what to write\n"
                "- DO NOT copy outline text into original_text/anchor_text fields\n"
                "- Outline text does not exist in the actual manuscript file\n"
                + ("="*80) + "\n\n"
                "**OPTION 1 (BEST): Use selection as anchor**\n"
                "- If user selected text, match it EXACTLY in 'original_text'\n"
                "- Use the selection plus minimal surrounding context (10-15 words) if needed for uniqueness\n"
                "- Copy from MANUSCRIPT TEXT sections, NOT from outline!\n\n"
                "**OPTION 2: Use left_context + right_context**\n"
                "- left_context: 30-50 chars BEFORE the target (exact text from manuscript)\n"
                "- right_context: 30-50 chars AFTER the target (exact text from manuscript)\n"
                "- Both must be copied from manuscript text, NOT from outline!\n\n"
                "**OPTION 3: Use long original_text**\n"
                "- Include 20-40 words of EXACT, VERBATIM text to replace\n"
                "- Include complete sentences with natural boundaries\n"
                "- Copy from MANUSCRIPT TEXT sections above, NEVER from outline/story overview text!\n\n"
                "**ABSOLUTE PROHIBITIONS:**\n"
                "- **NEVER** include chapter headers (##) in original_text - they will be deleted!\n"
                "- **NEVER** use outline text as anchor - THE OUTLINE DOES NOT EXIST IN THE MANUSCRIPT FILE!\n"
                "- **NEVER** use story overview text as anchor - IT DOES NOT EXIST IN THE MANUSCRIPT FILE!\n"
                "- **NEVER** create operations without proper anchors - operations will fail and content will be lost!\n"
                "- **NEVER** invent text that doesn't exist in the manuscript - copy EXACTLY from MANUSCRIPT TEXT sections!\n\n" +
                ("="*80) + "\n" +
                "FINAL REMINDER: RETURN JSON ONLY\n" +
                ("="*80) + "\n\n" +
                "Start your response with { and end with }\n"
                "Do NOT use any XML-style tags or YAML-style notation\n"
                "Your response must be valid JSON that can be parsed by json.loads()\n\n"
                "NOW GENERATE YOUR JSON RESPONSE:\n"
            )
            logger.debug(f"📊 STEP 6 - rest_of_template: {len(rest_of_template):,} chars")
            logger.debug(f"📊 STEP 6 - rest_of_template preview (first 500): {repr(rest_of_template[:500])}")
            logger.debug(f"📊 STEP 6 - rest_of_template preview (last 500): {repr(rest_of_template[-500:])}")
            
            # Final concatenation
            message_4_content = step5 + rest_of_template
            logger.debug(f"📊 STEP 7 - FINAL (step5 + rest_of_template): {len(message_4_content):,} chars (expected: {len(step5) + len(rest_of_template)})")
            
            # 🎯 ROOSEVELT DEBUG: Check message_4_content length before adding to messages
            logger.debug(f"📊 MESSAGE_4_CONTENT BUILT: {len(message_4_content):,} chars")
            logger.debug(f"📊 MESSAGE_4_CONTENT type: {type(message_4_content)}")
            
            # Append as single HumanMessage
            messages.append(HumanMessage(content=message_4_content))
        
        # 🎯 ROOSEVELT DEBUG: Log the actual size of Message 4 to identify bloat source
        if len(messages) > 3:
            msg4_content = messages[3].content if hasattr(messages[3], 'content') else str(messages[3])
            logger.debug(f"📊 MESSAGE 4 AFTER APPEND: length = {len(msg4_content):,} chars")
            logger.debug(f"📊 MESSAGE 4 AFTER APPEND first 500 chars: {msg4_content[:500]}")
            logger.debug(f"📊 MESSAGE 4 AFTER APPEND last 500 chars: {msg4_content[-500:]}")
        
        return {
            "generation_messages": messages,
            # CRITICAL: Preserve state for subsequent nodes
            "system_prompt": system_prompt,  # Use the processed system_prompt (not state.get)
            "datetime_context": state.get("datetime_context", ""),
            "generation_context_parts": state.get("generation_context_parts", []),  # CRITICAL: Needed for self-healing!
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # PRESERVE manuscript context!
            "manuscript": state.get("manuscript", ""),
            "filename": state.get("filename", ""),
            "frontmatter": state.get("frontmatter", {}),  # CRITICAL: Preserve frontmatter for temperature access
            "current_chapter_text": state.get("current_chapter_text", ""),
            "current_chapter_number": state.get("current_chapter_number"),
            "chapter_ranges": state.get("chapter_ranges", []),
            "current_request": state.get("current_request", ""),
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "cursor_offset": state.get("cursor_offset", -1),
            "requested_chapter_number": state.get("requested_chapter_number"),
        }
        
    except Exception as e:
        logger.error(f"Failed to build generation prompt: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "generation_messages": [],
            "error": str(e),
            "task_status": "error",
            # CRITICAL: Preserve state even on error
            "system_prompt": state.get("system_prompt", ""),  # PRESERVE system_prompt!
            "datetime_context": state.get("datetime_context", ""),  # PRESERVE datetime_context!
            "generation_context_parts": state.get("generation_context_parts", []),  # CRITICAL: Needed for self-healing!
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # PRESERVE manuscript context even on error!
            "manuscript": state.get("manuscript", ""),
            "filename": state.get("filename", ""),
            "frontmatter": state.get("frontmatter", {}),  # CRITICAL: Preserve frontmatter for temperature access
            "current_chapter_text": state.get("current_chapter_text", ""),
            "current_chapter_number": state.get("current_chapter_number"),
            "chapter_ranges": state.get("chapter_ranges", []),
            "current_request": state.get("current_request", ""),
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "cursor_offset": state.get("cursor_offset", -1),
            "requested_chapter_number": state.get("requested_chapter_number"),
        }


async def call_generation_llm_node(state: Dict[str, Any], llm_factory) -> Dict[str, Any]:
    """Call LLM for generation with structured output"""
    try:
        logger.info("Calling LLM for generation...")
        
        generation_messages = state.get("generation_messages", [])
        if not generation_messages:
            return {
                "llm_response": "",
                "error": "No generation messages available",
                "task_status": "error",
                # CRITICAL: Preserve state even on error
                "system_prompt": state.get("system_prompt", ""),
                "datetime_context": state.get("datetime_context", ""),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "manuscript": state.get("manuscript", ""),
                "filename": state.get("filename", ""),
                "frontmatter": state.get("frontmatter", {}),  # CRITICAL: Preserve frontmatter for temperature access
                "current_chapter_text": state.get("current_chapter_text", ""),
                "current_chapter_number": state.get("current_chapter_number"),
                "chapter_ranges": state.get("chapter_ranges", []),
                "current_request": state.get("current_request", ""),
                "selection_start": state.get("selection_start", -1),
                "selection_end": state.get("selection_end", -1),
                "cursor_offset": state.get("cursor_offset", -1),
                "requested_chapter_number": state.get("requested_chapter_number"),
            }
        
        # DEBUG: Log message sizes to identify token bloat
        total_chars = 0
        for i, msg in enumerate(generation_messages):
            msg_content = msg.content if hasattr(msg, 'content') else str(msg)
            msg_type = msg.__class__.__name__ if hasattr(msg, '__class__') else 'Unknown'
            msg_len = len(msg_content)
            total_chars += msg_len
            logger.info(f"📊 Message {i+1}/{len(generation_messages)} ({msg_type}): {msg_len:,} chars (~{msg_len // 4:,} tokens)")
        logger.info(f"📊 TOTAL: {total_chars:,} chars (~{total_chars // 4:,} tokens)")
        logger.info(f"📊 Expected token count: ~{total_chars // 4:,} (limit: 200,000)")
        
        # Check frontmatter for temperature override (default: 0.4 for fiction generation)
        frontmatter = state.get("frontmatter", {})
        temperature = frontmatter.get("temperature", 0.4)
        if temperature != 0.4:
            logger.info(f"🌡️ Using frontmatter temperature: {temperature} (default: 0.4)")
        
        # Get LLM from factory
        llm = llm_factory(temperature=temperature, state=state)
        
        start_time = datetime.now()
        response = await llm.ainvoke(generation_messages)
        
        raw_content = response.content if hasattr(response, 'content') else str(response)
        content = _unwrap_json_response(raw_content)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"LLM generation completed in {elapsed:.2f}s")
        logger.info(
            "LLM response debug: raw_len=%s, unwrapped_len=%s",
            len(raw_content),
            len(content),
        )
        logger.debug(
            "LLM response preview: raw=%s, unwrapped=%s",
            (raw_content[:300] + "..." if len(raw_content) > 300 else raw_content),
            (content[:300] + "..." if len(content) > 300 else content),
        )
        if len(content) == 0 and len(raw_content) > 0:
            logger.warning(
                "Unwrap produced empty content from %s chars of raw response; raw preview: %s",
                len(raw_content),
                raw_content[:500] + "..." if len(raw_content) > 500 else raw_content,
            )
        
        return {
            "llm_response": content,
            "llm_response_raw": raw_content,
            # CRITICAL: Preserve state for subsequent nodes
            "system_prompt": state.get("system_prompt", ""),
            "datetime_context": state.get("datetime_context", ""),
            "generation_context_parts": state.get("generation_context_parts", []),  # CRITICAL: Needed for self-healing!
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "manuscript": state.get("manuscript", ""),
            "filename": state.get("filename", ""),
            "frontmatter": state.get("frontmatter", {}),  # CRITICAL: Preserve frontmatter for temperature access
            "current_chapter_text": state.get("current_chapter_text", ""),
            "current_chapter_number": state.get("current_chapter_number"),
            "chapter_ranges": state.get("chapter_ranges", []),
            "current_request": state.get("current_request", ""),
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "cursor_offset": state.get("cursor_offset", -1),
            "requested_chapter_number": state.get("requested_chapter_number"),
        }
        
    except Exception as e:
        logger.error(f"Failed to call generation LLM: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "llm_response": "",
            "error": str(e),
            "task_status": "error",
            # CRITICAL: Preserve state even on error
            "system_prompt": state.get("system_prompt", ""),
            "datetime_context": state.get("datetime_context", ""),
            "generation_context_parts": state.get("generation_context_parts", []),  # CRITICAL: Needed for self-healing!
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "manuscript": state.get("manuscript", ""),
            "filename": state.get("filename", ""),
            "frontmatter": state.get("frontmatter", {}),  # CRITICAL: Preserve frontmatter for temperature access
            "current_chapter_text": state.get("current_chapter_text", ""),
            "current_chapter_number": state.get("current_chapter_number"),
            "chapter_ranges": state.get("chapter_ranges", []),
            "current_request": state.get("current_request", ""),
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "cursor_offset": state.get("cursor_offset", -1),
            "requested_chapter_number": state.get("requested_chapter_number"),
        }


async def validate_generated_output_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Validate generated output: check for outline copying, validate structure"""
    try:
        logger.info("Validating generated output...")
        
        content = state.get("llm_response", "")
        filename = state.get("filename", "manuscript.md")
        
        if not content:
            raw_preview = (state.get("llm_response_raw") or "")[:500]
            logger.error(
                "No LLM response to validate (content empty or missing). llm_response_raw preview: %s",
                raw_preview + "..." if len(raw_preview) >= 500 else raw_preview,
            )
            return {
                "structured_edit": None,
                "error": "No LLM response to validate",
                "task_status": "error",
                "system_prompt": state.get("system_prompt", ""),
                "datetime_context": state.get("datetime_context", ""),
                "generation_context_parts": state.get("generation_context_parts", []),
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
                "selection_start": state.get("selection_start", -1),
                "selection_end": state.get("selection_end", -1),
                "cursor_offset": state.get("cursor_offset", -1),
                "requested_chapter_number": state.get("requested_chapter_number"),
            }
        
        # Parse and validate structured response using Pydantic
        structured_edit = None
        try:
            # Parse JSON first
            raw = json.loads(content)
            
            # Ensure required fields have defaults
            if isinstance(raw, dict):
                # NORMALIZE: LLM sometimes returns a single operation object at top level
                # instead of {"operations": [...], "summary": ..., ...}
                if "op_type" in raw and "operations" not in raw:
                    logger.info("LLM returned single operation at top level - wrapping in ManuscriptEdit structure")
                    raw = {
                        "target_filename": raw.get("target_filename", filename),
                        "scope": raw.get("scope", "chapter"),
                        "summary": raw.get("summary", "Planned edit generated from context."),
                        "safety": raw.get("safety", "medium"),
                        "operations": [raw],
                    }
                raw.setdefault("target_filename", filename)
                # DEFENSIVE: Force scope to valid value if missing or invalid
                raw_scope = raw.get("scope", "paragraph")
                valid_scopes = ["paragraph", "chapter", "multi_chapter"]
                if raw_scope not in valid_scopes:
                    logger.warning(f"⚠️ Invalid scope value: {raw_scope} - defaulting to 'paragraph'")
                    raw["scope"] = "paragraph"
                else:
                    raw.setdefault("scope", "paragraph")
                
                raw.setdefault("summary", "Planned edit generated from context.")
                raw.setdefault("safety", "medium")
                raw.setdefault("operations", [])
                
                # For questions: empty operations array is valid (analysis-only response)
                # The summary will contain the answer
                request_type = state.get("request_type", "")
                if request_type == "question" and not raw.get("operations"):
                    logger.info("Question request with no operations - this is valid (analysis-only response)")
            
            # Validate with Pydantic model
            try:
                manuscript_edit = ManuscriptEdit(**raw)

                # Anti-copy safeguard: if the model reused outline phrasing in generated prose,
                # retry once with a rewrite instruction that preserves events but forces fresh prose.
                outline_text_for_check = state.get("outline_current_chapter_text") or state.get("outline_body")
                if outline_text_for_check and manuscript_edit.operations:
                    any_rewrite_needed = False
                    for op in manuscript_edit.operations:
                        op_text = op.text or ""
                        if op_text and _looks_like_outline_copied(op_text, outline_text_for_check):
                            any_rewrite_needed = True
                            break

                    if any_rewrite_needed:
                        logger.warning("Generated prose appears to reuse outline phrasing; retrying once with anti-copy rewrite instruction")
                        # Note: This would require another LLM call, which we'll handle in the main agent
                        # For now, we'll just log the warning and continue
                        # The main agent can handle retry logic if needed
                        pass

                # Convert to dict for state storage (TypedDict compatibility)
                structured_edit = manuscript_edit.model_dump()
                operations_count = len(manuscript_edit.operations)
                logger.info(f"Validated ManuscriptEdit with {operations_count} operations")
                
                # Check if empty operations are appropriate for this request type
                request_type = state.get("request_type", "")
                if request_type == "question" and operations_count == 0:
                    logger.info("Question request with no operations - this is valid (analysis in summary field)")
                elif request_type != "question" and operations_count == 0:
                    logger.warning(f"⚠️ Non-question request ({request_type}) returned 0 operations - this may indicate the LLM misunderstood the task")
                    logger.warning(f"⚠️ User request was: {state.get('current_request', 'unknown')}")
                    summary_preview = manuscript_edit.summary[:200] if manuscript_edit.summary else 'No summary'
                    logger.warning(f"⚠️ LLM summary: {summary_preview}")
                    logger.warning(f"⚠️ Full LLM response length: {len(content)} chars")
                    # Log a snippet of the actual response to help debug
                    if len(content) > 0:
                        response_preview = content[:500] if len(content) > 500 else content
                        logger.warning(f"⚠️ LLM response preview: {response_preview}")
            except ValidationError as ve:
                # Provide detailed validation error
                error_details = []
                for error in ve.errors():
                    field = " -> ".join(str(loc) for loc in error.get("loc", []))
                    msg = error.get("msg", "Validation error")
                    error_details.append(f"{field}: {msg}")
                
                error_msg = f"ManuscriptEdit validation failed:\n" + "\n".join(error_details)
                logger.error(f"{error_msg}")
                
                # SALVAGE OPERATION: For question requests, try to extract summary even if validation fails
                request_type = state.get("request_type", "")
                if request_type == "question" and isinstance(raw, dict):
                    summary = raw.get("summary", "")
                    if summary and len(summary) > 20:  # Meaningful summary exists
                        logger.warning(f"⚠️ Validation failed but this is a question request - salvaging summary ({len(summary)} chars)")
                        # Create minimal valid ManuscriptEdit with just the summary
                        salvaged_edit = {
                            "target_filename": raw.get("target_filename", filename),
                            "scope": "paragraph",  # Safe default
                            "summary": summary,
                            "safety": raw.get("safety", "medium"),
                            "operations": [],  # No operations needed for questions
                        }
                        logger.info(f"✅ Salvaged summary: {summary[:200]}...")
                        return {
                            "llm_response": content,
                            "structured_edit": salvaged_edit,
                            "system_prompt": state.get("system_prompt", ""),
                            "datetime_context": state.get("datetime_context", ""),
                            "generation_context_parts": state.get("generation_context_parts", []),
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
                            "selection_start": state.get("selection_start", -1),
                            "selection_end": state.get("selection_end", -1),
                            "cursor_offset": state.get("cursor_offset", -1),
                            "requested_chapter_number": state.get("requested_chapter_number"),
                        }
                
                # Not salvageable - return error
                return {
                    "llm_response": content,
                    "structured_edit": None,
                    "error": error_msg,
                    "task_status": "error",
                    "system_prompt": state.get("system_prompt", ""),
                    "datetime_context": state.get("datetime_context", ""),
                    "generation_context_parts": state.get("generation_context_parts", []),
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
                    "selection_start": state.get("selection_start", -1),
                    "selection_end": state.get("selection_end", -1),
                    "cursor_offset": state.get("cursor_offset", -1),
                    "requested_chapter_number": state.get("requested_chapter_number"),
                }
                
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON: %s. Content preview (first 500 chars): %s", e, content[:500] if content else "(empty)")
            return {
                "llm_response": content,
                "structured_edit": None,
                "error": f"Failed to parse JSON: {str(e)}",
                "task_status": "error",
                "system_prompt": state.get("system_prompt", ""),
                "datetime_context": state.get("datetime_context", ""),
                "generation_context_parts": state.get("generation_context_parts", []),
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
                "selection_start": state.get("selection_start", -1),
                "selection_end": state.get("selection_end", -1),
                "cursor_offset": state.get("cursor_offset", -1),
                "requested_chapter_number": state.get("requested_chapter_number"),
            }
        except Exception as e:
            logger.error("Failed to parse structured edit: %s. Content preview (first 500 chars): %s", e, content[:500] if content else "(empty)")
            import traceback
            logger.error("Traceback: %s", traceback.format_exc())
            return {
                "llm_response": content,
                "structured_edit": None,
                "error": f"Failed to parse edit plan: {str(e)}",
                "task_status": "error",
                "system_prompt": state.get("system_prompt", ""),
                "datetime_context": state.get("datetime_context", ""),
                "generation_context_parts": state.get("generation_context_parts", []),
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
                "selection_start": state.get("selection_start", -1),
                "selection_end": state.get("selection_end", -1),
                "cursor_offset": state.get("cursor_offset", -1),
                "requested_chapter_number": state.get("requested_chapter_number"),
            }
        
        if structured_edit is None:
            logger.error(
                "Structured edit is None after parse (no JSON/ValidationError path taken). Content len=%s, preview: %s",
                len(content),
                content[:500] + "..." if len(content) > 500 else content,
            )
            return {
                "llm_response": content,
                "structured_edit": None,
                "error": "Failed to produce a valid ManuscriptEdit. Ensure ONLY raw JSON ManuscriptEdit with operations is returned.",
                "task_status": "error",
                "system_prompt": state.get("system_prompt", ""),
                "datetime_context": state.get("datetime_context", ""),
                "generation_context_parts": state.get("generation_context_parts", []),
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
                "selection_start": state.get("selection_start", -1),
                "selection_end": state.get("selection_end", -1),
                "cursor_offset": state.get("cursor_offset", -1),
                "requested_chapter_number": state.get("requested_chapter_number"),
            }
        
        return {
            "llm_response": content,
            "structured_edit": structured_edit,
            "system_prompt": state.get("system_prompt", ""),
            "datetime_context": state.get("datetime_context", ""),
            "generation_context_parts": state.get("generation_context_parts", []),
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
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "cursor_offset": state.get("cursor_offset", -1),
            "requested_chapter_number": state.get("requested_chapter_number"),
        }
        
    except Exception as e:
        logger.error(f"Failed to validate generated output: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "structured_edit": None,
            "error": str(e),
            "task_status": "error",
            "system_prompt": state.get("system_prompt", ""),
            "datetime_context": state.get("datetime_context", ""),
            "generation_context_parts": state.get("generation_context_parts", []),
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
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "cursor_offset": state.get("cursor_offset", -1),
            "requested_chapter_number": state.get("requested_chapter_number"),
        }


async def validate_anchors_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that anchor_text and original_text actually exist in the manuscript.
    Separate operations into validated and invalid lists.
    Uses context-aware fuzzy matching limited to relevant chapters.
    """
    try:
        logger.info("Validating anchor texts against manuscript...")
        
        structured_edit = state.get("structured_edit")
        manuscript = state.get("manuscript", "")
        
        # CRITICAL: Get chapter context for scoped fuzzy matching
        current_chapter_text = state.get("current_chapter_text", "")
        prev_chapter_text = state.get("prev_chapter_text", "")
        next_chapter_text = state.get("next_chapter_text", "")
        
        # Build context-aware search scope (current + adjacent chapters only)
        # This prevents fuzzy matching from finding text in wrong chapters!
        search_scope_parts = []
        if prev_chapter_text:
            search_scope_parts.append(prev_chapter_text)
        if current_chapter_text:
            search_scope_parts.append(current_chapter_text)
        if next_chapter_text:
            search_scope_parts.append(next_chapter_text)
        
        # Fallback to full manuscript only if no chapter context available
        search_scope = "\n\n".join(search_scope_parts) if search_scope_parts else manuscript
        
        if search_scope_parts:
            logger.info(f"🎯 Context-aware fuzzy matching enabled: searching {len(search_scope)} chars across {len(search_scope_parts)} chapters")
        else:
            logger.warning(f"⚠️ No chapter context available - fuzzy matching will search entire manuscript ({len(manuscript)} chars)")
        
        # Handle both dict and Pydantic object
        if not structured_edit:
            logger.warning(
                "No structured_edit to validate (validate_output did not produce a ManuscriptEdit; check logs above for parse/validation errors)"
            )
            return {
                "validated_operations": [],
                "invalid_operations": [],
                "anchor_validation_passed": True,
                "anchor_validation_report": "No structured_edit to validate",
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
                "structured_edit": state.get("structured_edit"),
                "system_prompt": state.get("system_prompt", ""),
                "datetime_context": state.get("datetime_context", ""),
                "generation_context_parts": state.get("generation_context_parts", []),
            }
        
        # Extract operations (handle both dict and Pydantic object)
        if isinstance(structured_edit, dict):
            operations = structured_edit.get("operations", [])
        else:
            operations = structured_edit.operations if hasattr(structured_edit, 'operations') else []
        
        if not operations:
            logger.info("No operations to validate")
            return {
                "validated_operations": [],
                "invalid_operations": [],
                "anchor_validation_passed": True,
                "anchor_validation_report": "No operations to validate",
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
                "structured_edit": state.get("structured_edit"),
                "system_prompt": state.get("system_prompt", ""),
                "datetime_context": state.get("datetime_context", ""),
                "generation_context_parts": state.get("generation_context_parts", []),
            }
        
        validated_operations = []
        invalid_operations = []
        validation_report_lines = []
        
        # Check if file is empty (only frontmatter) - if so, anchor_text is optional for insert operations
        body_only = _strip_frontmatter_block(manuscript)
        is_empty_file = not body_only.strip()
        
        for i, op in enumerate(operations, 1):
            op_dict = op.dict() if hasattr(op, 'dict') else op
            op_type = op_dict.get("op_type", "")
            original_text = op_dict.get("original_text", "")
            anchor_text = op_dict.get("anchor_text", "")
            text = op_dict.get("text", "")
            
            # Determine what text needs validation
            text_to_validate = None
            text_type = None
            
            if op_type in ("replace_range", "delete_range"):
                if original_text and original_text.strip():
                    text_to_validate = original_text.strip()
                    text_type = "original_text"
                else:
                    # Missing required original_text
                    invalid_operations.append({
                        "operation_index": i,
                        "op_type": op_type,
                        "error": "Missing required 'original_text'",
                        "original_operation": op_dict,
                        "intended_text": text[:200] if text else ""
                    })
                    validation_report_lines.append(f"❌ Operation {i} ({op_type}): Missing required 'original_text'")
                    continue
                    
            elif op_type in ("insert_after_heading", "insert_after"):
                if is_empty_file:
                    # Empty file - anchor_text is optional
                    logger.debug(f"✅ Operation {i} ({op_type}): Empty file, anchor_text optional")
                    validated_operations.append(op_dict)
                    validation_report_lines.append(f"✅ Operation {i} ({op_type}): Empty file, anchor_text optional")
                    continue
                elif anchor_text and anchor_text.strip():
                    text_to_validate = anchor_text.strip()
                    text_type = "anchor_text"
                else:
                    # Missing required anchor_text for non-empty file
                    invalid_operations.append({
                        "operation_index": i,
                        "op_type": op_type,
                        "error": "Missing required 'anchor_text' for non-empty file",
                        "original_operation": op_dict,
                        "intended_text": text[:200] if text else ""
                    })
                    validation_report_lines.append(f"❌ Operation {i} ({op_type}): Missing required 'anchor_text'")
                    continue
            
            # Validate the text exists in manuscript
            if text_to_validate:
                # Try exact match first
                if text_to_validate in manuscript:
                    logger.debug(f"✅ Operation {i} ({op_type}): {text_type} found in manuscript (exact match)")
                    validated_operations.append(op_dict)
                    validation_report_lines.append(f"✅ Operation {i} ({op_type}): {text_type} found (exact match)")
                else:
                    # Try normalized match (whitespace normalization)
                    normalized_text = " ".join(text_to_validate.split())
                    normalized_manuscript = " ".join(manuscript.split())
                    
                    if normalized_text in normalized_manuscript:
                        logger.debug(f"✅ Operation {i} ({op_type}): {text_type} found in manuscript (normalized match)")
                        validated_operations.append(op_dict)
                        validation_report_lines.append(f"✅ Operation {i} ({op_type}): {text_type} found (normalized match)")
                    else:
                        # Text not found - try fuzzy matching for auto-correction
                        logger.warning(f"⚠️ Operation {i} ({op_type}): {text_type} NOT FOUND - attempting fuzzy match...")
                        logger.warning(f"   Searched for: {text_to_validate[:100]}...")
                        
                        # Try auto-correction with fuzzy matching (85% similarity threshold)
                        # CRITICAL: Use search_scope (current + adjacent chapters) NOT entire manuscript!
                        corrected_op, was_corrected = auto_correct_operation_anchor(
                            op_dict, 
                            search_scope,  # ✅ Context-aware: only current/prev/next chapters
                            threshold=0.85
                        )
                        
                        if was_corrected:
                            # Fuzzy match succeeded! Use corrected operation
                            logger.debug(f"✅ Operation {i} ({op_type}): AUTO-CORRECTED via fuzzy match")
                            score = corrected_op.get("_anchor_correction_score", 0.0)
                            original_attempt = corrected_op.get("_anchor_original_attempt", "")
                            logger.info(f"   Original attempt: {original_attempt[:100]}...")
                            logger.info(f"   Auto-corrected to: {corrected_op.get(text_type, '')[:100]}...")
                            logger.info(f"   Similarity score: {score:.2%}")
                            validated_operations.append(corrected_op)
                            validation_report_lines.append(f"✅ Operation {i} ({op_type}): AUTO-CORRECTED (fuzzy match, score={score:.2%})")
                        else:
                            # Fuzzy match failed - mark as invalid for self-healing
                            logger.warning(f"❌ Operation {i} ({op_type}): {text_type} NOT FOUND (fuzzy match below threshold)")
                            invalid_operations.append({
                                "operation_index": i,
                                "op_type": op_type,
                                "error": f"{text_type} not found in manuscript (fuzzy match failed)",
                                "invalid_text": text_to_validate,
                                "original_operation": op_dict,
                                "intended_text": text[:200] if text else ""
                            })
                            validation_report_lines.append(f"❌ Operation {i} ({op_type}): {text_type} NOT FOUND in manuscript")
                            validation_report_lines.append(f"   Searched for: {text_to_validate[:100]}...")
                            validation_report_lines.append(f"   Fuzzy match failed (similarity below 85%)")
            else:
                # No text to validate (shouldn't reach here, but handle gracefully)
                validated_operations.append(op_dict)
                validation_report_lines.append(f"✅ Operation {i} ({op_type}): No anchor validation needed")
        
        validation_report = "\n".join(validation_report_lines)
        validation_passed = len(invalid_operations) == 0
        
        logger.info(f"Anchor validation complete: {len(validated_operations)} valid, {len(invalid_operations)} invalid")
        if not validation_passed:
            logger.warning(f"⚠️ Anchor validation FAILED - {len(invalid_operations)} operations have invalid anchors")
        
        # Store invalid_operations in shared_memory so downstream nodes (self_heal, merge) can
        # read them even if top-level state key is dropped (LangGraph/subgraph state merge bug).
        shared_memory = dict(state.get("shared_memory", {}))
        if invalid_operations:
            shared_memory["_invalid_anchor_operations"] = invalid_operations
        
        return {
            "validated_operations": validated_operations,
            "invalid_operations": invalid_operations,
            "anchor_validation_passed": validation_passed,
            "anchor_validation_report": validation_report,
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": shared_memory,
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "manuscript": state.get("manuscript", ""),
            "filename": state.get("filename", ""),
            "current_chapter_text": state.get("current_chapter_text", ""),
            "current_chapter_number": state.get("current_chapter_number"),
            "chapter_ranges": state.get("chapter_ranges", []),
            "current_request": state.get("current_request", ""),
            "structured_edit": state.get("structured_edit"),
            "system_prompt": state.get("system_prompt", ""),
            "datetime_context": state.get("datetime_context", ""),
            "generation_context_parts": state.get("generation_context_parts", []),
        }
        
    except Exception as e:
        logger.error(f"Failed to validate anchors: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        # On error, pass through without validation
        return {
            "validated_operations": [],
            "invalid_operations": [],
            "anchor_validation_passed": True,  # Pass through on error
            "anchor_validation_report": f"Validation error: {str(e)}",
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
            "structured_edit": state.get("structured_edit"),
            "system_prompt": state.get("system_prompt", ""),
            "datetime_context": state.get("datetime_context", ""),
            "generation_context_parts": state.get("generation_context_parts", []),
        }


async def self_heal_anchors_node(state: Dict[str, Any], llm_factory) -> Dict[str, Any]:
    """
    Self-healing node: Ask LLM to fix invalid operations by finding REAL anchor text.
    Provides feedback about what failed and asks for corrections.
    """
    try:
        logger.info("Self-healing invalid anchor texts...")
        
        # Prefer top-level state; fallback to shared_memory in case state key was dropped (subgraph bug).
        invalid_operations = state.get("invalid_operations") or (
            state.get("shared_memory") or {}
        ).get("_invalid_anchor_operations", [])
        validated_operations = state.get("validated_operations", [])
        generation_context_parts = state.get("generation_context_parts", [])
        system_prompt = state.get("system_prompt", "")
        
        if not invalid_operations:
            logger.info("No invalid operations to heal")
            # Just return validated operations as final
            return {
                "healed_operations": [],
                "healing_successful": True,
                "invalid_operations": [],  # No invalid operations to preserve
                "validated_operations": state.get("validated_operations", []),
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
                "structured_edit": state.get("structured_edit"),
            }
        
        logger.info(f"Attempting to heal {len(invalid_operations)} invalid operations")
        
        # Build healing prompt
        healing_prompt_parts = []
        healing_prompt_parts.append("="*80 + "\n")
        healing_prompt_parts.append("⚠️ ANCHOR TEXT VALIDATION FAILED - SELF-HEALING REQUIRED ⚠️\n")
        healing_prompt_parts.append("="*80 + "\n\n")
        healing_prompt_parts.append("Your previous operations contained anchor texts that DO NOT EXIST in the manuscript.\n")
        healing_prompt_parts.append("This means you either:\n")
        healing_prompt_parts.append("1. Copied text from the OUTLINE instead of the MANUSCRIPT\n")
        healing_prompt_parts.append("2. Invented/hallucinated text that isn't in the manuscript\n")
        healing_prompt_parts.append("3. Paraphrased manuscript text instead of copying it EXACTLY\n\n")
        healing_prompt_parts.append("**YOUR TASK**: Fix these operations by finding REAL anchor text from the manuscript.\n\n")
        healing_prompt_parts.append("**FAILED OPERATIONS:**\n\n")
        
        for invalid_op in invalid_operations:
            op_idx = invalid_op.get("operation_index")
            op_type = invalid_op.get("op_type")
            error = invalid_op.get("error")
            invalid_text = invalid_op.get("invalid_text", "")
            intended_text = invalid_op.get("intended_text", "")
            
            healing_prompt_parts.append(f"Operation {op_idx} ({op_type}):\n")
            healing_prompt_parts.append(f"- **Error**: {error}\n")
            if invalid_text:
                healing_prompt_parts.append(f"- **Invalid anchor text you provided**: \"{invalid_text[:200]}...\"\n")
            if intended_text:
                healing_prompt_parts.append(f"- **Content you wanted to insert/edit**: \"{intended_text}...\"\n")
            healing_prompt_parts.append(f"- **What you need to do**: Find REAL text from the MANUSCRIPT TEXT sections above that exists where you want to insert/edit\n\n")
        
        healing_prompt_parts.append("\n" + "="*80 + "\n")
        healing_prompt_parts.append("HOW TO FIX THESE OPERATIONS:\n")
        healing_prompt_parts.append("="*80 + "\n\n")
        healing_prompt_parts.append("1. **Scroll UP** to the MANUSCRIPT TEXT sections (before '=== END OF MANUSCRIPT CONTEXT ===')\n")
        healing_prompt_parts.append("2. **Find the location** where you want to insert/edit content\n")
        healing_prompt_parts.append("3. **Copy EXACT text** from that location (VERBATIM, word-for-word)\n")
        healing_prompt_parts.append("4. **Use that EXACT text** as your anchor_text or original_text\n")
        healing_prompt_parts.append("5. **DO NOT** use text from OUTLINE, STORY OVERVIEW, or invent text\n\n")
        healing_prompt_parts.append("**RETURN ONLY THE CORRECTED OPERATIONS** in the same JSON format:\n\n")
        healing_prompt_parts.append("{\n")
        healing_prompt_parts.append('  "operations": [\n')
        healing_prompt_parts.append("    {\n")
        healing_prompt_parts.append('      "op_type": "insert_after_heading" or "replace_range",\n')
        healing_prompt_parts.append('      "anchor_text": "EXACT TEXT FROM MANUSCRIPT",  // for insert operations\n')
        healing_prompt_parts.append('      "original_text": "EXACT TEXT FROM MANUSCRIPT",  // for replace operations\n')
        healing_prompt_parts.append('      "text": "Your content to insert/replace"\n')
        healing_prompt_parts.append("    }\n")
        healing_prompt_parts.append("  ]\n")
        healing_prompt_parts.append("}\n\n")
        healing_prompt_parts.append("Return ONLY the corrected operations - do not include operations that were already valid.\n")
        
        healing_prompt = "".join(healing_prompt_parts)
        
        # Reconstruct context for healing (reuse generation context)
        context_text = "".join(generation_context_parts) if generation_context_parts else ""
        
        # Build messages for LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=context_text + "\n\n" + healing_prompt)
        ]
        
        # Call LLM for healing
        llm = llm_factory(temperature=0.3, state=state)  # Lower temperature for correction task
        logger.info("Calling LLM for anchor healing...")
        response = await llm.ainvoke(messages)
        llm_response = response.content if hasattr(response, 'content') else str(response)
        
        logger.info(f"LLM healing response received ({len(llm_response)} chars)")
        
        # Parse healed operations
        unwrapped = _unwrap_json_response(llm_response)
        
        try:
            healed_data = json.loads(unwrapped)
            healed_operations = healed_data.get("operations", [])
            
            logger.debug(f"✅ Parsed {len(healed_operations)} healed operations")
            
            return {
                "healed_operations": healed_operations,
                "healing_successful": True,
                "invalid_operations": state.get("invalid_operations", []),  # Preserve for merge node
                "validated_operations": state.get("validated_operations", []),  # Preserve for merge node
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
                "structured_edit": state.get("structured_edit"),
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse healed operations JSON: {e}")
            logger.error(f"LLM response: {llm_response[:500]}")
            # Healing failed - return empty healed operations but PRESERVE invalid_operations
            return {
                "healed_operations": [],
                "healing_successful": False,
                "invalid_operations": state.get("invalid_operations", []),  # CRITICAL: Preserve for conversion to failed_operations!
                "validated_operations": state.get("validated_operations", []),
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
                "structured_edit": state.get("structured_edit"),
            }
        
    except Exception as e:
        logger.error(f"Failed to heal anchors: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "healed_operations": [],
            "healing_successful": False,
            "invalid_operations": state.get("invalid_operations", []),  # CRITICAL: Preserve for conversion to failed_operations!
            "validated_operations": state.get("validated_operations", []),
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
            "structured_edit": state.get("structured_edit"),
        }


async def merge_operations_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge validated and healed operations into final structured_edit.
    Convert unhealed invalid operations to failed_operations for user display.
    """
    try:
        logger.info("Merging validated and healed operations...")
        
        validated_operations = state.get("validated_operations", [])
        healed_operations = state.get("healed_operations", [])
        # Prefer top-level state; fallback to shared_memory in case state key was dropped (subgraph bug).
        invalid_operations = state.get("invalid_operations") or (
            state.get("shared_memory") or {}
        ).get("_invalid_anchor_operations", [])
        healing_successful = state.get("healing_successful", True)
        structured_edit = state.get("structured_edit")
        
        # Combine successfully resolved operations
        final_operations = validated_operations + healed_operations
        
        logger.info(f"Final operations: {len(validated_operations)} validated + {len(healed_operations)} healed = {len(final_operations)} total")
        
        # CRITICAL: Convert unhealed invalid operations to failed_operations for user display
        failed_operations = []
        if invalid_operations and not healing_successful:
            logger.warning(f"⚠️ Self-healing failed - converting {len(invalid_operations)} invalid operations to failed_operations for manual placement")
            for invalid_op in invalid_operations:
                # Extract the original operation with generated content
                original_operation = invalid_op.get("original_operation", {})
                failed_operations.append({
                    "op_type": original_operation.get("op_type", "unknown"),
                    "anchor_text": original_operation.get("anchor_text"),
                    "original_text": original_operation.get("original_text"),
                    "text": original_operation.get("text", ""),  # This is the generated content!
                    "error": invalid_op.get("error", "Anchor text not found - healing failed")
                })
            logger.debug(f"✅ Created {len(failed_operations)} failed_operations entries for chat sidebar display")
        elif invalid_operations and len(healed_operations) < len(invalid_operations):
            # Some operations weren't healed
            unhealed_count = len(invalid_operations) - len(healed_operations)
            logger.warning(f"⚠️ {unhealed_count} operations were not healed - converting to failed_operations")
            # Take the last N invalid operations that weren't healed
            for invalid_op in invalid_operations[-unhealed_count:]:
                original_operation = invalid_op.get("original_operation", {})
                failed_operations.append({
                    "op_type": original_operation.get("op_type", "unknown"),
                    "anchor_text": original_operation.get("anchor_text"),
                    "original_text": original_operation.get("original_text"),
                    "text": original_operation.get("text", ""),
                    "error": invalid_op.get("error", "Anchor text not found")
                })
        
        logger.info(f"📊 Merge summary: {len(final_operations)} successful operations, {len(failed_operations)} failed operations")
        
        logger.info(f"📊 Merge summary: {len(final_operations)} successful operations, {len(failed_operations)} failed operations")
        
        # Update structured_edit with merged operations
        if structured_edit:
            # Create new ManuscriptEdit with merged operations
            updated_edit_dict = structured_edit.dict() if hasattr(structured_edit, 'dict') else structured_edit
            updated_edit_dict["operations"] = final_operations
            
            try:
                updated_structured_edit = ManuscriptEdit(**updated_edit_dict)
                logger.debug(f"✅ Updated ManuscriptEdit with {len(final_operations)} merged operations")
            except ValidationError as e:
                logger.error(f"Failed to create updated ManuscriptEdit: {e}")
                # Keep original structured_edit
                updated_structured_edit = structured_edit
        else:
            updated_structured_edit = None
        
        # Strip internal anchor key from shared_memory before returning to parent
        shared_memory_out = {
            k: v
            for k, v in (state.get("shared_memory") or {}).items()
            if k != "_invalid_anchor_operations"
        }
        return {
            "structured_edit": updated_structured_edit,
            "validated_operations": final_operations,  # CRITICAL: Pass operations for resolution subgraph!
            "failed_operations": failed_operations,  # CRITICAL: Pass failed ops for chat display!
            "generation_context_parts": state.get("generation_context_parts", []),
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": shared_memory_out,
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "manuscript": state.get("manuscript", ""),
            "filename": state.get("filename", ""),
            "current_chapter_text": state.get("current_chapter_text", ""),
            "current_chapter_number": state.get("current_chapter_number"),
            "chapter_ranges": state.get("chapter_ranges", []),
            "current_request": state.get("current_request", ""),
        }
        
    except Exception as e:
        logger.error(f"Failed to merge operations: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "structured_edit": state.get("structured_edit"),
            "validated_operations": [],  # CRITICAL: Pass empty list for resolution subgraph
            "failed_operations": [],
            "generation_context_parts": state.get("generation_context_parts", []),
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
        }


# ============================================
# Subgraph Builder
# ============================================

def build_generation_subgraph(checkpointer, llm_factory, get_datetime_context):
    """
    Build the generation subgraph for fiction editing.
    
    Args:
        checkpointer: LangGraph checkpointer for state persistence
        llm_factory: Function to get LLM instance (signature: (temperature, state) -> LLM)
        get_datetime_context: Function to get datetime context string
    
    Returns:
        Compiled StateGraph subgraph
    """
    workflow = StateGraph(FictionGenerationState)
    
    # Add nodes
    workflow.add_node("build_context", build_generation_context_node)
    workflow.add_node("build_prompt", build_generation_prompt_node)
    
    # LLM call node needs llm_factory - create wrapper
    async def call_llm_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        return await call_generation_llm_node(state, llm_factory)
    
    workflow.add_node("call_llm", call_llm_wrapper)
    workflow.add_node("validate_output", validate_generated_output_node)
    workflow.add_node("validate_anchors", validate_anchors_node)
    
    # Self-healing node needs llm_factory - create wrapper
    async def self_heal_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        return await self_heal_anchors_node(state, llm_factory)
    
    workflow.add_node("self_heal_anchors", self_heal_wrapper)
    workflow.add_node("merge_operations", merge_operations_node)
    
    # Set entry point
    workflow.set_entry_point("build_context")
    
    # Define edges
    workflow.add_edge("build_context", "build_prompt")
    workflow.add_edge("build_prompt", "call_llm")
    workflow.add_edge("call_llm", "validate_output")
    
    # Conditional routing after validate_output
    # If validation passes, go to validate_anchors
    # (validate_output always succeeds, it just validates Pydantic structure)
    workflow.add_edge("validate_output", "validate_anchors")
    
    # Conditional routing after validate_anchors
    def route_after_anchor_validation(state: Dict[str, Any]) -> str:
        """Route based on anchor validation results"""
        anchor_validation_passed = state.get("anchor_validation_passed", True)
        # Use shared_memory fallback in case top-level invalid_operations was dropped (subgraph bug)
        invalid_ops = state.get("invalid_operations") or (
            (state.get("shared_memory") or {}).get("_invalid_anchor_operations", [])
        )
        invalid_count = len(invalid_ops)
        if anchor_validation_passed and invalid_count == 0:
            # All anchors valid - skip healing, go straight to merge
            logger.debug("✅ All anchor validations passed - proceeding to merge")
            return "merge_operations"
        else:
            # Some anchors invalid - attempt self-healing
            logger.info(f"⚠️ {invalid_count} invalid anchors detected - routing to self-healing")
            return "self_heal_anchors"
    
    workflow.add_conditional_edges(
        "validate_anchors",
        route_after_anchor_validation,
        {
            "merge_operations": "merge_operations",
            "self_heal_anchors": "self_heal_anchors"
        }
    )
    
    # After self-healing, always go to merge (whether healing succeeded or not)
    workflow.add_edge("self_heal_anchors", "merge_operations")
    
    # After merge, we're done
    workflow.add_edge("merge_operations", END)
    
    return workflow.compile(checkpointer=checkpointer)

