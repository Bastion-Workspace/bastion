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
)

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
        logger.info("Building generation context...")
        
        manuscript = state.get("manuscript", "")
        filename = state.get("filename", "manuscript.md")
        frontmatter = state.get("frontmatter", {})
        current_request = state.get("current_request", "")
        
        current_chapter_text = state.get("current_chapter_text", "")
        current_chapter_number = state.get("current_chapter_number")
        prev_chapter_text = state.get("prev_chapter_text")
        next_chapter_text = state.get("next_chapter_text")
        
        outline_body = state.get("outline_body")
        rules_body = state.get("rules_body")
        style_body = state.get("style_body")
        characters_bodies = state.get("characters_bodies", [])
        outline_current_chapter_text = state.get("outline_current_chapter_text")
        
        selection_start = state.get("selection_start", -1)
        selection_end = state.get("selection_end", -1)
        cursor_offset = state.get("cursor_offset", -1)
        requested_chapter_number = state.get("requested_chapter_number")
        
        # Use requested chapter number if provided, otherwise use current
        target_chapter_number = requested_chapter_number if requested_chapter_number is not None else current_chapter_number
        
        # Determine chapter labels with actual chapter numbers
        prev_chapter_number = state.get("prev_chapter_number")
        next_chapter_number = state.get("next_chapter_number")
        
        if requested_chapter_number is not None:
            current_chapter_label = f"Chapter {requested_chapter_number}"
        else:
            current_chapter_label = f"Chapter {current_chapter_number}" if current_chapter_number else "Current Chapter"
        
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
            context_parts.append("⚠️ EMPTY FILE DETECTED: This file contains only frontmatter (no chapters yet)\n")
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
            context_parts.append("⚠️ CRITICAL: Maintain continuity with these previously generated chapters!\n")
            context_parts.append("Ensure character states, plot threads, and story flow connect seamlessly.\n\n")
        
        if prev_chapter_text:
            section_header = f"=== MANUSCRIPT TEXT: {prev_chapter_label.upper()} (PREVIOUS - FOR CONTEXT ONLY, USE FOR ANCHORS IF NEEDED) ===\n"
            context_parts.append(section_header)
            context_parts.append(f"{prev_chapter_text}\n\n")
            context_structure["sections"].append({
                "type": "manuscript_prev_chapter",
                "heading": section_header.strip(),
                "content_length": len(prev_chapter_text),
                "chapter_number": prev_chapter_number
            })
        
        section_header = f"=== MANUSCRIPT TEXT: {current_chapter_label.upper()} (CURRENT - EDITABLE, USE FOR ANCHORS) ===\n"
        context_parts.append(section_header)
        # CRITICAL: Send FULL chapter text - NO TRUNCATION
        context_parts.append(f"{current_chapter_text}\n\n")
        context_structure["sections"].append({
            "type": "manuscript_current_chapter",
            "heading": section_header.strip(),
            "content_length": len(current_chapter_text),
            "chapter_number": current_chapter_number
        })
        
        # Add secondary chapters explicitly mentioned in query
        explicit_secondary_chapters = state.get("explicit_secondary_chapters", [])
        chapter_ranges = state.get("chapter_ranges", [])
        
        if explicit_secondary_chapters and chapter_ranges:
            for sec_ch_num in explicit_secondary_chapters:
                # Find the chapter in chapter_ranges
                sec_chapter_range = None
                for ch_range in chapter_ranges:
                    if ch_range.chapter_number == sec_ch_num:
                        sec_chapter_range = ch_range
                        break
                
                if sec_chapter_range:
                    sec_chapter_text = _strip_frontmatter_block(
                        manuscript[sec_chapter_range.start:sec_chapter_range.end]
                    )
                    section_header = f"=== MANUSCRIPT TEXT: CHAPTER {sec_ch_num} (MENTIONED IN QUERY - FOR CONTEXT, USE FOR ANCHORS IF NEEDED) ===\n"
                    context_parts.append(section_header)
                    context_parts.append(f"{sec_chapter_text}\n\n")
                    context_structure["sections"].append({
                        "type": "manuscript_secondary_chapter",
                        "heading": section_header.strip(),
                        "content_length": len(sec_chapter_text),
                        "chapter_number": sec_ch_num,
                        "source": "explicit_query_mention"
                    })
        
        if next_chapter_text:
            section_header = f"=== MANUSCRIPT TEXT: {next_chapter_label.upper()} (NEXT - FOR CONTEXT ONLY, USE FOR ANCHORS IF NEEDED) ===\n"
            context_parts.append(section_header)
            context_parts.append(f"{next_chapter_text}\n\n")
            context_structure["sections"].append({
                "type": "manuscript_next_chapter",
                "heading": section_header.strip(),
                "content_length": len(next_chapter_text),
                "chapter_number": next_chapter_number
            })
        
        # Close manuscript section explicitly
        context_parts.append("=== END OF MANUSCRIPT CONTEXT ===\n")
        context_parts.append("⚠️ All text above this line is MANUSCRIPT TEXT (use for anchors and text matching)\n")
        context_parts.append("⚠️ All text below this line is REFERENCE DOCUMENTS (use for story context, NOT for text matching)\n\n")
        
        # Log manuscript section boundary
        manuscript_end_marker = "=== END OF MANUSCRIPT CONTEXT ==="
        context_structure["manuscript_end_marker"] = manuscript_end_marker
        
        # References (if present, add usage guidance)
        has_any_refs = bool(style_body or rules_body or characters_bodies or outline_body)
        if has_any_refs:
            context_parts.append("=== REFERENCE DOCUMENTS (USE FOR CONSISTENCY) ===\n")
            context_parts.append("The following references are provided to ensure consistency. Use them when generating narrative prose:\n")
            context_parts.append("- Style Guide: Internalize voice, POV, tense, pacing BEFORE writing - permeate every sentence\n")
            context_parts.append("- Outline: Follow story structure and plot beats through natural storytelling\n")
            context_parts.append("- Character Profiles: Reference traits, actions, dialogue patterns when writing character moments\n")
            context_parts.append("- Universe Rules: Verify world-building elements align with established constraints\n\n")
        
        if style_body:
            context_parts.append("=== STYLE GUIDE (voice and tone - READ FIRST) ===\n")
            context_parts.append("Use this guide to establish narrative voice, techniques, and craft. Internalize BEFORE writing:\n")
            context_parts.append(f"{style_body}\n\n")
        
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
        
        # Include outline for story context, but with EXTREME warnings about text matching
        if outline_current_chapter_text:
            warning_banner = "\n" + "="*80 + "\n" + "⚠️  WARNING: STORY OUTLINE BELOW - NOT MANUSCRIPT TEXT  ⚠️\n" + "="*80 + "\n"
            section_header = f"=== OUTLINE (STORY CONTEXT ONLY - NOT EDITABLE, NOT FOR TEXT MATCHING) ===\n\n"
            context_parts.append(warning_banner)
            context_parts.append(section_header)
            context_structure["sections"].append({
                "type": "reference_outline_current_chapter",
                "heading": section_header.strip(),
                "content_length": len(outline_current_chapter_text),
                "has_warning_banner": True,
                "warning": "OUTLINE TEXT - NOT MANUSCRIPT TEXT"
            })
            context_parts.append("🚫🚫🚫 ABSOLUTE PROHIBITION 🚫🚫🚫\n")
            context_parts.append("The text below is an OUTLINE - it tells you WHAT happens in the story.\n")
            context_parts.append("**DO NOT** use outline text for anchors, original_text, or any text matching!\n")
            context_parts.append("**DO NOT** copy, paraphrase, or reuse outline synopsis/beat text in your narrative prose!\n")
            context_parts.append("**DO** use the outline as inspiration for story structure and plot beats.\n")
            context_parts.append("**DO** write original narrative prose that achieves the outline's story goals.\n\n")
            context_parts.append(f"{outline_current_chapter_text}\n\n")
            context_parts.append("=== END OF OUTLINE ===\n")
            context_parts.append("**REMINDER**: The outline above is for STORY CONTEXT ONLY. Use it to understand what should happen, but write ORIGINAL prose.\n\n")
        elif outline_body:
            warning_banner = "\n" + "="*80 + "\n" + "⚠️  WARNING: STORY OUTLINE BELOW - NOT MANUSCRIPT TEXT  ⚠️\n" + "="*80 + "\n"
            section_header = f"=== OUTLINE (STORY CONTEXT ONLY - NOT EDITABLE, NOT FOR TEXT MATCHING) ===\n\n"
            context_parts.append(warning_banner)
            context_parts.append(section_header)
            context_parts.append("🚫🚫🚫 ABSOLUTE PROHIBITION 🚫🚫🚫\n")
            context_parts.append("The text below is an OUTLINE - it tells you WHAT happens in the story.\n")
            context_parts.append("**DO NOT** use outline text for anchors, original_text, or any text matching!\n")
            context_parts.append("**DO NOT** copy, paraphrase, or reuse outline synopsis/beat text in your narrative prose!\n")
            context_parts.append("**DO** use the outline as inspiration for story structure and plot beats.\n")
            context_parts.append("**DO** write original narrative prose that achieves the outline's story goals.\n\n")
            context_parts.append(f"{outline_body}\n\n")
            context_parts.append("=== END OF OUTLINE ===\n")
            context_parts.append("**REMINDER**: The outline above is for STORY CONTEXT ONLY. Use it to understand what should happen, but write ORIGINAL prose.\n\n")
        
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
                "\n⚠️ CREATIVE FREEDOM GRANTED: User has requested enhancements/additions. "
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
                "=== OUTLINE AS NARRATIVE BLUEPRINT ===\n"
                "The chapter outline provides story beats and structure, NOT a script to paraphrase or copy.\n\n"
                "**🚫 ABSOLUTE PROHIBITION: DO NOT COPY OR PARAPHRASE OUTLINE TEXT**\n"
                "- ❌ DO NOT copy outline synopsis text into your narrative prose\n"
                "- ❌ DO NOT paraphrase outline beats word-for-word\n"
                "- ❌ DO NOT convert outline bullets into simple prose sentences\n"
                "- ❌ DO NOT use outline language directly in your writing\n"
                "- ✅ DO creatively interpret outline beats into full narrative scenes\n"
                "- ✅ DO write original prose that achieves the outline's story goals\n"
                "- ✅ DO use the outline as inspiration, not as source text to copy\n\n"
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
        
        return {
            "generation_context_parts": context_parts,
            "generation_context_structure": context_structure,
            "is_empty_file": is_empty_file,
            "target_chapter_number": target_chapter_number,
            "current_chapter_label": current_chapter_label,
            # CRITICAL: Preserve state for subsequent nodes
            "system_prompt": state.get("system_prompt", ""),  # ✅ PRESERVE system_prompt!
            "datetime_context": state.get("datetime_context", ""),  # ✅ PRESERVE datetime_context!
            "metadata": state.get("metadata", {}),  # Contains user_chat_model!
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # ✅ PRESERVE manuscript context for next node!
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
        logger.error(f"Failed to build generation context: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "generation_context_parts": [],
            "error": str(e),
            "task_status": "error",
            # CRITICAL: Preserve state even on error
            "system_prompt": state.get("system_prompt", ""),  # ✅ PRESERVE system_prompt!
            "datetime_context": state.get("datetime_context", ""),  # ✅ PRESERVE datetime_context!
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # ✅ PRESERVE manuscript context even on error!
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


async def build_generation_prompt_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Build generation prompt: construct system prompt and user message"""
    try:
        logger.info("Building generation prompt...")
        
        # Get system prompt from state (built by main agent)
        system_prompt = state.get("system_prompt", "")
        logger.info(f"📊 SYSTEM PROMPT DEBUG: length = {len(system_prompt):,} chars (~{len(system_prompt) // 4:,} tokens)")
        logger.info(f"📊 SYSTEM PROMPT first 500 chars: {system_prompt[:500]}")
        logger.info(f"📊 SYSTEM PROMPT last 500 chars: {system_prompt[-500:]}")
        
        # ⚠️ CRITICAL DEBUG: Check if references are INSIDE the system prompt
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
        chapter_ranges = state.get("chapter_ranges", [])
        
        # datetime_context should be provided by main agent via state
        datetime_context = state.get("datetime_context", "")
        
        messages = [
            SystemMessage(content=system_prompt),
            SystemMessage(content=datetime_context) if datetime_context else None,
            HumanMessage(content="".join(generation_context_parts))
        ]
        # Remove None entries
        messages = [m for m in messages if m is not None]
        
        # Add selection/cursor context
        selection_context = ""
        if selection_start >= 0 and selection_end > selection_start:
            selected_text = manuscript[selection_start:selection_end]
            selection_context = (
                f"\n\n=== USER HAS SELECTED TEXT ===\n"
                f"Selected text (characters {selection_start}-{selection_end}):\n"
                f'"""{selected_text[:500]}{"..." if len(selected_text) > 500 else ""}"""\n\n'
                "⚠️ User selected this specific text! Use it as your anchor:\n"
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
            is_creating_new_chapter = (
                requested_chapter_number is not None and 
                state.get("current_chapter_text", "").strip() == "" and
                any(keyword in current_request.lower() for keyword in ["create", "craft", "write", "generate", "chapter"])
            )
            
            new_chapter_hints = ""
            if is_creating_new_chapter:
                # Find the last chapter in the manuscript
                if chapter_ranges:
                    last_chapter_range = chapter_ranges[-1]
                    last_chapter_num = last_chapter_range.chapter_number
                    # Get the last line of the last chapter
                    last_chapter_text = manuscript[last_chapter_range.start:last_chapter_range.end]
                    last_lines = last_chapter_text.strip().split('\n')
                    last_line = last_lines[-1] if last_lines else ""
                    
                    new_chapter_hints = (
                        f"\n=== CREATING NEW CHAPTER {requested_chapter_number} ===\n"
                        f"The chapter doesn't exist yet - you need to insert it after the last existing chapter.\n"
                        f"Last existing chapter: Chapter {last_chapter_num}\n"
                        f"**CRITICAL**: Use 'insert_after_heading' with anchor_text set to the LAST LINE of Chapter {last_chapter_num}\n"
                        f"Find the last line of Chapter {last_chapter_num} in the manuscript above and use it as anchor_text.\n\n"
                        f"Example JSON structure:\n"
                        f"{{\n"
                        f'  "target_filename": "manuscript.md",\n'
                        f'  "scope": "chapter",\n'
                        f'  "summary": "Generated Chapter {requested_chapter_number}",\n'
                        f'  "safety": "medium",\n'
                        f'  "operations": [{{\n'
                        f'    "op_type": "insert_after_heading",\n'
                        f'    "anchor_text": "She closed the door behind her.",\n'
                        f'    "text": "## Chapter {requested_chapter_number}\\n\\nYour chapter content here...",\n'
                        f'    "start": 0,\n'
                        f'    "end": 0\n'
                        f"  }}]\n"
                        f"}}\n\n"
                        f"**MANDATORY**: Your 'text' field MUST start with '## Chapter {requested_chapter_number}' followed by two newlines, then your chapter content.\n"
                        f"**DO NOT** use '## Chapter {requested_chapter_number}' as anchor_text - it doesn't exist yet!\n"
                        f"**DO NOT** insert at the beginning of the file - insert after the last chapter!\n"
                        f"**DO NOT** forget the chapter header - it is REQUIRED for all new chapters!\n\n"
                    )
            
            messages.append(HumanMessage(content=(
                "\n" + "="*80 + "\n"
                "⚠️⚠️⚠️ OUTPUT FORMAT REQUIREMENTS ⚠️⚠️⚠️\n"
                "="*80 + "\n\n"
                "YOU MUST RESPOND WITH **JSON ONLY**\n\n"
                "❌ DO NOT use XML tags like <operation> or <op_type>\n"
                "❌ DO NOT use YAML format (key: value without braces)\n"
                "❌ DO NOT use any format other than JSON\n"
                "✅ ONLY return valid JSON with curly braces { }\n"
                "✅ ONLY return a JSON object matching ManuscriptEdit structure\n\n"
                f"USER REQUEST: {current_request}\n\n"
                + selection_context +
                granular_correction_hints +
                new_chapter_hints +
                "\n" + "="*80 + "\n"
                "⚠️⚠️⚠️ BEFORE YOU GENERATE YOUR JSON RESPONSE ⚠️⚠️⚠️\n"
                "="*80 + "\n\n"
                "**STEP 1: SCROLL UP TO FIND MANUSCRIPT TEXT**\n"
                "Look for sections labeled 'MANUSCRIPT TEXT: CHAPTER N'.\n"
                "These sections contain the ACTUAL text from the manuscript file.\n\n"
                "**STEP 2: FIND THE EXACT TEXT TO EDIT IN THE MANUSCRIPT**\n"
                "Find the sentence or paragraph in the MANUSCRIPT TEXT sections that matches the user's request.\n"
                "Copy 20-40 words of EXACT, VERBATIM text from that MANUSCRIPT section.\n\n"
                "**STEP 3: DO NOT USE OUTLINE TEXT**\n"
                "The OUTLINE sections (below '=== END OF MANUSCRIPT CONTEXT ===') contain story beats.\n"
                "These words DO NOT EXIST in the manuscript file!\n"
                "If you copy outline text into 'original_text', the system will fail to find it!\n\n"
                "**⚠️ CRITICAL: ALL OPERATIONS MUST HAVE ANCHORS ⚠️**\n\n"
                "**FOR replace_range/delete_range:**\n"
                "- You MUST provide 'original_text' with EXACT text from the manuscript\n"
                "- If you don't have 'original_text', the operation will FAIL and fall back to cursor position\n"
                "- NEVER create a replace_range operation without 'original_text'\n\n"
                "**FOR insert_after_heading (NEW TEXT):**\n"
                "- Use this when adding NEW content that doesn't exist in the manuscript\n"
                "- You MUST provide 'anchor_text' with EXACT text from the manuscript to insert after\n"
                "- Find the sentence/paragraph where the new text should go\n"
                "- Copy that sentence/paragraph EXACTLY as 'anchor_text'\n"
                "- The new text will be inserted immediately after that anchor\n"
                "- **FOR NEW CHAPTERS**: Your 'text' field MUST start with '## Chapter N' followed by two newlines, then your chapter content\n"
                "- **MANDATORY**: All new chapter content must include the chapter header - do not omit it!\n\n"
                "**DECISION RULE:**\n"
                "- Editing EXISTING text? → Use 'replace_range' with 'original_text'\n"
                "- Adding NEW text? → Use 'insert_after_heading' with 'anchor_text'\n"
                "- NEVER use 'replace_range' without 'original_text' - it will fail!\n\n"
                "**VERIFICATION CHECKLIST BEFORE GENERATING JSON:**\n"
                "☐ I found the text in a 'MANUSCRIPT TEXT: CHAPTER N' section\n"
                "☐ I copied it EXACTLY as written in that section\n"
                "☐ I did NOT copy text from any OUTLINE section\n"
                "☐ The text I copied is BEFORE the '=== END OF MANUSCRIPT CONTEXT ===' marker\n\n"
                "="*80 + "\n"
                "NOW GENERATE YOUR JSON RESPONSE:\n"
                "="*80 + "\n\n"
                "⚠️⚠️⚠️ CRITICAL: REQUIRED JSON STRUCTURE ⚠️⚠️⚠️\n\n"
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
                "❌ DO NOT use XML tags: <operation>, <op_type>, <text>, etc.\n"
                "❌ DO NOT use YAML format: op_type: value\n"
                "❌ DO NOT return a single operation object!\n"
                "❌ DO NOT use 'operation' (singular) - must be 'operations' (array)!\n"
                "❌ DO NOT nest op_type at top level - it goes INSIDE operations array!\n"
                "✅ ONLY return JSON with curly braces { } and square brackets [ ]\n"
                "✅ ONLY use the exact field names shown above\n\n"
                "For REPLACE/DELETE operations in prose (no headers), you MUST provide robust anchors:\n\n"
                "**⚠️ PREFER GRANULAR EDITS: Use the SMALLEST possible 'original_text' match**\n"
                "- For word-level changes: 10-15 words of context (minimal, unique match)\n"
                "- For phrase changes: 15-20 words of context (minimal, unique match)\n"
                "- For sentence changes: Just the sentence(s) that need changing (15-30 words)\n"
                "- Only use 30-40 words when smaller matches aren't unique enough\n"
                "- **SMALLER IS BETTER** - precise alignment is more important than large context\n\n"
                "**OPTION 1 (BEST): Use selection as anchor**\n"
                "- If user selected text, match it EXACTLY in 'original_text'\n"
                "- Use the selection plus minimal surrounding context (10-15 words) if needed for uniqueness\n"
                "- ⚠️ Copy from manuscript text, NOT from outline!\n\n"
                "**OPTION 2: Use left_context + right_context**\n"
                "- left_context: 30-50 chars BEFORE the target (exact text from manuscript)\n"
                "- right_context: 30-50 chars AFTER the target (exact text from manuscript)\n"
                "- ⚠️ Both must be copied from manuscript text, NOT from outline!\n\n"
                "**OPTION 3: Use long original_text**\n"
                "- Include 20-40 words of EXACT, VERBATIM text to replace\n"
                "- Include complete sentences with natural boundaries\n"
                "- ⚠️ Copy from manuscript text above, NEVER from outline text!\n\n"
                "⚠️ NEVER include chapter headers (##) in original_text - they will be deleted!\n"
                "⚠️ NEVER use outline text as anchor - it doesn't exist in the manuscript and will fail to match!\n\n"
                "="*80 + "\n"
                "⚠️ FINAL REMINDER: RETURN JSON ONLY ⚠️\n"
                "="*80 + "\n\n"
                "Start your response with { and end with }\n"
                "Do NOT use any XML-style tags or YAML-style notation\n"
                "Your response must be valid JSON that can be parsed by json.loads()\n\n"
                "NOW GENERATE YOUR JSON RESPONSE:\n"
            )))
        
        return {
            "generation_messages": messages,
            "system_prompt": system_prompt,
            # CRITICAL: Preserve state for subsequent nodes
            "system_prompt": state.get("system_prompt", ""),
            "datetime_context": state.get("datetime_context", ""),
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # ✅ PRESERVE manuscript context!
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
        logger.error(f"Failed to build generation prompt: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "generation_messages": [],
            "error": str(e),
            "task_status": "error",
            # CRITICAL: Preserve state even on error
            "system_prompt": state.get("system_prompt", ""),  # ✅ PRESERVE system_prompt!
            "datetime_context": state.get("datetime_context", ""),  # ✅ PRESERVE datetime_context!
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # ✅ PRESERVE manuscript context even on error!
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
        
        # Get LLM from factory
        llm = llm_factory(temperature=0.4, state=state)
        
        start_time = datetime.now()
        response = await llm.ainvoke(generation_messages)
        
        content = response.content if hasattr(response, 'content') else str(response)
        
        content = _unwrap_json_response(content)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"LLM generation completed in {elapsed:.2f}s")
        
        return {
            "llm_response": content,
            "llm_response_raw": content,
            # CRITICAL: Preserve state for subsequent nodes
            "system_prompt": state.get("system_prompt", ""),
            "datetime_context": state.get("datetime_context", ""),
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


async def validate_generated_output_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Validate generated output: check for outline copying, validate structure"""
    try:
        logger.info("Validating generated output...")
        
        content = state.get("llm_response", "")
        filename = state.get("filename", "manuscript.md")
        
        if not content:
            return {
                "structured_edit": None,
                "error": "No LLM response to validate",
                "task_status": "error",
                "system_prompt": state.get("system_prompt", ""),
                "datetime_context": state.get("datetime_context", ""),
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
                raw.setdefault("target_filename", filename)
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
                logger.info(f"✅ Validated ManuscriptEdit with {operations_count} operations")
                
                # For questions: empty operations is valid (analysis-only response)
                request_type = state.get("request_type", "")
                if request_type == "question" and operations_count == 0:
                    logger.info("Question request with no operations - this is valid (analysis in summary field)")
            except ValidationError as ve:
                # Provide detailed validation error
                error_details = []
                for error in ve.errors():
                    field = " -> ".join(str(loc) for loc in error.get("loc", []))
                    msg = error.get("msg", "Validation error")
                    error_details.append(f"{field}: {msg}")
                
                error_msg = f"ManuscriptEdit validation failed:\n" + "\n".join(error_details)
                logger.error(f"❌ {error_msg}")
                return {
                    "llm_response": content,
                    "structured_edit": None,
                    "error": error_msg,
                    "task_status": "error",
                    "system_prompt": state.get("system_prompt", ""),
                    "datetime_context": state.get("datetime_context", ""),
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
            logger.error(f"Failed to parse JSON: {e}")
            return {
                "llm_response": content,
                "structured_edit": None,
                "error": f"Failed to parse JSON: {str(e)}",
                "task_status": "error",
                "system_prompt": state.get("system_prompt", ""),
                "datetime_context": state.get("datetime_context", ""),
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
            logger.error(f"Failed to parse structured edit: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "llm_response": content,
                "structured_edit": None,
                "error": f"Failed to parse edit plan: {str(e)}",
                "task_status": "error",
                "system_prompt": state.get("system_prompt", ""),
                "datetime_context": state.get("datetime_context", ""),
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
            return {
                "llm_response": content,
                "structured_edit": None,
                "error": "Failed to produce a valid ManuscriptEdit. Ensure ONLY raw JSON ManuscriptEdit with operations is returned.",
                "task_status": "error",
                "system_prompt": state.get("system_prompt", ""),
                "datetime_context": state.get("datetime_context", ""),
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
    
    # Set entry point
    workflow.set_entry_point("build_context")
    
    # Define edges
    workflow.add_edge("build_context", "build_prompt")
    workflow.add_edge("build_prompt", "call_llm")
    workflow.add_edge("call_llm", "validate_output")
    workflow.add_edge("validate_output", END)
    
    return workflow.compile(checkpointer=checkpointer)

