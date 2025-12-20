"""
Whole-Book Generation Subgraph for Fiction Agent

Handles end-to-end book generation:
1. Chapter planning from outline
2. Multi-chapter orchestration loop
3. Post-generation validation
4. Direct document writing (no operations to approve)
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage

from orchestrator.utils.fiction_utilities import (
    ChapterRange,
    find_chapter_ranges,
    strip_frontmatter_block as _strip_frontmatter_block,
)
from orchestrator.subgraphs.fiction_generation_subgraph import build_generation_subgraph
from orchestrator.subgraphs.fiction_validation_subgraph import build_validation_subgraph
from orchestrator.tools.document_editing_tools import update_document_content_tool
from orchestrator.tools.document_tools import get_document_content_tool, search_documents_structured

logger = logging.getLogger(__name__)


# ============================================
# State Schema
# ============================================

# Use Dict[str, Any] for compatibility with main agent state
BookGenerationState = Dict[str, Any]


# ============================================
# Utility Functions
# ============================================

def _extract_chapter_count_from_outline(outline_body: str) -> Optional[int]:
    """Extract chapter count from outline by scanning for chapter headers"""
    if not outline_body:
        return None
    
    # Look for patterns like "## Chapter 1", "## Chapter 2", etc.
    chapter_pattern = r'(?i)(?:^|\n)##?\s*(?:Chapter\s+)?(\d+)'
    matches = re.findall(chapter_pattern, outline_body)
    
    if matches:
        # Get the highest chapter number
        chapter_numbers = [int(m) for m in matches if m.isdigit()]
        if chapter_numbers:
            max_chapter = max(chapter_numbers)
            logger.info(f"📖 Found {max_chapter} chapters in outline")
            return max_chapter
    
    return None


def _estimate_chapters_from_outline(outline_body: str) -> int:
    """Estimate chapter count based on outline structure"""
    if not outline_body:
        return 20  # Default estimate
    
    # Count major sections (## headers)
    section_pattern = r'(?i)(?:^|\n)##\s+'
    sections = len(re.findall(section_pattern, outline_body))
    
    # Count major beats (### headers)
    beat_pattern = r'(?i)(?:^|\n)###\s+'
    beats = len(re.findall(beat_pattern, outline_body))
    
    # Estimate: 1-2 chapters per major section, or 1 chapter per 3-4 beats
    if sections > 0:
        estimated = max(sections, beats // 3)
    elif beats > 0:
        estimated = max(beats // 3, 10)
    else:
        # Fallback: estimate based on word count (80k words / 3k per chapter ≈ 27)
        word_count = len(outline_body.split())
        estimated = max(word_count // 3000, 20)
    
    logger.info(f"📖 Estimated {estimated} chapters from outline structure")
    return estimated


def _assemble_complete_manuscript(frontmatter: Dict[str, Any], generated_chapters: Dict[int, str]) -> str:
    """Assemble complete manuscript with frontmatter and all chapters"""
    # Build frontmatter block
    import yaml
    frontmatter_yaml = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)
    frontmatter_block = f"---\n{frontmatter_yaml}---\n\n"
    
    # Build chapter content
    chapter_content = []
    for ch_num in sorted(generated_chapters.keys()):
        chapter_text = generated_chapters[ch_num]
        # Ensure chapter has header
        if not chapter_text.strip().startswith("##"):
            chapter_text = f"## Chapter {ch_num}\n\n{chapter_text}"
        chapter_content.append(chapter_text)
    
    # Combine
    complete_manuscript = frontmatter_block + "\n\n".join(chapter_content)
    return complete_manuscript


# ============================================
# Node Functions
# ============================================

async def plan_chapters_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze outline and determine chapter structure"""
    try:
        logger.info("📖 Planning chapters for whole-book generation...")
        
        outline_body = state.get("outline_body")
        if not outline_body:
            logger.warning("⚠️ No outline found - cannot plan chapters")
            return {
                "error": "No outline found for whole-book generation",
                "task_status": "error",
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
            }
        
        # Try to extract chapter count from outline
        chapter_count = _extract_chapter_count_from_outline(outline_body)
        
        # If not found, estimate
        if not chapter_count:
            chapter_count = _estimate_chapters_from_outline(outline_body)
        
        logger.info(f"📖 Planned {chapter_count} chapters for generation")
        
        return {
            "estimated_chapter_count": chapter_count,
            "chapter_range": (1, chapter_count),
            "is_multi_chapter": True,
            "current_generation_chapter": 1,
            "generated_chapters": {},
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "outline_body": outline_body,
            "style_body": state.get("style_body"),
            "rules_body": state.get("rules_body"),
            "characters_bodies": state.get("characters_bodies", []),
            "manuscript": state.get("manuscript", ""),
            "filename": state.get("filename", ""),
            "frontmatter": state.get("frontmatter", {}),
            "system_prompt": state.get("system_prompt", ""),  # Preserve for generation subgraph
            "datetime_context": state.get("datetime_context", ""),  # Preserve for generation subgraph
        }
        
    except Exception as e:
        logger.error(f"Failed to plan chapters: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "error": str(e),
            "task_status": "error",
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }


async def prepare_generation_state_node(state: Dict[str, Any], get_datetime_context) -> Dict[str, Any]:
    """Prepare generation state: build system_prompt and datetime_context"""
    try:
        estimated_chapter_count = state.get("estimated_chapter_count", 20)
        
        logger.info(f"📖 Preparing whole-book generation for {estimated_chapter_count} chapters with emphasis on narrative depth")
        
        # Build system prompt (same as fiction_editing_agent)
        # This is a simplified version - the generation subgraph will use it or fallback
        system_prompt = (
            "CODE VERSION: 2025-12-14-22:15 UTC - IF YOU SEE THIS, CODE IS FRESH!\n\n"
            "You are a MASTER NOVELIST editor/generator. Persona disabled.\n\n"
            "=== REFERENCE USAGE (CRITICAL) ===\n\n"
            "**The Style Guide is HOW to write. The Outline is WHAT happens.**\n\n"
            "When references are available, you MUST use them appropriately:\n\n"
            "- **STYLE GUIDE**: Use for HOW to write narrative prose\n"
            "  - Internalize narrative voice, POV, tense, pacing BEFORE writing - the voice must permeate every sentence\n"
            "  - Apply dialogue style, sensory detail level, and show-don't-tell techniques\n"
            "  - Match sentence structure patterns, rhythm, and descriptive style\n"
            "  - Every sentence must sound like it was written in the Style Guide's voice\n\n"
            "- **OUTLINE**: Use for WHAT happens in the story (NEVER for text matching!)\n"
            "  - CRITICAL: Outline text does NOT exist in manuscript - NEVER use for anchors/original_text!\n"
            "  - ABSOLUTE PROHIBITION: DO NOT copy, paraphrase, or reuse outline synopsis/beat text in your narrative prose\n"
            "  - DO creatively interpret outline beats into original narrative scenes with full prose\n"
            "  - Follow story structure and plot beats as GUIDANCE, not as SOURCE TEXT to copy\n"
            "  - Achieve outline's story goals through natural storytelling (don't convert beats mechanically)\n"
            "  - For all text matching (anchors), use MANUSCRIPT text only, never outline text\n\n"
            "- **CHARACTER PROFILES**: Use when writing character appearances, actions, dialogue, and internal thoughts\n"
            "  - **CRITICAL**: Each character profile is for a DIFFERENT character with distinct traits, dialogue patterns, and behaviors\n"
            "  - Match the CORRECT character's profile - do not confuse characters\n"
            "  - Ensure dialogue patterns, actions, and appearances match profiles (vary phrasing, keep facts consistent)\n"
            "  - Draw from character profiles for authentic details, but express them differently each time\n\n"
            "- **UNIVERSE RULES**: Use to ensure world-building consistency\n"
            "  - Verify all world-building elements (magic systems, technology, physics) align with universe rules\n"
            "  - Ensure plot events and character actions don't violate established universe constraints\n"
            "  - Check timeline and events are consistent with universe history\n\n"
            "**IMPORTANT**: Maintain originality and do not copy from references. Adhere strictly to the project's Style Guide and Rules above all else.\n\n"
            "=== WHOLE-BOOK GENERATION MODE - NARRATIVE DEPTH FOCUS ===\n\n"
            "You are generating a complete book chapter-by-chapter.\n\n"
            f"**CRITICAL: NARRATIVE DEPTH OVER SUMMARY** (This book has {estimated_chapter_count} chapters)\n\n"
            "DO NOT write summaries or treatments. Write COMPLETE NARRATIVE PROSE.\n"
            "Each outline beat must be transformed into a FULL, IMMERSIVE SCENE.\n\n"
            "**WHAT NARRATIVE DEPTH MEANS:**\n\n"
            "1. **SCENE-LEVEL WRITING (Not Summary)**\n"
            "   ❌ BAD (Summary): 'Sarah confronted Tom about the missing files. He denied it.'\n"
            "   ✅ GOOD (Scene): Show the confrontation moment-by-moment with dialogue, body language,\n"
            "      setting details, internal reactions, tension building, and emotional beats.\n\n"
            "2. **SHOW, DON'T TELL**\n"
            "   ❌ BAD: 'Peterson was nervous.'\n"
            "   ✅ GOOD: 'Peterson's fingers drummed the desk, each tap louder than the last. His eyes\n"
            "      darted to the door, then back to the phone, willing it to ring.'\n\n"
            "3. **RICH DIALOGUE (Not Exposition)**\n"
            "   ❌ BAD: 'How are you?' 'Fine.' He left.\n"
            "   ✅ GOOD: Multi-turn exchanges that reveal character, build relationships, create subtext.\n"
            "      Include dialogue tags, action beats, pauses, reactions, internal thoughts.\n\n"
            "4. **SENSORY IMMERSION**\n"
            "   Every scene should ground the reader in specific sensory details:\n"
            "   - What characters SEE (light, shadow, colors, movement)\n"
            "   - What they HEAR (sounds, silence, music, voices)\n"
            "   - What they FEEL (textures, temperatures, physical sensations)\n"
            "   - What they SMELL and TASTE when relevant\n"
            "   - Create atmosphere through environmental details\n\n"
            "5. **CHARACTER INTERIORITY**\n"
            "   - Show character thoughts and emotional states through physical reactions\n"
            "   - Internal conflicts and decision-making processes\n"
            "   - Character voice in internal narration\n"
            "   - Emotional responses to events (not just stating 'he felt sad')\n\n"
            "6. **ORGANIC PACING**\n"
            "   - Let important moments BREATHE - don't rush through key scenes\n"
            "   - Include beats between major plot points (reactions, transitions, reflection)\n"
            "   - Build tension gradually through scene development\n"
            "   - Quiet moments matter as much as action sequences\n\n"
            "**OUTLINE BEATS ARE GOALS, NOT SCRIPTS:**\n"
            "- If outline says 'Character discovers secret' → Write the full discovery scene:\n"
            "  * Lead-up: What brings character to this moment?\n"
            "  * Discovery: Show the moment of realization through action and reaction\n"
            "  * Aftermath: How does character process this? What do they do next?\n"
            "- ONE outline beat can easily become 500-1000+ words of immersive scene\n\n"
            "**DEPTH INDICATORS (Use These as Quality Checks):**\n"
            "✓ Each major outline beat becomes a complete scene (not a paragraph)\n"
            "✓ Dialogue feels natural with multiple exchanges, not just quick information drops\n"
            "✓ Settings are vividly described with specific sensory details\n"
            "✓ Character emotions are SHOWN through action/reaction, not TOLD\n"
            "✓ Scenes have beginnings, middles, and ends with natural transitions\n"
            "✓ Reader can visualize and experience each moment\n"
            "✓ Pacing varies - intense moments and quieter beats both get proper development\n\n"
            "**YOU ARE WRITING A NOVEL:**\n"
            "Every chapter should read like immersive fiction that pulls readers into the story world.\n"
            "Quality narrative craft trumps efficiency. Develop each scene fully. Show, don't tell.\n"
            "Let characters breathe and interact naturally. Build atmosphere and emotional resonance.\n\n"
            "STRUCTURED OUTPUT REQUIRED:\n"
            "You MUST return ONLY raw JSON matching ManuscriptEdit structure.\n"
            "For each chapter, provide an operation with op_type='insert_after_heading'.\n"
            "The 'text' field MUST start with '## Chapter N' followed by two newlines, then your chapter content.\n"
        )
        
        # Get datetime context
        datetime_context = get_datetime_context() if get_datetime_context else ""
        
        return {
            "system_prompt": system_prompt,
            "datetime_context": datetime_context,
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }
        
    except Exception as e:
        logger.error(f"Failed to prepare generation state: {e}")
        return {
            "system_prompt": "",
            "datetime_context": "",
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }


async def orchestrate_chapter_generation_node(
    state: Dict[str, Any],
    generation_subgraph,
    validation_subgraph
) -> Dict[str, Any]:
    """Orchestrate chapter-by-chapter generation using existing subgraphs"""
    try:
        current_ch = state.get("current_generation_chapter")
        chapter_range = state.get("chapter_range")
        generated_chapters = state.get("generated_chapters", {})
        total_chapters = state.get("estimated_chapter_count", 0)
        
        if not current_ch or not chapter_range:
            logger.warning("⚠️ Invalid chapter generation state")
            return {
                "generation_complete": True,
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
            }
        
        start_ch, end_ch = chapter_range
        
        # Check if we're done
        if current_ch > end_ch:
            logger.info(f"✅ All {end_ch} chapters generated!")
            return {
                "generation_complete": True,
                "generated_chapters": generated_chapters,
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
            }
        
        logger.info(f"📖 Generating Chapter {current_ch}/{total_chapters}...")
        
        # Prepare state for generation subgraph
        # Get previous chapter text for continuity
        prev_chapter_text = None
        if current_ch > start_ch:
            prev_chapter_text = generated_chapters.get(current_ch - 1)
        
        # Extract outline section for current chapter
        outline_body = state.get("outline_body", "")
        outline_current_chapter_text = None
        if outline_body:
            chapter_pattern = rf"(?i)(?:^|\n)##?\s*(?:Chapter\s+)?{current_ch}[:\s]*(.*?)(?=\n##?\s*(?:Chapter\s+)?\d+|\Z)"
            match = re.search(chapter_pattern, outline_body, re.DOTALL)
            if match:
                outline_current_chapter_text = match.group(1).strip()
        
        # Get system prompt and datetime context (need to build them)
        # For now, we'll let the generation subgraph build them
        # But we need to preserve the state structure
        
        # Build chapter generation state
        # Note: The generation subgraph's build_generation_prompt_node expects system_prompt
        # to be in state, but if it's missing, it will build a fallback. We'll pass through
        # whatever is in state, and the generation subgraph will handle it.
        chapter_state = {
            "manuscript": state.get("manuscript", ""),
            "filename": state.get("filename", ""),
            "frontmatter": state.get("frontmatter", {}),
            "current_chapter_number": current_ch,
            "requested_chapter_number": current_ch,
            "current_chapter_text": "",  # Empty for new generation
            "prev_chapter_text": prev_chapter_text,
            "prev_chapter_number": current_ch - 1 if current_ch > start_ch else None,
            "next_chapter_text": None,
            "next_chapter_number": None,
            "outline_body": outline_body,
            "outline_current_chapter_text": outline_current_chapter_text,
            "style_body": state.get("style_body"),
            "rules_body": state.get("rules_body"),
            "characters_bodies": state.get("characters_bodies", []),
            "is_multi_chapter": True,
            "is_whole_book_generation": True,  # Flag for whole-book mode
            "generated_chapters": generated_chapters,  # For continuity context
            "current_request": f"Generate Chapter {current_ch} following the outline and style guide",
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "system_prompt": state.get("system_prompt", ""),  # Pass through if available
            "datetime_context": state.get("datetime_context", ""),  # Pass through if available
            "chapter_ranges": state.get("chapter_ranges", []),
        }
        
        # Call generation subgraph
        generation_result = await generation_subgraph.ainvoke(chapter_state)
        
        # Extract generated chapter text from structured_edit
        structured_edit = generation_result.get("structured_edit")
        if structured_edit and structured_edit.get("operations"):
            # Extract text from first operation
            first_op = structured_edit["operations"][0]
            chapter_text = first_op.get("text", "")
            
            # Ensure chapter has header
            if chapter_text and not chapter_text.strip().startswith("##"):
                chapter_text = f"## Chapter {current_ch}\n\n{chapter_text}"
            
            if chapter_text:
                generated_chapters[current_ch] = chapter_text
                word_count = len(chapter_text.split())
                
                logger.info(f"✅ Chapter {current_ch} generated ({len(chapter_text)} chars, ~{word_count:,} words)")
                
                # Provide narrative depth assessment (not strict length requirements)
                if word_count < 1500:
                    logger.warning(f"⚠️ Chapter {current_ch} appears VERY SHORT ({word_count:,} words)")
                    logger.warning(f"   This suggests outline beats may have been summarized rather than developed into full scenes")
                elif word_count < 2500:
                    logger.info(f"ℹ️ Chapter {current_ch}: {word_count:,} words - check if scenes have sufficient depth and immersion")
                else:
                    logger.info(f"✅ Chapter {current_ch}: {word_count:,} words - likely has good narrative development")
        
        # Run validation on this chapter
        validation_state = {
            **chapter_state,
            "structured_edit": structured_edit,
            "current_chapter_text": chapter_text if chapter_text else "",
        }
        
        validation_result = await validation_subgraph.ainvoke(validation_state)
        
        # Update to next chapter
        next_ch = current_ch + 1 if current_ch < end_ch else None
        
        return {
            "current_generation_chapter": next_ch,
            "generated_chapters": generated_chapters,
            "generation_complete": next_ch is None,
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }
        
    except Exception as e:
        logger.error(f"Failed to orchestrate chapter generation: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "error": str(e),
            "task_status": "error",
            "generation_complete": True,
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }


async def validate_style_consistency_node(state: Dict[str, Any], llm_factory) -> Dict[str, Any]:
    """Validate style consistency across all generated chapters using fast model"""
    try:
        logger.info("🔍 Validating style consistency...")
        
        generated_chapters = state.get("generated_chapters", {})
        style_body = state.get("style_body")
        
        if not generated_chapters:
            return {
                "style_issues": [],
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
            }
        
        # Build manuscript text (sample first 5 chapters for validation)
        chapter_texts = []
        for ch_num in sorted(generated_chapters.keys())[:5]:
            chapter_texts.append(f"=== Chapter {ch_num} ===\n{generated_chapters[ch_num]}")
        
        manuscript_sample = "\n\n".join(chapter_texts)
        
        # Use fast model for validation
        from orchestrator.config import FAST_MODEL
        llm = llm_factory(temperature=0.1, state={**state, "user_chat_model": FAST_MODEL})
        
        prompt = f"""You are validating style consistency across a generated book.

STYLE GUIDE:
{style_body[:5000] if style_body else "No style guide available"}

MANUSCRIPT SAMPLE (Chapters 1-5):
{manuscript_sample[:10000]}

TASK: Identify any deviations from the style guide across chapters.
Focus on: voice, tone, POV, tense, pacing, dialogue patterns.
Return JSON list: [{{"chapter": N, "issue": "...", "severity": "minor|moderate|major"}}]
"""
        
        messages = [
            SystemMessage(content="You are a style consistency validator. Return only valid JSON."),
            HumanMessage(content=prompt)
        ]
        
        response = await llm.ainvoke(messages)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Parse JSON response
        try:
            # Remove markdown code blocks if present
            if '```json' in content:
                match = re.search(r'```json\s*\n(.*?)\n```', content, re.DOTALL)
                if match:
                    content = match.group(1).strip()
            elif '```' in content:
                match = re.search(r'```\s*\n(.*?)\n```', content, re.DOTALL)
                if match:
                    content = match.group(1).strip()
            
            style_issues = json.loads(content)
            if not isinstance(style_issues, list):
                style_issues = []
        except Exception as e:
            logger.warning(f"Failed to parse style validation response: {e}")
            style_issues = []
        
        logger.info(f"🔍 Found {len(style_issues)} style issues")
        
        return {
            "style_issues": style_issues,
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }
        
    except Exception as e:
        logger.error(f"Failed to validate style consistency: {e}")
        return {
            "style_issues": [],
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }


async def validate_continuity_coherence_node(state: Dict[str, Any], llm_factory) -> Dict[str, Any]:
    """Validate continuity coherence across all generated chapters"""
    try:
        logger.info("🔍 Validating continuity coherence...")
        
        generated_chapters = state.get("generated_chapters", {})
        characters_bodies = state.get("characters_bodies", [])
        rules_body = state.get("rules_body")
        
        if not generated_chapters:
            return {
                "continuity_issues": [],
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
            }
        
        # Build manuscript text (sample first 5 chapters)
        chapter_texts = []
        for ch_num in sorted(generated_chapters.keys())[:5]:
            chapter_texts.append(f"=== Chapter {ch_num} ===\n{generated_chapters[ch_num]}")
        
        manuscript_sample = "\n\n".join(chapter_texts)
        
        characters_text = "\n\n".join([f"=== Character {i+1} ===\n{char}" for i, char in enumerate(characters_bodies[:3])])
        
        # Use fast model
        from orchestrator.config import FAST_MODEL
        llm = llm_factory(temperature=0.1, state={**state, "user_chat_model": FAST_MODEL})
        
        prompt = f"""You are checking continuity across a generated book.

CHARACTER PROFILES:
{characters_text[:3000] if characters_text else "No character profiles"}

UNIVERSE RULES:
{rules_body[:2000] if rules_body else "No universe rules"}

MANUSCRIPT SAMPLE:
{manuscript_sample[:10000]}

TASK: Identify continuity errors:
- Character inconsistencies (age, appearance, backstory contradictions)
- Plot holes or dropped threads
- Timeline issues
- Rule violations

Return JSON list: [{{"type": "character|plot|timeline|rule", "chapters": [N, M], "issue": "..."}}]
"""
        
        messages = [
            SystemMessage(content="You are a continuity validator. Return only valid JSON."),
            HumanMessage(content=prompt)
        ]
        
        response = await llm.ainvoke(messages)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Parse JSON response
        try:
            if '```json' in content:
                match = re.search(r'```json\s*\n(.*?)\n```', content, re.DOTALL)
                if match:
                    content = match.group(1).strip()
            elif '```' in content:
                match = re.search(r'```\s*\n(.*?)\n```', content, re.DOTALL)
                if match:
                    content = match.group(1).strip()
            
            continuity_issues = json.loads(content)
            if not isinstance(continuity_issues, list):
                continuity_issues = []
        except Exception as e:
            logger.warning(f"Failed to parse continuity validation response: {e}")
            continuity_issues = []
        
        logger.info(f"🔍 Found {len(continuity_issues)} continuity issues")
        
        return {
            "continuity_issues": continuity_issues,
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }
        
    except Exception as e:
        logger.error(f"Failed to validate continuity: {e}")
        return {
            "continuity_issues": [],
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }


async def validate_outline_alignment_node(state: Dict[str, Any], llm_factory) -> Dict[str, Any]:
    """Validate outline alignment - check if generated content matches outline"""
    try:
        logger.info("🔍 Validating outline alignment...")
        
        generated_chapters = state.get("generated_chapters", {})
        outline_body = state.get("outline_body")
        
        if not generated_chapters or not outline_body:
            return {
                "outline_discrepancies": [],
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
            }
        
        # Build manuscript text (sample)
        chapter_texts = []
        for ch_num in sorted(generated_chapters.keys())[:5]:
            chapter_texts.append(f"=== Chapter {ch_num} ===\n{generated_chapters[ch_num]}")
        
        manuscript_sample = "\n\n".join(chapter_texts)
        
        # Use fast model
        from orchestrator.config import FAST_MODEL
        llm = llm_factory(temperature=0.1, state={**state, "user_chat_model": FAST_MODEL})
        
        prompt = f"""You are checking outline alignment.

OUTLINE:
{outline_body[:5000]}

GENERATED MANUSCRIPT SAMPLE:
{manuscript_sample[:10000]}

TASK: Compare generated content to outline. Identify missing beats or extra content.
Return JSON: {{"coverage": 0.95, "missing_beats": [...], "extra_content": [...]}}
"""
        
        messages = [
            SystemMessage(content="You are an outline alignment validator. Return only valid JSON."),
            HumanMessage(content=prompt)
        ]
        
        response = await llm.ainvoke(messages)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Parse JSON response
        try:
            if '```json' in content:
                match = re.search(r'```json\s*\n(.*?)\n```', content, re.DOTALL)
                if match:
                    content = match.group(1).strip()
            elif '```' in content:
                match = re.search(r'```\s*\n(.*?)\n```', content, re.DOTALL)
                if match:
                    content = match.group(1).strip()
            
            outline_alignment = json.loads(content)
            if not isinstance(outline_alignment, dict):
                outline_alignment = {"coverage": 0.0, "missing_beats": [], "extra_content": []}
        except Exception as e:
            logger.warning(f"Failed to parse outline alignment response: {e}")
            outline_alignment = {"coverage": 0.0, "missing_beats": [], "extra_content": []}
        
        logger.info(f"🔍 Outline alignment: {outline_alignment.get('coverage', 0.0):.0%} coverage")
        
        return {
            "outline_alignment": outline_alignment,
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }
        
    except Exception as e:
        logger.error(f"Failed to validate outline alignment: {e}")
        return {
            "outline_alignment": {"coverage": 0.0, "missing_beats": [], "extra_content": []},
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }


async def compile_validation_report_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Compile validation report and write manuscript directly to document"""
    try:
        logger.info("📊 Compiling validation report...")
        
        style_issues = state.get("style_issues", [])
        continuity_issues = state.get("continuity_issues", [])
        outline_alignment = state.get("outline_alignment", {})
        generated_chapters = state.get("generated_chapters", {})
        frontmatter = state.get("frontmatter", {})
        
        # Assemble complete manuscript
        complete_manuscript = _assemble_complete_manuscript(frontmatter, generated_chapters)
        
        # Get document ID from active_editor
        shared_memory = state.get("shared_memory", {})
        active_editor = shared_memory.get("active_editor", {})
        document_id = active_editor.get("document_id")
        user_id = state.get("user_id", "system")
        
        if not document_id:
            # Try to get from filename
            filename = state.get("filename", "")
            if filename:
                # Search for document by filename
                results = await search_documents_structured(
                    query=filename,
                    user_id=user_id,
                    limit=5
                )
                for result in results:
                    doc_metadata = result.get('document', {})
                    if doc_metadata.get('filename') == filename:
                        document_id = doc_metadata.get('document_id')
                        break
        
        # Write directly to document
        if document_id:
            logger.info(f"📝 Writing complete manuscript to document {document_id}...")
            update_result = await update_document_content_tool(
                document_id=document_id,
                content=complete_manuscript,
                user_id=user_id,
                append=False
            )
            
            if update_result.get("success"):
                logger.info(f"✅ Manuscript written successfully ({len(complete_manuscript)} chars)")
            else:
                logger.warning(f"⚠️ Failed to write manuscript: {update_result.get('error')}")
        else:
            logger.warning("⚠️ No document_id found - cannot write manuscript")
        
        # Compile validation report
        total_issues = len(style_issues) + len(continuity_issues)
        coverage = outline_alignment.get("coverage", 0.0)
        
        validation_report = {
            "style_issues": style_issues,
            "continuity_issues": continuity_issues,
            "outline_alignment": outline_alignment,
            "total_issues": total_issues,
            "coverage": coverage,
            "chapters_generated": len(generated_chapters),
            "document_written": document_id is not None,
            "chapter_word_counts": chapter_word_counts,
            "total_words": total_words,
            "avg_words_per_chapter": avg_words_per_chapter,
            "shallow_chapters": shallow_chapters,
        }
        
        # Calculate chapter length statistics
        chapter_word_counts = {}
        total_words = 0
        shallow_chapters = []  # Chapters that might lack narrative depth
        
        for ch_num, ch_text in generated_chapters.items():
            word_count = len(ch_text.split())
            chapter_word_counts[ch_num] = word_count
            total_words += word_count
            
            # Flag potentially shallow chapters (< 1500 words likely indicates summary writing)
            if word_count < 1500:
                shallow_chapters.append((ch_num, word_count))
        
        avg_words_per_chapter = total_words // len(generated_chapters) if generated_chapters else 0
        
        # Build response message
        response_text = f"Generated {len(generated_chapters)} chapters (~{total_words:,} total words, ~{avg_words_per_chapter:,} words/chapter)"
        if document_id:
            response_text += " and written directly to document"
        
        response_text += f".\n\n**Narrative Depth Analysis:**\n"
        response_text += f"- Total manuscript: ~{total_words:,} words\n"
        response_text += f"- Average per chapter: ~{avg_words_per_chapter:,} words\n"
        
        if shallow_chapters:
            response_text += f"\n⚠️ **POTENTIAL DEPTH ISSUES** ({len(shallow_chapters)} chapters under 1,500 words):\n"
            for ch_num, word_count in shallow_chapters:
                response_text += f"  - Chapter {ch_num}: {word_count:,} words (likely summary-style rather than full scenes)\n"
            response_text += "\n💡 **To improve depth:** Regenerate these chapters individually with explicit instructions like:\n"
            response_text += "   'Generate Chapter N with full scene development - rich dialogue, sensory details,\n"
            response_text += "    character interiority, and atmospheric description. Show, don't tell.'\n"
        else:
            response_text += f"✅ All chapters show reasonable narrative development\n"
        
        response_text += f"\n**Validation Summary:**\n"
        response_text += f"- Style Consistency: {len(style_issues)} issues found\n"
        response_text += f"- Continuity: {len(continuity_issues)} issues found\n"
        response_text += f"- Outline Alignment: {coverage:.0%} coverage\n"
        
        if total_issues > 0:
            response_text += f"\n**Issues Detected:** {total_issues} total\n"
            if style_issues:
                response_text += f"  - Style issues: {len(style_issues)}\n"
            if continuity_issues:
                response_text += f"  - Continuity issues: {len(continuity_issues)}\n"
        
        return {
            "validation_report": validation_report,
            "response": {
                "response": response_text,
                "task_status": "complete",
                "agent_type": "fiction_editing_agent",
                "validation_report": validation_report,
            },
            "editor_operations": [],  # Empty - no operations to approve!
            "task_status": "complete",
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }
        
    except Exception as e:
        logger.error(f"Failed to compile validation report: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "error": str(e),
            "task_status": "error",
            "response": {
                "response": f"Book generation failed: {str(e)}",
                "task_status": "error",
            },
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }


# ============================================
# Subgraph Builder
# ============================================

def build_book_generation_subgraph(
    checkpointer,
    llm_factory,
    get_datetime_context,
    continuity_tracker
):
    """
    Build the whole-book generation subgraph.
    
    Args:
        checkpointer: LangGraph checkpointer for state persistence
        llm_factory: Function to get LLM instance
        get_datetime_context: Function to get datetime context string
        continuity_tracker: FictionContinuityTracker instance
    
    Returns:
        Compiled StateGraph subgraph
    """
    workflow = StateGraph(BookGenerationState)
    
    # Build dependent subgraphs
    generation_subgraph = build_generation_subgraph(
        checkpointer,
        llm_factory,
        get_datetime_context
    )
    validation_subgraph = build_validation_subgraph(
        checkpointer,
        llm_factory,
        get_datetime_context,
        continuity_tracker
    )
    
    # Add nodes
    workflow.add_node("plan_chapters", plan_chapters_node)
    
    # Prepare generation state (build system_prompt and datetime_context)
    async def prepare_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        return await prepare_generation_state_node(state, get_datetime_context)
    
    workflow.add_node("prepare_generation_state", prepare_wrapper)
    
    # Chapter generation loop node (wraps generation and validation)
    async def orchestrate_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        return await orchestrate_chapter_generation_node(
            state, generation_subgraph, validation_subgraph
        )
    
    workflow.add_node("generate_chapter", orchestrate_wrapper)
    
    # Validation nodes
    async def validate_style_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        return await validate_style_consistency_node(state, llm_factory)
    
    async def validate_continuity_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        return await validate_continuity_coherence_node(state, llm_factory)
    
    async def validate_outline_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        return await validate_outline_alignment_node(state, llm_factory)
    
    workflow.add_node("validate_style", validate_style_wrapper)
    workflow.add_node("validate_continuity", validate_continuity_wrapper)
    workflow.add_node("validate_outline", validate_outline_wrapper)
    workflow.add_node("compile_report", compile_validation_report_node)
    
    # Set entry point
    workflow.set_entry_point("plan_chapters")
    
    # Define flow
    workflow.add_edge("plan_chapters", "prepare_generation_state")
    workflow.add_edge("prepare_generation_state", "generate_chapter")
    
    # Loop: continue generating chapters until done
    def route_chapter_loop(state: Dict[str, Any]) -> str:
        if state.get("generation_complete", False):
            return "validate_style"
        return "generate_chapter"
    
    workflow.add_conditional_edges(
        "generate_chapter",
        route_chapter_loop,
        {
            "generate_chapter": "generate_chapter",  # Loop
            "validate_style": "validate_style"       # Done generating
        }
    )
    
    # Validation flow
    workflow.add_edge("validate_style", "validate_continuity")
    workflow.add_edge("validate_continuity", "validate_outline")
    workflow.add_edge("validate_outline", "compile_report")
    workflow.add_edge("compile_report", END)
    
    return workflow.compile(checkpointer=checkpointer)

