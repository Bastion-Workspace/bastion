"""
Two-Pass Chapter Generation for Fiction Editing

When the user requests "two-pass" for new chapter creation:
- Pass 1: Generate pure prose (no JSON) with creative-only prompt at higher temperature.
- Pass 2: Refine draft for readability, redundancy, style compliance; output ManuscriptEdit JSON.

Activated only when: (1) creating a new chapter (empty current chapter + creation keywords),
and (2) "two-pass" or "two pass" appears in current_request.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from orchestrator.utils.fiction_utilities import unwrap_json_response as _unwrap_json_response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preserve state helper (critical 5 + fiction-specific keys)
# ---------------------------------------------------------------------------

def _preserve_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return dict of state keys to preserve across two-pass nodes."""
    return {
        "metadata": state.get("metadata", {}),
        "user_id": state.get("user_id", "system"),
        "shared_memory": state.get("shared_memory", {}),
        "messages": state.get("messages", []),
        "query": state.get("query", ""),
        "system_prompt": state.get("system_prompt", ""),
        "datetime_context": state.get("datetime_context", ""),
        "manuscript": state.get("manuscript", ""),
        "filename": state.get("filename", ""),
        "frontmatter": state.get("frontmatter", {}),
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
        "outline_body": state.get("outline_body"),
        "story_overview": state.get("story_overview"),
        "book_map": state.get("book_map"),
        "style_body": state.get("style_body"),
        "rules_body": state.get("rules_body"),
        "characters_bodies": state.get("characters_bodies", []),
        "outline_current_chapter_text": state.get("outline_current_chapter_text"),
        "target_chapter_number": state.get("target_chapter_number"),
        "prev_chapter_last_line": state.get("prev_chapter_last_line"),
        "generation_context_parts": state.get("generation_context_parts", []),
    }


# ---------------------------------------------------------------------------
# Pass 1: Creative-only system prompt (no JSON, no operations)
# ---------------------------------------------------------------------------

DRAFT_SYSTEM_PROMPT = """You are a MASTER NOVELIST. Your job is to write fiction — not to expand outlines.

=== YOUR CREATIVE PROCESS (READ THIS FIRST) ===

**1. ABSORB THE STYLE GUIDE** — This is your primary creative lens.
   The Style Guide defines HOW you write: voice, tone, sentence length, pacing,
   words and phrasing to use, words and phrasing to AVOID, level of detail,
   dialogue style, POV, tense, and narrative technique.
   If a writing sample is provided, it is your north star — study its rhythms,
   its choices, what it lingers on, what it skips. Your prose must read as though
   it came from the same author as that sample.

**2. UNDERSTAND THE STORY** — Read the Outline to know what happens.
   The Outline tells you the events, plot beats, and story goals for the chapter.
   Internalize these as background knowledge — the way an author knows their plot
   before sitting down to write. You are NOT converting an outline into prose.
   You are writing a chapter of a novel, and you happen to know what needs to happen.

**3. WRITE IN SCENES** — Think like a novelist, not a summarizer.
   Combine related beats into cohesive scenes with natural flow.
   Not every beat needs its own scene or paragraph. Some beats are a single line
   of dialogue; others deserve a full page. Let the Style Guide's pacing sensibility
   determine how much space each moment gets.
   Build tension, create atmosphere, develop character through action and dialogue.
   The reader should never feel they are reading an expanded outline.

=== REFERENCE HIERARCHY ===

1. **STYLE GUIDE** (highest authority — HOW to write)
2. **OUTLINE** (story content — WHAT happens). NEVER copy or paraphrase outline language into your prose.
3. **CHARACTER PROFILES** (character authenticity)
4. **UNIVERSE RULES** (world-building consistency)
5. **SERIES TIMELINE** (if provided — cross-book continuity)

**When in doubt:** The Style Guide is always right. Write the way it tells you to write.

=== WRITING QUALITY STANDARDS ===

**Scene Construction:** Build scenes with narrative purpose. Open in medias res or with sensory grounding. Let dialogue carry exposition. End scenes with forward momentum.

**Variation and Freshness:** Vary character and location descriptions. Mix dialogue tags, action beats, and internal thoughts. Avoid repetitive patterns.

**Show, Don't Tell:** Reveal emotions through actions, dialogue, and physical reactions. Build atmosphere through sensory details. Let readers infer meaning.

**Scene-Building vs Summary:** Write complete scenes with setting, action, dialogue. Avoid summary prose that reports events. Build scenes moment-by-moment.

**Character Voice and Dialogue:** Dialogue must sound natural and character-specific. Each character's speech patterns should reflect their personality and emotional state.

**Organic Pacing:** Write scenes that flow naturally. Don't rush through beats to cover outline points. Outline beats are story goals to achieve, not items to check off sequentially.

=== OUTPUT INSTRUCTION ===

Output ONLY the chapter prose. Start with "## Chapter N" (use the exact chapter number you are writing). No JSON, no explanations, no markdown code blocks. Just the chapter text. Write as much as needed for a complete, compelling chapter. Complete sentences, proper grammar. No YAML frontmatter in the output."""


# ---------------------------------------------------------------------------
# Routing: when to use two-pass
# ---------------------------------------------------------------------------

def route_generation_mode(state: Dict[str, Any]) -> str:
    """
    Route after build_generation_context:
    - Two-pass new chapter: new chapter + "two-pass" requested -> build_draft_prompt
    - Single-pass new chapter: new chapter -> build_generation_prompt
    - Editing existing chapter: paragraph-numbered two-phase -> identify_paragraph_edits
    """
    current_chapter_text = (state.get("current_chapter_text") or "").strip()
    current_request = (state.get("current_request") or "").lower()
    explicit_primary_chapter = state.get("explicit_primary_chapter")
    target_chapter_number = state.get("target_chapter_number")

    is_new_chapter = (
        explicit_primary_chapter is not None
        and target_chapter_number is not None
        and target_chapter_number == explicit_primary_chapter
        and current_chapter_text == ""
        and any(kw in current_request for kw in ["create", "craft", "write", "generate", "chapter"])
    )
    two_pass_requested = "two-pass" in current_request or "two pass" in current_request

    if is_new_chapter and two_pass_requested:
        logger.info("Two-pass chapter generation activated (new chapter + two-pass requested)")
        return "build_draft_prompt"
    if is_new_chapter:
        return "build_generation_prompt"
    logger.info("Editing existing chapter: routing to paragraph-numbered two-phase editing")
    return "identify_paragraph_edits"


# ---------------------------------------------------------------------------
# Pass 1: Build draft prompt (creative-only, no JSON)
# ---------------------------------------------------------------------------

async def build_draft_prompt_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Build messages for Pass 1: creative draft only. No JSON schema."""
    try:
        generation_context_parts = state.get("generation_context_parts", [])
        if not generation_context_parts:
            return {
                "draft_messages": [],
                "error": "No generation context available for draft",
                "task_status": "error",
                **_preserve_state(state),
            }

        user_content_parts = list(generation_context_parts)
        user_content_parts.append(
            "\n\n=== OUTPUT INSTRUCTION ===\n"
            "Output ONLY the chapter prose. Start with ## Chapter N (use the chapter number you are writing). "
            "No JSON, no explanations, no markdown code blocks. Just the chapter text.\n"
        )
        user_content = "".join(user_content_parts)

        draft_messages = [
            SystemMessage(content=DRAFT_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        return {
            "draft_messages": draft_messages,
            **_preserve_state(state),
        }
    except Exception as e:
        logger.error("Failed to build draft prompt: %s", e)
        return {
            "draft_messages": [],
            "error": str(e),
            "task_status": "error",
            **_preserve_state(state),
        }


# ---------------------------------------------------------------------------
# Pass 1: Call draft LLM (temperature 0.6, raw prose)
# ---------------------------------------------------------------------------

async def call_draft_llm_node(state: Dict[str, Any], llm_factory) -> Dict[str, Any]:
    """Call LLM for Pass 1 draft at temperature 0.6. Returns raw prose, no JSON."""
    try:
        draft_messages = state.get("draft_messages", [])
        if not draft_messages:
            return {
                "draft_prose": "",
                "error": "No draft messages available",
                "task_status": "error",
                **_preserve_state(state),
            }

        llm = llm_factory(temperature=0.6, state=state)
        start_time = datetime.now()
        response = await llm.ainvoke(draft_messages)
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info("Draft LLM (Pass 1) completed in %.2fs", elapsed)

        raw = response.content if hasattr(response, "content") else str(response)
        # Strip markdown code fence if model wrapped prose in ```...```
        draft_prose = raw.strip()
        if draft_prose.startswith("```"):
            match = re.match(r"^```(?:\w*)\n?(.*?)```", draft_prose, re.DOTALL)
            if match:
                draft_prose = match.group(1).strip()

        return {
            "draft_prose": draft_prose,
            **_preserve_state(state),
        }
    except Exception as e:
        logger.error("Failed to call draft LLM: %s", e)
        return {
            "draft_prose": "",
            "error": str(e),
            "task_status": "error",
            **_preserve_state(state),
        }


# ---------------------------------------------------------------------------
# Pass 2: Refinement system prompt (refine draft -> ManuscriptEdit JSON)
# ---------------------------------------------------------------------------

def _build_refinement_system_prompt(
    target_chapter_number: int,
    filename: str,
    prev_chapter_last_line: str,
) -> str:
    """Build Pass 2 system prompt: refine draft and output ManuscriptEdit JSON."""
    anchor_instruction = ""
    if prev_chapter_last_line:
        anchor_instruction = (
            f'Use insert_after_heading with anchor_text set to this EXACT line (copy verbatim): "{prev_chapter_last_line}"\n'
            "Your 'text' field must contain the refined chapter starting with ## Chapter N.\n"
        )

    return f"""You are refining a first draft of Chapter {target_chapter_number} into final prose and producing editor operations.

=== YOUR TASK ===

1. **REFINE THE DRAFT** for:
   - Remove redundancy (repeated descriptions, repeated phrases, unnecessary repetition)
   - Tighten wordy passages; make every sentence earn its place
   - Verify Style Guide compliance (voice, tone, pacing, word choices)
   - Verify outline beat coverage (all required story beats present, no outline language copied)
   - Assess pacing (scene length, transitions, emotional arc)

2. **OUTPUT FORMAT**: You MUST respond with a single JSON object only (no markdown fences, no explanatory text). Structure:

{{
  "target_filename": "{filename}",
  "scope": "chapter",
  "summary": "Refined Chapter {target_chapter_number}: tightened prose, ensured style and outline compliance",
  "safety": "medium",
  "chapter_index": {target_chapter_number - 1 if target_chapter_number else 0},
  "operations": [
    {{
      "op_type": "insert_after_heading",
      "anchor_text": "<EXACT last line of previous chapter from context>",
      "text": "## Chapter {target_chapter_number}\\n\\n<refined chapter prose>",
      "start": 0,
      "end": 0
    }}
  ]
}}

=== CRITICAL ===

- "scope" MUST be exactly "chapter". "safety" MUST be exactly "low", "medium", or "high".
- operations[0].text MUST start with "## Chapter {target_chapter_number}" then two newlines, then the refined chapter prose.
- operations[0].anchor_text MUST be the EXACT, VERBATIM last line of the previous chapter (from the MANUSCRIPT CONTEXT above). Do NOT use outline text.
{anchor_instruction}
- Output MUST be valid JSON only. No triple backticks, no explanation before or after.
"""


# ---------------------------------------------------------------------------
# Pass 2: Build refinement prompt (draft + refinement instructions -> generation_messages)
# ---------------------------------------------------------------------------

async def build_refinement_prompt_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Build messages for Pass 2: refine draft_prose against style guide and outline, output ManuscriptEdit JSON."""
    try:
        draft_prose = state.get("draft_prose", "")
        if not draft_prose:
            return {
                "generation_messages": [],
                "error": "No draft prose available for refinement",
                "task_status": "error",
                **_preserve_state(state),
            }

        target_chapter_number = state.get("target_chapter_number") or 1
        filename = state.get("filename", "manuscript.md")
        prev_chapter_last_line = state.get("prev_chapter_last_line") or ""

        refinement_system = _build_refinement_system_prompt(
            target_chapter_number, filename, prev_chapter_last_line
        )

        # Build reference sections so Pass 2 can verify compliance (not in a vacuum)
        ref_parts = []
        style_body = state.get("style_body") or ""
        if style_body:
            ref_parts.append(
                "=== STYLE GUIDE (PRIMARY — CHECK COMPLIANCE) ===\n"
                "Use this to verify voice, tone, pacing, word choices, and sentence rhythm. "
                "Tighten or rewrite any prose that does not match.\n\n"
                f"{style_body}\n\n"
            )
        outline_current = state.get("outline_current_chapter_text") or ""
        if outline_current:
            ref_parts.append(
                "=== CHAPTER OUTLINE (VERIFY BEAT COVERAGE) ===\n"
                "Ensure all required beats are present. Remove any outline language that leaked into the draft. "
                "Do not copy outline phrasing into the refined prose.\n\n"
                f"{outline_current}\n\n"
            )
        rules_body = state.get("rules_body") or ""
        if rules_body:
            ref_parts.append(
                "=== UNIVERSE RULES (WORLD-BUILDING) ===\n"
                "Ensure the refined prose does not violate these constraints.\n\n"
                f"{rules_body}\n\n"
            )
        characters_bodies = state.get("characters_bodies") or []
        if characters_bodies:
            ref_parts.append("=== CHARACTER PROFILES ===\n")
            for i, char_body in enumerate(characters_bodies, 1):
                ref_parts.append(f"--- Character {i} ---\n{char_body}\n\n")
            ref_parts.append("Match dialogue and behavior to these profiles.\n\n")

        ref_block = "".join(ref_parts) if ref_parts else ""

        user_content = (
            ref_block
            + "=== FIRST DRAFT (REFINE THIS) ===\n\n"
            + draft_prose
            + "\n\n=== END OF FIRST DRAFT ===\n\n"
            + "Refine the above draft against the Style Guide and Chapter Outline above: remove redundancy, "
            "tighten prose, fix any style or outline violations. Then output a single ManuscriptEdit JSON "
            "with one insert_after_heading operation containing the refined chapter. "
            "Use the exact anchor_text from the manuscript context (last line of previous chapter). "
            "Your response must be JSON only, no markdown fences."
        )

        generation_messages = [
            SystemMessage(content=refinement_system),
            HumanMessage(content=user_content),
        ]

        return {
            "generation_messages": generation_messages,
            **_preserve_state(state),
        }
    except Exception as e:
        logger.error("Failed to build refinement prompt: %s", e)
        return {
            "generation_messages": [],
            "error": str(e),
            "task_status": "error",
            **_preserve_state(state),
        }


# ---------------------------------------------------------------------------
# Pass 2: Call refinement LLM (temperature 0.3, ManuscriptEdit JSON)
# ---------------------------------------------------------------------------

async def call_refinement_llm_node(state: Dict[str, Any], llm_factory) -> Dict[str, Any]:
    """Call LLM for Pass 2 refinement at temperature 0.3. Returns ManuscriptEdit JSON."""
    try:
        generation_messages = state.get("generation_messages", [])
        if not generation_messages:
            return {
                "llm_response": "",
                "error": "No refinement messages available",
                "task_status": "error",
                **_preserve_state(state),
            }

        llm = llm_factory(temperature=0.3, state=state)
        start_time = datetime.now()
        response = await llm.ainvoke(generation_messages)
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info("Refinement LLM (Pass 2) completed in %.2fs", elapsed)

        raw_content = response.content if hasattr(response, "content") else str(response)
        content = _unwrap_json_response(raw_content)

        return {
            "llm_response": content,
            "llm_response_raw": raw_content,
            "generation_context_parts": state.get("generation_context_parts", []),
            **_preserve_state(state),
        }
    except Exception as e:
        logger.error("Failed to call refinement LLM: %s", e)
        return {
            "llm_response": "",
            "error": str(e),
            "task_status": "error",
            "generation_context_parts": state.get("generation_context_parts", []),
            **_preserve_state(state),
        }
