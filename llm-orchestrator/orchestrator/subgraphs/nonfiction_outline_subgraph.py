"""
Non-Fiction Outline Editing Subgraph

Reusable subgraph for non-fiction outline document editing workflows.
Supports arbitrary sections (## <Section Name>), style guide, and flexible reference loading.

Supports:
- Type gating: Strict validation for type: nfoutline documents
- Section detection: Finds section ranges by ## <Section Name> (arbitrary names)
- Structure analysis: Section count, outline completeness
- Reference loading: Style guide + arbitrary reference_* frontmatter keys
- Question vs Edit Detection: Distinguishes analysis questions from edit requests

Produces EditorOperations suitable for Prefer Editor HITL application.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from orchestrator.models.agent_response_contract import AgentResponse, ManuscriptEditMetadata
from orchestrator.utils.editor_operation_resolver import resolve_editor_operation
from orchestrator.utils.writing_subgraph_utilities import (
    preserve_critical_state,
    create_writing_error_response,
    extract_user_request,
    strip_frontmatter_block,
    slice_hash,
    build_response_text_for_question,
    build_response_text_for_edit,
    build_failed_operations_section,
    create_manuscript_edit_metadata,
    prepare_writing_context,
)

logger = logging.getLogger(__name__)


# ============================================
# Section Detection (Non-Fiction)
# ============================================

@dataclass
class SectionRange:
    """Represents a section in a non-fiction outline."""
    heading_text: str
    section_name: str
    start: int
    end: int


SECTION_PATTERN = re.compile(r"^##\s+(.+)$", re.MULTILINE)


def find_section_ranges(text: str) -> List[SectionRange]:
    """Find all section ranges in text (## <Section Name>)."""
    if not text:
        return []
    matches = list(SECTION_PATTERN.finditer(text))
    if not matches:
        return []
    ranges: List[SectionRange] = []
    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        section_name = (m.group(1) or "").strip()
        ranges.append(SectionRange(
            heading_text=m.group(0),
            section_name=section_name,
            start=start,
            end=end
        ))
    return ranges


def locate_section_index(ranges: List[SectionRange], cursor_offset: int) -> int:
    """Locate which section contains the cursor."""
    if cursor_offset < 0:
        return -1
    for i, r in enumerate(ranges):
        if r.start <= cursor_offset < r.end:
            return i
    return -1


def get_adjacent_sections(
    ranges: List[SectionRange], idx: int
) -> Tuple[Optional[SectionRange], Optional[SectionRange]]:
    """Get previous and next sections."""
    prev = ranges[idx - 1] if 0 <= idx - 1 < len(ranges) else None
    next_ = ranges[idx + 1] if 0 <= idx + 1 < len(ranges) else None
    return prev, next_


def find_last_line_of_last_section(outline: str) -> Optional[str]:
    """Find the last non-empty line of the last section for anchor insertion."""
    if not outline:
        return None
    section_ranges = find_section_ranges(strip_frontmatter_block(outline))
    if not section_ranges:
        lines = outline.rstrip().split("\n")
        for line in reversed(lines):
            if line.strip():
                return line.rstrip()
        return None
    last_section = section_ranges[-1]
    body_only = strip_frontmatter_block(outline)
    section_content = body_only[last_section.start:last_section.end]
    lines = section_content.split("\n")
    for line in reversed(lines):
        stripped = line.strip()
        if stripped and not re.match(r"^##\s+", stripped):
            return line.rstrip()
    return last_section.heading_text.rstrip()


# ============================================
# Utility Functions
# ============================================

def _frontmatter_end_index(text: str) -> int:
    """Return the end index of a leading YAML frontmatter block if present, else 0."""
    try:
        m = re.match(r"^(---\s*\n[\s\S]*?\n---\s*\n)", text, flags=re.MULTILINE)
        if m:
            return m.end()
        return 0
    except Exception:
        return 0


def _unwrap_json_response(content: str) -> str:
    """Extract raw JSON from LLM output if wrapped in code fences or prose."""
    try:
        json.loads(content)
        return content
    except Exception:
        pass
    try:
        text = content.strip()
        text = re.sub(r"^```(?:json)?\s*\n([\s\S]*?)\n```\s*$", r"\1", text)
        try:
            json.loads(text)
            return text
        except Exception:
            pass
        start = text.find("{")
        if start == -1:
            return content
        brace = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                brace += 1
            elif ch == "}":
                brace -= 1
                if brace == 0:
                    snippet = text[start : i + 1]
                    try:
                        json.loads(snippet)
                        return snippet
                    except Exception:
                        break
        return content
    except Exception:
        return content


def _assess_reference_quality(content: str, ref_type: str) -> Tuple[float, List[str]]:
    """Assess reference quality; returns (quality_score, warnings)."""
    if not content or len(content.strip()) < 50:
        return 0.0, ["Reference content is too short or empty"]
    quality_score = 0.5
    warnings = []
    content_length = len(content.strip())
    if ref_type == "style":
        if "example" in content.lower() or "```" in content:
            quality_score += 0.2
        else:
            warnings.append("Style guide lacks examples")
        if content_length > 300:
            quality_score += 0.3
        elif content_length < 150:
            warnings.append("Style guide is quite brief")
        style_keywords = ["voice", "tone", "structure", "heading", "section"]
        if any(kw in content.lower() for kw in style_keywords):
            quality_score += 0.1
    else:
        if "## " in content or "- " in content:
            quality_score += 0.2
        if content_length > 400:
            quality_score += 0.3
        elif content_length < 200:
            warnings.append("Reference content is quite brief")
        ref_keywords = ["data", "source", "research", "reference", "citation"]
        if any(kw in content.lower() for kw in ref_keywords):
            quality_score += 0.1
    return min(1.0, quality_score), warnings


def _extract_conversation_history(messages: List[Any], limit: int = 10) -> List[Dict[str, str]]:
    """Extract conversation history from LangChain messages."""
    try:
        history = []
        for msg in messages[-limit:]:
            if hasattr(msg, "content"):
                role = "assistant" if getattr(msg, "type", None) == "ai" else "user"
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                history.append({"role": role, "content": content})
        return history
    except Exception as e:
        logger.error(f"Failed to extract conversation history: {e}")
        return []


def _build_editing_agent_messages(
    system_prompt: str,
    context_parts: List[str],
    current_request: str,
    messages_list: List[Any],
    get_datetime_context: Callable,
    look_back_limit: int = 6,
) -> List[Any]:
    """Build message list for editing agents with conversation history."""
    messages = [
        SystemMessage(content=system_prompt),
        SystemMessage(content=get_datetime_context()),
    ]
    if messages_list:
        conversation_history = _extract_conversation_history(messages_list, limit=look_back_limit)
        if conversation_history and conversation_history[-1].get("content") == current_request:
            conversation_history = conversation_history[:-1]
        for msg in conversation_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
    if context_parts:
        messages.append(HumanMessage(content="".join(context_parts)))
    if current_request:
        messages.append(HumanMessage(content=current_request))
    return messages


def _determine_reference_mode(
    style_body: Optional[str], reference_bodies: List[str]
) -> Tuple[str, str]:
    """Determine generation mode for non-fiction outline."""
    has_style = style_body and len(style_body.strip()) >= 50
    has_references = bool(reference_bodies and len(reference_bodies) > 0)
    if has_style and has_references:
        return "fully_referenced", "Style guide and reference materials available - use them for consistency and grounding."
    if has_style:
        return "with_style", "Style guide available - use it for tone and structure guidance."
    if has_references:
        return "with_references", "Reference materials available - use them for factual grounding."
    return "freehand", "No references - use best practices for non-fiction outlining."


# ============================================
# System Prompt Builder
# ============================================

def _build_system_prompt() -> str:
    """Build system prompt for non-fiction outline editing."""
    return (
        "You are a non-fiction outline editor. Generate operations to edit non-fiction outlines.\n\n"
        "**CRITICAL: WORK WITH AVAILABLE INFORMATION FIRST**\n"
        "Always start by working with what you know from the request, existing outline content, and references:\n"
        "- Make edits based on available information - don't wait for clarification\n"
        "- Use context from style guide and reference materials to inform your work\n"
        "- Add or revise content based on reasonable inferences from the request\n"
        "- **FOR EMPTY FILES**: When the outline is empty (only frontmatter), ASK QUESTIONS FIRST before creating content\n"
        "  * Don't create the entire outline structure at once\n"
        "  * Ask about topic, audience, key sections, or reference materials\n"
        "  * Build incrementally based on user responses\n"
        "- Only proceed without questions when you have enough information to make meaningful edits\n\n"
        "**WHEN TO ASK QUESTIONS**:\n"
        "- **ALWAYS for empty files**: When outline is empty, ask questions about topic and structure before creating content\n"
        "- When the request is vague and you cannot make reasonable edits\n"
        "- When user requests a large amount of content - break it into steps and ask about priorities\n"
        "- When asking, you can provide operations for what you CAN do, then ask questions in the summary about what you need\n\n"
        "**HOW TO ASK QUESTIONS**: Include operations for work you CAN do, then add questions/suggestions in the summary field.\n"
        "For empty files, it's acceptable to return a clarification request with questions instead of operations.\n"
        "DO NOT return empty operations array for edit requests - always provide edits OR ask questions.\n\n"
        "**HANDLING QUESTIONS THAT DON'T REQUIRE EDITS**:\n"
        "- If the user is asking a question that can be answered WITHOUT making edits to the outline\n"
        "- Examples: \"What sections do we have?\", \"Summarize the outline\", \"Analyze the structure\"\n"
        "- OR if the user explicitly says \"don't edit\", \"no edits\", \"just answer\", \"only analyze\"\n"
        "- THEN return a ManuscriptEdit with:\n"
        "  * Use standard scope value (\"paragraph\" is fine for questions)\n"
        "  * EMPTY operations array ([])\n"
        "  * Your complete answer in the summary field\n\n"
        "NON-FICTION OUTLINE STRUCTURE:\n"
        "- Use arbitrary section names with ## <Section Name> (e.g., ## Introduction, ## Methods, ## Results)\n"
        "- No numbering required for sections\n"
        "- Each section can have subsections (###), bullet points, or prose as appropriate\n"
        "- Optional: # Overview, # Notes at the top; then ## sections for main content\n\n"
        "**SECTION HEADING FORMAT**:\n"
        "- Section headings MUST be \"## <Section Name>\" where the name is descriptive and arbitrary\n"
        "- Examples: \"## Introduction\", \"## Literature Review\", \"## Methodology\", \"## Key Findings\"\n"
        "- Do NOT use \"## Chapter N\" - this is non-fiction; use meaningful section names\n\n"
        "OPERATIONS:\n\n"
        "**1. replace_range** - Use ONLY when changing existing content. Provide 'original_text' with EXACT text from the file.\n\n"
        "**2. insert_after_heading** - Use when ADDING new content. For empty files, omit 'anchor_text'. "
        "For files with content, provide 'anchor_text' with EXACT line to insert after.\n\n"
        "**3. delete_range** - Use when removing content. Provide 'original_text' with EXACT text to remove.\n\n"
        "OUTPUT FORMAT - ManuscriptEdit JSON:\n"
        "{\n"
        '  "type": "ManuscriptEdit",\n'
        '  "target_filename": "filename.md",\n'
        '  "scope": "paragraph|section|multi_section",\n'
        '  "summary": "brief description (or questions if seeking clarification)",\n'
        '  "operations": [\n'
        "    {\n"
        '      "op_type": "replace_range|delete_range|insert_after_heading|insert_after",\n'
        '      "start": 0,\n'
        '      "end": 0,\n'
        '      "text": "content to insert/replace",\n'
        '      "original_text": "exact text from file (for replace/delete)",\n'
        '      "anchor_text": "exact line to insert after (for insert_after_heading)"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "**OUTPUT RULES**:\n"
        "- Return raw JSON only (no markdown fences, no explanatory text)\n"
        "- Always provide operations based on available information\n"
        "- If you need clarification, include it in the summary field AFTER describing the work you've done\n"
        "- Never return empty operations array unless the request is a question that doesn't require edits\n"
    )


# ============================================
# Subgraph Nodes
# ============================================

async def prepare_context_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare context: extract editor context, validate type: nfoutline."""
    try:
        logger.info("Preparing context for non-fiction outline editing...")
        context = await prepare_writing_context(
            state=state,
            doc_type="nfoutline",
            default_filename="nfoutline.md",
            content_key="outline",
            validate_type=True,
            clarification_key="pending_nfoutline_clarification",
        )
        if context.get("error"):
            return context
        shared_memory = state.get("shared_memory", {}) or {}
        previous_clarification = shared_memory.get("pending_nfoutline_clarification")
        clarification_context = ""
        if previous_clarification:
            clarification_context = (
                "\n\n=== PREVIOUS CLARIFICATION REQUEST ===\n"
                f"Context: {previous_clarification.get('context', '')}\n"
                "Questions Asked:\n"
            )
            for i, q in enumerate(previous_clarification.get("questions", []), 1):
                clarification_context += f"{i}. {q}\n"
            clarification_context += "\nThe user's response is in their latest message. Use this context to proceed.\n"
        current_request = extract_user_request(state)
        context.update({
            "clarification_context": clarification_context,
            "current_request": current_request.strip(),
        })
        return context
    except Exception as e:
        logger.error(f"Failed to prepare context: {e}")
        return create_writing_error_response(str(e), "nonfiction_outline_subgraph", state)


async def load_references_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Load style guide + arbitrary references from frontmatter (reference, reference_*, references)."""
    try:
        logger.info("Loading referenced context files from nfoutline frontmatter...")
        from orchestrator.tools.reference_file_loader import load_referenced_files

        active_editor = state.get("active_editor", {})
        user_id = state.get("user_id", "system")
        frontmatter = state.get("frontmatter", {})

        reference_config = {
            "style": ["style"],
            "references": ["reference", "reference_*", "references"],
        }

        result = await load_referenced_files(
            active_editor=active_editor,
            user_id=user_id,
            reference_config=reference_config,
            doc_type_filter="nfoutline",
            cascade_config=None,
        )

        loaded_files = result.get("loaded_files", {})

        style_body = None
        style_quality = 0.0
        style_warnings = []
        if loaded_files.get("style") and len(loaded_files["style"]) > 0:
            style_body = loaded_files["style"][0].get("content", "")
            if style_body:
                style_body = strip_frontmatter_block(style_body)
                style_quality, style_warnings = _assess_reference_quality(style_body, "style")

        reference_bodies = []
        reference_titles = []
        ref_qualities = []
        ref_warnings = []
        if loaded_files.get("references"):
            for ref_file in loaded_files["references"]:
                content = ref_file.get("content", "")
                title = ref_file.get("filename", ref_file.get("title", "reference"))
                if content:
                    content = strip_frontmatter_block(content)
                    quality, warnings = _assess_reference_quality(content, "reference")
                    reference_bodies.append(content)
                    reference_titles.append(title)
                    ref_qualities.append(quality)
                    ref_warnings.extend(warnings)

        avg_ref_quality = sum(ref_qualities) / len(ref_qualities) if ref_qualities else 0.0
        all_warnings = list(style_warnings)
        if style_quality < 0.4 and style_body:
            all_warnings.append(f"Style guide quality is low ({style_quality:.0%})")
        all_warnings.extend(ref_warnings[:5])

        has_style = style_body and len(style_body.strip()) > 50 and style_quality >= 0.4
        has_references = len(reference_bodies) > 0 and avg_ref_quality >= 0.4

        generation_mode, mode_guidance = _determine_reference_mode(style_body, reference_bodies)

        ref_parts = []
        if has_style:
            ref_parts.append(f"Style guide (quality: {style_quality:.0%})")
        if has_references:
            ref_parts.append(f"{len(reference_bodies)} reference(s) (quality: {avg_ref_quality:.0%})")
        reference_summary = ", ".join(ref_parts) if ref_parts else "No references available - freehand mode"
        if all_warnings:
            reference_summary += f"\nWarnings: {'; '.join(all_warnings[:3])}"

        body_only = state.get("body_only", "")
        section_ranges = find_section_ranges(body_only)
        section_count = len(section_ranges)
        sections_list = [r.section_name for r in section_ranges]

        has_overview = bool(re.search(r"^#\s+(Overview|Summary)\s*$", body_only, re.MULTILINE | re.IGNORECASE))
        has_notes = bool(re.search(r"^#\s+Notes\s*$", body_only, re.MULTILINE | re.IGNORECASE))

        structure_warnings = []
        if section_count == 0 and len(body_only.strip()) > 100:
            structure_warnings.append("Content exists but no ## sections defined")
        completeness = min(1.0, (section_count * 0.2 + (0.4 if has_overview else 0) + (0.2 if has_notes else 0)))

        section_list_str = ", ".join(sections_list) if sections_list else "No sections yet"
        if completeness < 0.25:
            structure_guidance = f"Outline {completeness:.0%} complete. Sections: {section_list_str}. Build structure or edit existing sections."
        elif completeness < 0.75:
            structure_guidance = f"Outline {completeness:.0%} complete. Sections: {section_list_str}. Continue developing or edit existing sections."
        else:
            structure_guidance = f"Structure complete. Sections: {section_list_str}. You can edit any existing section."

        available_references = {"style": has_style, "references": has_references}
        reference_quality = {"style": style_quality, "references": avg_ref_quality}

        return {
            "style_body": style_body,
            "reference_bodies": reference_bodies,
            "reference_titles": reference_titles,
            "reference_quality": reference_quality,
            "reference_warnings": all_warnings,
            "available_references": available_references,
            "generation_mode": generation_mode,
            "reference_summary": reference_summary,
            "mode_guidance": mode_guidance,
            "outline_completeness": completeness,
            "section_count": section_count,
            "structure_warnings": structure_warnings,
            "structure_guidance": structure_guidance,
            "has_overview": has_overview,
            "has_notes": has_notes,
            "sections_list": sections_list,
            **preserve_critical_state(state),
            "active_editor": state.get("active_editor", {}),
            "outline": state.get("outline", ""),
            "filename": state.get("filename", ""),
            "cursor_offset": state.get("cursor_offset", -1),
            "selection_start": state.get("selection_start", -1),
            "selection_end": state.get("selection_end", -1),
            "body_only": state.get("body_only", ""),
            "clarification_context": state.get("clarification_context", ""),
            "current_request": state.get("current_request", ""),
            "frontmatter": state.get("frontmatter", {}),
        }
    except Exception as e:
        logger.error(f"Failed to load references: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "style_body": None,
            "reference_bodies": [],
            "reference_titles": [],
            "reference_quality": {},
            "reference_warnings": [],
            "available_references": {},
            "generation_mode": "freehand",
            "reference_summary": "Error loading references - defaulting to freehand",
            "mode_guidance": "Freehand mode - proceed with best practices.",
            "outline_completeness": 0.0,
            "section_count": 0,
            "structure_warnings": [],
            "structure_guidance": "Unable to analyze structure - proceed with caution.",
            "has_overview": False,
            "has_notes": False,
            "sections_list": [],
            "error": str(e),
            **preserve_critical_state(state),
            "active_editor": state.get("active_editor", {}),
            "outline": state.get("outline", ""),
            "filename": state.get("filename", ""),
            "body_only": state.get("body_only", ""),
            "current_request": state.get("current_request", ""),
            "frontmatter": state.get("frontmatter", {}),
        }


async def detect_request_type_node(
    state: Dict[str, Any],
    llm_factory: Callable,
) -> Dict[str, Any]:
    """Detect if user request is a question or an edit request."""
    try:
        logger.info("Detecting request type (question vs edit request)...")
        current_request = state.get("current_request", "")
        if not current_request:
            return _preserve_nf_state(state, {"request_type": "edit_request"})

        body_only = state.get("body_only", "")
        style_body = state.get("style_body")
        reference_bodies = state.get("reference_bodies", [])

        prompt = f"""Analyze the user's request and determine if it's a QUESTION or an EDIT REQUEST.

**USER REQUEST**: {current_request}

**CONTEXT**:
- Current outline: {body_only[:500] if body_only else "Empty outline"}
- Has style reference: {bool(style_body)}
- Has {len(reference_bodies)} reference(s)

**INTENT DETECTION**:
- QUESTIONS: User is asking a question - may or may not want edits. Route to edit path; LLM can decide if edits are needed.
- EDIT REQUESTS: User wants to create, modify, or generate content - NO question asked.

**OUTPUT**: Return ONLY valid JSON:
{{
  "request_type": "question" | "edit_request",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation"
}}
"""

        llm = llm_factory(temperature=0.1, state=state)
        messages = [
            SystemMessage(content="You are an intent classifier. Return only valid JSON."),
            HumanMessage(content=prompt),
        ]
        response = await llm.ainvoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        content = _unwrap_json_response(content)

        if not content or not content.strip():
            return _preserve_nf_state(state, {"request_type": "edit_request"})

        try:
            result = json.loads(content)
            request_type = result.get("request_type", "edit_request")
            confidence = result.get("confidence", 0.5)
            if confidence < 0.6:
                request_type = "edit_request"
            logger.info(f"Request type detected: {request_type}")
            return _preserve_nf_state(state, {"request_type": request_type})
        except Exception:
            return _preserve_nf_state(state, {"request_type": "edit_request"})
    except Exception as e:
        logger.error(f"Failed to detect request type: {e}")
        return _preserve_nf_state(state, {"request_type": "edit_request"})


def _preserve_nf_state(state: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Preserve all nonfiction outline state keys for subsequent nodes."""
    out = {
        **preserve_critical_state(state),
        "active_editor": state.get("active_editor", {}),
        "outline": state.get("outline", ""),
        "filename": state.get("filename", ""),
        "frontmatter": state.get("frontmatter", {}),
        "body_only": state.get("body_only", ""),
        "cursor_offset": state.get("cursor_offset", -1),
        "selection_start": state.get("selection_start", -1),
        "selection_end": state.get("selection_end", -1),
        "current_request": state.get("current_request", ""),
        "clarification_context": state.get("clarification_context", ""),
        "style_body": state.get("style_body"),
        "reference_bodies": state.get("reference_bodies", []),
        "reference_titles": state.get("reference_titles", []),
        "generation_mode": state.get("generation_mode", ""),
        "available_references": state.get("available_references", {}),
        "reference_summary": state.get("reference_summary", ""),
        "mode_guidance": state.get("mode_guidance", ""),
        "outline_completeness": state.get("outline_completeness", 0.0),
        "section_count": state.get("section_count", 0),
        "structure_guidance": state.get("structure_guidance", ""),
        "structure_warnings": state.get("structure_warnings", []),
        "has_overview": state.get("has_overview", False),
        "has_notes": state.get("has_notes", False),
        "sections_list": state.get("sections_list", []),
        "reference_quality": state.get("reference_quality", {}),
        "reference_warnings": state.get("reference_warnings", []),
        "structured_edit": state.get("structured_edit"),
        "request_type": state.get("request_type", "edit_request"),
        "task_status": state.get("task_status", "complete"),
        "clarification_request": state.get("clarification_request"),
        "response": state.get("response"),
        "failed_operations": state.get("failed_operations", []),
    }
    if extra:
        out.update(extra)
    return out


async def generate_edit_plan_node(
    state: Dict[str, Any],
    llm_factory: Callable,
    get_datetime_context: Callable,
) -> Dict[str, Any]:
    """Generate edit plan using LLM for non-fiction outline."""
    try:
        logger.info("Generating non-fiction outline edit plan...")
        outline = state.get("outline", "")
        filename = state.get("filename", "nfoutline.md")
        body_only = state.get("body_only", "")
        current_request = state.get("current_request", "")
        clarification_context = state.get("clarification_context", "")

        style_body = state.get("style_body")
        reference_bodies = state.get("reference_bodies", [])
        reference_titles = state.get("reference_titles", [])

        mode_guidance = state.get("mode_guidance", "")
        structure_guidance = state.get("structure_guidance", "")
        reference_summary = state.get("reference_summary", "")

        system_prompt = _build_system_prompt()

        context_parts = []
        context_parts.append("=== CURRENT NON-FICTION OUTLINE ===\n")
        context_parts.append(f"File: {filename}\n")
        if not body_only.strip():
            context_parts.append("\nEmpty file (only frontmatter). Ask questions before creating content.\n\n")
        else:
            section_ranges = find_section_ranges(body_only)
            cursor_offset = state.get("cursor_offset", -1)
            current_section_idx = locate_section_index(section_ranges, cursor_offset - _frontmatter_end_index(outline)) if cursor_offset >= 0 else -1

            if section_ranges:
                if current_section_idx >= 0:
                    curr = section_ranges[current_section_idx]
                    context_parts.append(f"\n**CURRENT SECTION: {curr.section_name}**\n\n")
                for i, r in enumerate(section_ranges):
                    section_text = body_only[r.start:r.end].strip()
                    context_parts.append(f"=== SECTION: {r.section_name} ===\n")
                    context_parts.append(f"{section_text}\n\n")
            else:
                context_parts.append("\n" + body_only + "\n\n")

        if style_body:
            context_parts.append("=== STYLE GUIDE ===\n")
            context_parts.append(f"{style_body}\n\n")
        if reference_bodies:
            context_parts.append("=== REFERENCE MATERIALS ===\n")
            for i, (body, title) in enumerate(zip(reference_bodies, reference_titles)):
                context_parts.append(f"--- {title} ---\n{body}\n\n")
        if clarification_context:
            context_parts.append(clarification_context)

        context_parts.append(
            "\n=== USER REQUEST ===\n"
            "Analyze the request. For questions, return 0 operations and put the answer in summary. "
            "For edit requests, generate ManuscriptEdit operations (replace_range, insert_after_heading, delete_range). "
            "Use ## <Section Name> for new sections; do not use ## Chapter N.\n\n"
        )

        request_with_instructions = f"=== USER REQUEST ===\n{current_request}\n\nGenerate ManuscriptEdit JSON. Use insert_after_heading for new content, replace_range to change existing text, delete_range to remove. Section format: ## <Section Name>."
        llm = llm_factory(temperature=0.3, state=state)
        messages = _build_editing_agent_messages(
            system_prompt,
            context_parts,
            request_with_instructions,
            state.get("messages", []),
            get_datetime_context,
        )
        response = await llm.ainvoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        content = _unwrap_json_response(content)

        if not content or not content.strip():
            return _preserve_nf_state(state, {"structured_edit": None, "task_status": "error", "error": "Empty LLM response"})

        try:
            parsed = json.loads(content)
            if not isinstance(parsed.get("operations"), list):
                parsed["operations"] = []
            return _preserve_nf_state(state, {
                "llm_response": content,
                "structured_edit": parsed,
                "system_prompt": system_prompt,
            })
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse ManuscriptEdit JSON: {e}")
            return _preserve_nf_state(state, {"structured_edit": None, "task_status": "error", "error": str(e)})
    except Exception as e:
        logger.error(f"Failed to generate edit plan: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return _preserve_nf_state(state, {"structured_edit": None, "task_status": "error", "error": str(e)})


async def resolve_operations_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve editor operations for non-fiction outline."""
    try:
        logger.info("Resolving editor operations...")
        outline = state.get("outline", "")
        structured_edit = state.get("structured_edit")
        selection_start = state.get("selection_start", -1)
        selection_end = state.get("selection_end", -1)

        if not structured_edit or not isinstance(structured_edit.get("operations"), list):
            return _preserve_nf_state(state, {
                "editor_operations": [],
                "task_status": "error",
                "error": "No operations to resolve",
            })

        fm_end_idx = _frontmatter_end_index(outline)
        selection = {"start": selection_start, "end": selection_end} if selection_start >= 0 and selection_end >= 0 else None
        body_only = strip_frontmatter_block(outline)
        is_empty_file = not body_only.strip()

        editor_operations = []
        failed_operations = state.get("failed_operations", []) or []
        operations = structured_edit.get("operations", [])

        for op in operations:
            op_text = op.get("text", "")
            if isinstance(op_text, str):
                op_text = strip_frontmatter_block(op_text)
                op_text = re.sub(r"\n{3,}", "\n\n", op_text)

            try:
                cursor_pos = state.get("cursor_offset", -1)
                cursor_pos = cursor_pos if cursor_pos >= 0 else None
                resolved_start, resolved_end, resolved_text, resolved_confidence = resolve_editor_operation(
                    content=outline,
                    op_dict=op,
                    selection=selection,
                    frontmatter_end=fm_end_idx,
                    cursor_offset=cursor_pos,
                )

                if resolved_start == -1 and resolved_end == -1:
                    if is_empty_file and op.get("op_type") == "insert_after_heading":
                        resolved_start = resolved_end = fm_end_idx
                        resolved_confidence = 0.7
                    else:
                        resolved_start = resolved_end = fm_end_idx
                        resolved_confidence = 0.5

                if is_empty_file and resolved_start < fm_end_idx:
                    resolved_start = resolved_end = fm_end_idx
                    resolved_confidence = 0.7

                pre_slice = outline[resolved_start:resolved_end]
                resolved_op = {
                    "op_type": op.get("op_type", "replace_range"),
                    "start": resolved_start,
                    "end": resolved_end,
                    "text": resolved_text,
                    "pre_hash": slice_hash(pre_slice),
                    "original_text": op.get("original_text"),
                    "anchor_text": op.get("anchor_text"),
                    "confidence": resolved_confidence,
                }
                editor_operations.append(resolved_op)
            except Exception as e:
                logger.warning(f"Operation resolution failed: {e}")
                fallback_start = fm_end_idx
                fallback_end = fm_end_idx
                editor_operations.append({
                    "op_type": op.get("op_type", "replace_range"),
                    "start": fallback_start,
                    "end": fallback_end,
                    "text": op_text,
                    "pre_hash": slice_hash(outline[fallback_start:fallback_end]),
                    "confidence": 0.3,
                })

        return _preserve_nf_state(state, {
            "editor_operations": editor_operations,
            "failed_operations": failed_operations,
        })
    except Exception as e:
        logger.error(f"Failed to resolve operations: {e}")
        return _preserve_nf_state(state, {
            "editor_operations": [],
            "failed_operations": state.get("failed_operations", []),
            "task_status": "error",
            "error": str(e),
        })


async def format_response_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Format final response with editor operations."""
    try:
        from datetime import datetime

        structured_edit = state.get("structured_edit", {})
        editor_operations = state.get("editor_operations", [])
        clarification_request = state.get("clarification_request")
        task_status = state.get("task_status", "complete")
        request_type = state.get("request_type", "edit_request")

        if request_type == "question" and structured_edit and structured_edit.get("summary") and len(editor_operations) == 0:
            return {
                "response": {
                    "response": structured_edit.get("summary"),
                    "task_status": "complete",
                    "agent_type": "nonfiction_outline_subgraph",
                    "timestamp": datetime.now().isoformat(),
                },
                "task_status": "complete",
                "editor_operations": [],
                **preserve_critical_state(state),
            }

        if clarification_request:
            response = state.get("response", {})
            if isinstance(response, dict) and all(k in response for k in ["response", "task_status", "agent_type", "timestamp"]):
                return {
                    "response": response,
                    "task_status": "incomplete",
                    **preserve_critical_state(state),
                }
            response_text = response.get("response", "") if isinstance(response, dict) else str(response) if response else "Clarification needed"
            return {
                "response": AgentResponse(
                    response=response_text,
                    task_status="incomplete",
                    agent_type="nonfiction_outline_subgraph",
                    timestamp=datetime.now().isoformat(),
                ).dict(exclude_none=True),
                "task_status": "incomplete",
                **preserve_critical_state(state),
            }

        if task_status == "error":
            error_msg = state.get("error", "Unknown error")
            return {
                "response": AgentResponse(
                    response=f"Non-fiction outline editing failed: {error_msg}",
                    task_status="error",
                    agent_type="nonfiction_outline_subgraph",
                    timestamp=datetime.now().isoformat(),
                    error=error_msg,
                ).dict(exclude_none=True),
                "task_status": "error",
                **preserve_critical_state(state),
            }

        failed_operations = state.get("failed_operations", [])
        if request_type == "question":
            response_text = build_response_text_for_question(structured_edit, editor_operations, fallback="Analysis complete.")
        else:
            response_text = build_response_text_for_edit(structured_edit, editor_operations, fallback="Edit plan ready.")
            if failed_operations:
                response_text += build_failed_operations_section(failed_operations, "outline")

        manuscript_edit_metadata = create_manuscript_edit_metadata(structured_edit, editor_operations)
        standard_response = AgentResponse(
            response=response_text,
            task_status=task_status,
            agent_type="nonfiction_outline_subgraph",
            timestamp=datetime.now().isoformat(),
        )

        shared_memory = (state.get("shared_memory") or {}).copy()
        shared_memory.pop("pending_nfoutline_clarification", None)

        return {
            "response": standard_response.dict(exclude_none=True),
            "editor_operations": editor_operations,
            "manuscript_edit": manuscript_edit_metadata.dict(exclude_none=True) if manuscript_edit_metadata else None,
            "task_status": task_status,
            "shared_memory": shared_memory,
            **preserve_critical_state(state),
        }
    except Exception as e:
        logger.error(f"Failed to format response: {e}")
        return create_writing_error_response(str(e), "nonfiction_outline_subgraph", state)


# ============================================
# Subgraph Builder
# ============================================

def build_nonfiction_outline_subgraph(
    checkpointer,
    llm_factory: Callable,
    get_datetime_context: Callable,
):
    """
    Build non-fiction outline editing subgraph.

    Args:
        checkpointer: LangGraph checkpointer for state persistence
        llm_factory: Callable that returns LLM instance (temperature, state)
        get_datetime_context: Callable that returns datetime context string

    Returns:
        Compiled StateGraph for integration into Writing Assistant Agent
    """
    from typing import TypedDict

    class NonfictionOutlineSubgraphState(TypedDict, total=False):
        metadata: Dict[str, Any]
        user_id: str
        shared_memory: Dict[str, Any]
        messages: List[Any]
        query: str
        active_editor: Dict[str, Any]
        outline: str
        filename: str
        frontmatter: Dict[str, Any]
        cursor_offset: int
        selection_start: int
        selection_end: int
        body_only: str
        style_body: Optional[str]
        reference_bodies: List[str]
        reference_titles: List[str]
        clarification_context: str
        current_request: str
        system_prompt: str
        llm_response: str
        structured_edit: Optional[Dict[str, Any]]
        clarification_request: Optional[Dict[str, Any]]
        request_type: str
        editor_operations: List[Dict[str, Any]]
        failed_operations: List[Dict[str, Any]]
        response: Dict[str, Any]
        task_status: str
        error: str
        generation_mode: str
        available_references: Dict[str, Any]
        reference_summary: str
        mode_guidance: str
        reference_quality: Dict[str, Any]
        reference_warnings: List[str]
        outline_completeness: float
        section_count: int
        structure_warnings: List[str]
        structure_guidance: str
        has_overview: bool
        has_notes: bool
        sections_list: List[str]

    workflow = StateGraph(NonfictionOutlineSubgraphState)

    async def prepare_context_wrapper(s):
        return await prepare_context_node(s)

    async def load_references_wrapper(s):
        return await load_references_node(s)

    async def detect_request_type_wrapper(s):
        return await detect_request_type_node(s, llm_factory)

    async def generate_edit_plan_wrapper(s):
        return await generate_edit_plan_node(s, llm_factory, get_datetime_context)

    async def resolve_operations_wrapper(s):
        return await resolve_operations_node(s)

    async def format_response_wrapper(s):
        return await format_response_node(s)

    workflow.add_node("prepare_context", prepare_context_wrapper)
    workflow.add_node("load_references", load_references_wrapper)
    workflow.add_node("detect_request_type", detect_request_type_wrapper)
    workflow.add_node("generate_edit_plan", generate_edit_plan_wrapper)
    workflow.add_node("resolve_operations", resolve_operations_wrapper)
    workflow.add_node("format_response", format_response_wrapper)

    workflow.set_entry_point("prepare_context")
    workflow.add_edge("prepare_context", "load_references")
    workflow.add_edge("load_references", "detect_request_type")
    workflow.add_edge("detect_request_type", "generate_edit_plan")
    workflow.add_edge("generate_edit_plan", "resolve_operations")
    workflow.add_edge("resolve_operations", "format_response")
    workflow.add_edge("format_response", END)

    return workflow.compile(checkpointer=checkpointer)
