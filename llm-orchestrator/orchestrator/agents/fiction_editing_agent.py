"""
Fiction Editing Agent - LangGraph Implementation
Gated to fiction manuscripts. Consumes active editor manuscript, cursor, and
referenced outline/rules/style/characters. Produces ManuscriptEdit with HITL.
"""

import hashlib
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict, Tuple
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import ValidationError

from .base_agent import BaseAgent, TaskStatus

logger = logging.getLogger(__name__)


# ============================================
# Chapter Scope Utilities
# ============================================

@dataclass
class ChapterRange:
    heading_text: str
    chapter_number: Optional[int]
    start: int
    end: int


CHAPTER_PATTERN = re.compile(r"^##\s+Chapter\s+(\d+)\b.*$", re.MULTILINE)


def find_chapter_ranges(text: str) -> List[ChapterRange]:
    """Find all chapter ranges in text."""
    if not text:
        return []
    matches = list(CHAPTER_PATTERN.finditer(text))
    if not matches:
        return []
    ranges: List[ChapterRange] = []
    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        chapter_num: Optional[int] = None
        try:
            chapter_num = int(m.group(1))
        except Exception:
            chapter_num = None
        ranges.append(ChapterRange(heading_text=m.group(0), chapter_number=chapter_num, start=start, end=end))
    return ranges


def locate_chapter_index(ranges: List[ChapterRange], cursor_offset: int) -> int:
    """Locate which chapter contains the cursor."""
    if cursor_offset < 0:
        return -1
    for i, r in enumerate(ranges):
        if r.start <= cursor_offset < r.end:
            return i
    return -1


def get_adjacent_chapters(ranges: List[ChapterRange], idx: int) -> Tuple[Optional[ChapterRange], Optional[ChapterRange]]:
    """Get previous and next chapters."""
    prev_c = ranges[idx - 1] if 0 <= idx - 1 < len(ranges) else None
    next_c = ranges[idx + 1] if 0 <= idx + 1 < len(ranges) else None
    return prev_c, next_c


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


# ============================================
# Utility Functions
# ============================================

def _slice_hash(text: str) -> str:
    """Match frontend simple hash (31-bit rolling, hex)."""
    try:
        h = 0
        for ch in text:
            h = (h * 31 + ord(ch)) & 0xFFFFFFFF
        return format(h, 'x')
    except Exception:
        return ""


def _strip_frontmatter_block(text: str) -> str:
    """Strip YAML frontmatter from text."""
    try:
        return re.sub(r'^---\s*\n[\s\S]*?\n---\s*\n', '', text, flags=re.MULTILINE)
    except Exception:
        return text


def _frontmatter_end_index(text: str) -> int:
    """Return the end index of a leading YAML frontmatter block if present, else 0."""
    try:
        m = re.match(r'^(---\s*\n[\s\S]*?\n---\s*\n)', text, flags=re.MULTILINE)
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
        start = text.find('{')
        if start == -1:
            return content
        brace = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == '{':
                brace += 1
            elif ch == '}':
                brace -= 1
                if brace == 0:
                    snippet = text[start:i+1]
                    try:
                        json.loads(snippet)
                        return snippet
                    except Exception:
                        break
        return content
    except Exception:
        return content


def _extract_chapter_number_from_request(request: str) -> Optional[int]:
    """Extract chapter number from user request like 'Chapter 1', 'generate chapter 2', etc."""
    if not request:
        return None
    # Pattern: "Chapter N" or "chapter N" (case insensitive)
    patterns = [
        r'(?:^|\s)(?:chapter|ch\.?)\s+(\d+)(?:\s|$|[^\d])',
        r'(?:^|\s)(\d+)(?:\s+chapter|$)',
    ]
    for pattern in patterns:
        match = re.search(pattern, request, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                continue
    return None


def _extract_chapter_range_from_request(request: str) -> Optional[Tuple[int, int]]:
    """
    Extract chapter range from user request.
    Returns (start_chapter, end_chapter) inclusive, or None if not a range request.
    
    Examples:
    - "generate the first few chapters" -> (1, 3)  # Default: first 3
    - "generate chapters 1-3" -> (1, 3)
    - "generate chapters 1 through 5" -> (1, 5)
    - "generate the first 5 chapters" -> (1, 5)
    - "generate chapter 1" -> None (single chapter, use _extract_chapter_number_from_request)
    """
    if not request:
        return None
    
    request_lower = request.lower()
    
    # Pattern 1: Explicit range "chapters 1-3" or "chapters 1 through 5"
    range_patterns = [
        r'(?:chapters?|ch\.?)\s+(\d+)\s*[-–—]\s*(\d+)',  # "chapters 1-3"
        r'(?:chapters?|ch\.?)\s+(\d+)\s+through\s+(\d+)',  # "chapters 1 through 5"
        r'(?:chapters?|ch\.?)\s+(\d+)\s+to\s+(\d+)',  # "chapters 1 to 5"
    ]
    for pattern in range_patterns:
        match = re.search(pattern, request_lower)
        if match:
            try:
                start = int(match.group(1))
                end = int(match.group(2))
                if start <= end:
                    return (start, end)
            except (ValueError, IndexError):
                continue
    
    # Pattern 2: "first N chapters" or "first few chapters"
    first_patterns = [
        r'first\s+(\d+)\s+chapters?',  # "first 5 chapters"
        r'first\s+few\s+chapters?',  # "first few chapters" -> default to 3
    ]
    for pattern in first_patterns:
        match = re.search(pattern, request_lower)
        if match:
            try:
                if match.group(1):
                    count = int(match.group(1))
                    return (1, count)
                else:
                    # "first few" -> default to 3 chapters
                    return (1, 3)
            except (ValueError, IndexError):
                if 'few' in request_lower:
                    return (1, 3)
    
    # Pattern 3: "chapters N through M" (alternative wording)
    through_pattern = r'chapter\s+(\d+)\s+through\s+chapter\s+(\d+)'
    match = re.search(through_pattern, request_lower)
    if match:
        try:
            start = int(match.group(1))
            end = int(match.group(2))
            if start <= end:
                return (start, end)
        except (ValueError, IndexError):
            pass
    
    return None


def _ensure_chapter_heading(text: str, chapter_number: int) -> str:
    """Ensure the text begins with '## Chapter N' heading."""
    try:
        if re.match(r'^\s*#{1,6}\s*Chapter\s+\d+\b', text, flags=re.IGNORECASE):
            return text
        heading = f"## Chapter {chapter_number}\n\n"
        return heading + text.lstrip('\n')
    except Exception:
        return text


# ============================================
# Simplified Resolver (Progressive Search)
# ============================================

def _resolve_operation_simple(
    manuscript: str,
    op_dict: Dict[str, Any],
    selection: Optional[Dict[str, int]] = None,
    frontmatter_end: int = 0
) -> Tuple[int, int, str, float]:
    """
    Simplified operation resolver using progressive search.
    Returns (start, end, text, confidence)
    """
    op_type = op_dict.get("op_type", "replace_range")
    original_text = op_dict.get("original_text")
    anchor_text = op_dict.get("anchor_text")
    left_context = op_dict.get("left_context")
    right_context = op_dict.get("right_context")
    occurrence_index = op_dict.get("occurrence_index", 0)
    text = op_dict.get("text", "")
    
    # Use selection if available
    if selection and selection.get("start", -1) >= 0:
        sel_start = selection["start"]
        sel_end = selection["end"]
        if op_type == "replace_range":
            return sel_start, sel_end, text, 1.0
    
    # Strategy 1: Exact match with original_text
    if original_text and op_type in ("replace_range", "delete_range"):
        count = 0
        search_from = 0
        while True:
            pos = manuscript.find(original_text, search_from)
            if pos == -1:
                break
            if count == occurrence_index:
                end_pos = pos + len(original_text)
                # Guard frontmatter: ensure operations never occur before frontmatter end
                pos = max(pos, frontmatter_end)
                end_pos = max(end_pos, pos)
                return pos, end_pos, text, 1.0
            count += 1
            search_from = pos + 1
    
    # Strategy 2: Anchor text for insert_after_heading
    if anchor_text and op_type == "insert_after_heading":
        pos = manuscript.find(anchor_text)
        if pos != -1:
            # For chapter headings, find the end of the entire chapter (not just the heading line)
            # Look for the next chapter heading or end of document
            if anchor_text.startswith("## Chapter"):
                # Find end of this chapter by looking for next chapter heading
                next_chapter_pattern = re.compile(r"\n##\s+Chapter\s+\d+", re.MULTILINE)
                match = next_chapter_pattern.search(manuscript, pos + len(anchor_text))
                if match:
                    # Insert before the next chapter
                    end_pos = match.start()
                else:
                    # This is the last chapter, insert at end of document
                    end_pos = len(manuscript)
            else:
                # For non-chapter headings, find end of line/paragraph
                end_pos = manuscript.find("\n", pos)
                if end_pos == -1:
                    end_pos = len(manuscript)
                else:
                    end_pos += 1
            # Guard frontmatter: ensure insertions never occur before frontmatter end
            end_pos = max(end_pos, frontmatter_end)
            return end_pos, end_pos, text, 0.9
    
    # Strategy 3: Left + right context
    if left_context and right_context:
        pattern = re.escape(left_context) + r"([\s\S]{0,400}?)" + re.escape(right_context)
        m = re.search(pattern, manuscript)
        if m:
            # Guard frontmatter: ensure operations never occur before frontmatter end
            start = max(m.start(1), frontmatter_end)
            end = max(m.end(1), start)
            return start, end, text, 0.8
    
    # Fallback: use approximate positions from op_dict
    start = op_dict.get("start", 0)
    end = op_dict.get("end", 0)
    
    # Guard frontmatter: ensure operations never occur before frontmatter end
    start = max(start, frontmatter_end)
    end = max(end, start)
    
    return start, end, text, 0.5


# ============================================
# LangGraph State
# ============================================

class FictionEditingState(TypedDict):
    """State for fiction editing agent LangGraph workflow"""
    query: str
    user_id: str
    metadata: Dict[str, Any]
    messages: List[Any]
    shared_memory: Dict[str, Any]
    active_editor: Dict[str, Any]
    manuscript: str
    filename: str
    frontmatter: Dict[str, Any]
    cursor_offset: int
    selection_start: int
    selection_end: int
    chapter_ranges: List[ChapterRange]
    active_chapter_idx: int
    current_chapter_text: str
    current_chapter_number: Optional[int]
    prev_chapter_text: Optional[str]
    next_chapter_text: Optional[str]
    paragraph_text: str
    para_start: int
    para_end: int
    outline_body: Optional[str]
    rules_body: Optional[str]
    style_body: Optional[str]
    characters_bodies: List[str]
    outline_current_chapter_text: Optional[str]
    current_request: str
    requested_chapter_number: Optional[int]
    system_prompt: str
    llm_response: str
    structured_edit: Optional[Dict[str, Any]]
    editor_operations: List[Dict[str, Any]]
    response: Dict[str, Any]
    task_status: str
    error: str
    # New fields for mode tracking and validation
    generation_mode: str
    creative_freedom_requested: bool
    mode_guidance: str
    reference_quality: Dict[str, Any]
    reference_warnings: List[str]
    reference_guidance: str
    consistency_warnings: List[str]
    # Multi-chapter generation fields
    is_multi_chapter: bool
    chapter_range: Optional[Tuple[int, int]]  # (start, end) inclusive
    current_generation_chapter: Optional[int]  # Current chapter being generated in multi-chapter mode
    generated_chapters: Dict[int, str]  # Map of chapter_number -> generated text for continuity
    # Outline sync detection
    outline_sync_analysis: Optional[Dict[str, Any]]  # Analysis of outline vs manuscript discrepancies
    outline_needs_sync: bool  # Whether manuscript needs updates to match outline


# ============================================
# Fiction Editing Agent
# ============================================

class FictionEditingAgent(BaseAgent):
    """
    Fiction Editing Agent for manuscript editing and generation
    
    Gated to fiction manuscripts. Consumes active editor manuscript, cursor, and
    referenced outline/rules/style/characters. Produces ManuscriptEdit with HITL.
    Uses LangGraph workflow for explicit state management
    """
    
    def __init__(self):
        super().__init__("fiction_editing_agent")
        self._grpc_client = None
        logger.info("Fiction Editing Agent ready!")
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """Build LangGraph workflow for fiction editing agent"""
        workflow = StateGraph(FictionEditingState)
        
        # Phase 1: Context preparation
        workflow.add_node("prepare_context", self._prepare_context_node)
        workflow.add_node("analyze_scope", self._analyze_scope_node)
        workflow.add_node("load_references", self._load_references_node)
        
        # Phase 2: Pre-generation assessment
        workflow.add_node("assess_references", self._assess_reference_quality_node)
        workflow.add_node("detect_outline_changes", self._detect_outline_changes_node)
        workflow.add_node("detect_mode", self._detect_mode_and_intent_node)
        
        # Phase 3: Multi-chapter loop control
        workflow.add_node("check_multi_chapter", self._check_multi_chapter_node)
        workflow.add_node("prepare_chapter_context", self._prepare_chapter_context_node)
        
        # Phase 4: Generation
        workflow.add_node("generate_edit_plan", self._generate_edit_plan_node)
        
        # Phase 5: Post-generation validation
        workflow.add_node("validate_consistency", self._validate_consistency_node)
        
        # Phase 6: Resolution and response
        workflow.add_node("resolve_operations", self._resolve_operations_node)
        workflow.add_node("accumulate_chapter", self._accumulate_chapter_node)
        workflow.add_node("format_response", self._format_response_node)
        
        # Entry point
        workflow.set_entry_point("prepare_context")
        
        # Flow
        workflow.add_edge("prepare_context", "analyze_scope")
        workflow.add_edge("analyze_scope", "load_references")
        workflow.add_edge("load_references", "assess_references")
        workflow.add_edge("assess_references", "detect_outline_changes")
        workflow.add_edge("detect_outline_changes", "detect_mode")
        workflow.add_edge("detect_mode", "check_multi_chapter")
        
        # Multi-chapter routing
        workflow.add_conditional_edges(
            "check_multi_chapter",
            self._route_multi_chapter,
            {
                "multi_chapter_loop": "prepare_chapter_context",
                "single_chapter": "generate_edit_plan"
            }
        )
        
        # Multi-chapter loop flow
        workflow.add_edge("prepare_chapter_context", "generate_edit_plan")
        
        # Shared flow: both single and multi-chapter go through generation pipeline
        workflow.add_edge("generate_edit_plan", "validate_consistency")
        workflow.add_edge("validate_consistency", "resolve_operations")
        
        # Route after resolve_operations: single goes to format, multi goes to accumulate
        workflow.add_conditional_edges(
            "resolve_operations",
            self._route_single_vs_multi,
            {
                "format_response": "format_response",
                "accumulate_chapter": "accumulate_chapter"
            }
        )
        
        # Multi-chapter loop: check if more chapters needed
        workflow.add_conditional_edges(
            "accumulate_chapter",
            self._route_chapter_completion,
            {
                "next_chapter": "prepare_chapter_context",
                "complete": "format_response"
            }
        )
        
        workflow.add_edge("format_response", END)
        
        return workflow.compile(checkpointer=checkpointer)
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for fiction editing"""
        return (
            "You are a MASTER NOVELIST editor/generator. Persona disabled.\n\n"
            "=== STYLE GUIDE FIRST PRINCIPLE ===\n\n"
            "**The Style Guide is HOW to write. The Outline is WHAT happens.**\n\n"
            "When generating narrative prose:\n"
            "- The Style Guide establishes your narrative voice, techniques, and craft (POV, tense, pacing, dialogue style, sensory detail level)\n"
            "- The Outline provides story structure and plot beats (what events occur, character arcs, story progression)\n"
            "- Your task: Write natural, compelling narrative in the Style Guide's voice that achieves the Outline's story goals\n"
            "- NEVER convert outline beats mechanically - craft scenes that flow naturally and happen to hit those beats\n"
            "- The Style Guide voice must permeate every sentence - internalize it BEFORE writing, not as an afterthought\n\n"
            "Maintain originality and do not copy from references. Adhere strictly to the project's Style Guide and Rules above all else.\n\n"
            "STRUCTURED OUTPUT REQUIRED: You MUST return ONLY raw JSON (no prose, no markdown, no code fences) matching this schema:\n"
            "{\n"
            '  "type": "ManuscriptEdit",\n'
            '  "target_filename": string (REQUIRED),\n'
            '  "scope": one of ["paragraph", "chapter", "multi_chapter"] (REQUIRED),\n'
            '  "summary": string (REQUIRED),\n'
            '  "chapter_index": integer|null (optional),\n'
            '  "safety": one of ["low", "medium", "high"] (REQUIRED),\n'
            '  "operations": [\n'
            "    {\n"
            '      "op_type": one of ["replace_range", "delete_range", "insert_after_heading"] (REQUIRED),\n'
            '      "start": integer (REQUIRED - approximate, anchors take precedence),\n'
            '      "end": integer (REQUIRED - approximate, anchors take precedence),\n'
            '      "text": string (REQUIRED - new prose for replace/insert),\n'
            '      "original_text": string (⚠️ REQUIRED for replace_range/delete_range - EXACT 20-40 words from manuscript!),\n'
            '      "anchor_text": string (⚠️ REQUIRED for insert_after_heading - EXACT complete line/paragraph to insert after!),\n'
            '      "left_context": string (optional - text before target),\n'
            '      "right_context": string (optional - text after target),\n'
            '      "occurrence_index": integer (optional - which occurrence, 0-based, default 0)\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "⚠️ ⚠️ ⚠️ CRITICAL FIELD REQUIREMENTS:\n"
            "- replace_range → MUST include 'original_text' with EXACT 20-40 words from manuscript\n"
            "- delete_range → MUST include 'original_text' with EXACT text to delete\n"
            "- insert_after_heading → MUST include 'anchor_text' with EXACT complete line/paragraph to insert after\n"
            "- If you don't provide these fields, the operation will FAIL!\n\n"
            "OUTPUT RULES:\n"
            "- Output MUST be a single JSON object only.\n"
            "- Do NOT include triple backticks or language tags.\n"
            "- Do NOT include explanatory text before or after the JSON.\n\n"
            "=== THREE FUNDAMENTAL OPERATIONS ===\n\n"
            "**1. replace_range**: Replace existing text with new text\n"
            "   USE WHEN: User wants to revise, improve, change, modify, or rewrite existing prose\n"
            "   ANCHORING: Provide 'original_text' with EXACT, VERBATIM text from manuscript (20-40 words)\n\n"
            "   **GRANULAR CORRECTIONS (SPECIFIC WORD/PHRASE REPLACEMENT)**:\n"
            "   - When user says 'not X', 'should be Y not X', 'change X to Y', 'instead of X' (with specific words)\n"
            "   - Example: 'It should be a boat, not a canoe' or 'Change 'canoe' to 'boat''\n"
            "   - Find the EXACT sentence or paragraph containing the word/phrase in the CURRENT CHAPTER or PARAGRAPH AROUND CURSOR\n"
            "   - Set 'original_text' to the FULL sentence or paragraph (20-40 words) containing the word/phrase\n"
            "   - Set 'text' to the same sentence/paragraph with ONLY the specific word/phrase changed\n"
            "   - Example: If manuscript has 'He paddled the canoe across the river' and user says 'boat not canoe':\n"
            "     * original_text: 'He paddled the canoe across the river'\n"
            "     * text: 'He paddled the boat across the river'\n"
            "   - **PRESERVE ALL SURROUNDING TEXT** - only change the specific word/phrase\n"
            "   - **DO NOT regenerate entire paragraphs** when making word-level corrections\n\n"
            "**2. insert_after_heading**: Insert new text AFTER a specific location WITHOUT replacing\n"
            "   USE WHEN: User wants to add, append, or insert new content (not replace existing)\n"
            "   ANCHORING: Provide 'anchor_text' with EXACT, COMPLETE, VERBATIM paragraph/sentence to insert after\n\n"
            "**3. delete_range**: Remove text\n"
            "   USE WHEN: User wants to delete, remove, or cut content\n"
            "   ANCHORING: Provide 'original_text' with EXACT text to delete\n\n"
            "=== CHAPTER BOUNDARIES ARE SACRED ===\n\n"
            "Chapters are marked by \"## Chapter N\" headings.\n"
            "⚠️ CRITICAL: NEVER include the next chapter's heading in your operation!\n\n"
            "=== CRITICAL TEXT PRECISION REQUIREMENTS ===\n\n"
            "For 'original_text' and 'anchor_text' fields:\n"
            "- Must be EXACT, COMPLETE, and VERBATIM from the current manuscript\n"
            "- Include ALL whitespace, line breaks, and formatting exactly as written\n"
            "- Include complete sentences or natural text boundaries (periods, paragraph breaks)\n"
            "- NEVER paraphrase, summarize, or reformat the text\n"
            "- Minimum 10-20 words for unique identification\n"
            "- ⚠️ NEVER include chapter headers (##) in original_text for replace_range!\n\n"
            "=== CREATIVE ADDITIONS POLICY ===\n\n"
            "**You have creative freedom to enhance the story with additions:**\n\n"
            "When the user requests additions, enhancements, or expansions, you may add story elements\n"
            "that are NOT explicitly in the outline, as long as they maintain consistency.\n\n"
            "**MANDATORY CONSISTENCY CHECKS for all additions:**\n"
            "Before adding ANY new story element, verify:\n"
            "1. ✅ Style Guide compliance - matches established voice/tone/pacing\n"
            "2. ✅ Universe Rules compliance - no violations of established physics/magic/tech\n"
            "3. ✅ Character consistency - behavior matches character profiles\n"
            "4. ✅ Manuscript continuity - no contradictions with established facts\n"
            "5. ✅ Timeline coherence - events fit logically in story sequence\n\n"
            "**ALLOWED additions (enhance without changing plot):**\n"
            "- Sensory details and atmospheric descriptions\n"
            "- Internal character thoughts and emotional reactions\n"
            "- Brief character interactions that deepen relationships\n"
            "- Worldbuilding details that enrich the setting\n"
            "- Transitional moments that improve flow\n"
            "- Foreshadowing elements for later story beats\n"
            "- Tension-building moments within existing scenes\n"
            "- Character vulnerability or growth moments\n"
            "- Dialogue that reveals character or advances relationships\n\n"
            "**FORBIDDEN additions (require user approval):**\n"
            "- Major plot events not in outline (character deaths, revelations, etc.)\n"
            "- New characters with significant roles\n"
            "- World-altering events or discoveries\n"
            "- Changes to story direction or timeline\n"
            "- Events that contradict outline's plot structure\n\n"
            "**When uncertain about an addition:**\n"
            "Use 'clarifying_questions' field to ask:\n"
            "- 'Adding [X] would enhance [Y], but it's not in the outline. Should I include it?'\n"
            "- 'This addition might affect [later plot point]. Should I proceed?'\n"
            "- 'The outline doesn't specify [detail]. Should I add [specific element]?'\n\n"
            "**Default behavior:**\n"
            "- If generating full chapter from outline → Follow outline structure, add enriching details\n"
            "- If user says 'stick to outline exactly' → Strict adherence, minimal additions\n"
            "- If user says 'add/enhance/expand/enrich' → Creative freedom with consistency checks\n"
            "- If editing existing prose → Full creative freedom with consistency checks\n\n"
            "=== CONSISTENCY VALIDATION FRAMEWORK ===\n\n"
            "For EVERY operation, especially creative additions, validate against ALL references:\n\n"
            "**Style Guide Validation:**\n"
            "- Narrative voice (POV, tense, formality level)\n"
            "- Sentence structure patterns and rhythm\n"
            "- Dialogue style and character speech patterns\n"
            "- Pacing and scene construction approach\n"
            "- Descriptive style (minimalist vs. rich, etc.)\n\n"
            "**Universe Rules Validation:**\n"
            "- Physics/magic/technology constraints\n"
            "- Cultural and social rules\n"
            "- Historical facts and timeline\n"
            "- Geographic and environmental limits\n"
            "- Power systems and their limitations\n\n"
            "**Character Profile Validation:**\n"
            "- Core personality traits and behaviors\n"
            "- Motivations and goals\n"
            "- Speech patterns and vocabulary\n"
            "- Relationships and dynamics with other characters\n"
            "- Backstory and experiences that shape decisions\n"
            "- Emotional state and character arc position\n\n"
            "**Manuscript Continuity Validation:**\n"
            "- Established facts from earlier chapters\n"
            "- Character emotional states and development\n"
            "- Ongoing plot threads and setups\n"
            "- Previously mentioned details (locations, objects, events)\n"
            "- Cause-and-effect relationships\n"
            "- Character knowledge and awareness\n\n"
            "**When adding new elements, ask yourself:**\n"
            "- Would this character realistically do/say this based on their profile?\n"
            "- Does this violate any established universe rules?\n"
            "- Does this contradict anything in previous chapters?\n"
            "- Does this match the established narrative voice?\n"
            "- Will this create problems for outlined future events?\n\n"
            "**If ANY consistency check fails → Use clarifying_questions to ask the user!**\n\n"
            "=== NARRATIVE CRAFT PRINCIPLES ===\n\n"
            "**Show, Don't Tell:**\n"
            "- Reveal character emotions through actions, dialogue, and physical reactions, not statements\n"
            "- Build atmosphere through sensory details (sight, sound, smell, texture, temperature)\n"
            "- Let readers infer meaning from scene details rather than explaining directly\n"
            "- Example: Instead of 'Peterson was worried,' show: 'Peterson's fingers drummed the desk, each tap louder than the last.'\n\n"
            "**Scene-Building vs Summary:**\n"
            "- Write complete scenes with setting, action, dialogue, and character internality\n"
            "- Avoid summary prose that reports events ('He went to the office and found the file')\n"
            "- Build scenes moment-by-moment with specific details and natural pacing\n"
            "- Let story events emerge organically within scenes, not as mechanical beat-conversion\n"
            "- Transitions between beats should flow naturally, not feel like checklist items\n\n"
            "**Character Voice and Dialogue:**\n"
            "- Dialogue must sound natural and character-specific, not expository or mechanical\n"
            "- Each character's speech patterns should reflect their personality, background, and emotional state\n"
            "- Avoid dialogue that merely conveys plot information - let characters speak as real people would\n"
            "- Internal thoughts should match the character's voice and perspective (POV)\n\n"
            "**Sensory Details and Atmosphere:**\n"
            "- Ground every scene in specific sensory details (what characters see, hear, feel, smell, taste)\n"
            "- Use atmospheric details to establish mood and tone, not just setting\n"
            "- Balance sensory detail level according to Style Guide requirements\n"
            "- Create immersive scenes that readers can experience, not just observe\n\n"
            "**Organic Pacing vs Mechanical Beat-Following:**\n"
            "- Write scenes that flow naturally with appropriate pacing for the moment\n"
            "- Don't rush through beats to 'cover' all outline points - let scenes breathe\n"
            "- Build tension, emotion, and character development organically within scenes\n"
            "- Outline beats are story goals to achieve, not items to check off sequentially\n"
            "- A single scene can achieve multiple outline beats naturally if the story flows that way\n"
            "- Conversely, a single outline beat might require multiple scenes if the story demands it\n\n"
            "**Style Guide Integration:**\n"
            "- Every sentence must sound like it was written in the Style Guide's voice\n"
            "- Apply Style Guide techniques (POV, tense, pacing, dialogue style) consistently throughout\n"
            "- If Style Guide includes writing samples, emulate their technique and voice, never copy content\n"
            "- The Style Guide voice should be so internalized that it feels natural, not forced\n\n"
            "=== CONTENT GENERATION RULES ===\n\n"
            "1. operations[].text MUST contain final prose (no placeholders or notes)\n"
            "2. For chapter generation: aim 800-1200 words, begin with '## Chapter N'\n"
            "3. If outline present: Transform outline beats into full narrative prose - craft vivid scenes, not outline paraphrasing\n"
            "   - The outline is a blueprint for story structure, not a script to convert line-by-line\n"
            "   - Write complete narrative with dialogue, action, description, character voice, and emotional depth\n"
            "   - Add all enriching details needed to bring the beats to life as compelling prose\n"
            "4. NO YAML frontmatter in operations[].text\n"
            "5. Match established voice and style\n"
            "6. Complete sentences with proper grammar\n"
            "7. NEVER simply paraphrase outline beats - always craft full narrative prose\n"
        )
    
    async def _prepare_context_node(self, state: FictionEditingState) -> Dict[str, Any]:
        """Prepare context: extract active editor, validate fiction type"""
        try:
            logger.info("Preparing context for fiction editing...")
            
            shared_memory = state.get("shared_memory", {}) or {}
            active_editor = shared_memory.get("active_editor", {}) or {}
            
            manuscript = active_editor.get("content", "") or ""
            filename = active_editor.get("filename") or "manuscript.md"
            frontmatter = active_editor.get("frontmatter", {}) or {}
            cursor_offset = int(active_editor.get("cursor_offset", -1))
            selection_start = int(active_editor.get("selection_start", -1))
            selection_end = int(active_editor.get("selection_end", -1))
            
            # Hard gate: require fiction
            fm_type = str(frontmatter.get("type", "")).lower()
            if fm_type != "fiction":
                return {
                    "error": "Active editor is not a fiction manuscript; editing agent skipping.",
                    "task_status": "error",
                    "response": {
                        "response": "Active editor is not a fiction manuscript; editing agent skipping.",
                        "task_status": "error",
                        "agent_type": "fiction_editing_agent"
                    }
                }
            
            # Extract user request
            messages = state.get("messages", [])
            try:
                if messages:
                    latest_message = messages[-1]
                    current_request = latest_message.content if hasattr(latest_message, 'content') else str(latest_message)
                else:
                    current_request = state.get("query", "")
            except Exception:
                current_request = ""
            
            return {
                "active_editor": active_editor,
                "manuscript": manuscript,
                "filename": filename,
                "frontmatter": frontmatter,
                "cursor_offset": cursor_offset,
                "selection_start": selection_start,
                "selection_end": selection_end,
                "current_request": current_request.strip()
            }
            
        except Exception as e:
            logger.error(f"Failed to prepare context: {e}")
            return {
                "error": str(e),
                "task_status": "error"
            }
    
    async def _analyze_scope_node(self, state: FictionEditingState) -> Dict[str, Any]:
        """Analyze chapter scope: find chapters, determine current/prev/next"""
        try:
            logger.info("Analyzing chapter scope...")
            
            manuscript = state.get("manuscript", "")
            cursor_offset = state.get("cursor_offset", -1)
            
            # Find chapter ranges
            chapter_ranges = find_chapter_ranges(manuscript)
            active_idx = locate_chapter_index(chapter_ranges, cursor_offset if cursor_offset >= 0 else 0)
            
            prev_c, next_c = (None, None)
            current_chapter_text = manuscript
            current_chapter_number: Optional[int] = None
            
            if active_idx != -1:
                current = chapter_ranges[active_idx]
                prev_c, next_c = get_adjacent_chapters(chapter_ranges, active_idx)
                current_chapter_text = manuscript[current.start:current.end]
                current_chapter_number = current.chapter_number
            
            # Get paragraph bounds
            para_start, para_end = paragraph_bounds(manuscript, cursor_offset if cursor_offset >= 0 else 0)
            paragraph_text = manuscript[para_start:para_end]
            
            # Get adjacent chapter text
            prev_chapter_text = None
            next_chapter_text = None
            
            if prev_c:
                prev_chapter_text = _strip_frontmatter_block(manuscript[prev_c.start:prev_c.end])
            if next_c:
                next_chapter_text = _strip_frontmatter_block(manuscript[next_c.start:next_c.end])
            
            # Strip frontmatter from current chapter
            context_current_chapter_text = _strip_frontmatter_block(current_chapter_text)
            context_paragraph_text = _strip_frontmatter_block(paragraph_text)
            
            return {
                "chapter_ranges": chapter_ranges,
                "active_chapter_idx": active_idx,
                "current_chapter_text": context_current_chapter_text,
                "current_chapter_number": current_chapter_number,
                "prev_chapter_text": prev_chapter_text,
                "next_chapter_text": next_chapter_text,
                "paragraph_text": context_paragraph_text,
                "para_start": para_start,
                "para_end": para_end
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze scope: {e}")
            return {
                "error": str(e),
                "task_status": "error"
            }
    
    async def _load_references_node(self, state: FictionEditingState) -> Dict[str, Any]:
        """Load referenced context files (outline, rules, style, characters)"""
        try:
            logger.info("Loading referenced context files...")
            
            from orchestrator.tools.reference_file_loader import load_referenced_files
            
            active_editor = state.get("active_editor", {})
            user_id = state.get("user_id", "system")
            
            # Fiction reference configuration
            # Manuscript frontmatter has: outline: "./outline.md"
            reference_config = {
                "outline": ["outline"]
            }
            
            # Cascading: outline frontmatter has rules, style, characters
            cascade_config = {
                "outline": {
                    "rules": ["rules"],
                    "style": ["style"],
                    "characters": ["characters", "character_*"]  # Support both list and individual keys
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
                # Try to extract chapter-specific outline section
                # This is a simplified extraction - could be enhanced
                import re
                chapter_pattern = rf"(?i)(?:^|\n)##?\s*(?:Chapter\s+)?{current_chapter_number}[:\s]*(.*?)(?=\n##?\s*(?:Chapter\s+)?\d+|\Z)"
                match = re.search(chapter_pattern, outline_body, re.DOTALL)
                if match:
                    outline_current_chapter_text = match.group(1).strip()
            
            return {
                "outline_body": outline_body,
                "rules_body": rules_body,
                "style_body": style_body,
                "characters_bodies": characters_bodies,
                "outline_current_chapter_text": outline_current_chapter_text
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
                "error": str(e)
            }
    
    async def _detect_mode_and_intent_node(self, state: FictionEditingState) -> Dict[str, Any]:
        """Detect generation mode and creative freedom intent from user request"""
        try:
            logger.info("Detecting mode and creative intent...")
            
            current_request = state.get("current_request", "")
            current_request_lower = current_request.lower()
            current_chapter_text = state.get("current_chapter_text", "")
            outline_current_chapter_text = state.get("outline_current_chapter_text")
            
            # Check for multi-chapter request first
            chapter_range = _extract_chapter_range_from_request(current_request)
            is_multi_chapter = chapter_range is not None
            
            # Extract requested chapter number from user request (for single chapter)
            requested_chapter_number = _extract_chapter_number_from_request(current_request) if not is_multi_chapter else None
            
            # Detect creative freedom keywords
            creative_keywords = [
                "add", "enhance", "expand", "enrich", "include", 
                "give", "show", "more", "develop", "deepen"
            ]
            creative_freedom_requested = any(kw in current_request_lower for kw in creative_keywords)
            
            # Detect strict adherence keywords
            strict_keywords = [
                "stick to outline", "follow outline exactly", "only outline", 
                "strictly follow", "outline only"
            ]
            strict_mode_requested = any(kw in current_request for kw in strict_keywords)
            
            # Determine mode
            if len(current_chapter_text.strip()) < 100:
                # Empty or very short chapter - likely generation
                mode = "generation"
            elif creative_freedom_requested and not strict_mode_requested:
                mode = "enhancement"
            else:
                mode = "editing"
            
            # Build mode-specific guidance for LLM
            if mode == "generation":
                mode_guidance = (
                    "\n\n=== MODE: GENERATION ===\n"
                    "You are generating NEW narrative prose from outline beats.\n"
                    "- Transform outline beats into full, vivid narrative scenes\n"
                    "- Craft complete prose with dialogue, action, description, and character voice\n"
                    "- Follow outline structure as story blueprint (not a script to paraphrase)\n"
                    "- Add enriching details: sensory details, character thoughts, emotional depth\n"
                    "- **CRITICAL: Follow ALL Style Guide narrative instructions** (voice, tone, POV, tense, pacing, techniques)\n"
                    "- Maintain Style Guide voice precisely - it overrides default assumptions\n"
                    "- Respect Universe Rules absolutely\n"
                    "- Use Character profiles for authentic behavior\n"
                    "- Write 800-1200 words of engaging narrative prose, not outline paraphrasing\n"
                )
            elif mode == "enhancement":
                mode_guidance = (
                    "\n\n=== MODE: CREATIVE ENHANCEMENT ===\n"
                    "You have creative freedom to add story elements.\n"
                    "- User has requested additions/enhancements\n"
                    "- Add elements that enrich the narrative\n"
                    "- CRITICAL: Maintain consistency with all references\n"
                    "- Validate additions against Style/Rules/Characters/Continuity\n"
                    "- Use clarifying_questions if additions might conflict with outline\n"
                )
            else:
                mode_guidance = (
                    "\n\n=== MODE: EDITING ===\n"
                    "You are revising EXISTING content.\n"
                    "- Maintain consistency with established story\n"
                    "- Respect existing character development\n"
                    "- Keep tone consistent unless explicitly asked to change\n"
                )
            
            return {
                "generation_mode": mode,
                "creative_freedom_requested": creative_freedom_requested or mode == "enhancement",
                "mode_guidance": mode_guidance,
                "requested_chapter_number": requested_chapter_number,
                "is_multi_chapter": is_multi_chapter,
                "chapter_range": chapter_range
            }
            
        except Exception as e:
            logger.error(f"Mode detection failed: {e}")
            return {
                "generation_mode": "editing",
                "creative_freedom_requested": False,
                "mode_guidance": "",
                "requested_chapter_number": None,
                "is_multi_chapter": False,
                "chapter_range": None
            }
    
    async def _detect_outline_changes_node(self, state: FictionEditingState) -> Dict[str, Any]:
        """Detect if outline has changed and manuscript needs updates"""
        try:
            logger.info("Detecting outline changes...")
            
            current_chapter_text = state.get("current_chapter_text", "")
            outline_current_chapter_text = state.get("outline_current_chapter_text")
            current_chapter_number = state.get("current_chapter_number")
            current_request = state.get("current_request", "").lower()
            
            # Skip if no outline or no existing chapter text
            if not outline_current_chapter_text or len(current_chapter_text.strip()) < 100:
                return {
                    "outline_sync_analysis": None,
                    "outline_needs_sync": False
                }
            
            # Always check outline sync when we have both outline and existing chapter text
            # The outline is always provided as context, so we can proactively detect discrepancies
            # Skip only if user has a very specific editing request that would conflict
            current_request_lower = current_request.lower().strip()
            
            # Skip sync detection only if user has a very specific, conflicting request
            # (e.g., "add a scene" - that's adding new content, not syncing with outline)
            # But if user says "revise" or "update" or is just editing, we should check sync
            conflicting_keywords = [
                "add", "insert", "new scene", "new paragraph", "write new",
                "generate new", "create new", "expand with", "also include"
            ]
            has_conflicting_request = any(kw in current_request_lower for kw in conflicting_keywords)
            
            if has_conflicting_request:
                logger.info("User has conflicting request (adding new content) - skipping outline sync detection")
                return {
                    "outline_sync_analysis": None,
                    "outline_needs_sync": False
                }
            
            # Otherwise, always check outline sync (outline is always provided as context)
            logger.info("Checking outline sync (outline always provided as context)")
            
            # Use LLM to compare outline to manuscript
            llm = self._get_llm(temperature=0.2, state=state)
            
            comparison_prompt = f"""Compare the current outline for Chapter {current_chapter_number} with the existing manuscript chapter.

**CURRENT OUTLINE FOR CHAPTER {current_chapter_number}**:
{outline_current_chapter_text}

**EXISTING MANUSCRIPT CHAPTER {current_chapter_number}**:
{current_chapter_text[:2000] if len(current_chapter_text) > 2000 else current_chapter_text}

**YOUR TASK**: Determine if the manuscript chapter needs updates to match the outline.

**ANALYSIS CRITERIA**:
1. **Plot Events**: Does the manuscript include all plot events/beats from the outline?
2. **Missing Elements**: Are there outline beats that are not present in the manuscript?
3. **Changed Elements**: Has the outline changed plot points that exist in the manuscript?
4. **Character Actions**: Do character actions in manuscript match outline expectations?
5. **Story Progression**: Does the manuscript follow the outline's story progression?

**OUTPUT FORMAT**: Return ONLY valid JSON:
{{
  "needs_sync": true|false,
  "discrepancies": [
    {{
      "type": "missing_beat|changed_beat|character_action_mismatch|story_progression_issue",
      "outline_expectation": "What the outline says should happen",
      "manuscript_current": "What the manuscript currently has (or 'missing')",
      "severity": "high|medium|low",
      "suggestion": "Specific revision needed"
    }}
  ],
  "summary": "Brief summary of what needs updating"
}}

**CRITICAL**: Only flag significant discrepancies. Minor differences in how events are written (but same events) are OK.
Only flag when:
- Outline beats are completely missing from manuscript
- Outline has changed plot points that contradict manuscript
- Character actions fundamentally differ from outline
- Story progression doesn't match outline structure

Return ONLY the JSON object, no markdown, no code blocks."""
            
            messages = [
                SystemMessage(content="You are an outline-manuscript synchronization analyzer. Compare outlines to manuscripts and detect when updates are needed."),
                HumanMessage(content=comparison_prompt)
            ]
            
            try:
                # Try structured output
                structured_llm = llm.with_structured_output({
                    "type": "object",
                    "properties": {
                        "needs_sync": {"type": "boolean"},
                        "discrepancies": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string"},
                                    "outline_expectation": {"type": "string"},
                                    "manuscript_current": {"type": "string"},
                                    "severity": {"type": "string"},
                                    "suggestion": {"type": "string"}
                                },
                                "required": ["type", "outline_expectation", "manuscript_current", "severity", "suggestion"]
                            }
                        },
                        "summary": {"type": "string"}
                    },
                    "required": ["needs_sync", "discrepancies", "summary"]
                })
                result = await structured_llm.ainvoke(messages)
                sync_analysis = result if isinstance(result, dict) else (result.dict() if hasattr(result, 'dict') else result.model_dump())
            except Exception as e:
                logger.warning(f"Structured output failed, using fallback: {e}")
                # Fallback: parse JSON response
                response = await llm.ainvoke(messages)
                content = response.content if hasattr(response, 'content') else str(response)
                content = _unwrap_json_response(content)
                try:
                    sync_analysis = json.loads(content)
                except Exception as parse_error:
                    logger.error(f"Failed to parse outline sync analysis: {parse_error}")
                    return {
                        "outline_sync_analysis": None,
                        "outline_needs_sync": False
                    }
            
            needs_sync = sync_analysis.get("needs_sync", False)
            discrepancies = sync_analysis.get("discrepancies", [])
            
            if needs_sync and discrepancies:
                logger.info(f"⚠️ Outline sync needed: {len(discrepancies)} discrepancy(ies) detected")
                for i, disc in enumerate(discrepancies, 1):
                    logger.info(f"  {i}. {disc.get('type')}: {disc.get('suggestion', '')[:100]}")
            else:
                logger.info("✅ Manuscript appears in sync with outline")
            
            return {
                "outline_sync_analysis": sync_analysis,
                "outline_needs_sync": needs_sync
            }
            
        except Exception as e:
            logger.error(f"Failed to detect outline changes: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "outline_sync_analysis": None,
                "outline_needs_sync": False
            }
    
    async def _check_multi_chapter_node(self, state: FictionEditingState) -> Dict[str, Any]:
        """Check if this is a multi-chapter generation request and initialize state"""
        try:
            is_multi_chapter = state.get("is_multi_chapter", False)
            chapter_range = state.get("chapter_range")
            
            if is_multi_chapter and chapter_range:
                start_ch, end_ch = chapter_range
                # Initialize multi-chapter state
                return {
                    "is_multi_chapter": True,
                    "chapter_range": chapter_range,
                    "current_generation_chapter": start_ch,
                    "generated_chapters": {}
                }
            else:
                return {
                    "is_multi_chapter": False,
                    "current_generation_chapter": None,
                    "generated_chapters": {}
                }
        except Exception as e:
            logger.error(f"Multi-chapter check failed: {e}")
            return {
                "is_multi_chapter": False,
                "current_generation_chapter": None,
                "generated_chapters": {}
            }
    
    def _route_multi_chapter(self, state: FictionEditingState) -> str:
        """Route to multi-chapter loop or single chapter generation"""
        if state.get("is_multi_chapter", False):
            return "multi_chapter_loop"
        return "single_chapter"
    
    async def _prepare_chapter_context_node(self, state: FictionEditingState) -> Dict[str, Any]:
        """Prepare context for current chapter in multi-chapter generation loop"""
        try:
            current_ch = state.get("current_generation_chapter")
            chapter_range = state.get("chapter_range")
            generated_chapters = state.get("generated_chapters", {})
            manuscript = state.get("manuscript", "")
            chapter_ranges = state.get("chapter_ranges", [])
            
            if not current_ch or not chapter_range:
                return {"error": "Invalid multi-chapter state", "task_status": "error"}
            
            start_ch, end_ch = chapter_range
            
            # Get previous chapter text from generated chapters or manuscript
            prev_chapter_text = None
            if current_ch > start_ch:
                # Previous chapter was just generated
                prev_chapter_text = generated_chapters.get(current_ch - 1)
            elif current_ch == start_ch and start_ch > 1:
                # First chapter in range, but previous chapters exist in manuscript
                for r in chapter_ranges:
                    if r.chapter_number == current_ch - 1:
                        prev_chapter_text = _strip_frontmatter_block(manuscript[r.start:r.end])
                        break
            
            # Get next chapter text from manuscript (if exists)
            next_chapter_text = None
            for r in chapter_ranges:
                if r.chapter_number == current_ch + 1:
                    next_chapter_text = _strip_frontmatter_block(manuscript[r.start:r.end])
                    break
            
            # Extract outline for current chapter
            outline_body = state.get("outline_body")
            outline_current_chapter_text = None
            if outline_body and current_ch:
                chapter_pattern = rf"(?i)(?:^|\n)##?\s*(?:Chapter\s+)?{current_ch}[:\s]*(.*?)(?=\n##?\s*(?:Chapter\s+)?\d+|\Z)"
                match = re.search(chapter_pattern, outline_body, re.DOTALL)
                if match:
                    outline_current_chapter_text = match.group(1).strip()
            
            # Update state for current chapter generation
            return {
                "current_chapter_number": current_ch,
                "requested_chapter_number": current_ch,
                "current_chapter_text": "",  # Empty for new generation
                "prev_chapter_text": prev_chapter_text,
                "next_chapter_text": next_chapter_text,
                "outline_current_chapter_text": outline_current_chapter_text
            }
            
        except Exception as e:
            logger.error(f"Failed to prepare chapter context: {e}")
            return {
                "error": str(e),
                "task_status": "error"
            }
    
    async def _accumulate_chapter_node(self, state: FictionEditingState) -> Dict[str, Any]:
        """Accumulate generated chapter and update state for next iteration"""
        try:
            current_ch = state.get("current_generation_chapter")
            chapter_range = state.get("chapter_range")
            generated_chapters = state.get("generated_chapters", {})
            editor_operations = state.get("editor_operations", [])
            
            if not current_ch or not chapter_range:
                return {}
            
            # Extract generated text from operations
            generated_text = "\n\n".join([
                op.get("text", "").strip()
                for op in editor_operations
                if op.get("text", "").strip()
            ]).strip()
            
            if generated_text:
                generated_chapters[current_ch] = generated_text
                logger.info(f"Accumulated Chapter {current_ch} ({len(generated_text)} chars)")
            
            # Determine next chapter number for continuation
            start_ch, end_ch = chapter_range
            next_ch = None
            if current_ch < end_ch:
                next_ch = current_ch + 1
            
            return {
                "generated_chapters": generated_chapters,
                "current_generation_chapter": next_ch  # Update to next chapter or None if done
            }
            
        except Exception as e:
            logger.error(f"Failed to accumulate chapter: {e}")
            return {}
    
    def _route_chapter_completion(self, state: FictionEditingState) -> str:
        """Check if more chapters need to be generated"""
        current_ch = state.get("current_generation_chapter")
        chapter_range = state.get("chapter_range")
        
        if not chapter_range:
            return "complete"
        
        start_ch, end_ch = chapter_range
        
        if current_ch and current_ch < end_ch:
            # More chapters to generate - continue loop
            return "next_chapter"
        
        return "complete"
    
    def _route_single_vs_multi(self, state: FictionEditingState) -> str:
        """Route after resolve_operations: single chapter goes to format, multi goes to accumulate"""
        if state.get("is_multi_chapter", False):
            return "accumulate_chapter"
        return "format_response"
    
    async def _assess_reference_quality_node(self, state: FictionEditingState) -> Dict[str, Any]:
        """Assess completeness of reference materials and provide guidance"""
        try:
            logger.info("Assessing reference quality...")
            
            outline_body = state.get("outline_body")
            rules_body = state.get("rules_body")
            style_body = state.get("style_body")
            characters_bodies = state.get("characters_bodies", [])
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
            
            return {
                "reference_quality": reference_quality,
                "reference_warnings": warnings,
                "reference_guidance": reference_guidance
            }
            
        except Exception as e:
            logger.error(f"Reference assessment failed: {e}")
            return {
                "reference_quality": {"completeness_score": 0.0},
                "reference_warnings": [],
                "reference_guidance": ""
            }
    
    async def _validate_consistency_node(self, state: FictionEditingState) -> Dict[str, Any]:
        """Validate generated content for potential consistency issues"""
        try:
            logger.info("Validating consistency...")
            
            structured_edit = state.get("structured_edit")
            if not structured_edit:
                return {"consistency_warnings": []}
            
            operations = structured_edit.get("operations", [])
            if not operations:
                return {"consistency_warnings": []}
            
            # Extract generated text
            generated_texts = [op.get("text", "") for op in operations if op.get("text")]
            if not generated_texts:
                return {"consistency_warnings": []}
            
            combined_text = "\n\n".join(generated_texts)
            
            warnings = []
            
            # Check 1: Manuscript continuity - look for potential contradictions
            manuscript = state.get("manuscript", "")
            if manuscript and combined_text:
                # Simple heuristic checks
                if "## Chapter" in combined_text and "## Chapter" in manuscript:
                    # Check for duplicate chapter numbers
                    existing_chapters = set(re.findall(r'## Chapter (\d+)', manuscript))
                    new_chapters = set(re.findall(r'## Chapter (\d+)', combined_text))
                    duplicates = existing_chapters & new_chapters
                    if duplicates:
                        warnings.append(
                            f"⚠️ Chapter number collision: Chapter(s) {', '.join(duplicates)} "
                            f"already exist in manuscript"
                        )
            
            # Check 2: Style consistency - basic checks
            style_body = state.get("style_body")
            if style_body:
                # Check tense consistency
                if "present tense" in style_body.lower() and " had " in combined_text.lower():
                    warnings.append("⚠️ Possible tense inconsistency: Style guide specifies present tense")
                elif "past tense" in style_body.lower() and any(
                    combined_text.count(f" {verb} ") > 3 
                    for verb in ["am", "is", "are"]
                ):
                    warnings.append("⚠️ Possible tense inconsistency: Style guide specifies past tense")
            
            # Check 3: Character profile consistency
            characters_bodies = state.get("characters_bodies", [])
            if characters_bodies and combined_text:
                # Extract character names from profiles
                for char_body in characters_bodies:
                    # Look for name patterns
                    name_matches = re.findall(r'(?:name|Name|NAME)[:：]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', char_body)
                    for name in name_matches:
                        if name in combined_text:
                            # Character appears in generated text
                            # Could add more sophisticated behavioral checks here
                            pass
            
            # Check 4: Universe rules - look for common violations
            rules_body = state.get("rules_body")
            if rules_body and combined_text:
                # Check for rule violation indicators
                if "no magic" in rules_body.lower() and any(
                    word in combined_text.lower() 
                    for word in ["spell", "magic", "enchant", "wizard"]
                ):
                    warnings.append("⚠️ Possible universe rule violation: Magic use detected but rules forbid it")
            
            return {"consistency_warnings": warnings}
            
        except Exception as e:
            logger.error(f"Consistency validation failed: {e}")
            return {"consistency_warnings": []}
    
    async def _generate_edit_plan_node(self, state: FictionEditingState) -> Dict[str, Any]:
        """Generate edit plan using LLM"""
        try:
            logger.info("Generating fiction edit plan...")
            
            manuscript = state.get("manuscript", "")
            filename = state.get("filename", "manuscript.md")
            frontmatter = state.get("frontmatter", {})
            current_request = state.get("current_request", "")
            
            current_chapter_text = state.get("current_chapter_text", "")
            current_chapter_number = state.get("current_chapter_number")
            prev_chapter_text = state.get("prev_chapter_text")
            next_chapter_text = state.get("next_chapter_text")
            paragraph_text = state.get("paragraph_text", "")
            
            outline_body = state.get("outline_body")
            rules_body = state.get("rules_body")
            style_body = state.get("style_body")
            characters_bodies = state.get("characters_bodies", [])
            outline_current_chapter_text = state.get("outline_current_chapter_text")
            
            para_start = state.get("para_start", 0)
            selection_start = state.get("selection_start", -1)
            selection_end = state.get("selection_end", -1)
            cursor_offset = state.get("cursor_offset", -1)
            requested_chapter_number = state.get("requested_chapter_number")
            
            # Use requested chapter number if provided, otherwise use current
            target_chapter_number = requested_chapter_number if requested_chapter_number is not None else current_chapter_number
            
            # Determine chapter labels
            if requested_chapter_number is not None:
                current_chapter_label = f"Chapter {requested_chapter_number}"
            else:
                current_chapter_label = f"Chapter {current_chapter_number}" if current_chapter_number else "Current Chapter"
            prev_chapter_label = f"Chapter {prev_chapter_text and 'Previous'}" if prev_chapter_text else None
            next_chapter_label = f"Chapter {next_chapter_text and 'Next'}" if next_chapter_text else None
            
            # Build system prompt
            system_prompt = self._build_system_prompt()
            
            # Build context message
            context_parts = [
                "=== MANUSCRIPT CONTEXT ===\n",
                f"Primary file: {filename}\n",
                f"Working area: {current_chapter_label}\n",
                f"Cursor position: paragraph shown below\n\n"
            ]
            
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
                context_parts.append(f"=== {prev_chapter_label.upper()} (FOR CONTEXT - DO NOT EDIT) ===\n{prev_chapter_text}\n\n")
            
            context_parts.append(f"=== {current_chapter_label.upper()} (CURRENT WORK AREA) ===\n{current_chapter_text}\n\n")
            context_parts.append(f"=== PARAGRAPH AROUND CURSOR ===\n{paragraph_text}\n\n")
            
            if next_chapter_text:
                context_parts.append(f"=== {next_chapter_label.upper()} (FOR CONTEXT - DO NOT EDIT) ===\n{next_chapter_text}\n\n")
            
            if style_body:
                context_parts.append(f"=== STYLE GUIDE (voice and tone - READ FIRST) ===\n{style_body}\n\n")
            
            if rules_body:
                context_parts.append(f"=== RULES (universe constraints) ===\n{rules_body}\n\n")
            
            if characters_bodies:
                context_parts.append("".join([f"=== CHARACTER DOC ===\n{body}\n\n" for body in characters_bodies]))
            
            if outline_current_chapter_text:
                context_parts.append(f"=== CURRENT CHAPTER OUTLINE (beats to follow) ===\n{outline_current_chapter_text}\n\n")
            
            if outline_body:
                context_parts.append(f"=== FULL OUTLINE (story structure) ===\n{outline_body}\n\n")
            
            if not outline_body:
                context_parts.append(
                    "=== NO OUTLINE PRESENT ===\n"
                    "The user's request serves as the story directive (WHAT happens).\n"
                    "**Your task:** Write narrative prose in the Style Guide's voice that fulfills the user's request.\n"
                    "- Use the user's request as the story goal (e.g., 'Generate a paragraph where X happens')\n"
                    "- Apply Style Guide voice, techniques, and narrative craft to fulfill that request\n"
                    "- Write natural, compelling prose that matches the Style Guide, not generic narrative\n"
                    "- Continue from manuscript context if present, maintaining established voice and style\n\n"
                )
            
            # Add outline sync warnings if detected
            outline_needs_sync = state.get("outline_needs_sync", False)
            outline_sync_analysis = state.get("outline_sync_analysis")
            if outline_needs_sync and outline_sync_analysis:
                discrepancies = outline_sync_analysis.get("discrepancies", [])
                summary = outline_sync_analysis.get("summary", "")
                if discrepancies:
                    context_parts.append(
                        "\n⚠️ ⚠️ ⚠️ OUTLINE SYNC ALERT ⚠️ ⚠️ ⚠️\n"
                        "The outline for this chapter has changed and the manuscript needs updates!\n\n"
                        f"Summary: {summary}\n\n"
                        "Discrepancies detected:\n"
                    )
                    for i, disc in enumerate(discrepancies, 1):
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
                        "**YOUR TASK**: Update the manuscript to match the current outline.\n"
                        "Generate operations that revise the chapter to include missing beats, "
                        "update changed plot points, and align character actions with outline expectations.\n"
                        "Preserve good prose and Style Guide voice while making necessary updates.\n\n"
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
            elif outline_current_chapter_text:
                context_parts.append(
                    "=== OUTLINE AS NARRATIVE BLUEPRINT ===\n"
                    "The chapter outline provides story beats and structure, NOT a script to paraphrase.\n\n"
                    "**CRITICAL DISTINCTION:**\n"
                    "- **Outline = WHAT happens** (story structure, plot points, character arcs)\n"
                    "- **Style Guide = HOW to write** (voice, techniques, narrative craft)\n"
                    "- **Your prose = Natural storytelling in Style Guide voice that achieves Outline goals**\n\n"
                    "**BEFORE writing prose - Style Guide Application Process:**\n"
                    "1. Read the Style Guide completely - internalize voice, POV, tense, pacing\n"
                    "2. Note specific techniques: dialogue style, sensory detail level, show-don't-tell ratio\n"
                    "3. Identify any writing samples - these demonstrate the TARGET voice (emulate technique, never copy content)\n"
                    "4. Extract character voice patterns if specified in Style Guide\n"
                    "5. Understand the narrative voice so deeply it feels natural, not forced\n\n"
                    "**WHILE writing prose - Natural Storytelling Process:**\n"
                    "1. Write naturally in the established Style Guide voice - don't think about outline beats yet\n"
                    "2. Build complete scenes with setting, action, dialogue, internality, sensory details\n"
                    "3. Let story events emerge organically within scenes - don't convert beats sequentially\n"
                    "4. Check: Does each paragraph sound like it's from this Style Guide? Does it match the voice?\n"
                    "5. Only after scene flows naturally: verify outline beats are achieved (they should be, if you wrote the scene well)\n\n"
                    "**BAD EXAMPLE - Outline-Following (Mechanical Beat Conversion):**\n"
                    "Outline Beat: 'Peterson discovers the anomaly in the financial records'\n\n"
                    "❌ BAD (Checklist prose):\n"
                    "> Peterson discovered the anomaly in the financial records. He looked at the numbers. They didn't add up. He decided to investigate further.\n\n"
                    "Problems: Summary prose, no sensory details, no character voice, mechanical beat-conversion, generic narrative\n\n"
                    "**GOOD EXAMPLE - Style-Guided Narrative (Natural Storytelling):**\n"
                    "Outline Beat: 'Peterson discovers the anomaly in the financial records'\n\n"
                    "✅ GOOD (Style-guided narrative with comprehensive style):\n"
                    "> The spreadsheet blurred as Peterson's eyes traced the same column for the third time. Something gnawed at him—not the numbers themselves, but the spaces between them. Entry 2847: $50,000. Entry 2848: skipped. Entry 2849: $50,000. His finger hovered over the trackpad. 'Why would they skip an entry?' he muttered, already knowing the answer would cost him sleep tonight.\n\n"
                    "Why it works: Sensory details (blurred, traced, gnawed), character internality (knowing the answer would cost sleep), natural dialogue, specific details (entry numbers), atmospheric tension, Style Guide voice throughout\n\n"
                    "**Technique Checklist for Style Application:**\n"
                    "- [ ] Every sentence matches Style Guide voice (POV, tense, tone)\n"
                    "- [ ] Sensory details present (sight, sound, smell, texture, temperature) at appropriate level\n"
                    "- [ ] Show-don't-tell: emotions revealed through actions, not statements\n"
                    "- [ ] Complete scenes with setting, action, dialogue, internality - not summary prose\n"
                    "- [ ] Character voice authentic and natural (dialogue, thoughts, actions)\n"
                    "- [ ] Natural pacing - scenes breathe, don't rush through beats\n"
                    "- [ ] Outline beats achieved organically within natural story flow\n"
                    "- [ ] Transitions between beats feel natural, not mechanical\n\n"
                    "**What the outline provides:** Story structure, key events, plot progression\n"
                    "**What the Style Guide provides:** Narrative voice, writing style, technical requirements\n"
                    "**What you must add:** Full narrative prose, character voice, scene details, emotional depth\n"
                    "**Constraint:** Do not add major plot events not in the outline (new characters, deaths, revelations)\n"
                    "**Freedom:** Add all enriching details needed to make the prose compelling and complete\n"
                    "**Goal:** Write 800-1200 words of engaging narrative prose that sounds like it came from the Style Guide, not outline paraphrasing\n\n"
                )
            
            context_parts.append(f"⚠️ CRITICAL: Your operations must target {current_chapter_label.upper()} ONLY. ")
            context_parts.append("Adjacent chapters are for context (tone, transitions, continuity) - DO NOT edit them!\n\n")
            
            # Explicitly tell LLM which chapter number to use if requested
            if requested_chapter_number is not None:
                context_parts.append(f"⚠️ USER REQUESTED: Generate content for Chapter {requested_chapter_number}. ")
                context_parts.append(f"Set 'chapter_index' to {requested_chapter_number - 1} (0-indexed) in your JSON response. ")
                context_parts.append(f"Begin your generated text with '## Chapter {requested_chapter_number}' heading.\n\n")
            
            context_parts.append("Provide a ManuscriptEdit JSON plan for the current work area.")
            
            messages = [
                SystemMessage(content=system_prompt),
                SystemMessage(content=f"Current Date/Time: {datetime.now().isoformat()}"),
                HumanMessage(content="".join(context_parts))
            ]
            
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
                        "2. Find the FULL sentence or paragraph (20-40 words) that contains the word/phrase\n"
                        "3. Set 'original_text' to that FULL sentence/paragraph (not just the word)\n"
                        "4. Set 'text' to the same sentence/paragraph with ONLY the specific word/phrase changed\n"
                        "5. **PRESERVE ALL OTHER TEXT** - do not rewrite or regenerate the sentence\n"
                        "6. **DO NOT replace entire paragraphs** - only change the specific word/phrase\n\n"
                        "Example: If user says 'boat not canoe' and manuscript has:\n"
                        "  'He paddled the canoe across the river, feeling the current pull against him.'\n"
                        "Then:\n"
                        "  original_text: 'He paddled the canoe across the river, feeling the current pull against him.'\n"
                        "  text: 'He paddled the boat across the river, feeling the current pull against him.'\n\n"
                    )
                
                # Check if creating a new chapter
                is_creating_new_chapter = (
                    requested_chapter_number is not None and 
                    current_chapter_text.strip() == "" and
                    any(keyword in current_request.lower() for keyword in ["create", "craft", "write", "generate", "chapter"])
                )
                
                new_chapter_hints = ""
                if is_creating_new_chapter:
                    # Find the last chapter in the manuscript
                    chapter_ranges = state.get("chapter_ranges", [])
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
                            f"Find the last line of Chapter {last_chapter_num} in the manuscript above and use it as anchor_text.\n"
                            f"Example: If the last line is 'She closed the door behind her.', then:\n"
                            f"  op_type: 'insert_after_heading'\n"
                            f"  anchor_text: 'She closed the door behind her.'\n"
                            f"  text: '## Chapter {requested_chapter_number}\\n\\n[your chapter content]'\n"
                            f"**DO NOT** use '## Chapter {requested_chapter_number}' as anchor_text - it doesn't exist yet!\n"
                            f"**DO NOT** insert at the beginning of the file - insert after the last chapter!\n\n"
                        )
                
                messages.append(HumanMessage(content=(
                    f"USER REQUEST: {current_request}\n\n"
                    + selection_context +
                    granular_correction_hints +
                    new_chapter_hints +
                    "\n=== ANCHORING REQUIREMENTS FOR PROSE ===\n"
                    "For REPLACE/DELETE operations in prose (no headers), you MUST provide robust anchors:\n\n"
                    "**OPTION 1 (BEST): Use selection as anchor**\n"
                    "- If user selected text, match it EXACTLY in 'original_text'\n"
                    "- Include at least 20-30 words for reliable matching\n\n"
                    "**OPTION 2: Use left_context + right_context**\n"
                    "- left_context: 30-50 chars BEFORE the target (exact text)\n"
                    "- right_context: 30-50 chars AFTER the target (exact text)\n\n"
                    "**OPTION 3: Use long original_text**\n"
                    "- Include 20-40 words of EXACT, VERBATIM text to replace\n"
                    "- Include complete sentences with natural boundaries\n\n"
                    "⚠️ NEVER include chapter headers (##) in original_text - they will be deleted!\n"
                )))
            
            # Call LLM - pass state to access user's model selection from metadata
            llm = self._get_llm(temperature=0.4, state=state)
            start_time = datetime.now()
            response = await llm.ainvoke(messages)
            
            content = response.content if hasattr(response, 'content') else str(response)
            content = _unwrap_json_response(content)
            
            # Parse structured response
            structured_edit = None
            try:
                raw = json.loads(content)
                if isinstance(raw, dict) and isinstance(raw.get("operations"), list):
                    raw.setdefault("target_filename", filename)
                    raw.setdefault("scope", "paragraph")
                    raw.setdefault("summary", "Planned edit generated from context.")
                    raw.setdefault("safety", "medium")
                    structured_edit = raw
                else:
                    structured_edit = json.loads(content)
            except Exception as e:
                logger.error(f"Failed to parse structured edit: {e}")
                return {
                    "llm_response": content,
                    "structured_edit": None,
                    "error": f"Failed to parse edit plan: {str(e)}",
                    "task_status": "error"
                }
            
            if structured_edit is None:
                return {
                    "llm_response": content,
                    "structured_edit": None,
                    "error": "Failed to produce a valid ManuscriptEdit. Ensure ONLY raw JSON ManuscriptEdit with operations is returned.",
                    "task_status": "error"
                }
            
            return {
                "llm_response": content,
                "structured_edit": structured_edit,
                "system_prompt": system_prompt
            }
            
        except Exception as e:
            logger.error(f"Failed to generate edit plan: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "llm_response": "",
                "structured_edit": None,
                "error": str(e),
                "task_status": "error"
            }
    
    async def _resolve_operations_node(self, state: FictionEditingState) -> Dict[str, Any]:
        """Resolve editor operations with progressive search"""
        try:
            logger.info("Resolving editor operations...")
            
            manuscript = state.get("manuscript", "")
            structured_edit = state.get("structured_edit")
            selection_start = state.get("selection_start", -1)
            selection_end = state.get("selection_end", -1)
            para_start = state.get("para_start", 0)
            para_end = state.get("para_end", 0)
            current_chapter_number = state.get("current_chapter_number")
            requested_chapter_number = state.get("requested_chapter_number")
            chapter_ranges = state.get("chapter_ranges", [])
            
            if not structured_edit or not isinstance(structured_edit.get("operations"), list):
                return {
                    "editor_operations": [],
                    "error": "No operations to resolve",
                    "task_status": "error"
                }
            
            fm_end_idx = _frontmatter_end_index(manuscript)
            selection = {"start": selection_start, "end": selection_end} if selection_start >= 0 and selection_end >= 0 else None
            
            editor_operations = []
            operations = structured_edit.get("operations", [])
            
            # Determine desired chapter number
            # Priority: 1) User requested chapter, 2) LLM chapter_index (0-indexed, convert to 1-indexed), 3) Current chapter, 4) Next chapter
            if requested_chapter_number is not None:
                # User explicitly requested a chapter number - use it directly
                desired_ch_num = requested_chapter_number
            else:
                llm_chapter_index = structured_edit.get("chapter_index")
                if llm_chapter_index is not None:
                    # LLM provides 0-indexed chapter_index, convert to 1-indexed
                    desired_ch_num = int(llm_chapter_index) + 1
                elif current_chapter_number:
                    desired_ch_num = current_chapter_number
                else:
                    max_num = max([r.chapter_number for r in chapter_ranges if r.chapter_number is not None], default=0)
                    desired_ch_num = (max_num or 0) + 1
            
            for op in operations:
                # Resolve operation
                try:
                    resolved_start, resolved_end, resolved_text, resolved_confidence = _resolve_operation_simple(
                        manuscript,
                        op,
                        selection=selection,
                        frontmatter_end=fm_end_idx
                    )
                    
                    # If resolution failed (0:0) and this is a new chapter, find the last chapter
                    is_chapter_scope = (str(structured_edit.get("scope", "")).lower() == "chapter")
                    is_new_chapter = (resolved_start == resolved_end == 0) and is_chapter_scope
                    
                    if is_new_chapter and chapter_ranges:
                        # Find the last existing chapter and insert after it
                        last_chapter_range = chapter_ranges[-1]
                        resolved_start = last_chapter_range.end
                        resolved_end = last_chapter_range.end
                        resolved_confidence = 0.8
                        logger.info(f"New chapter detected - inserting after last chapter (Chapter {last_chapter_range.chapter_number}) at position {resolved_start}")
                    
                    logger.info(f"Resolved {op.get('op_type')} [{resolved_start}:{resolved_end}] confidence={resolved_confidence:.2f}")
                    
                    # Ensure chapter heading for new chapters
                    if is_chapter_scope and is_new_chapter and not resolved_text.strip().startswith('#'):
                        chapter_num = desired_ch_num or current_chapter_number or 1
                        resolved_text = _ensure_chapter_heading(resolved_text, int(chapter_num))
                    
                    # Calculate pre_hash
                    pre_slice = manuscript[resolved_start:resolved_end]
                    pre_hash = _slice_hash(pre_slice)
                    
                    # Build operation dict
                    resolved_op = {
                        "op_type": op.get("op_type", "replace_range"),
                        "start": resolved_start,
                        "end": resolved_end,
                        "text": resolved_text,
                        "pre_hash": pre_hash,
                        "original_text": op.get("original_text"),
                        "anchor_text": op.get("anchor_text"),
                        "left_context": op.get("left_context"),
                        "right_context": op.get("right_context"),
                        "occurrence_index": op.get("occurrence_index", 0),
                        "confidence": resolved_confidence
                    }
                    
                    editor_operations.append(resolved_op)
                    
                except Exception as e:
                    logger.warning(f"Operation resolution failed: {e}, using fallback")
                    # Fallback positioning
                    scope = str(structured_edit.get("scope", "")).lower()
                    if scope == "chapter" and desired_ch_num and chapter_ranges:
                        found = False
                        for r in chapter_ranges:
                            if r.chapter_number == desired_ch_num:
                                fallback_start = r.start
                                fallback_end = r.end
                                found = True
                                break
                        if not found and chapter_ranges:
                            fallback_start = chapter_ranges[-1].end
                            fallback_end = chapter_ranges[-1].end
                        else:
                            fallback_start = fm_end_idx
                            fallback_end = fm_end_idx
                    else:
                        fallback_start = para_start
                        fallback_end = para_end
                    
                    pre_slice = manuscript[fallback_start:fallback_end]
                    resolved_op = {
                        "op_type": op.get("op_type", "replace_range"),
                        "start": fallback_start,
                        "end": fallback_end,
                        "text": op.get("text", ""),
                        "pre_hash": _slice_hash(pre_slice),
                        "confidence": 0.3
                    }
                    editor_operations.append(resolved_op)
            
            return {
                "editor_operations": editor_operations
            }
            
        except Exception as e:
            logger.error(f"Failed to resolve operations: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "editor_operations": [],
                "error": str(e),
                "task_status": "error"
            }
    
    async def _format_response_node(self, state: FictionEditingState) -> Dict[str, Any]:
        """Format final response with editor operations"""
        try:
            logger.info("Formatting response...")
            
            structured_edit = state.get("structured_edit", {})
            editor_operations = state.get("editor_operations", [])
            task_status = state.get("task_status", "complete")
            is_multi_chapter = state.get("is_multi_chapter", False)
            generated_chapters = state.get("generated_chapters", {})
            
            if task_status == "error":
                error_msg = state.get("error", "Unknown error")
                return {
                    "response": {
                        "response": f"Fiction editing failed: {error_msg}",
                        "task_status": "error",
                        "agent_type": "fiction_editing_agent"
                    },
                    "task_status": "error"
                }
            
            # Build prose preview
            if is_multi_chapter and generated_chapters:
                # Multi-chapter: combine all generated chapters
                chapter_texts = []
                for ch_num in sorted(generated_chapters.keys()):
                    chapter_texts.append(generated_chapters[ch_num])
                generated_preview = "\n\n".join(chapter_texts).strip()
                response_text = f"Generated {len(generated_chapters)} chapters:\n\n{generated_preview}"
            else:
                # Single chapter
                generated_preview = "\n\n".join([
                    op.get("text", "").strip()
                    for op in editor_operations
                    if op.get("text", "").strip()
                ]).strip()
                response_text = generated_preview if generated_preview else (structured_edit.get("summary", "Edit plan ready."))
            
            # Add clarifying questions if present
            clarifying_questions = structured_edit.get("clarifying_questions", [])
            if clarifying_questions:
                questions_section = "\n\n**Questions for clarification:**\n" + "\n".join([
                    f"- {q}" for q in clarifying_questions
                ])
                response_text = response_text + questions_section
            
            # Add outline sync analysis if detected
            outline_needs_sync = state.get("outline_needs_sync", False)
            outline_sync_analysis = state.get("outline_sync_analysis")
            if outline_needs_sync and outline_sync_analysis:
                discrepancies = outline_sync_analysis.get("discrepancies", [])
                summary = outline_sync_analysis.get("summary", "")
                if discrepancies:
                    sync_section = "\n\n**⚠️ OUTLINE SYNC ALERT**\n"
                    sync_section += f"The outline for this chapter has changed. {summary}\n\n"
                    sync_section += "**Discrepancies detected:**\n"
                    for i, disc in enumerate(discrepancies, 1):
                        disc_type = disc.get("type", "unknown")
                        outline_exp = disc.get("outline_expectation", "")
                        manuscript_curr = disc.get("manuscript_current", "")
                        severity = disc.get("severity", "medium")
                        suggestion = disc.get("suggestion", "")
                        sync_section += f"{i}. [{severity.upper()}] {disc_type}:\n"
                        sync_section += f"   Outline expects: {outline_exp}\n"
                        sync_section += f"   Manuscript has: {manuscript_curr}\n"
                        sync_section += f"   → {suggestion}\n\n"
                    sync_section += "**I've generated operations to update the manuscript to match the outline.**\n"
                    response_text = sync_section + "\n\n" + response_text
            
            # Add consistency warnings if present
            consistency_warnings = state.get("consistency_warnings", [])
            reference_warnings = state.get("reference_warnings", [])
            
            all_warnings = consistency_warnings + reference_warnings
            if all_warnings:
                warnings_section = "\n\n**⚠️ Validation Notices:**\n" + "\n".join(all_warnings)
                response_text = response_text + warnings_section
            
            # Build response with editor operations
            response = {
                "response": response_text,
                "task_status": task_status,
                "agent_type": "fiction_editing_agent",
                "timestamp": datetime.now().isoformat()
            }
            
            # For multi-chapter, combine all operations from all chapters
            if is_multi_chapter and generated_chapters:
                # Build combined operations for all chapters
                all_operations = []
                chapter_range = state.get("chapter_range")
                if chapter_range:
                    start_ch, end_ch = chapter_range
                    for ch_num in range(start_ch, end_ch + 1):
                        ch_text = generated_chapters.get(ch_num, "")
                        if ch_text:
                            # Create operation for this chapter
                            all_operations.append({
                                "op_type": "insert_after_heading",
                                "text": ch_text,
                                "chapter_index": ch_num - 1  # 0-indexed
                            })
                
                if all_operations:
                    response["editor_operations"] = all_operations
                    response["manuscript_edit"] = {
                        "target_filename": structured_edit.get("target_filename", state.get("filename", "manuscript.md")),
                        "scope": "multi_chapter",
                        "summary": f"Generated chapters {start_ch}-{end_ch}",
                        "safety": structured_edit.get("safety", "medium"),
                        "operations": all_operations
                    }
            elif editor_operations:
                response["editor_operations"] = editor_operations
                response["manuscript_edit"] = {
                    "target_filename": structured_edit.get("target_filename"),
                    "scope": structured_edit.get("scope"),
                    "summary": structured_edit.get("summary"),
                    "chapter_index": structured_edit.get("chapter_index"),
                    "safety": structured_edit.get("safety"),
                    "operations": editor_operations
                }
            
            return {
                "response": response,
                "task_status": task_status
            }
            
        except Exception as e:
            logger.error(f"Failed to format response: {e}")
            return {
                "response": self._create_error_response(str(e)),
                "task_status": "error"
            }
    
    async def process(self, query: str, metadata: Dict[str, Any] = None, messages: List[Any] = None) -> Dict[str, Any]:
        """Process fiction editing query using LangGraph workflow"""
        try:
            logger.info(f"Fiction editing agent processing: {query[:100]}...")
            
            # Extract user_id and shared_memory from metadata
            metadata = metadata or {}
            user_id = metadata.get("user_id", "system")
            shared_memory = metadata.get("shared_memory", {}) or {}
            
            # Prepare new messages (current query)
            new_messages = self._prepare_messages_with_query(messages, query)
            
            # Get workflow to access checkpoint
            workflow = await self._get_workflow()
            config = self._get_checkpoint_config(metadata)
            
            # Load and merge checkpointed messages to preserve conversation history
            conversation_messages = await self._load_and_merge_checkpoint_messages(
                workflow, config, new_messages
            )
            
            # Load shared_memory from checkpoint if available
            checkpoint_state = await workflow.aget_state(config)
            existing_shared_memory = {}
            if checkpoint_state and checkpoint_state.values:
                existing_shared_memory = checkpoint_state.values.get("shared_memory", {})
            
            # Merge shared_memory: start with checkpoint, then update with NEW data (so new active_editor overwrites old)
            shared_memory_merged = existing_shared_memory.copy()
            shared_memory_merged.update(shared_memory)  # New data (including updated active_editor) takes precedence
            
            # Initialize state for LangGraph workflow
            initial_state: FictionEditingState = {
                "query": query,
                "user_id": user_id,
                "metadata": metadata,
                "messages": conversation_messages,
                "shared_memory": shared_memory_merged,
                "active_editor": {},
                "manuscript": "",
                "filename": "manuscript.md",
                "frontmatter": {},
                "cursor_offset": -1,
                "selection_start": -1,
                "selection_end": -1,
                "chapter_ranges": [],
                "active_chapter_idx": -1,
                "current_chapter_text": "",
                "current_chapter_number": None,
                "prev_chapter_text": None,
                "next_chapter_text": None,
                "paragraph_text": "",
                "para_start": 0,
                "para_end": 0,
                "outline_body": None,
                "rules_body": None,
                "style_body": None,
                "characters_bodies": [],
                "outline_current_chapter_text": None,
                "current_request": "",
                "requested_chapter_number": None,
                "system_prompt": "",
                "llm_response": "",
                "structured_edit": None,
                "editor_operations": [],
                "response": {},
                "task_status": "",
                "error": "",
                # New fields for mode tracking and validation
                "generation_mode": "",
                "creative_freedom_requested": False,
                "mode_guidance": "",
                "reference_quality": {},
                "reference_warnings": [],
                "reference_guidance": "",
                "consistency_warnings": [],
                # Multi-chapter generation fields
                "is_multi_chapter": False,
                "chapter_range": None,
                "current_generation_chapter": None,
                "generated_chapters": {},
                # Outline sync detection
                "outline_sync_analysis": None,
                "outline_needs_sync": False
            }
            
            # Run LangGraph workflow with checkpointing (workflow and config already created above)
            result_state = await workflow.ainvoke(initial_state, config=config)
            
            # Extract final response
            response = result_state.get("response", {})
            task_status = result_state.get("task_status", "complete")
            
            if task_status == "error":
                error_msg = result_state.get("error", "Unknown error")
                logger.error(f"Fiction editing agent failed: {error_msg}")
                return {
                    "response": f"Fiction editing failed: {error_msg}",
                    "task_status": "error",
                    "agent_results": {}
                }
            
            # Build result dict matching character_development_agent pattern
            result = {
                "response": response.get("response", "Fiction editing complete"),
                "task_status": task_status,
                "agent_results": {
                    "editor_operations": response.get("editor_operations", []),
                    "manuscript_edit": response.get("manuscript_edit")
                }
            }
            
            # Add editor operations at top level for compatibility
            if response.get("editor_operations"):
                result["editor_operations"] = response["editor_operations"]
            if response.get("manuscript_edit"):
                result["manuscript_edit"] = response["manuscript_edit"]
            
            logger.info(f"Fiction editing agent completed: {task_status}")
            return result
            
        except Exception as e:
            logger.error(f"Fiction editing agent failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "response": f"Fiction editing failed: {str(e)}",
                "task_status": "error",
                "agent_results": {}
            }


def get_fiction_editing_agent() -> FictionEditingAgent:
    """Get singleton fiction editing agent instance"""
    global _fiction_editing_agent
    if _fiction_editing_agent is None:
        _fiction_editing_agent = FictionEditingAgent()
    return _fiction_editing_agent


_fiction_editing_agent: Optional[FictionEditingAgent] = None

