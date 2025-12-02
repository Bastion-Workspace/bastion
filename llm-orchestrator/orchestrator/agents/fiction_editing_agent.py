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
                return pos, end_pos, text, 1.0
            count += 1
            search_from = pos + 1
    
    # Strategy 2: Anchor text for insert_after_heading
    if anchor_text and op_type == "insert_after_heading":
        pos = manuscript.find(anchor_text)
        if pos != -1:
            # Find end of line/paragraph
            end_pos = manuscript.find("\n", pos)
            if end_pos == -1:
                end_pos = len(manuscript)
            else:
                end_pos += 1
            return end_pos, end_pos, text, 0.9
    
    # Strategy 3: Left + right context
    if left_context and right_context:
        pattern = re.escape(left_context) + r"([\s\S]{0,400}?)" + re.escape(right_context)
        m = re.search(pattern, manuscript)
        if m:
            return m.start(1), m.end(1), text, 0.8
    
    # Fallback: use approximate positions from op_dict
    start = op_dict.get("start", 0)
    end = op_dict.get("end", 0)
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
        workflow.add_node("detect_mode", self._detect_mode_and_intent_node)
        
        # Phase 3: Generation
        workflow.add_node("generate_edit_plan", self._generate_edit_plan_node)
        
        # Phase 4: Post-generation validation
        workflow.add_node("validate_consistency", self._validate_consistency_node)
        
        # Phase 5: Resolution and response
        workflow.add_node("resolve_operations", self._resolve_operations_node)
        workflow.add_node("format_response", self._format_response_node)
        
        # Entry point
        workflow.set_entry_point("prepare_context")
        
        # Flow
        workflow.add_edge("prepare_context", "analyze_scope")
        workflow.add_edge("analyze_scope", "load_references")
        workflow.add_edge("load_references", "assess_references")
        workflow.add_edge("assess_references", "detect_mode")
        workflow.add_edge("detect_mode", "generate_edit_plan")
        workflow.add_edge("generate_edit_plan", "validate_consistency")
        workflow.add_edge("validate_consistency", "resolve_operations")
        workflow.add_edge("resolve_operations", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow.compile(checkpointer=checkpointer)
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for fiction editing"""
        return (
            "You are a MASTER NOVELIST editor/generator. Persona disabled. Adhere strictly to the project's Style "
            "Guide and Rules above all else. Maintain originality and do not copy from references.\n\n"
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
            "=== CONTENT GENERATION RULES ===\n\n"
            "1. operations[].text MUST contain final prose (no placeholders or notes)\n"
            "2. For chapter generation: aim 800-1200 words, begin with '## Chapter N'\n"
            "3. If outline present: Follow outline structure as story blueprint, add enriching details\n"
            "4. NO YAML frontmatter in operations[].text\n"
            "5. Match established voice and style\n"
            "6. Complete sentences with proper grammar\n"
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
            
            current_request = state.get("current_request", "").lower()
            current_chapter_text = state.get("current_chapter_text", "")
            outline_current_chapter_text = state.get("outline_current_chapter_text")
            
            # Detect creative freedom keywords
            creative_keywords = [
                "add", "enhance", "expand", "enrich", "include", 
                "give", "show", "more", "develop", "deepen"
            ]
            creative_freedom_requested = any(kw in current_request for kw in creative_keywords)
            
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
                    "You are generating NEW content from outline beats.\n"
                    "- Follow outline structure as story blueprint\n"
                    "- Add enriching details that enhance outlined events\n"
                    "- Maintain Style Guide voice precisely\n"
                    "- Respect Universe Rules absolutely\n"
                    "- Use Character profiles for authentic behavior\n"
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
                "mode_guidance": mode_guidance
            }
            
        except Exception as e:
            logger.error(f"Mode detection failed: {e}")
            return {
                "generation_mode": "editing",
                "creative_freedom_requested": False,
                "mode_guidance": ""
            }
    
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
            
            # Determine chapter labels
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
            
            if prev_chapter_text:
                context_parts.append(f"=== {prev_chapter_label.upper()} (FOR CONTEXT - DO NOT EDIT) ===\n{prev_chapter_text}\n\n")
            
            context_parts.append(f"=== {current_chapter_label.upper()} (CURRENT WORK AREA) ===\n{current_chapter_text}\n\n")
            context_parts.append(f"=== PARAGRAPH AROUND CURSOR ===\n{paragraph_text}\n\n")
            
            if next_chapter_text:
                context_parts.append(f"=== {next_chapter_label.upper()} (FOR CONTEXT - DO NOT EDIT) ===\n{next_chapter_text}\n\n")
            
            if outline_current_chapter_text:
                context_parts.append(f"=== CURRENT CHAPTER OUTLINE (beats to follow) ===\n{outline_current_chapter_text}\n\n")
            
            if outline_body:
                context_parts.append(f"=== FULL OUTLINE (story structure) ===\n{outline_body}\n\n")
            
            if rules_body:
                context_parts.append(f"=== RULES (universe constraints) ===\n{rules_body}\n\n")
            
            if style_body:
                context_parts.append(f"=== STYLE GUIDE (voice and tone) ===\n{style_body}\n\n")
            
            if characters_bodies:
                context_parts.append("".join([f"=== CHARACTER DOC ===\n{body}\n\n" for body in characters_bodies]))
            
            if not outline_body:
                context_parts.append("⚠️ NO OUTLINE PRESENT: Continue from manuscript context in established voice and style.\n\n")
            
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
                context_parts.append("⚠️ STRICT OUTLINE CONSTRAINTS: When chapter outline is present, convert ONLY those beats into prose. Do NOT add events not in outline.\n\n")
            
            context_parts.append(f"⚠️ CRITICAL: Your operations must target {current_chapter_label.upper()} ONLY. ")
            context_parts.append("Adjacent chapters are for context (tone, transitions, continuity) - DO NOT edit them!\n\n")
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
                messages.append(HumanMessage(content=(
                    f"USER REQUEST: {current_request}\n\n"
                    + selection_context +
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
            
            # Call LLM
            llm = self._get_llm(temperature=0.4)
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
            desired_ch_num = structured_edit.get("chapter_index")
            if desired_ch_num is not None:
                desired_ch_num = int(desired_ch_num) + 1
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
                    
                    logger.info(f"Resolved {op.get('op_type')} [{resolved_start}:{resolved_end}] confidence={resolved_confidence:.2f}")
                    
                    # Ensure chapter heading for new chapters
                    is_chapter_scope = (str(structured_edit.get("scope", "")).lower() == "chapter")
                    is_new_chapter = (resolved_start == resolved_end)
                    
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
            
            if editor_operations:
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
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process fiction editing query using LangGraph workflow"""
        try:
            # Extract query from state
            messages = state.get("messages", [])
            query = ""
            if messages:
                latest_message = messages[-1]
                query = latest_message.content if hasattr(latest_message, 'content') else str(latest_message)
            else:
                query = state.get("query", "")
            
            logger.info(f"Fiction editing agent processing: {query[:100]}...")
            
            shared_memory = state.get("shared_memory", {}) or {}
            metadata = state.get("metadata", {}) or {}
            user_id = state.get("user_id", metadata.get("user_id", "system"))
            
            # Initialize state for LangGraph workflow
            initial_state: FictionEditingState = {
                "query": query,
                "user_id": user_id,
                "metadata": metadata,
                "messages": messages,
                "shared_memory": shared_memory,
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
                "consistency_warnings": []
            }
            
            # Get workflow (lazy initialization with checkpointer)
            workflow = await self._get_workflow()
            
            # Get checkpoint config (handles thread_id from conversation_id/user_id)
            config = self._get_checkpoint_config(metadata)
            
            # Run LangGraph workflow with checkpointing
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

