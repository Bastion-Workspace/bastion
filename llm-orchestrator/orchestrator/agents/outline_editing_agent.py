"""
Outline Editing Agent - LangGraph Implementation
Gated to outline documents. Consumes full outline body (frontmatter stripped),
loads Style, Rules, and Character references directly from this file's
frontmatter, and emits editor operations for Prefer Editor HITL.
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

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


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


# ============================================
# Simplified Resolver (Progressive Search)
# ============================================

def _assess_reference_quality(content: str, ref_type: str) -> Tuple[float, List[str]]:
    """
    Assess reference quality and return (quality_score, warnings).
    Returns quality score 0.0-1.0 and list of warnings.
    """
    if not content or len(content.strip()) < 50:
        return 0.0, ["Reference content is too short or empty"]
    
    quality_score = 0.5  # Base score for existing content
    warnings = []
    
    content_length = len(content.strip())
    
    if ref_type == "rules":
        # Good rules have structure and specificity
        if "## " in content or "- " in content:  # Has structure
            quality_score += 0.2
        else:
            warnings.append("Rules lack clear structure (no headings or bullets)")
        
        if content_length > 500:  # Substantial content
            quality_score += 0.3
        elif content_length < 200:
            warnings.append("Rules content is quite brief")
        
        # Check for key rule indicators
        rule_keywords = ["rule", "constraint", "limit", "cannot", "must", "cannot", "forbidden"]
        if any(kw in content.lower() for kw in rule_keywords):
            quality_score += 0.1
    
    elif ref_type == "style":
        # Good style guides have examples and specifics
        if "example" in content.lower() or "```" in content:
            quality_score += 0.2
        else:
            warnings.append("Style guide lacks examples")
        
        if content_length > 300:
            quality_score += 0.3
        elif content_length < 150:
            warnings.append("Style guide is quite brief")
        
        # Check for style indicators
        style_keywords = ["voice", "tone", "pacing", "dialogue", "narrative", "tense", "pov"]
        if any(kw in content.lower() for kw in style_keywords):
            quality_score += 0.1
    
    elif ref_type == "characters":
        # Good character profiles have detail
        if content_length > 400:
            quality_score += 0.3
        elif content_length < 200:
            warnings.append("Character profile is quite brief")
        
        # Check for character depth indicators
        char_keywords = ["motivation", "personality", "goal", "backstory", "trait", "relationship"]
        if any(kw in content.lower() for kw in char_keywords):
            quality_score += 0.2
        else:
            warnings.append("Character profile lacks depth indicators")
    
    return min(1.0, quality_score), warnings


def _resolve_operation_simple(
    outline: str,
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
            pos = outline.find(original_text, search_from)
            if pos == -1:
                break
            if count == occurrence_index:
                end_pos = pos + len(original_text)
                return pos, end_pos, text, 1.0
            count += 1
            search_from = pos + 1
    
    # Strategy 2: Anchor text for insert_after_heading
    if anchor_text and op_type == "insert_after_heading":
        pos = outline.find(anchor_text)
        if pos != -1:
            # For chapter headings, find the end of the entire chapter (not just the heading line)
            # Look for the next chapter heading or end of document
            if anchor_text.startswith("## Chapter"):
                # Find end of this chapter by looking for next chapter heading
                next_chapter_pattern = re.compile(r"\n##\s+Chapter\s+\d+", re.MULTILINE)
                match = next_chapter_pattern.search(outline, pos + len(anchor_text))
                if match:
                    # Insert before the next chapter
                    end_pos = match.start()
                else:
                    # This is the last chapter, insert at end of document
                    end_pos = len(outline)
            else:
                # For non-chapter headings, just find end of line/paragraph
                end_pos = outline.find("\n", pos)
                if end_pos == -1:
                    end_pos = len(outline)
                else:
                    end_pos += 1
            return end_pos, end_pos, text, 0.9
        else:
            # Anchor text not found - check if it's a heading pattern
            # If anchor_text looks like a heading (starts with #), create the section
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', anchor_text.strip())
            if heading_match:
                # This is a heading that doesn't exist - insert at end of document
                # Prepend the heading to the text so the section gets created
                heading_level = heading_match.group(1)
                heading_text = heading_match.group(2)
                section_header = f"\n\n{heading_level} {heading_text}\n\n"
                # Ensure text starts with section header if it doesn't already
                if not text.strip().startswith(heading_level):
                    text = section_header + text
                else:
                    # Text already has header, just ensure proper spacing
                    if not text.startswith("\n"):
                        text = "\n\n" + text
                # Insert at end of document (after frontmatter)
                end_pos = len(outline)
                return end_pos, end_pos, text, 0.7
    
    # Strategy 3: Left + right context
    if left_context and right_context:
        pattern = re.escape(left_context) + r"([\s\S]{0,400}?)" + re.escape(right_context)
        m = re.search(pattern, outline)
        if m:
            return m.start(1), m.end(1), text, 0.8
    
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

class OutlineEditingState(TypedDict):
    """State for outline editing agent LangGraph workflow"""
    query: str
    user_id: str
    metadata: Dict[str, Any]
    messages: List[Any]
    shared_memory: Dict[str, Any]
    active_editor: Dict[str, Any]
    outline: str
    filename: str
    frontmatter: Dict[str, Any]
    cursor_offset: int
    selection_start: int
    selection_end: int
    body_only: str
    para_start: int
    para_end: int
    rules_body: Optional[str]
    style_body: Optional[str]
    characters_bodies: List[str]
    clarification_context: str
    current_request: str
    system_prompt: str
    llm_response: str
    structured_edit: Optional[Dict[str, Any]]
    clarification_request: Optional[Dict[str, Any]]
    editor_operations: List[Dict[str, Any]]
    response: Dict[str, Any]
    task_status: str
    error: str
    # NEW: Mode tracking
    generation_mode: str  # "fully_referenced" | "partial_references" | "freehand"
    available_references: Dict[str, bool]  # Which refs loaded successfully
    reference_summary: str  # Human-readable summary
    mode_guidance: str  # Dynamic prompt guidance
    reference_quality: Dict[str, float]  # Quality scores (0-1)
    reference_warnings: List[str]  # Quality warnings
    # NEW: Structure analysis
    outline_completeness: float  # 0.0-1.0
    chapter_count: int
    structure_warnings: List[str]
    structure_guidance: str
    has_synopsis: bool
    has_notes: bool
    has_characters: bool
    has_outline_section: bool
    # NEW: Content routing plan
    routing_plan: Optional[Dict[str, Any]]  # Structured routing plan from analysis


# ============================================
# Outline Editing Agent
# ============================================

class OutlineEditingAgent(BaseAgent):
    """
    Outline Editing Agent for outline development and editing
    
    Gated to outline documents. Consumes full outline body (frontmatter stripped),
    loads Style, Rules, and Character references directly from this file's
    frontmatter, and emits editor operations for Prefer Editor HITL.
    Uses LangGraph workflow for explicit state management
    """
    
    def __init__(self):
        super().__init__("outline_editing_agent")
        logger.info("Outline Editing Agent ready!")
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """Build LangGraph workflow for outline editing agent"""
        workflow = StateGraph(OutlineEditingState)
        
        # Add nodes
        workflow.add_node("prepare_context", self._prepare_context_node)
        workflow.add_node("load_references", self._load_references_node)
        workflow.add_node("analyze_mode", self._analyze_mode_node)
        workflow.add_node("analyze_outline_structure", self._analyze_outline_structure_node)
        workflow.add_node("analyze_and_route_request", self._analyze_and_route_request_node)
        workflow.add_node("generate_edit_plan", self._generate_edit_plan_node)
        workflow.add_node("resolve_operations", self._resolve_operations_node)
        workflow.add_node("format_response", self._format_response_node)
        
        # Entry point
        workflow.set_entry_point("prepare_context")
        
        # Flow: prepare_context -> load_references -> analyze_mode -> analyze_outline_structure -> analyze_and_route_request -> generate_edit_plan -> resolve_operations -> format_response -> END
        workflow.add_edge("prepare_context", "load_references")
        workflow.add_edge("load_references", "analyze_mode")
        workflow.add_edge("analyze_mode", "analyze_outline_structure")
        workflow.add_edge("analyze_outline_structure", "analyze_and_route_request")
        workflow.add_edge("analyze_and_route_request", "generate_edit_plan")
        workflow.add_edge("generate_edit_plan", "resolve_operations")
        workflow.add_edge("resolve_operations", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow.compile(checkpointer=checkpointer)
    
    def _build_system_prompt(self, state: Optional[OutlineEditingState] = None) -> str:
        """Build context-aware system prompt for outline editing"""
        
        # Base prompt (always included)
        base_prompt = (
            "You are an Outline Development Assistant. Generate outlines based on user requests.\n\n"
            "🎯 CORE PRINCIPLE: USER'S REQUEST IS PRIMARY\n"
            "Generate what the user asks for. References (style/rules/characters) are consistency guidelines, not content sources.\n\n"
            "OUTLINE STRUCTURE:\n"
            "# Overall Synopsis\n"
            "# Notes\n"
            "# Characters\n"
            "  - Protagonists: Name - Role\n"
            "  - Antagonists: Name - Role\n"
            "  - Supporting Characters: Name - Role\n"
            "# Outline\n"
            "## Chapter 1\n"
            "[3-5 sentence summary of chapter events]\n"
            "- Specific plot event or story beat\n"
            "- Another plot event or story beat\n"
            "- Continue with bullet points of what happens in the chapter\n"
            "(Typically 6-12 bullet points per chapter, but adjust based on chapter complexity)\n\n"
            "**CONTENT ROUTING RULES (CRITICAL)**:\n"
            "When the user provides information, you MUST route it to the appropriate section:\n\n"
            "- **# Overall Synopsis** -> Story-wide summary (2-3 sentences about the entire story)\n"
            "  - Overall plot summary\n"
            "  - Main story arc description\n"
            "  - High-level story premise\n\n"
            "- **# Notes** -> Story-wide rules, themes, and constraints that apply to the ENTIRE story\n"
            "  - Universe rules and constraints (magic systems, world-building rules, etc.)\n"
            "  - Themes and motifs that run throughout the story\n"
            "  - Tone and mood guidelines for the entire work\n"
            "  - Writing techniques and stylistic approaches\n"
            "  - Symbols and recurring elements\n"
            "  - Genre considerations and conventions\n"
            "  - Any information that applies broadly to ALL chapters, not just one\n\n"
            "- **# Characters** -> Character information (protagonists, antagonists, supporting)\n"
            "  - Character names and roles\n"
            "  - Character relationships\n"
            "  - Character overviews (detailed profiles go in character reference files)\n\n"
            "- **## Chapter N** -> Chapter-specific plot events and beats\n"
            "  - Specific plot events that happen in THIS chapter\n"
            "  - Character actions within a specific chapter\n"
            "  - Chapter-specific conflicts and resolutions\n"
            "  - Scene-by-scene beats for the chapter\n\n"
            "**ROUTING DECISION LOGIC**:\n"
            "- If information applies to the ENTIRE story (rules, themes, universe constraints) -> # Notes\n"
            "- If information is a high-level story summary -> # Overall Synopsis\n"
            "- If information describes what happens in a SPECIFIC chapter -> ## Chapter N\n"
            "- If information is about characters (who they are, their roles) -> # Characters\n"
            "- When in doubt: Story-wide information goes in Notes, chapter-specific events go in Chapters\n\n"
            "**REVISION CAPABILITIES**:\n"
            "- You CAN and SHOULD edit existing sections (Notes, Synopsis, Characters, Chapters)\n"
            "- Use 'replace_range' operations when user wants to change/replace existing content\n"
            "- Use 'insert_after_heading' to ADD new content to existing sections (preserves all existing content)\n"
            "- **CRITICAL FOR CHAPTERS**: When user wants to ADD to a chapter (not replace), use 'insert_after_heading'\n"
            "  - Find the last bullet point in the chapter and insert new bullets after it\n"
            "  - Or insert after the chapter heading if the chapter only has a summary\n"
            "  - **NEVER regenerate the entire chapter** when adding - preserve all existing beats\n"
            "- **GRANULAR CORRECTIONS**: When user corrects a specific word/phrase (e.g., 'boat not canoe')\n"
            "  - Find the specific bullet point or sentence containing the word/phrase in the CURRENT OUTLINE\n"
            "  - Use 'replace_range' with original_text = the FULL bullet point/sentence (20+ words for uniqueness)\n"
            "  - Set content = same text with ONLY the specific word/phrase changed\n"
            "  - Example: original_text: '- Character escapes in a canoe across the river'\n"
            "    content: '- Character escapes in a boat across the river'\n"
            "  - **DO NOT replace the entire chapter** - only replace the specific bullet point\n"
            "- Always check if a section exists before creating new content - update existing sections when appropriate\n\n"
            "**ROUTING PLAN USAGE (CRITICAL)**:\n"
            "- You will receive a routing plan that identifies ALL distinct pieces of information from the user's request\n"
            "- The routing plan specifies which section each piece belongs to and what operation type to use\n"
            "- You MUST generate operations for EVERY piece in the routing plan - do not skip any\n"
            "- Use the routing plan's target_section, anchor_text, and operation_type for each operation\n"
            "- If the routing plan indicates structural changes (character role changes, etc.), handle them appropriately\n\n"
            "CHAPTER RULES:\n"
            "- Every chapter needs a summary paragraph (3-5 sentences) followed by bullet points of plot events\n"
            "- Use simple bullet points (just '- ') - no need for 'Beat 1:', 'Beat 2:' labels\n"
            "- Focus on plot events, character actions, and story progression - not prose or dialogue\n"
            "- Typically 6-12 bullet points per chapter (adjust based on chapter complexity and pacing needs)\n"
            "- Use '## Chapter N' format (numbers only, no titles)\n"
            "- NO empty chapters or placeholder text\n\n"
            "**CHAPTER COUNT RESPECT (CRITICAL)**:\n"
            "- **ALWAYS respect the exact number of chapters requested by the user**\n"
            "- If user asks for \"1 chapter\" or \"Chapter 1\", generate ONLY 1 chapter - do NOT continue the story beyond what's requested\n"
            "- If user asks for \"2 chapters\" and provides material for 2 chapters, generate exactly 2 chapters\n"
            "- If user asks for \"chapters 3-5\", generate exactly those 3 chapters (3, 4, and 5)\n"
            "- **DO NOT carry the story forward beyond what's requested** - stop at the exact number of chapters requested\n"
            "- If user provides material that could span multiple chapters but only asks for 1 chapter, generate only 1 chapter with the most relevant content\n"
            "- If user provides material for 2 chapters and asks for 2 chapters, generate exactly 2 chapters\n"
            "- When in doubt, generate the minimum number of chapters that satisfies the request - do not add extra chapters\n\n"
            "OUTPUT FORMAT - ManuscriptEdit JSON:\n"
            "{\n"
            '  "type": "ManuscriptEdit",\n'
            '  "target_filename": "filename.md",\n'
            '  "scope": "paragraph|chapter|multi_chapter",\n'
            '  "summary": "brief description",\n'
            '  "operations": [\n'
            "    {\n"
            '      "op_type": "replace_range|delete_range|insert_after_heading",\n'
            '      "start": 0,\n'
            '      "end": 0,\n'
            '      "text": "content to insert/replace",\n'
            '      "original_text": "exact text from file (for replace/delete)",\n'
            '      "anchor_text": "exact header line (for insert_after_heading)"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "ANCHORING (CRITICAL):\n"
            "- For replace/delete: Include 'original_text' with EXACT text from file (20+ words)\n"
            "- For insert after header: Use 'anchor_text' with exact header line (e.g., '## Chapter 1')\n"
            "- Never include headers in 'original_text' for replace operations\n"
            "- Text should end immediately after last character (no trailing newlines)\n\n"
            "Output raw JSON only (no markdown fences, no explanatory text).\n"
        )
        
        # If no state provided, return base prompt
        if not state:
            return base_prompt
        
        # Build dynamic sections based on mode and structure
        mode_guidance = state.get("mode_guidance", "")
        structure_guidance = state.get("structure_guidance", "")
        generation_mode = state.get("generation_mode", "freehand")
        available_references = state.get("available_references", {})
        reference_summary = state.get("reference_summary", "")
        
        # Build dynamic prompt sections
        sections = [base_prompt]
        
        # Add mode section
        if mode_guidance:
            sections.append("\n\n=== GENERATION MODE ===\n")
            sections.append(mode_guidance)
        
        # Add reference summary
        if reference_summary:
            sections.append(f"\n\n=== REFERENCE STATUS ===\n{reference_summary}\n")
        
        # Add structure guidance
        if structure_guidance:
            sections.append("\n\n=== OUTLINE STRUCTURE STATUS ===\n")
            sections.append(structure_guidance)
        
        # Add brief reference guidance if references are available
        has_any_refs = any(available_references.values())
        if has_any_refs:
            ref_types = [k for k, v in available_references.items() if v]
            sections.append(
                f"\n\n=== REFERENCES AVAILABLE ===\n"
                f"You have: {', '.join(ref_types)}\n"
                f"Use these as consistency guidelines within the user's requested story.\n"
            )
        
        return "".join(sections)
    
    async def _prepare_context_node(self, state: OutlineEditingState) -> Dict[str, Any]:
        """Prepare context: extract active editor, validate outline type"""
        try:
            logger.info("Preparing context for outline editing...")
            
            shared_memory = state.get("shared_memory", {}) or {}
            active_editor = shared_memory.get("active_editor", {}) or {}
            
            outline = active_editor.get("content", "") or ""
            filename = active_editor.get("filename") or "outline.md"
            frontmatter = active_editor.get("frontmatter", {}) or {}
            cursor_offset = int(active_editor.get("cursor_offset", -1))
            selection_start = int(active_editor.get("selection_start", -1))
            selection_end = int(active_editor.get("selection_end", -1))
            
            # Hard gate: require outline
            doc_type = str(frontmatter.get("type", "")).strip().lower()
            if doc_type != "outline":
                return {
                    "error": "Active editor is not an outline file; skipping.",
                    "task_status": "error",
                    "response": {
                        "response": "Active editor is not an outline file; skipping.",
                        "task_status": "error",
                        "agent_type": "outline_editing_agent"
                    }
                }
            
            # Check if responding to previous clarification request
            previous_clarification = shared_memory.get("pending_outline_clarification")
            clarification_context = ""
            if previous_clarification:
                logger.info("Detected previous clarification request, including context")
                clarification_context = (
                    "\n\n=== PREVIOUS CLARIFICATION REQUEST ===\n"
                    f"Context: {previous_clarification.get('context', '')}\n"
                    f"Questions Asked:\n"
                )
                for i, q in enumerate(previous_clarification.get('questions', []), 1):
                    clarification_context += f"{i}. {q}\n"
                clarification_context += "\nThe user's response is in their latest message. Use this context to proceed with the outline development.\n"
            
            # Extract user request
            messages = state.get("messages", [])
            try:
                if messages:
                    latest_message = messages[-1]
                    current_request = latest_message.content if hasattr(latest_message, 'content') else str(latest_message)
                else:
                    current_request = state.get("query", "")
            except Exception as e:
                logger.error(f"Failed to extract user request: {e}")
                current_request = ""
            
            # Get paragraph bounds
            body_only = _strip_frontmatter_block(outline)
            para_start, para_end = paragraph_bounds(outline, cursor_offset if cursor_offset >= 0 else 0)
            if selection_start >= 0 and selection_end > selection_start:
                para_start, para_end = selection_start, selection_end
            
            return {
                "active_editor": active_editor,
                "outline": outline,
                "filename": filename,
                "frontmatter": frontmatter,
                "cursor_offset": cursor_offset,
                "selection_start": selection_start,
                "selection_end": selection_end,
                "body_only": body_only,
                "para_start": para_start,
                "para_end": para_end,
                "clarification_context": clarification_context,
                "current_request": current_request.strip()
            }
            
        except Exception as e:
            logger.error(f"Failed to prepare context: {e}")
            return {
                "error": str(e),
                "task_status": "error"
            }
    
    async def _load_references_node(self, state: OutlineEditingState) -> Dict[str, Any]:
        """Load referenced context files (style, rules, characters) directly from outline frontmatter"""
        try:
            logger.info("Loading referenced context files from outline frontmatter...")
            
            from orchestrator.tools.reference_file_loader import load_referenced_files
            
            active_editor = state.get("active_editor", {})
            user_id = state.get("user_id", "system")
            frontmatter = state.get("frontmatter", {})
            
            # Outline reference configuration - load directly from outline's frontmatter (no cascading)
            reference_config = {
                "style": ["style"],
                "rules": ["rules"],
                "characters": ["characters", "character_*"]  # Support both list and individual keys
            }
            
            # Use unified loader (no cascade_config - outline loads directly)
            result = await load_referenced_files(
                active_editor=active_editor,
                user_id=user_id,
                reference_config=reference_config,
                doc_type_filter="outline",
                cascade_config=None  # No cascading for outline
            )
            
            loaded_files = result.get("loaded_files", {})
            
            # Extract content from loaded files and assess quality
            style_body = None
            style_quality = 0.0
            style_warnings = []
            
            if loaded_files.get("style") and len(loaded_files["style"]) > 0:
                style_body = loaded_files["style"][0].get("content", "")
                if style_body:
                    style_body = _strip_frontmatter_block(style_body)
                    style_quality, style_warnings = _assess_reference_quality(style_body, "style")
            
            rules_body = None
            rules_quality = 0.0
            rules_warnings = []
            
            if loaded_files.get("rules") and len(loaded_files["rules"]) > 0:
                rules_body = loaded_files["rules"][0].get("content", "")
                if rules_body:
                    rules_body = _strip_frontmatter_block(rules_body)
                    rules_quality, rules_warnings = _assess_reference_quality(rules_body, "rules")
            
            characters_bodies = []
            characters_qualities = []
            characters_warnings = []
            
            if loaded_files.get("characters"):
                for char_file in loaded_files["characters"]:
                    char_content = char_file.get("content", "")
                    if char_content:
                        char_content = _strip_frontmatter_block(char_content)
                        char_quality, char_warnings = _assess_reference_quality(char_content, "characters")
                        characters_bodies.append(char_content)
                        characters_qualities.append(char_quality)
                        characters_warnings.extend(char_warnings)
            
            # Calculate average character quality
            avg_character_quality = sum(characters_qualities) / len(characters_qualities) if characters_qualities else 0.0
            
            # Collect all warnings
            all_warnings = []
            if style_quality < 0.4 and style_body:
                all_warnings.append(f"Style guide quality is low ({style_quality:.0%})")
            all_warnings.extend(style_warnings)
            
            if rules_quality < 0.4 and rules_body:
                all_warnings.append(f"Rules quality is low ({rules_quality:.0%})")
            all_warnings.extend(rules_warnings)
            
            if avg_character_quality < 0.4 and characters_bodies:
                all_warnings.append(f"Character profiles quality is low ({avg_character_quality:.0%})")
            all_warnings.extend(characters_warnings)
            
            return {
                "rules_body": rules_body,
                "style_body": style_body,
                "characters_bodies": characters_bodies,
                "reference_quality": {
                    "style": style_quality,
                    "rules": rules_quality,
                    "characters": avg_character_quality
                },
                "reference_warnings": all_warnings
            }
            
        except Exception as e:
            logger.error(f"Failed to load references: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "rules_body": None,
                "style_body": None,
                "characters_bodies": [],
                "reference_quality": {},
                "reference_warnings": [],
                "error": str(e)
            }
    
    async def _analyze_mode_node(self, state: OutlineEditingState) -> Dict[str, Any]:
        """Analyze generation mode based on available references and quality"""
        try:
            logger.info("Analyzing outline generation mode...")
            
            rules_body = state.get("rules_body")
            style_body = state.get("style_body")
            characters_bodies = state.get("characters_bodies", [])
            reference_quality = state.get("reference_quality", {})
            reference_warnings = state.get("reference_warnings", [])
            current_request = state.get("current_request", "").lower()
            
            # Detect available references (consider quality - low quality < 0.4 treated as not present)
            style_quality = reference_quality.get("style", 0.0)
            rules_quality = reference_quality.get("rules", 0.0)
            characters_quality = reference_quality.get("characters", 0.0)
            
            has_style = style_body is not None and len(style_body.strip()) > 50 and style_quality >= 0.4
            has_rules = rules_body is not None and len(rules_body.strip()) > 50 and rules_quality >= 0.4
            has_characters = len(characters_bodies) > 0 and characters_quality >= 0.4
            
            available_references = {
                "style": has_style,
                "rules": has_rules,
                "characters": has_characters
            }
            
            # Detect creative freedom keywords
            freehand_keywords = ["freehand", "creative freedom", "ignore references", 
                                  "new direction", "fresh start", "brainstorm", "from scratch"]
            creative_freedom_requested = any(kw in current_request for kw in freehand_keywords)
            
            # Determine mode
            ref_count = sum(available_references.values())
            has_any_refs = ref_count > 0
            
            if creative_freedom_requested:
                generation_mode = "freehand"
                mode_guidance = "CREATIVE FREEDOM MODE - Full creative latitude. References available for optional consistency checks."
            elif has_any_refs and ref_count == 3:
                generation_mode = "fully_referenced"
                mode_guidance = "FULLY REFERENCED - Complete universe context available. Use as guidelines, user's request is primary."
            elif has_any_refs:
                generation_mode = "partial_references"
                refs_available = [k for k, v in available_references.items() if v]
                mode_guidance = f"PARTIAL REFERENCES - Available: {', '.join(refs_available)}. Fill gaps with creativity."
            else:
                generation_mode = "freehand"
                mode_guidance = "FREEHAND MODE - No references. Full creative freedom based on user's request."
            
            # Build reference summary
            ref_parts = []
            if has_style:
                ref_parts.append(f"Style guide (quality: {style_quality:.0%})")
            if has_rules:
                ref_parts.append(f"Universe rules (quality: {rules_quality:.0%})")
            if has_characters:
                ref_parts.append(f"{len(characters_bodies)} character profile(s) (quality: {characters_quality:.0%})")
            
            if ref_parts:
                reference_summary = f"Available: {', '.join(ref_parts)}"
            else:
                reference_summary = "No references available - freehand mode"
            
            if reference_warnings:
                reference_summary += f"\nWarnings: {'; '.join(reference_warnings[:3])}"  # Limit to 3 warnings
            
            logger.info(f"Outline generation mode: {generation_mode}")
            logger.info(f"Available references: {', '.join([k for k, v in available_references.items() if v]) or 'none'}")
            if not has_any_refs:
                logger.info("Freehand mode - no references available")
            
            return {
                "generation_mode": generation_mode,
                "available_references": available_references,
                "reference_summary": reference_summary,
                "mode_guidance": mode_guidance
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze mode: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "generation_mode": "freehand",
                "available_references": {},
                "reference_summary": "Error analyzing references - defaulting to freehand",
                "mode_guidance": "Freehand mode - proceed with creative freedom."
            }
    
    async def _analyze_outline_structure_node(self, state: OutlineEditingState) -> Dict[str, Any]:
        """Analyze existing outline structure and quality"""
        try:
            logger.info("Analyzing outline structure...")
            
            body_only = state.get("body_only", "")
            
            # Detect existing sections
            has_synopsis = bool(re.search(r"^#\s+(Overall\s+)?Synopsis\s*$", body_only, re.MULTILINE | re.IGNORECASE))
            has_notes = bool(re.search(r"^#\s+Notes\s*$", body_only, re.MULTILINE | re.IGNORECASE))
            has_characters = bool(re.search(r"^#\s+Characters?\s*$", body_only, re.MULTILINE | re.IGNORECASE))
            has_outline = bool(re.search(r"^#\s+Outline\s*$", body_only, re.MULTILINE | re.IGNORECASE))
            
            # Count chapters
            chapter_matches = list(CHAPTER_PATTERN.finditer(body_only))
            chapter_count = len(chapter_matches)
            
            # Assess completeness
            sections_present = sum([has_synopsis, has_notes, has_characters, has_outline])
            completeness_score = sections_present / 4.0 if sections_present > 0 else 0.0
            
            # Detect structural issues
            structure_warnings = []
            if chapter_count == 0 and has_outline:
                structure_warnings.append("Outline section exists but no chapters defined")
            if not has_synopsis and chapter_count > 0:
                structure_warnings.append("Chapters exist without Overall Synopsis")
            if has_characters and not re.search(r"Protagonist|Antagonist|Supporting", body_only, re.IGNORECASE):
                structure_warnings.append("Characters section missing protagonist/antagonist designation")
            if chapter_count > 0:
                # Check chapter numbering
                chapter_nums = []
                for m in chapter_matches:
                    try:
                        chapter_nums.append(int(m.group(1)))
                    except Exception:
                        pass
                if chapter_nums:
                    expected = list(range(1, len(chapter_nums) + 1))
                    if chapter_nums != expected:
                        structure_warnings.append(f"Chapter numbering is non-sequential: {chapter_nums}")
            
            # Generate structure-specific guidance with section availability
            section_list = []
            if has_synopsis:
                section_list.append("Overall Synopsis (can edit)")
            if has_notes:
                section_list.append("Notes (can edit)")
            if has_characters:
                section_list.append("Characters (can edit)")
            if has_outline:
                section_list.append("Outline")
            if chapter_count > 0:
                section_list.append(f"{chapter_count} chapter(s) (can edit)")
            
            sections_available = ", ".join(section_list) if section_list else "No sections yet"
            
            if completeness_score < 0.25:
                structure_guidance = f"Outline {completeness_score:.0%} complete ({sections_present}/4 sections). Available sections: {sections_available}. Build full structure or edit existing sections."
            elif completeness_score < 0.75:
                structure_guidance = f"Outline {completeness_score:.0%} complete. Available sections: {sections_available}. Continue developing or edit existing sections."
            elif structure_warnings:
                structure_guidance = f"Available sections: {sections_available}. Issues: {'; '.join(structure_warnings[:2])}"  # Limit to 2
            else:
                structure_guidance = f"Structure complete. Available sections: {sections_available}. You can edit any existing section."
            
            logger.info(f"Outline completeness: {completeness_score:.0%} ({sections_present}/4 sections)")
            logger.info(f"Chapter count: {chapter_count}")
            if structure_warnings:
                logger.warning(f"Structure issues: {'; '.join(structure_warnings)}")
            
            return {
                "outline_completeness": completeness_score,
                "chapter_count": chapter_count,
                "structure_warnings": structure_warnings,
                "structure_guidance": structure_guidance,
                "has_synopsis": has_synopsis,
                "has_notes": has_notes,
                "has_characters": has_characters,
                "has_outline_section": has_outline
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze outline structure: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "outline_completeness": 0.0,
                "chapter_count": 0,
                "structure_warnings": [],
                "structure_guidance": "Unable to analyze structure - proceed with caution.",
                "has_synopsis": False,
                "has_notes": False,
                "has_characters": False,
                "has_outline_section": False
            }
    
    def _build_content_routing_prompt(self, state: OutlineEditingState) -> str:
        """Build prompt for content analysis and routing"""
        current_request = state.get("current_request", "")
        body_only = state.get("body_only", "")
        has_synopsis = state.get("has_synopsis", False)
        has_notes = state.get("has_notes", False)
        has_characters = state.get("has_characters", False)
        chapter_count = state.get("chapter_count", 0)
        structure_guidance = state.get("structure_guidance", "")
        
        # Build section availability context
        sections_available = []
        if has_synopsis:
            sections_available.append("# Overall Synopsis")
        if has_notes:
            sections_available.append("# Notes")
        if has_characters:
            sections_available.append("# Characters")
        if chapter_count > 0:
            sections_available.append(f"## Chapter 1-{chapter_count}")
        
        sections_text = ", ".join(sections_available) if sections_available else "No sections exist yet"
        
        prompt = f"""**YOUR TASK**: Analyze the user's request and create a routing plan for ALL content pieces.

**USER REQUEST**:
{current_request}

**CURRENT OUTLINE STRUCTURE**:
{structure_guidance}
Available sections: {sections_text}

**CURRENT OUTLINE CONTENT** (for reference):
{body_only[:2000] if body_only else "Empty outline"}

**MULTI-PART DETECTION (CRITICAL)**:
- User may provide multiple types of information in one message
- Each piece may belong in a different outline section
- You MUST identify and route ALL pieces - missing any piece is a failure
- Examples of multi-part requests:
  * "Add character John as antagonist. Universe rule: time travel is impossible" → 2 pieces (character + rule)
  * "Chapter 2 needs action scene. Also add theme about redemption to the overall story" → 2 pieces (chapter beat + theme)
  * "Sarah should become the main protagonist instead of supporting character" → 1 piece (structural change)

**ROUTING RULES**:
- Character info → # Characters (detect if role change: supporting → protagonist)
- Universe-wide rules/themes → # Notes
- Story summary → # Overall Synopsis
- Chapter-specific events → ## Chapter N
- When in doubt: story-wide goes to Notes, specific events to Chapters

**STRUCTURAL CHANGE DETECTION (CRITICAL)**:
- "X becomes the protagonist" → Move X from supporting to protagonists (is_structural_change: true)
- "This applies to all chapters" → Route to # Notes, not a specific chapter
- "Move character X from Y to Z" → Update character categorization (is_structural_change: true)
- "X should be more prominent" → May indicate protagonist reclassification (is_structural_change: true)
- "X is now the main character" → Move to protagonists (is_structural_change: true)

**ADDITION vs REPLACEMENT vs NEW CHAPTER DETECTION (CRITICAL)**:
- **CREATING NEW CHAPTER**: User says "create", "add chapter", "chapter N", "outline for chapter N" where N doesn't exist yet
  - Check CURRENT OUTLINE CONTENT above - if the chapter heading (e.g., "## Chapter 6") does NOT exist, this is a NEW chapter
  - For NEW chapters: operation_type: "insert_after_heading"
  - **CRITICAL**: Find the LAST LINE of the LAST existing chapter in CURRENT OUTLINE CONTENT above
  - Set anchor_text to that LAST LINE (could be a bullet point, summary sentence, or chapter heading)
  - Include the new chapter heading "## Chapter N" in the content itself
  - Example: If creating Chapter 6 and last chapter is Chapter 5, find the last line of Chapter 5:
    * If last line is "- Character reaches the destination", then anchor_text: "- Character reaches the destination"
    * content: "## Chapter 6\n\n[summary paragraph]\n\n- Beat 1\n- Beat 2\n..."
  - **DO NOT** set anchor_text to the new chapter heading (it doesn't exist yet!)
  - **The LLM can figure out the exact insertion point from the context**

- **ADDING to existing chapter**: User says "add", "also", "include", "insert" for a chapter that EXISTS → Use "insert_after_heading" operation
  - Example: "Add a scene where X happens to Chapter 2" → operation_type: "insert_after_heading", anchor_text: "## Chapter 2"
  - Example: "Chapter 3 also needs a confrontation" → operation_type: "insert_after_heading", anchor_text: "## Chapter 3"
  - **PRESERVE ALL EXISTING CONTENT** - only add new bullet points, don't regenerate the chapter
  
- **GRANULAR CORRECTIONS (SPECIFIC WORD/PHRASE REPLACEMENT)**: User says "not X", "should be Y not X", "change X to Y" → Use "replace_range" with specific original_text
  - Example: "It should be a boat in chapter 2, not a canoe" → operation_type: "replace_range"
    - Find the bullet point in Chapter 2 that mentions "canoe"
    - Set original_text to the ENTIRE bullet point (or sentence) containing "canoe" with enough context (20+ words)
    - Set content to the same text but with "canoe" replaced with "boat"
    - Example original_text: "- Character escapes in a canoe across the river"
    - Example content: "- Character escapes in a boat across the river"
  - Example: "Chapter 1 should say 'betrayal' not 'mistake'" → Find text with "mistake", replace with "betrayal"
  - **CRITICAL**: Include enough context in original_text (the full bullet point or sentence) so the system can find it uniquely
  - **DO NOT replace the entire chapter** - only replace the specific bullet point or sentence containing the word/phrase
  
- **REPLACING larger chapter content**: User says "change", "replace", "instead of", "update" (without specific word) → Use "replace_range" operation
  - Example: "Change Chapter 1 to focus on X instead of Y" → operation_type: "replace_range"
  - Only use replace_range when user explicitly wants to change existing content

- **DEFAULT for new content**: If chapter exists and user is adding (not replacing), ALWAYS use "insert_after_heading"
  - Find the last bullet point in the chapter and insert after it
  - Or insert after the chapter heading if chapter is empty

**CONFLICT DETECTION**:
- Check if character already exists in different role (e.g., in Supporting but should be in Protagonists)
- Check if rule contradicts existing rule
- Check if content overlaps with existing sections
- Flag conflicts for replacement operations (operation_type: "replace_range")

**OUTPUT FORMAT**: Return ONLY valid JSON:
{{
  "content_pieces": [
    {{
      "piece_type": "character_update|rule|synopsis|chapter_beat|structural_change",
      "target_section": "# Characters|# Notes|# Overall Synopsis|## Chapter N",
      "operation_type": "replace_range|insert_after_heading|delete_range",
      "content": "The actual content to insert/update (markdown formatted)",
      "anchor_text": "Exact heading or text to anchor to (e.g., '## Chapter 2' or '# Characters')",
      "original_text": "For replace_range operations: the EXACT text from CURRENT OUTLINE to replace (20+ words for uniqueness). For granular corrections, include the full bullet point/sentence containing the word/phrase. Empty string for insert operations.",
      "reasoning": "Why this routing decision was made",
      "is_structural_change": false,
      "conflicts_with": "Description of conflicting content if any, or empty string"
    }}
  ],
  "completeness_check": "Verification that all parts of request are covered - list each piece identified"
}}

**CRITICAL INSTRUCTIONS**:
1. Identify EVERY distinct piece of information in the user's request
2. Route each piece to the appropriate section based on routing rules
3. **DETECT ADDITION vs REPLACEMENT vs GRANULAR CORRECTION**:
   - Keywords for ADDITION: "add", "also", "include", "insert", "and", "plus" → Use "insert_after_heading"
   - Keywords for GRANULAR CORRECTION: "not X", "should be Y not X", "change X to Y", "instead of X" (with specific words) → Use "replace_range"
     - Find the specific bullet point or sentence containing the word/phrase to replace
     - Include the FULL bullet point (or sentence) in original_text (20+ words for uniqueness)
     - Replace only the specific word/phrase in the content
   - Keywords for REPLACEMENT: "change", "replace", "instead of", "update", "revise" (without specific words) → Use "replace_range"
   - When in doubt for chapters: If chapter exists and user is adding content, use "insert_after_heading"
4. Detect structural changes (character role changes, scope changes)
5. Flag conflicts with existing content
6. Generate content for each piece (markdown formatted)
7. **For granular corrections**: 
   - Read the CURRENT OUTLINE CONTENT above to find the exact text containing the word/phrase
   - Set original_text to the full bullet point or sentence (not just the word)
   - Set content to the same text with only the specific word/phrase changed
   - Example: If user says "boat not canoe" and outline has "- Character escapes in a canoe across the river"
     - original_text: "- Character escapes in a canoe across the river"
     - content: "- Character escapes in a boat across the river"
8. **For NEW chapters** (chapter doesn't exist yet):
   - Check CURRENT OUTLINE CONTENT to verify the chapter heading doesn't exist
   - Find the LAST LINE of the LAST existing chapter in CURRENT OUTLINE CONTENT
   - Set anchor_text to that LAST LINE (the actual last line of text - could be a bullet point, summary sentence, etc.)
   - Include the new chapter heading "## Chapter N" in the content itself
   - Example: If last line of Chapter 5 is "- Character reaches the destination", then anchor_text: "- Character reaches the destination", content: "## Chapter 6\n\n[summary]\n\n- Beat 1..."
   - **The LLM can figure out the exact insertion point from the context - use the actual last line, not a heading**
9. **For chapter additions** (chapter exists): Set anchor_text to the chapter heading (e.g., "## Chapter 2") and note that content should be inserted after the last existing bullet point
10. Verify completeness - ensure ALL parts of the request are covered

Return ONLY the JSON object, no markdown, no code blocks."""
        
        return prompt
    
    async def _analyze_and_route_request_node(self, state: OutlineEditingState) -> Dict[str, Any]:
        """Analyze user request and create routing plan for all content pieces"""
        try:
            logger.info("Analyzing and routing user request...")
            
            current_request = state.get("current_request", "")
            if not current_request:
                logger.warning("No current request found - skipping routing analysis")
                return {
                    "routing_plan": {
                        "content_pieces": [],
                        "completeness_check": "No user request provided"
                    }
                }
            
            # Build routing prompt
            routing_prompt = self._build_content_routing_prompt(state)
            
            # Include conversation history for context (standardized 6-message look-back)
            messages_list = state.get("messages", [])
            conversation_history = self._format_conversation_history_for_prompt(messages_list, look_back_limit=6, max_message_length=300)
            
            # Call LLM with structured output
            llm = self._get_llm(temperature=0.2, state=state)
            
            messages = [
                SystemMessage(content="You are an outline content routing expert. Analyze user requests and route content pieces to appropriate outline sections."),
                HumanMessage(content=conversation_history + routing_prompt)
            ]
            
            # Parse JSON response (structured output requires Pydantic model, so we use JSON parsing)
            response = await llm.ainvoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)
            content = _unwrap_json_response(content)
            try:
                routing_plan = json.loads(content)
            except Exception as parse_error:
                logger.error(f"Failed to parse routing plan: {parse_error}")
                return {
                    "routing_plan": {
                        "content_pieces": [],
                        "completeness_check": f"Error parsing routing plan: {str(parse_error)}"
                    },
                    "error": str(parse_error)
                }
            
            # Validate routing plan structure
            if not isinstance(routing_plan, dict):
                routing_plan = {"content_pieces": [], "completeness_check": "Invalid routing plan format"}
            if "content_pieces" not in routing_plan:
                routing_plan["content_pieces"] = []
            if "completeness_check" not in routing_plan:
                routing_plan["completeness_check"] = "Completeness check not provided"
            
            piece_count = len(routing_plan.get("content_pieces", []))
            logger.info(f"Routing plan created: {piece_count} content piece(s) identified")
            
            if piece_count > 0:
                for i, piece in enumerate(routing_plan["content_pieces"]):
                    piece_type = piece.get("piece_type", "unknown")
                    target_section = piece.get("target_section", "unknown")
                    is_structural = piece.get("is_structural_change", False)
                    logger.info(f"  Piece {i+1}: {piece_type} → {target_section} (structural_change: {is_structural})")
            
            return {
                "routing_plan": routing_plan
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze and route request: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "routing_plan": {
                    "content_pieces": [],
                    "completeness_check": f"Error during routing analysis: {str(e)}"
                },
                "error": str(e)
            }
    
    async def _generate_edit_plan_node(self, state: OutlineEditingState) -> Dict[str, Any]:
        """Generate edit plan using LLM"""
        try:
            logger.info("Generating outline edit plan...")
            
            outline = state.get("outline", "")
            filename = state.get("filename", "outline.md")
            body_only = state.get("body_only", "")
            current_request = state.get("current_request", "")
            clarification_context = state.get("clarification_context", "")
            
            rules_body = state.get("rules_body")
            style_body = state.get("style_body")
            characters_bodies = state.get("characters_bodies", [])
            
            para_start = state.get("para_start", 0)
            selection_start = state.get("selection_start", -1)
            selection_end = state.get("selection_end", -1)
            
            # Get mode and structure context
            mode_guidance = state.get("mode_guidance", "")
            structure_guidance = state.get("structure_guidance", "")
            reference_summary = state.get("reference_summary", "")
            
            # Get routing plan from analysis node
            routing_plan = state.get("routing_plan", {})
            content_pieces = routing_plan.get("content_pieces", [])
            completeness_check = routing_plan.get("completeness_check", "")
            
            # Build dynamic system prompt
            system_prompt = self._build_system_prompt(state)
            
            # Build context message - lean and focused
            context_parts = []
            
            # Include conversation history for context (standardized 6-message look-back)
            messages = state.get("messages", [])
            conversation_history = self._format_conversation_history_for_prompt(messages, look_back_limit=6)
            if conversation_history:
                context_parts.append(conversation_history)
            
            # User request first (brief but clear)
            if current_request:
                context_parts.append("=== CURRENT USER REQUEST ===\n")
                context_parts.append(f"{current_request}\n\n")
            else:
                logger.error("current_request is empty - user's request will not be sent to LLM")
            
            # Routing plan (CRITICAL - use this to generate operations)
            if content_pieces:
                context_parts.append("=== ROUTING PLAN (CRITICAL) ===\n")
                context_parts.append(f"Completeness check: {completeness_check}\n\n")
                context_parts.append("You MUST generate operations for ALL content pieces listed below:\n\n")
                for i, piece in enumerate(content_pieces, 1):
                    piece_type = piece.get("piece_type", "unknown")
                    target_section = piece.get("target_section", "unknown")
                    operation_type = piece.get("operation_type", "replace_range")
                    anchor_text = piece.get("anchor_text", "")
                    is_structural = piece.get("is_structural_change", False)
                    conflicts = piece.get("conflicts_with", "")
                    reasoning = piece.get("reasoning", "")
                    
                    context_parts.append(f"**Content Piece {i}**:\n")
                    context_parts.append(f"- Type: {piece_type}\n")
                    context_parts.append(f"- Target Section: {target_section}\n")
                    context_parts.append(f"- Operation Type: {operation_type}\n")
                    context_parts.append(f"- Anchor Text: {anchor_text}\n")
                    original_text = piece.get("original_text", "")
                    if original_text:
                        context_parts.append(f"- Original Text (to replace): {original_text[:200]}...\n")
                    if is_structural:
                        context_parts.append(f"- ⚠️ STRUCTURAL CHANGE: This is a structural change (e.g., character role change)\n")
                    if conflicts:
                        context_parts.append(f"- ⚠️ CONFLICT: {conflicts}\n")
                    context_parts.append(f"- Reasoning: {reasoning}\n")
                    context_parts.append(f"- Content to insert/update:\n{piece.get('content', '')}\n\n")
                context_parts.append("**IMPORTANT**: Generate one operation for EACH content piece above. Do not skip any pieces.\n\n")
            else:
                logger.warning("No content pieces in routing plan - proceeding without routing guidance")
            
            # Current outline state
            context_parts.append("=== CURRENT OUTLINE ===\n")
            context_parts.append(f"File: {filename}\n")
            if mode_guidance:
                context_parts.append(f"Mode: {mode_guidance}\n")
            if structure_guidance:
                context_parts.append(f"Status: {structure_guidance}\n")
            context_parts.append("\n" + body_only + "\n\n")
            
            # References (if present, keep concise)
            if rules_body:
                context_parts.append("=== UNIVERSE RULES ===\n")
                context_parts.append(f"{rules_body}\n\n")
            
            if style_body:
                context_parts.append("=== STYLE GUIDE ===\n")
                context_parts.append(f"{style_body}\n\n")
            
            if characters_bodies:
                context_parts.append("=== CHARACTER PROFILES ===\n")
                context_parts.append("".join([f"{b}\n---\n" for b in characters_bodies]))
                context_parts.append("\n")
            
            if clarification_context:
                context_parts.append(clarification_context)
            
            # Final instruction
            if content_pieces:
                context_parts.append(f"Generate ManuscriptEdit JSON with operations for ALL {len(content_pieces)} content piece(s) from the routing plan above.\n")
                context_parts.append("Use the routing plan's target_section, anchor_text, operation_type, and original_text (if provided) for each operation.\n")
                context_parts.append("**CRITICAL**: If the routing plan provides original_text, you MUST use that EXACT text in the operation's original_text field.\n")
                context_parts.append("\n**CRITICAL FOR CHAPTER OPERATIONS**:\n")
                context_parts.append("**FOR NEW CHAPTERS** (chapter doesn't exist in CURRENT OUTLINE):\n")
                context_parts.append("- Check CURRENT OUTLINE above - if the chapter heading (e.g., '## Chapter 6') does NOT exist, this is a NEW chapter\n")
                context_parts.append("- For NEW chapters: Find the LAST LINE of the LAST existing chapter in CURRENT OUTLINE above\n")
                context_parts.append("- Set anchor_text to that LAST LINE (the actual last line of text - could be a bullet point, summary sentence, etc.)\n")
                context_parts.append("- **INCLUDE the new chapter heading in the content**: content should start with '## Chapter N' followed by summary and beats\n")
                context_parts.append("- Example: If last line of Chapter 5 is '- Character reaches the destination', then:\n")
                context_parts.append("  anchor_text: '- Character reaches the destination'\n")
                context_parts.append("  content: '## Chapter 6\\n\\n[summary paragraph]\\n\\n- Beat 1\\n- Beat 2...'\n")
                context_parts.append("- **DO NOT** set anchor_text to the new chapter heading (it doesn't exist yet!)\n")
                context_parts.append("- **The LLM can figure out the exact insertion point from the context - use the actual last line**\n\n")
                context_parts.append("**FOR EXISTING CHAPTERS** (chapter exists in CURRENT OUTLINE):\n")
                context_parts.append("- If operation_type is 'insert_after_heading' for an existing chapter, find the LAST bullet point in that chapter\n")
                context_parts.append("- Insert new content AFTER the last existing bullet point (preserves all existing content)\n")
                context_parts.append("- If chapter only has summary paragraph, insert bullets after the summary\n")
                context_parts.append("- **DO NOT replace the entire chapter** when adding - only insert new content\n")
                context_parts.append("\n**FOR GRANULAR CORRECTIONS (replace_range with specific word changes)**:\n")
                context_parts.append("- Read the CURRENT OUTLINE above to find the exact bullet point containing the word/phrase\n")
                context_parts.append("- Set original_text to the FULL bullet point or sentence (20+ words for uniqueness)\n")
                context_parts.append("- Set content to the same text with ONLY the specific word/phrase changed\n")
                context_parts.append("- Example: If routing plan says to change 'canoe' to 'boat' in Chapter 2:\n")
                context_parts.append("  * Find: '- Character escapes in a canoe across the river'\n")
                context_parts.append("  * original_text: '- Character escapes in a canoe across the river'\n")
                context_parts.append("  * content: '- Character escapes in a boat across the river'\n")
                context_parts.append("- **DO NOT replace the entire chapter** - only replace the specific bullet point\n")
            else:
                context_parts.append("Generate ManuscriptEdit JSON for the user's request above.\n")
            if current_request:
                context_parts.append(f'User asked for: "{current_request[:100]}..."\n')
            
            messages = [
                SystemMessage(content=system_prompt),
                SystemMessage(content=f"Current Date/Time: {datetime.now().isoformat()}"),
                HumanMessage(content="".join(context_parts))
            ]
            
            # Call LLM - pass state to access user's model selection from metadata
            llm = self._get_llm(temperature=0.3, state=state)
            start_time = datetime.now()
            response = await llm.ainvoke(messages)
            
            content = response.content if hasattr(response, 'content') else str(response)
            content = _unwrap_json_response(content)
            
            # Log the raw LLM response for debugging
            logger.info(f"LLM generated edit plan (first 500 chars): {content[:500]}")
            
            # Try parsing as clarification request first (RARE)
            clarification_request = None
            structured_edit = None
            
            try:
                raw = json.loads(content)
                if isinstance(raw, dict) and raw.get("clarification_needed") is True:
                    confidence = raw.get("confidence_without_clarification", 0.5)
                    if confidence < 0.4:
                        clarification_request = raw
                        logger.warning(f"Requesting clarification (confidence={confidence:.2f}) - RARE PATH")
                    else:
                        logger.info(f"Agent wanted clarification but confidence={confidence:.2f} is too high - expecting content instead")
            except Exception:
                pass
            
            # If it's a genuine clarification request, return early
            if clarification_request:
                processing_time = (datetime.now() - start_time).total_seconds()
                questions = clarification_request.get("questions", [])
                questions_formatted = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
                response_text = (
                    f"**⚠️ Critical Ambiguity Detected**\n\n"
                    f"{clarification_request.get('context', '')}\n\n"
                    f"I cannot proceed without clarification on:\n\n"
                    f"{questions_formatted}\n\n"
                    f"(Confidence without clarification: {clarification_request.get('confidence_without_clarification', 0.3):.0%})"
                )
                if clarification_request.get("suggested_direction"):
                    response_text += f"\n\n**Suggestion:** {clarification_request.get('suggested_direction')}\n"
                
                # Store clarification request in shared_memory for next turn
                shared_memory = state.get("shared_memory", {}) or {}
                shared_memory_out = shared_memory.copy()
                shared_memory_out["pending_outline_clarification"] = clarification_request
                
                return {
                    "clarification_request": clarification_request,
                    "llm_response": content,
                    "response": {
                        "response": response_text,
                        "requires_user_input": True,
                        "task_status": "incomplete",
                        "agent_type": "outline_editing_agent",
                        "shared_memory": shared_memory_out
                    },
                    "task_status": "incomplete"
                }
            
            # Otherwise, parse as ManuscriptEdit
            try:
                raw = json.loads(content)
                if isinstance(raw, dict) and isinstance(raw.get("operations"), list):
                    raw.setdefault("target_filename", filename)
                    raw.setdefault("scope", "paragraph")
                    raw.setdefault("summary", "Planned outline edit generated from context.")
                    raw.setdefault("safety", "medium")
                    
                    # Process operations to preserve anchor fields
                    ops = []
                    for op in raw["operations"]:
                        if not isinstance(op, dict):
                            continue
                        op_type = op.get("op_type")
                        if op_type not in ("replace_range", "delete_range", "insert_after_heading"):
                            op_type = "replace_range"
                        
                        op_text = op.get("text", "")
                        # Log operation text for debugging
                        if op_text:
                            logger.info(f"Operation {len(ops)} text length: {len(op_text)} chars, preview: {op_text[:100]}")
                        else:
                            logger.warning(f"Operation {len(ops)} has EMPTY text field!")
                        
                        ops.append({
                            "op_type": op_type,
                            "start": op.get("start", para_start),
                            "end": op.get("end", para_start),
                            "text": op_text,
                            "original_text": op.get("original_text"),
                            "anchor_text": op.get("anchor_text"),
                            "left_context": op.get("left_context"),
                            "right_context": op.get("right_context"),
                            "occurrence_index": op.get("occurrence_index", 0)
                        })
                    raw["operations"] = ops
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
                    "error": "Failed to produce a valid Outline edit plan. Ensure ONLY raw JSON ManuscriptEdit with operations is returned.",
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
    
    async def _resolve_operations_node(self, state: OutlineEditingState) -> Dict[str, Any]:
        """Resolve editor operations with progressive search"""
        try:
            logger.info("Resolving editor operations...")
            
            outline = state.get("outline", "")
            structured_edit = state.get("structured_edit")
            selection_start = state.get("selection_start", -1)
            selection_end = state.get("selection_end", -1)
            para_start = state.get("para_start", 0)
            para_end = state.get("para_end", 0)
            
            if not structured_edit or not isinstance(structured_edit.get("operations"), list):
                return {
                    "editor_operations": [],
                    "error": "No operations to resolve",
                    "task_status": "error"
                }
            
            fm_end_idx = _frontmatter_end_index(outline)
            selection = {"start": selection_start, "end": selection_end} if selection_start >= 0 and selection_end >= 0 else None
            
            editor_operations = []
            operations = structured_edit.get("operations", [])
            
            for op in operations:
                # Sanitize op text
                op_text = op.get("text", "")
                if isinstance(op_text, str):
                    op_text = _strip_frontmatter_block(op_text)
                    # Strip placeholder filler lines
                    op_text = re.sub(r"^\s*-\s*\[To be.*?\]|^\s*\[To be.*?\]|^\s*TBD\s*$", "", op_text, flags=re.IGNORECASE | re.MULTILINE)
                    op_text = re.sub(r"\n{3,}", "\n\n", op_text)
                
                # Resolve operation
                try:
                    resolved_start, resolved_end, resolved_text, resolved_confidence = _resolve_operation_simple(
                        outline,
                        op,
                        selection=selection,
                        frontmatter_end=fm_end_idx
                    )
                    
                    logger.info(f"Resolved {op.get('op_type')} [{resolved_start}:{resolved_end}] confidence={resolved_confidence:.2f}")
                    
                    # Calculate pre_hash
                    pre_slice = outline[resolved_start:resolved_end]
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
                    fallback_start = para_start
                    fallback_end = para_end
                    
                    pre_slice = outline[fallback_start:fallback_end]
                    resolved_op = {
                        "op_type": op.get("op_type", "replace_range"),
                        "start": fallback_start,
                        "end": fallback_end,
                        "text": op_text,
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
    
    async def _format_response_node(self, state: OutlineEditingState) -> Dict[str, Any]:
        """Format final response with editor operations"""
        try:
            logger.info("Formatting response...")
            
            structured_edit = state.get("structured_edit", {})
            editor_operations = state.get("editor_operations", [])
            clarification_request = state.get("clarification_request")
            task_status = state.get("task_status", "complete")
            
            # If we have a clarification request, it was already formatted in generate_edit_plan
            if clarification_request:
                response = state.get("response", {})
                return {
                    "response": response,
                    "task_status": "incomplete"
                }
            
            if task_status == "error":
                error_msg = state.get("error", "Unknown error")
                return {
                    "response": {
                        "response": f"Outline editing failed: {error_msg}",
                        "task_status": "error",
                        "agent_type": "outline_editing_agent"
                    },
                    "task_status": "error"
                }
            
            # Build preview
            generated_preview = "\n\n".join([
                op.get("text", "").strip()
                for op in editor_operations
                if op.get("text", "").strip()
            ]).strip()
            
            response_text = generated_preview if generated_preview else (structured_edit.get("summary", "Edit plan ready."))
            
            # Build response with editor operations
            response = {
                "response": response_text,
                "task_status": task_status,
                "agent_type": "outline_editing_agent",
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
            
            # Clear any pending clarification since we're completing successfully
            shared_memory = state.get("shared_memory", {}) or {}
            shared_memory_out = shared_memory.copy()
            shared_memory_out.pop("pending_outline_clarification", None)
            response["shared_memory"] = shared_memory_out
            
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
        """Process outline editing query using LangGraph workflow"""
        try:
            # Ensure query is a string
            if not isinstance(query, str):
                query = str(query) if query else ""
            
            logger.info(f"Outline editing agent processing: {query[:100] if query else 'empty'}...")
            
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
            initial_state: OutlineEditingState = {
                "query": query,
                "user_id": user_id,
                "metadata": metadata,
                "messages": conversation_messages,
                "shared_memory": shared_memory_merged,
                "active_editor": {},
                "outline": "",
                "filename": "outline.md",
                "frontmatter": {},
                "cursor_offset": -1,
                "selection_start": -1,
                "selection_end": -1,
                "body_only": "",
                "para_start": 0,
                "para_end": 0,
                "rules_body": None,
                "style_body": None,
                "characters_bodies": [],
                "clarification_context": "",
                "current_request": "",
                "system_prompt": "",
                "llm_response": "",
                "structured_edit": None,
                "clarification_request": None,
                "editor_operations": [],
                "response": {},
                "task_status": "",
                "error": "",
                # NEW: Mode tracking
                "generation_mode": "freehand",
                "available_references": {},
                "reference_summary": "",
                "mode_guidance": "",
                "reference_quality": {},
                "reference_warnings": [],
                # NEW: Structure analysis
                "outline_completeness": 0.0,
                "chapter_count": 0,
                "structure_warnings": [],
                "structure_guidance": "",
                "has_synopsis": False,
                "has_notes": False,
                "has_characters": False,
                "has_outline_section": False,
                # NEW: Content routing plan
                "routing_plan": None
            }
            
            # Run LangGraph workflow with checkpointing (workflow and config already created above)
            result_state = await workflow.ainvoke(initial_state, config=config)
            
            # Extract final response
            response = result_state.get("response", {})
            task_status = result_state.get("task_status", "complete")
            
            if task_status == "error":
                error_msg = result_state.get("error", "Unknown error")
                logger.error(f"Outline editing agent failed: {error_msg}")
                return {
                    "response": f"Outline editing failed: {error_msg}",
                    "task_status": "error",
                    "agent_results": {}
                }
            
            # Build result dict matching fiction_editing_agent pattern
            result = {
                "response": response.get("response", "Outline editing complete"),
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
            if response.get("shared_memory"):
                result["shared_memory"] = response["shared_memory"]
            
            logger.info(f"Outline editing agent completed: {task_status}")
            return result
            
        except Exception as e:
            logger.error(f"Outline editing agent failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "response": f"Outline editing failed: {str(e)}",
                "task_status": "error",
                "agent_results": {}
            }


def get_outline_editing_agent() -> OutlineEditingAgent:
    """Get singleton outline editing agent instance"""
    global _outline_editing_agent
    if _outline_editing_agent is None:
        _outline_editing_agent = OutlineEditingAgent()
    return _outline_editing_agent


_outline_editing_agent: Optional[OutlineEditingAgent] = None

