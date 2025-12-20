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
from pydantic import ValidationError

from .base_agent import BaseAgent, OpenRouterError
from orchestrator.models.editor_models import EditorOperation, ManuscriptEdit
from orchestrator.utils.editor_operation_resolver import resolve_editor_operation

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


def find_last_line_of_last_chapter(outline: str) -> Optional[str]:
    """Find the last non-empty line of the last chapter in the outline.
    
    Returns the actual last line of text (could be a bullet point, summary sentence, etc.)
    that can be used as anchor_text for inserting a new chapter.
    Preserves original whitespace for exact matching in the resolver.
    """
    if not outline:
        return None
    
    chapter_ranges = find_chapter_ranges(outline)
    if not chapter_ranges:
        # No chapters found - return last non-empty line of entire document
        lines = outline.rstrip().split('\n')
        for line in reversed(lines):
            if line.strip():
                # Preserve original line (with original whitespace) for exact matching
                return line.rstrip()
        return None
    
    # Get the last chapter
    last_chapter = chapter_ranges[-1]
    chapter_content = outline[last_chapter.start:last_chapter.end]
    
    # Find the last non-empty line in this chapter
    # Split preserving line endings - we'll reconstruct with original content
    lines = chapter_content.split('\n')
    for line in reversed(lines):
        stripped = line.strip()
        # Skip the chapter heading itself and empty lines
        if stripped and not stripped.startswith('## Chapter'):
            # Return the line with original whitespace preserved (just strip trailing newline)
            return line.rstrip()
    
    # If no content found (empty chapter), return the chapter heading
    return last_chapter.heading_text.rstrip()


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


# Removed: _resolve_operation_simple - now using centralized resolver from orchestrator.utils.editor_operation_resolver


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
    # NEW: Request type detection
    request_type: str  # "question" | "edit_request" | "unknown"


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
        
        # Add nodes - simplified workflow
        workflow.add_node("prepare_context", self._prepare_context_node)
        workflow.add_node("load_references", self._load_references_node)
        workflow.add_node("generate_edit_plan", self._generate_edit_plan_node)
        workflow.add_node("resolve_operations", self._resolve_operations_node)
        workflow.add_node("format_response", self._format_response_node)
        
        # Entry point
        workflow.set_entry_point("prepare_context")
        
        # Simplified flow: prepare_context -> load_references -> generate_edit_plan -> resolve_operations -> format_response
        workflow.add_edge("prepare_context", "load_references")
        workflow.add_edge("load_references", "generate_edit_plan")
        workflow.add_edge("generate_edit_plan", "resolve_operations")
        workflow.add_edge("resolve_operations", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow.compile(checkpointer=checkpointer)
    
    def _build_system_prompt(self, state: Optional[OutlineEditingState] = None) -> str:
        """Build system prompt for outline editing"""
        return (
            "You are an outline editor. Generate operations to edit story outlines.\n\n"
            "**CRITICAL: WORK WITH AVAILABLE INFORMATION FIRST**\n"
            "Always start by working with what you know from the request, existing outline content, and references:\n"
            "- Make edits based on available information - don't wait for clarification\n"
            "- Use context from rules, style guide, and character profiles to inform your work\n"
            "- Add or revise content based on reasonable inferences from the request\n"
            "- **FOR EMPTY FILES**: When the outline is empty (only frontmatter), ASK QUESTIONS FIRST before creating content\n"
            "  * Don't create the entire outline structure at once\n"
            "  * Ask about story genre, main characters, key plot points, or chapter count\n"
            "  * Build incrementally based on user responses\n"
            "- Only proceed without questions when you have enough information to make meaningful edits\n"
            "\n"
            "**WHEN TO ASK QUESTIONS**:\n"
            "- **ALWAYS for empty files**: When outline is empty, ask questions about story basics before creating content\n"
            "- When the request is vague and you cannot make reasonable edits (e.g., 'improve outline' with no existing content)\n"
            "- When there's a critical plot conflict that requires user decision (e.g., existing beat directly contradicts new request)\n"
            "- When user requests a large amount of content (e.g., 'create the whole outline') - break it into steps and ask about priorities\n"
            "- When asking, you can provide operations for what you CAN do, then ask questions in the summary about what you need\n"
            "\n"
            "**HOW TO ASK QUESTIONS**: Include operations for work you CAN do, then add questions/suggestions in the summary field.\n"
            "For empty files, it's acceptable to return a clarification request with questions instead of operations.\n"
            "DO NOT return empty operations array for edit requests - always provide edits OR ask questions.\n\n"
            "OUTLINE STRUCTURE:\n"
            "# Overall Synopsis (high-level story summary - major elements only)\n"
            "# Notes (rules, themes, worldbuilding)\n"
            "# Characters (BRIEF list only: names and roles like 'Protagonist: John', 'Antagonist: Sarah')\n"
            "  **CRITICAL**: Character DETAILS belong in character profile files, NOT in the outline!\n"
            "  The outline should only have brief character references (name + role), not full profiles.\n"
            "  Do NOT copy character descriptions, backstories, or traits into the outline.\n"
            "## Chapter N (with summary paragraph + bullet point beats)\n\n"
            "BEAT FORMATTING:\n"
            "- Every beat MUST start with '- ' (dash space)\n"
            "- Beats are specific plot events/actions\n"
            "- Use as many beats as needed\n"
            "- Each chapter needs: (1) 3-5 sentence summary paragraph AND (2) detailed beats\n\n"
            "OPERATIONS:\n\n"
            "**1. replace_range - CHANGING EXISTING TEXT**:\n"
            "- Use this when modifying, rewriting, or continuing text that ALREADY EXISTS in the outline\n"
            "- You MUST provide 'original_text' with EXACT text from the current outline\n"
            "- Copy 20+ words of EXACT text that you want to replace\n"
            "- If your new content starts with the same text as existing content (overlap), use replace_range\n"
            "- **CRITICAL FOR \"CONTINUE\"**: When adding beats to a chapter, if you include existing beats for context,\n"
            "  use replace_range with original_text set to those existing beats you're including\n"
            "- Example: Chapter has beats A, B, C. You want to add D, E. If you generate \"B, C, D, E\",\n"
            "  then original_text=\"- Beat B\\n- Beat C\" and text=\"- Beat B\\n- Beat C\\n- Beat D\\n- Beat E\"\n"
            "- This marks B and C as strikethrough (existing) and D and E as green (new)\n"
            "- If you can't find exact text in the outline, DO NOT use replace_range\n\n"
            "**2. insert_after_heading - ADDING COMPLETELY NEW TEXT**:\n"
            "- Use this when adding NEW content that DOES NOT overlap with existing text\n"
            "- **CRITICAL FOR EMPTY FILES**: If the outline file is empty (only frontmatter), DO NOT provide 'anchor_text'\n"
            "  * Empty file = no content exists yet, so there's nothing to anchor to\n"
            "  * Omit the 'anchor_text' field entirely - the system will insert after frontmatter automatically\n"
            "  * Example for empty file: {\"op_type\": \"insert_after_heading\", \"text\": \"## Chapter 1\\n\\n[content]\"}\n"
            "- **For files with content**: You MUST provide 'anchor_text' with EXACT text from outline to insert after\n"
            "- For new chapters: anchor_text = last line of previous chapter\n"
            "- For truly new beats (no overlap): anchor_text = last existing beat or chapter heading\n"
            "- **ONLY use this if your generated content is 100% new** (no existing beats included)\n"
            "- ⚠️ CRITICAL WARNING: When adding beats to existing chapter, if chapter already has beats:\n"
            "  * anchor_text should be the LAST beat of that chapter, NOT the chapter heading\n"
            "  * Using chapter heading as anchor when beats exist will INSERT BETWEEN heading and first beat!\n"
            "  * This splits the chapter and breaks structure - NEVER do this!\n"
            "  * Example WRONG: '## Chapter 2\\n[INSERT HERE]\\n- Existing beat 1' ← splits chapter!\n"
            "  * Example CORRECT: anchor_text='- Existing beat 3' (last beat) → inserts after all beats\n"
            "- ⚠️ If chapter has NO beats yet (empty), THEN you can use chapter heading as anchor\n"
            "- ⚠️ **NEVER use anchor_text that references context headers** - Text like '=== CURRENT OUTLINE ===' or 'File: filename.md' does NOT exist in the file\n\n"
            "**3. delete_range - REMOVING CONTENT**:\n"
            "- Provide original_text (exact text to remove)\n\n"
            "**DECISION RULE**:\n"
            "**STEP 1: Check what exists in the target chapter/section**\n"
            "- Does the chapter already have beats? How many?\n"
            "- Is the section empty or does it have content?\n"
            "\n"
            "**STEP 2: Choose operation and anchor:**\n"
            "- Generated content overlaps with existing text? → Use 'replace_range' with 'original_text'\n"
            "- Generated content is 100% new (no overlap)? → Use 'insert_after_heading' with 'anchor_text'\n"
            "- Adding beats to chapter with existing beats? → anchor_text = LAST existing beat, NOT chapter heading\n"
            "- Adding beats to empty chapter? → anchor_text = chapter heading (no beats to split)\n"
            "- When in doubt: if you copy ANY existing beats, use 'replace_range'\n\n"
            "OUTPUT FORMAT - ManuscriptEdit JSON:\n"
            "{\n"
            '  "type": "ManuscriptEdit",\n'
            '  "target_filename": "filename.md",\n'
            '  "scope": "paragraph|chapter|multi_chapter",\n'
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
            "- Always provide operations based on available information - work with what you know\n"
            "- If you need clarification, include it in the summary field AFTER describing the work you've done\n"
            "- Never return empty operations array unless the request is completely impossible to fulfill\n"
        )
    
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
                    },
                    # ✅ CRITICAL: Preserve state even on error
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", "")
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
                "current_request": current_request.strip(),
                # ✅ CRITICAL: Preserve state for subsequent nodes
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.error(f"Failed to prepare context: {e}")
            return {
                "error": str(e),
                "task_status": "error",
                # ✅ CRITICAL: Preserve state even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
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
                "reference_warnings": all_warnings,
                # ✅ CRITICAL: Preserve state for subsequent nodes
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
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
                "error": str(e),
                # ✅ CRITICAL: Preserve state even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
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
    
    def _build_content_routing_prompt_removed(self, state: OutlineEditingState) -> str:
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
        
        # For routing, show both beginning AND end of outline to ensure last chapter is visible
        outline_preview = ""
        if body_only:
            if len(body_only) <= 3000:
                # Small outline - show all
                outline_preview = body_only
            else:
                # Large outline - show first 1500 and last 1500 chars
                first_part = body_only[:1500]
                last_part = body_only[-1500:]
                outline_preview = f"{first_part}\n\n[... middle content omitted ...]\n\n{last_part}"
        
        prompt = f"""**YOUR TASK**: Analyze the user's request and create a routing plan for ALL content pieces.

**USER REQUEST**:
{current_request}

**CURRENT OUTLINE STRUCTURE**:
{structure_guidance}
Available sections: {sections_text}
Chapter count: {chapter_count}

**CURRENT OUTLINE CONTENT** (showing beginning and end for context):
{outline_preview if outline_preview else "Empty outline"}

**MULTI-PART DETECTION (CRITICAL)**:
- User may provide multiple types of information in one message
- Each piece may belong in a different outline section
- You MUST identify and route ALL pieces - missing any piece is a failure
- Examples of multi-part requests:
  * "Add character John as antagonist. Universe rule: time travel is impossible" → 2 pieces (character + rule)
  * "Chapter 2 needs action scene. Also add theme about redemption to the overall story" → 2 pieces (chapter beat + theme)
  * "Sarah should become the main protagonist instead of supporting character" → 1 piece (structural change)

**ROUTING RULES**:
- Character info → # Characters (BRIEF references only: name + role like 'Protagonist: John')
  **CRITICAL**: Character DETAILS (descriptions, backstories, traits) belong in character profile files, NOT in the outline!
  The outline # Characters section should only list names and roles, not full character information.
- Universe-wide rules/themes → # Notes
- High-level story summary (MAJOR elements only) → # Overall Synopsis
- Chapter-specific events → ## Chapter N
- When in doubt: story-wide goes to Notes, specific events to Chapters

**SYNOPSIS MANAGEMENT (CRITICAL)**:
- The Overall Synopsis should stay focused on MAJOR story elements only
- As the outline grows with more chapters, the synopsis should be periodically PRUNED to remove minor details
- Only include plot points that are essential to understanding the core story arc
- If a detail is covered in chapter beats but not crucial to the big picture, it should NOT be in the synopsis
- Think of the synopsis as a "back of the book" description - what would make someone want to read the story?

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
  - **CRITICAL - CHAPTER ORDERING**: Chapters are numbered sequentially (1, 2, 3, 4...). Chapter N must come AFTER Chapter N-1.
  - **CRITICAL**: Find the LAST LINE of the LAST existing chapter in CURRENT OUTLINE CONTENT above
    * Look at the END of the outline content shown above - the last chapter is the one with the highest chapter number
    * If you see "## Chapter 3" at the end, that's the last chapter - use its last line as anchor
    * If you see "## Chapter 2" at the end, that's the last chapter - use its last line as anchor
    * **DO NOT** use a line from an earlier chapter (e.g., Chapter 2) when Chapter 3 exists - always use the LAST chapter
  - Set anchor_text to that LAST LINE (could be a bullet point, summary sentence, or chapter heading)
  - Include the new chapter heading "## Chapter N" in the content itself
  - Example: If creating Chapter 4 and the outline has Chapters 1, 2, 3:
    * Find the LAST line of Chapter 3 (the last chapter that exists)
    * If last line is "- Character reaches the destination", then anchor_text: "- Character reaches the destination"
    * content: "## Chapter 4\n\n[summary paragraph]\n\n- Beat 1\n- Beat 2\n..."
  - **DO NOT** set anchor_text to the new chapter heading (it doesn't exist yet!)
  - **DO NOT** use anchor_text from Chapter 2 if Chapter 3 exists - always use the LAST chapter's last line

- **ADDING to existing chapter** (ADDITION INTENT): When user wants to add new content to an existing chapter without removing existing content → Use "insert_after_heading" operation
  - Semantic understanding: User provides new beats/content, wants to extend/continue, wants to add more detail
  - Examples: "Add a scene where X happens to Chapter 2" → operation_type: "insert_after_heading", anchor_text: "## Chapter 2"
  - Examples: "Chapter 3 also needs a confrontation" → operation_type: "insert_after_heading", anchor_text: "## Chapter 3"
  - Examples: "Continue Chapter 2 with these beats: ..." → operation_type: "insert_after_heading", anchor_text: "## Chapter 2"
  - Examples: "Here are more beats for Chapter 1: ..." → operation_type: "insert_after_heading", anchor_text: "## Chapter 1"
  - **PRESERVE ALL EXISTING CONTENT** - only add new bullet points, don't regenerate the chapter
  - **CRITICAL FOR "continue" or providing new beats**: When user says "continue chapter", "continue with", or provides a list of new beats for an existing chapter, they want to ADD beats, not replace the chapter
    - Find the LAST bullet point in the existing chapter
    - Insert new beats AFTER that last bullet point
    - Do NOT include the chapter heading in your content - just the new beats
    - **IMPORTANT: Consider updating summaries when adding significant beats**:
      * If the new beats significantly change or expand the chapter's scope, you may need to update the chapter summary paragraph
      * If the new beats are major plot points that affect the overall story, you may need to update the Overall Synopsis section
      * You can provide MULTIPLE operations: one to add beats, and additional operations to update summaries if needed
      * Use your judgment - minor additions may not need summary updates, but major plot developments should be reflected
  
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
  
- **REPLACING larger chapter content** (REPLACEMENT INTENT): When user wants to change/rewrite existing chapter content → Use "replace_range" operation
  - Semantic understanding: User wants to change the focus/direction, rewrite/regenerate, replace existing structure
  - Examples: "Change Chapter 1 to focus on X instead of Y" → operation_type: "replace_range"
  - Examples: "Rewrite Chapter 2" → operation_type: "replace_range"
  - Examples: "Regenerate Chapter 3" → operation_type: "replace_range"
  - **CRITICAL**: If replacing entire chapter, you MUST provide BOTH operations:
    - Operation 1: delete_range to remove the old chapter content (from "## Chapter N" heading to next "## Chapter" heading or end of document)
    - Operation 2: insert_after_heading to add the new chapter content (anchor to the line before where the old chapter was)
  - Only use replace_range when user explicitly wants to change existing content (not when adding new content)

- **DEFAULT for new content**: If chapter exists and user is adding (not replacing), ALWAYS use "insert_after_heading"
  - Find the last bullet point in the chapter and insert after it
  - Or insert after the chapter heading if chapter is empty
  - **IMPORTANT: When adding significant beats, consider if summaries need updating**:
    * If new beats significantly change/expand chapter scope → May need to update chapter summary
    * If new beats are major plot points affecting overall story → May need to update Overall Synopsis
    * You can route MULTIPLE content pieces: one for beats, and additional pieces for summary updates if needed

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
      "operation_type": "replace_range|insert_after_heading|insert_after|delete_range",
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
3. **DETECT ADDITION vs REPLACEMENT vs GRANULAR CORRECTION** (SEMANTIC UNDERSTANDING REQUIRED):
   - **UNDERSTAND USER INTENT SEMANTICALLY** - Don't rely solely on keywords. Understand what the user wants to accomplish.
   
   - **ADDITION INTENT** (user wants to add new content without removing existing): Use "insert_after_heading"
     - Semantic indicators: User provides new beats/content to add, wants to extend/continue a chapter, wants to add more detail
     - Examples: "Continue Chapter 2 with these beats...", "Add more scenes to Chapter 3", "Extend the chapter with...", "Here are more beats for Chapter 1"
     - **CRITICAL FOR "continue"**: When user says "continue chapter" or provides new beats for an existing chapter, they want to ADD beats, not replace
       - Use "insert_after_heading" with anchor_text set to the chapter heading (e.g., "## Chapter 2")
       - Find the LAST bullet point in the chapter and insert new beats after it
       - **PRESERVE ALL EXISTING CONTENT** - only add new bullet points
       - Do NOT include the chapter heading in your content - just the new beats
   
  - **SYNOPSIS UPDATE INTENT** (user adds major plot points or outline grows significantly): Consider updating Overall Synopsis
    - When user adds significant chapters or major plot developments, evaluate if Synopsis needs updating
    - **PRUNE while updating**: Remove minor details from synopsis that have accumulated over time
    - Keep only MAJOR story elements: main conflict, protagonist's journey, key turning points, resolution
    - Think "back of the book" level - what's essential to understand the core story?
  
  - **GRANULAR CORRECTION INTENT** (user wants to change a specific word/phrase): Use "replace_range"
    - Semantic indicators: User identifies a specific word/phrase to change, wants to correct a detail, wants to fix a specific element
    - Examples: "It should be a boat, not a canoe", "Change 'betrayal' to 'mistake'", "That should say X instead of Y"
    - Find the specific bullet point or sentence containing the word/phrase to replace
    - Include the FULL bullet point (or sentence) in original_text (20+ words for uniqueness)
    - Replace only the specific word/phrase in the content
   
   - **REPLACEMENT INTENT** (user wants to change/rewrite existing content): Use "replace_range"
     - Semantic indicators: User wants to change the focus/direction of content, wants to rewrite/regenerate, wants to replace existing structure
     - Examples: "Change Chapter 1 to focus on X instead of Y", "Rewrite Chapter 2", "Regenerate this chapter"
     - **CRITICAL**: If replacing entire chapter, you MUST provide BOTH operations:
       - Operation 1: delete_range to remove the old chapter content (from chapter heading to next chapter heading or end)
       - Operation 2: insert_after_heading to add the new chapter content
     - Only use replace_range when user explicitly wants to change existing content
   
   - **DEFAULT**: When in doubt for chapters, if chapter exists and user is providing new content, assume ADDITION intent and use "insert_after_heading"
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
    
    async def _detect_request_type_node(self, state: OutlineEditingState) -> Dict[str, Any]:
        """Detect if user request is a question or an edit request - trust LLM to figure it out"""
        try:
            logger.info("Detecting request type (question vs edit request)...")
            
            current_request = state.get("current_request", "")
            if not current_request:
                logger.warning("No current request found - defaulting to edit_request")
                return {"request_type": "edit_request"}
            
            body_only = state.get("body_only", "")
            rules_body = state.get("rules_body")
            style_body = state.get("style_body")
            characters_bodies = state.get("characters_bodies", [])
            
            # Build simple prompt for LLM to determine intent
            prompt = f"""Analyze the user's request and determine if it's a QUESTION or an EDIT REQUEST.

**USER REQUEST**: {current_request}

**CONTEXT**:
- Current outline: {body_only[:500] if body_only else "Empty outline"}
- Has rules reference: {bool(rules_body)}
- Has style reference: {bool(style_body)}
- Has {len(characters_bodies)} character reference(s)

**INTENT DETECTION**:
- QUESTIONS (including pure questions and conditional edits): User is asking a question - may or may not want edits
  - Pure questions: "Do you see our characters?", "What rules are loaded?", "How many chapters do we have?"
  - Conditional edits: "Do we have a synopsis? Add one if not", "How many chapters? Add 3 more if less than 10", "Is Chapter 2 complete? Finish it if not"
  - Questions often start with: "Do you", "What", "Can you", "Are there", "How many", "Show me", "Is", "Does", "Are we"
  - **Key insight**: Questions can be answered, and IF edits are needed based on the answer, they can be made
  - Route ALL questions to edit path - LLM can decide if edits are needed
  
- EDIT REQUESTS: User wants to create, modify, or generate content - NO question asked
  - Examples: "Add a chapter", "Create an outline", "Update the synopsis", "Generate outline for chapter 2"
  - Edit requests are action-oriented: "add", "create", "update", "generate", "change", "replace"
  - Edit requests specify what content to create or modify
  - **Key indicator**: Action verbs present, no question asked

**OUTPUT**: Return ONLY valid JSON:
{{
  "request_type": "question" | "edit_request",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of why this classification"
}}

**CRITICAL**: 
- If request contains a question (even with action verbs) → "question" (will route to edit path, LLM decides if edits needed)
- If request is ONLY action verbs with NO question → "edit_request"
- Trust your semantic understanding - questions go to edit path where LLM can analyze and optionally edit"""
            
            # Call LLM with structured output
            llm = self._get_llm(temperature=0.1, state=state)  # Low temperature for consistent classification
            
            datetime_context = self._get_datetime_context()
            messages = [
                SystemMessage(content="You are an intent classifier. Analyze user requests and determine if they are questions or edit requests. Return only valid JSON."),
                SystemMessage(content=datetime_context),
                HumanMessage(content=prompt)
            ]
            
            try:
                response = await self._safe_llm_invoke(llm, messages, error_context="Analyze outline structure")
            except OpenRouterError as e:
                logger.error(f"Failed to analyze outline structure: {e}")
                return {
                    "error": e.user_message,
                    "task_status": "error"
                }
            content = response.content if hasattr(response, 'content') else str(response)
            content = _unwrap_json_response(content)
            
            try:
                result = json.loads(content)
                request_type = result.get("request_type", "edit_request")
                confidence = result.get("confidence", 0.5)
                reasoning = result.get("reasoning", "")
                
                logger.info(f"Request type detected: {request_type} (confidence: {confidence:.0%}, reasoning: {reasoning})")
                
                # Default to edit_request if confidence is low
                if confidence < 0.6:
                    logger.warning(f"Low confidence ({confidence:.0%}) - defaulting to edit_request")
                    request_type = "edit_request"
                
                return {
                    "request_type": request_type
                }
                
            except Exception as parse_error:
                logger.error(f"Failed to parse request type detection: {parse_error}")
                logger.warning("Defaulting to edit_request due to parse error")
                return {"request_type": "edit_request"}
            
        except Exception as e:
            logger.error(f"Failed to detect request type: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Default to edit_request on error
            return {"request_type": "edit_request"}
    
    def _route_from_request_type(self, state: OutlineEditingState) -> str:
        """Route based on detected request type"""
        request_type = state.get("request_type", "edit_request")
        # ALL questions route to edit path (can analyze and optionally edit)
        # Both question and edit_request route to the same path (analyze_and_route_request)
        return "edit_request"  # Both question and edit_request go to edit path
    
    async def _generate_question_answer(self, state: OutlineEditingState) -> Optional[str]:
        """Generate a conversational answer to a user's question about the outline"""
        try:
            current_request = state.get("current_request", "")
            body_only = state.get("body_only", "")
            rules_body = state.get("rules_body")
            style_body = state.get("style_body")
            characters_bodies = state.get("characters_bodies", [])
            
            # Build context for answering
            context_parts = []
            
            # Current outline state
            if body_only:
                context_parts.append("=== CURRENT OUTLINE ===\n")
                context_parts.append(f"{body_only}\n\n")
            else:
                context_parts.append("=== CURRENT OUTLINE ===\n")
                context_parts.append("Empty outline (no content yet)\n\n")
            
            # Reference information (summaries for context)
            if rules_body:
                context_parts.append("=== UNIVERSE RULES (SUMMARY) ===\n")
                rules_summary = rules_body[:1000] + "..." if len(rules_body) > 1000 else rules_body
                context_parts.append(f"{rules_summary}\n\n")
            
            if style_body:
                context_parts.append("=== STYLE GUIDE (SUMMARY) ===\n")
                style_summary = style_body[:1000] + "..." if len(style_body) > 1000 else style_body
                context_parts.append(f"{style_summary}\n\n")
            
            if characters_bodies:
                context_parts.append(f"=== CHARACTER PROFILES ({len(characters_bodies)} character(s)) ===\n")
                for i, char_body in enumerate(characters_bodies, 1):
                    char_summary = char_body[:500] + "..." if len(char_body) > 500 else char_body
                    context_parts.append(f"**Character {i}**:\n{char_summary}\n\n")
            
            # Build request with instructions
            request_with_instructions = f"""=== USER QUESTION ===
{current_request}

**YOUR TASK**: Answer the user's question clearly and helpfully with specific details from the outline.

- If they're asking about the outline structure, provide specific details about chapters, beats, characters, etc.
- If they're asking for an assessment, provide a thorough analysis with specific examples from the outline
- If they're asking about references (characters, rules, style), confirm what's loaded and provide relevant information
- Be conversational, detailed, and helpful - provide real insights, not just generic summaries
- Use specific examples from the outline to support your answer
- If you don't have certain information, say so honestly
- Keep your answer focused and relevant to the question

**OUTPUT**: Provide a natural, conversational answer to the user's question. Do NOT generate JSON or editor operations. Be specific and detailed."""
            
            # Use standardized helper for message construction with conversation history
            messages_list = state.get("messages", [])
            system_prompt = "You are a helpful outline assistant. Answer user questions about the outline, references, and related information. Be conversational, detailed, and helpful."
            messages = self._build_editing_agent_messages(
                system_prompt=system_prompt,
                context_parts=context_parts,
                current_request=request_with_instructions,
                messages_list=messages_list,
                look_back_limit=6
            )
            
            # Call LLM for conversational response
            llm = self._get_llm(temperature=0.7, state=state)  # Higher temperature for natural conversation
            
            response = await llm.ainvoke(messages)
            answer_text = response.content if hasattr(response, 'content') else str(response)
            
            logger.info(f"Generated conversational answer (first 200 chars): {answer_text[:200]}...")
            return answer_text
            
        except Exception as e:
            logger.error(f"Failed to generate question answer: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    async def _answer_question_node(self, state: OutlineEditingState) -> Dict[str, Any]:
        """Answer user questions about the outline, references, or general information"""
        try:
            logger.info("Answering user question...")
            
            current_request = state.get("current_request", "")
            body_only = state.get("body_only", "")
            rules_body = state.get("rules_body")
            style_body = state.get("style_body")
            characters_bodies = state.get("characters_bodies", [])
            reference_summary = state.get("reference_summary", "")
            structure_guidance = state.get("structure_guidance", "")
            
            # Build context for answering
            context_parts = []
            
            # Current outline state
            if body_only:
                context_parts.append("=== CURRENT OUTLINE ===\n")
                context_parts.append(f"{body_only}\n\n")
            else:
                context_parts.append("=== CURRENT OUTLINE ===\n")
                context_parts.append("Empty outline (no content yet)\n\n")
            
            # Reference information
            if reference_summary:
                context_parts.append(f"=== REFERENCES ===\n{reference_summary}\n\n")
            
            if rules_body:
                context_parts.append("=== UNIVERSE RULES (SUMMARY) ===\n")
                # Include first 500 chars as summary
                rules_summary = rules_body[:500] + "..." if len(rules_body) > 500 else rules_body
                context_parts.append(f"{rules_summary}\n\n")
            
            if style_body:
                context_parts.append("=== STYLE GUIDE (SUMMARY) ===\n")
                style_summary = style_body[:500] + "..." if len(style_body) > 500 else style_body
                context_parts.append(f"{style_summary}\n\n")
            
            if characters_bodies:
                context_parts.append(f"=== CHARACTER PROFILES ({len(characters_bodies)} character(s)) ===\n")
                for i, char_body in enumerate(characters_bodies, 1):
                    char_summary = char_body[:300] + "..." if len(char_body) > 300 else char_body
                    context_parts.append(f"**Character {i}**:\n{char_summary}\n\n")
            
            if structure_guidance:
                context_parts.append(f"=== OUTLINE STRUCTURE ===\n{structure_guidance}\n\n")
            
            # Build request with instructions
            request_with_instructions = f"""=== USER QUESTION ===
{current_request}

**YOUR TASK**: Answer the user's question clearly and helpfully.

- If they're asking about references (characters, rules, style), confirm what's loaded and provide relevant information
- If they're asking about the outline, describe what's currently in it
- If they're asking for verification, confirm what you can see
- Be conversational and helpful - this is a question, not a content generation request
- If you don't have certain information, say so honestly
- Keep your answer focused and relevant to the question

**OUTPUT**: Provide a natural, conversational answer to the user's question. Do NOT generate JSON or editor operations."""
            
            # Use standardized helper for message construction with conversation history
            messages_list = state.get("messages", [])
            system_prompt = "You are a helpful outline assistant. Answer user questions about the outline, references, and related information. Be conversational and helpful."
            messages = self._build_editing_agent_messages(
                system_prompt=system_prompt,
                context_parts=context_parts,
                current_request=request_with_instructions,
                messages_list=messages_list,
                look_back_limit=6
            )
            
            # Call LLM for conversational response
            llm = self._get_llm(temperature=0.7, state=state)  # Higher temperature for natural conversation
            
            try:
                response = await self._safe_llm_invoke(llm, messages, error_context="Answer question")
            except OpenRouterError as e:
                logger.error(f"Failed to answer question: {e}")
                return {
                    "response": {
                        "response": e.user_message,
                        "task_status": "error",
                        "agent_type": "outline_editing_agent"
                    },
                    "task_status": "error",
                    "error": e.user_message
                }
            answer_text = response.content if hasattr(response, 'content') else str(response)
            
            logger.info(f"Generated answer (first 200 chars): {answer_text[:200]}...")
            
            # Store as response (no operations needed for questions)
            return {
                "response": {
                    "response": answer_text,
                    "task_status": "complete",
                    "agent_type": "outline_editing_agent",
                    "timestamp": datetime.now().isoformat()
                },
                "task_status": "complete",
                "editor_operations": []  # No operations for questions
            }
            
        except Exception as e:
            logger.error(f"Failed to answer question: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "response": {
                    "response": f"I encountered an error while trying to answer your question: {str(e)}",
                    "task_status": "error",
                    "agent_type": "outline_editing_agent"
                },
                "task_status": "error",
                "error": str(e)
            }
    
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
            
            # Use standardized helper for message construction with conversation history
            messages_list = state.get("messages", [])
            system_prompt = "You are an outline content routing expert. Analyze user requests and route content pieces to appropriate outline sections."
            messages = self._build_editing_agent_messages(
                system_prompt=system_prompt,
                context_parts=[],  # Routing doesn't need file context
                current_request=routing_prompt,
                messages_list=messages_list,
                look_back_limit=6
            )
            
            # Call LLM with structured output
            llm = self._get_llm(temperature=0.2, state=state)
            
            # Parse JSON response (structured output requires Pydantic model, so we use JSON parsing)
            try:
                response = await self._safe_llm_invoke(llm, messages, error_context="Analyze and route request")
            except OpenRouterError as e:
                logger.error(f"Failed to analyze and route request: {e}")
                return {
                    "routing_plan": {
                        "content_pieces": [],
                        "completeness_check": e.user_message
                    },
                    "error": e.user_message,
                    "task_status": "error"
                }
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
            
            # Auto-correct anchor_text for new chapter operations in routing plan
            body_only = state.get("body_only", "")
            if body_only:
                chapter_ranges = find_chapter_ranges(body_only)
                existing_chapter_numbers = {ch.chapter_number for ch in chapter_ranges if ch.chapter_number is not None}
                
                for piece in routing_plan.get("content_pieces", []):
                    target_section = piece.get("target_section", "")
                    operation_type = piece.get("operation_type", "")
                    content = piece.get("content", "")
                    
                    # Check if this is creating a new chapter
                    if operation_type == "insert_after_heading" and target_section.startswith("## Chapter"):
                        # Extract chapter number from target_section
                        chapter_match = re.match(r'^##\s+Chapter\s+(\d+)', target_section)
                        if chapter_match:
                            new_chapter_num = int(chapter_match.group(1))
                            # If this chapter doesn't exist yet, it's a new chapter
                            if new_chapter_num not in existing_chapter_numbers:
                                # Find the last line of the last chapter
                                last_line = find_last_line_of_last_chapter(body_only)
                                if last_line:
                                    old_anchor = piece.get("anchor_text", "")
                                    logger.info(f"🔧 Auto-correcting routing plan anchor_text for new Chapter {new_chapter_num}")
                                    logger.info(f"   Old anchor_text: {old_anchor[:100] if old_anchor else 'None'}...")
                                    logger.info(f"   New anchor_text: {last_line[:100]}...")
                                    piece["anchor_text"] = last_line
            
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
            
            # Build dynamic system prompt
            system_prompt = self._build_system_prompt(state)
            
            # Build context message - simple and focused
            context_parts = []
            
            # User request will be separated into request_with_instructions below
            if not current_request:
                logger.error("current_request is empty - user's request will not be sent to LLM")
            
            # Current outline
            is_empty_file = not body_only.strip()
            context_parts.append("=== CURRENT OUTLINE ===\n")
            context_parts.append(f"File: {filename}\n")
            if is_empty_file:
                context_parts.append("\n⚠️ EMPTY FILE DETECTED: This file contains only frontmatter (no content yet)\n\n")
            else:
                context_parts.append("\n" + body_only + "\n\n")
            
            # References (if present)
            if rules_body:
                context_parts.append("=== UNIVERSE RULES ===\n")
                context_parts.append(f"{rules_body}\n\n")
            
            if style_body:
                context_parts.append("=== STYLE GUIDE ===\n")
                context_parts.append(f"{style_body}\n\n")
            
            if characters_bodies:
                context_parts.append("=== CHARACTER PROFILES ===\n")
                context_parts.append("**NOTE**: Character details (descriptions, backstories, traits) belong in character profile files.\n")
                context_parts.append("The outline should only reference characters briefly (name + role), not include full character information.\n\n")
                context_parts.append("".join([f"{b}\n---\n" for b in characters_bodies]))
                context_parts.append("\n")
            
            if clarification_context:
                context_parts.append(clarification_context)
            
            # Add "WORK FIRST" guidance and cross-reference instructions (like character agent)
            context_parts.append(
                "\n=== EDIT REQUEST: WORK WITH AVAILABLE INFORMATION ===\n"
                "The user wants you to add or revise outline content.\n\n"
                "**YOUR APPROACH**:\n"
                "1. **FOR EMPTY FILES**: If the outline is empty, ASK QUESTIONS FIRST before creating content\n"
                "   - Don't create the entire outline structure at once\n"
                "   - Ask about story basics: genre, main characters, key plot points, chapter count\n"
                "   - Build incrementally - create one section at a time based on user responses\n"
                "2. **FOR FILES WITH CONTENT**: Make edits based on the request and available context (outline file, rules, style, characters)\n"
                "3. **USE INFERENCE**: Make reasonable inferences from the request - but ask if starting from scratch\n"
                "4. **ASK ALONG THE WAY**: If you need specific details, include questions in the summary AFTER describing the work you've done\n"
                "5. **NEVER EMPTY OPERATIONS FOR EDIT REQUESTS**: Always provide operations OR ask questions (clarification request)\n"
                "6. **CHARACTER INFORMATION**: Keep character details in character profiles, not in the outline!\n"
                "   - Outline should only have brief character references (name + role)\n"
                "   - Do NOT copy character descriptions, backstories, or traits into the outline\n\n"
                "CRITICAL: CHECK FOR DUPLICATES FIRST\n"
                "Before adding ANY new content:\n"
                "1. **CHECK FOR SIMILAR CONTENT** - Does similar plot/beat information already exist in related chapters?\n"
                "2. **CONSOLIDATE IF NEEDED** - If plot point appears in multiple places, ensure each adds unique perspective\n"
                "3. **AVOID REDUNDANCY** - Don't add identical information to multiple chapters\n"
                "\n"
                "CRITICAL: CROSS-REFERENCE RELATED SECTIONS\n"
                "After checking for duplicates:\n"
                "1. **SCAN THE ENTIRE OUTLINE** - Read through ALL chapters to identify related plot information\n"
                "2. **IDENTIFY ALL AFFECTED SECTIONS** - When adding/updating plot content, find ALL places it should appear\n"
                "3. **GENERATE MULTIPLE OPERATIONS** - If plot content affects multiple chapters, create operations for EACH affected section\n"
                "4. **ENSURE CONSISTENCY** - Related chapters must be updated together to maintain plot coherence\n"
                "\n"
                "Examples of when to generate multiple operations:\n"
                "- Adding character introduction → Update chapter where introduced AND character list section\n"
                "- Adding plot twist → Update chapter with twist AND any earlier chapters that need foreshadowing\n"
                "- Updating character arc → Update relevant chapters AND character section if arc affects character description\n"
                "- Adding worldbuilding detail → Update chapter where detail appears AND Notes section if it's a major world element\n\n"
            )
            
            # Build request with instructions
            request_with_instructions = ""
            if current_request:
                request_with_instructions = f"""=== USER REQUEST ===
{current_request}

Generate ManuscriptEdit JSON with operations to fulfill the user's request.
Use replace_range for changing existing content, insert_after_heading for adding new content, delete_range for removing content."""
                if is_empty_file:
                    request_with_instructions += """

⚠️ CRITICAL: EMPTY FILE INSTRUCTIONS
Since this file is empty (only frontmatter), follow these rules:
1. **DO NOT use anchor_text** - The file has no content to anchor to
2. **Use insert_after_heading WITHOUT anchor_text** - The system will automatically insert after frontmatter
3. **Include section headings in your text** - For example, if creating the first chapter, include '## Chapter 1' in the text
4. **Example operation for empty file**:
   {"op_type": "insert_after_heading", "text": "## Chapter 1\\n\\n[your content]"}
   (Note: NO anchor_text field needed - omit it entirely)
5. **DO NOT reference context headers** - Text like '=== CURRENT OUTLINE ===' or 'File: filename.md' does NOT exist in the file
6. **DO NOT use anchor_text like '# Notes' or '# Characters'** - These sections don't exist yet in an empty file"""
            
            # Use standardized helper for message construction with conversation history
            messages_list = state.get("messages", [])
            messages = self._build_editing_agent_messages(
                system_prompt=system_prompt,
                context_parts=context_parts,
                current_request=request_with_instructions,
                messages_list=messages_list,
                look_back_limit=6
            )
            
            # Call LLM - pass state to access user's model selection from metadata
            llm = self._get_llm(temperature=0.3, state=state)
            start_time = datetime.now()
            try:
                response = await self._safe_llm_invoke(llm, messages, error_context="Generate edit plan")
            except OpenRouterError as e:
                logger.error(f"Failed to generate edit plan: {e}")
                return {
                    "structured_edit": None,
                    "error": e.user_message,
                    "task_status": "error"
                }
            
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
                    f"**Critical Ambiguity Detected**\n\n"
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
            
            # Otherwise, parse as ManuscriptEdit with Pydantic validation
            try:
                # Parse JSON first
                raw = json.loads(content)
                
                # Ensure required fields have defaults
                if isinstance(raw, dict):
                    raw.setdefault("target_filename", filename)
                    raw.setdefault("scope", "paragraph")
                    raw.setdefault("summary", "Planned outline edit generated from context.")
                    raw.setdefault("safety", "medium")
                    raw.setdefault("operations", [])
                
                # Validate with Pydantic model
                try:
                    manuscript_edit = ManuscriptEdit(**raw)
                    
                    # Log operation details (preserve existing logging behavior)
                    for idx, op in enumerate(manuscript_edit.operations):
                        op_text = op.text
                        if op_text:
                            logger.info(f"Operation {idx} text length: {len(op_text)} chars, preview: {op_text[:100]}")
                        else:
                            logger.warning(f"Operation {idx} has EMPTY text field!")
                    
                    
                    # Detect and fix content overlap for insert_after_heading operations
                    # When LLM generates "continue chapter" content, it may include the last beats for context
                    # This causes duplication when we insert after those beats
                    body_only = state.get("body_only", "")
                    if body_only:
                        chapter_ranges = find_chapter_ranges(body_only)
                        existing_chapter_numbers = {ch.chapter_number for ch in chapter_ranges if ch.chapter_number is not None}
                        
                        corrected_operations = []
                        for op in manuscript_edit.operations:
                            if op.op_type == "insert_after_heading" and op.text and op.anchor_text:
                                # Check if anchor is a chapter heading for an existing chapter
                                anchor_chapter_match = re.match(r'^##\s+Chapter\s+(\d+)', op.anchor_text.strip())
                                if anchor_chapter_match:
                                    anchor_chapter_num = int(anchor_chapter_match.group(1))
                                    
                                    # If this chapter exists, check for content overlap
                                    if anchor_chapter_num in existing_chapter_numbers:
                                        # Find this chapter's range
                                        target_chapter = None
                                        for ch in chapter_ranges:
                                            if ch.chapter_number == anchor_chapter_num:
                                                target_chapter = ch
                                                break
                                        
                                        if target_chapter:
                                            # Get chapter content (without heading)
                                            chapter_content = body_only[target_chapter.start:target_chapter.end]
                                            lines = chapter_content.split('\n')[1:]  # Skip heading line
                                            existing_content_body = '\n'.join(lines).strip()
                                            
                                            # Check for content overlap - compare ALL existing bullets with generated text
                                            if existing_content_body and len(existing_content_body) > 50:
                                                # Find ALL bullet points in existing chapter
                                                bullet_lines = [line for line in lines if line.strip().startswith('- ')]
                                                
                                                if len(bullet_lines) >= 1:
                                                    # Normalize generated text for comparison
                                                    generated_text_normalized = ' '.join(op.text.strip().split())
                                                    
                                                    # Check if ANY existing bullets appear in generated text
                                                    overlapping_bullets = []
                                                    for bullet in bullet_lines:
                                                        bullet_normalized = ' '.join(bullet.strip().split())
                                                        # Check if this bullet (first 100 chars) appears in generated text
                                                        if len(bullet_normalized) > 20:
                                                            bullet_sample = bullet_normalized[:100]
                                                            if bullet_sample in generated_text_normalized:
                                                                overlapping_bullets.append(bullet)
                                                    
                                                    # If we found overlapping bullets, convert to replace_range
                                                    if overlapping_bullets:
                                                        logger.warning(f"⚠️ Content overlap detected for Chapter {anchor_chapter_num}")
                                                        logger.warning(f"   Found {len(overlapping_bullets)} overlapping bullet(s) in generated text")
                                                        logger.warning(f"   Converting insert_after_heading to replace_range")
                                                        
                                                        # Use overlapping bullets as original_text
                                                        overlapping_text = '\n'.join(overlapping_bullets).strip()
                                                        op.op_type = "replace_range"
                                                        op.original_text = overlapping_text
                                                        op.anchor_text = None  # Not needed for replace_range
                                                        
                                                        logger.info(f"✅ Converted to replace_range with {len(overlapping_bullets)} bullet(s)")
                                                        logger.info(f"   original_text: {overlapping_text[:150]}...")
                            
                            corrected_operations.append(op)
                        
                        # Replace operations with corrected ones
                        manuscript_edit.operations = corrected_operations
                    
                    # Convert to dict for state storage (TypedDict compatibility)
                    structured_edit = manuscript_edit.model_dump()
                    logger.info(f"✅ Validated ManuscriptEdit with {len(manuscript_edit.operations)} operations")
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
                        "task_status": "error"
                    }
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}")
                return {
                    "llm_response": content,
                    "structured_edit": None,
                    "error": f"Failed to parse JSON: {str(e)}",
                    "task_status": "error"
                }
            except Exception as e:
                logger.error(f"Failed to parse structured edit: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
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
            
            # Check if this is a question with no operations - generate conversational answer
            operations = structured_edit.get("operations", [])
            if len(operations) == 0:
                # Check if user request looks like a question
                current_request = state.get("current_request", "").lower()
                is_question = any(keyword in current_request for keyword in [
                    "?", "how", "what", "why", "when", "where", "who", "which",
                    "assess", "evaluate", "review", "analyze", "check", "examine",
                    "tell me", "explain", "describe", "summarize", "looks like", "looking"
                ])
                
                if is_question:
                    logger.info("Question detected with no operations - generating conversational answer")
                    # Generate a proper conversational answer
                    answer = await self._generate_question_answer(state)
                    if answer:
                        # Store answer in structured_edit summary for format_response to use
                        structured_edit["summary"] = answer
                        structured_edit["is_question_answer"] = True
                else:
                    # Edit request with no operations - this is an error!
                    # The LLM should have generated operations for edit requests
                    logger.error("⚠️ Edit request detected but no operations generated - this should not happen!")
                    logger.error(f"   Request: {current_request[:200]}")
                    logger.error(f"   Summary: {structured_edit.get('summary', '')[:200]}")
                    # Force an error response
                    return {
                        "llm_response": content,
                        "structured_edit": None,
                        "error": "Edit request received but no operations were generated. Please try rephrasing your request or be more specific about what you want to add or change.",
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
            
            # Check if file is empty (only frontmatter)
            body_only = _strip_frontmatter_block(outline)
            is_empty_file = not body_only.strip()
            
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
                    # Use centralized resolver
                    cursor_pos = state.get("cursor_offset", -1)
                    cursor_pos = cursor_pos if cursor_pos >= 0 else None
                    resolved_start, resolved_end, resolved_text, resolved_confidence = resolve_editor_operation(
                        content=outline,
                        op_dict=op,
                        selection=selection,
                        frontmatter_end=fm_end_idx,
                        cursor_offset=cursor_pos
                    )
                    
                    # Check if resolver returned failure signal (anchor text not found)
                    if resolved_start == -1 and resolved_end == -1:
                        # Anchor text not found - check if file is empty first
                        op_type = op.get("op_type", "")
                        anchor_text = op.get("anchor_text", "")
                        
                        # For empty files, if anchor_text was provided but not found, insert after frontmatter
                        if is_empty_file and op_type == "insert_after_heading":
                            resolved_start = fm_end_idx
                            resolved_end = fm_end_idx
                            resolved_confidence = 0.7
                            logger.info(f"Empty file: anchor_text not found or invalid, inserting after frontmatter at {fm_end_idx}")
                        elif op_type == "insert_after_heading" and anchor_text:
                            # Try to find the last chapter heading as fallback
                            chapter_pattern = re.compile(r"\n##\s+Chapter\s+\d+", re.MULTILINE)
                            matches = list(chapter_pattern.finditer(outline, fm_end_idx))
                            if matches:
                                # Found chapters - insert after the last one
                                last_chapter_match = matches[-1]
                                # Find end of that chapter (next chapter or end of doc)
                                next_match = chapter_pattern.search(outline, last_chapter_match.end())
                                if next_match:
                                    resolved_start = next_match.start()
                                    resolved_end = next_match.start()
                                else:
                                    # Last chapter - insert at end
                                    resolved_start = len(outline)
                                    resolved_end = len(outline)
                                resolved_confidence = 0.6
                                logger.info(f"Anchor text not found, using fallback: Inserting after last chapter at position {last_chapter_match.start()}")
                            else:
                                # No chapters found - insert after frontmatter
                                resolved_start = fm_end_idx
                                resolved_end = fm_end_idx
                                resolved_confidence = 0.5
                                logger.info(f"Anchor text not found, no chapters found, inserting after frontmatter at {fm_end_idx}")
                        else:
                            # Not a chapter insertion - use standard fallback
                            body_only = _strip_frontmatter_block(outline)
                            if not body_only.strip():
                                # Empty file - insert after frontmatter
                                resolved_start = fm_end_idx
                                resolved_end = fm_end_idx
                                resolved_confidence = 0.5
                            else:
                                # Use paragraph bounds or frontmatter end, whichever is larger
                                resolved_start = max(para_start if para_start is not None else 0, fm_end_idx)
                                resolved_end = max(para_end if para_end is not None else 0, resolved_start)
                                resolved_confidence = 0.4
                    
                    # Special handling for empty files: ensure operations insert after frontmatter
                    if is_empty_file and resolved_start < fm_end_idx:
                        resolved_start = fm_end_idx
                        resolved_end = fm_end_idx
                        resolved_confidence = 0.7
                        logger.info(f"Empty file detected - adjusting operation to insert after frontmatter at {fm_end_idx}")
                    
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
                    # Standard fallback positioning - for empty files, insert after frontmatter
                    body_only = _strip_frontmatter_block(outline)
                    if not body_only.strip():
                        # Empty file - insert after frontmatter
                        fallback_start = fm_end_idx
                        fallback_end = fm_end_idx
                    else:
                        # Use paragraph bounds or frontmatter end, whichever is larger
                        fallback_start = max(para_start if para_start is not None else 0, fm_end_idx)
                        fallback_end = max(para_end if para_end is not None else 0, fallback_start)
                    
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
            request_type = state.get("request_type", "edit_request")
            
            # Handle question requests - if no operations and summary looks like an answer, use it
            if not editor_operations and structured_edit:
                is_question_answer = structured_edit.get("is_question_answer", False)
                summary = structured_edit.get("summary", "")
                
                # Check if summary is a real answer (not just a generic summary)
                if is_question_answer or (summary and len(summary) > 50 and not summary.startswith("Assessment of")):
                    logger.info("Returning question response (no operations needed)")
                    return {
                        "response": {
                            "response": summary,
                            "task_status": "complete",
                            "agent_type": "outline_editing_agent",
                            "timestamp": datetime.now().isoformat()
                        },
                        "task_status": "complete",
                        "editor_operations": []  # No operations for pure questions
                    }
            
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
                logger.info(f"✅ Adding {len(editor_operations)} editor operations to response")
                response["editor_operations"] = editor_operations
                response["manuscript_edit"] = {
                    "target_filename": structured_edit.get("target_filename"),
                    "scope": structured_edit.get("scope"),
                    "summary": structured_edit.get("summary"),
                    "chapter_index": structured_edit.get("chapter_index"),
                    "safety": structured_edit.get("safety"),
                    "operations": editor_operations
                }
            else:
                logger.warning(f"⚠️ No editor_operations to add (editor_operations={editor_operations}, type={type(editor_operations)})")
            
            # Clear any pending clarification since we're completing successfully
            shared_memory = state.get("shared_memory", {}) or {}
            shared_memory_out = shared_memory.copy()
            shared_memory_out.pop("pending_outline_clarification", None)
            response["shared_memory"] = shared_memory_out
            
            logger.info(f"🔍 _format_response_node returning: response_keys={list(response.keys())}, has_editor_ops={bool(response.get('editor_operations'))}")
            
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
                "routing_plan": None,
                # NEW: Request type detection
                "request_type": "unknown"
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
                logger.info(f"✅ Added {len(response['editor_operations'])} editor operations to result")
            else:
                logger.warning(f"⚠️ No editor_operations in response dict (response keys: {list(response.keys())})")
            if response.get("manuscript_edit"):
                result["manuscript_edit"] = response["manuscript_edit"]
            if response.get("shared_memory"):
                result["shared_memory"] = response["shared_memory"]
            
            logger.info(f"Outline editing agent completed: {task_status}, result_keys={list(result.keys())}")
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

