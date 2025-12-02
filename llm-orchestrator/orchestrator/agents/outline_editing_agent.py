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
            # Find end of line/paragraph
            end_pos = outline.find("\n", pos)
            if end_pos == -1:
                end_pos = len(outline)
            else:
                end_pos += 1
            return end_pos, end_pos, text, 0.9
    
    # Strategy 3: Left + right context
    if left_context and right_context:
        pattern = re.escape(left_context) + r"([\s\S]{0,400}?)" + re.escape(right_context)
        m = re.search(pattern, outline)
        if m:
            return m.start(1), m.end(1), text, 0.8
    
    # Fallback: use approximate positions from op_dict
    start = op_dict.get("start", 0)
    end = op_dict.get("end", 0)
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
        workflow.add_node("generate_edit_plan", self._generate_edit_plan_node)
        workflow.add_node("resolve_operations", self._resolve_operations_node)
        workflow.add_node("format_response", self._format_response_node)
        
        # Entry point
        workflow.set_entry_point("prepare_context")
        
        # Flow: prepare_context -> load_references -> generate_edit_plan -> resolve_operations -> format_response -> END
        workflow.add_edge("prepare_context", "load_references")
        workflow.add_edge("load_references", "generate_edit_plan")
        workflow.add_edge("generate_edit_plan", "resolve_operations")
        workflow.add_edge("resolve_operations", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow.compile(checkpointer=checkpointer)
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for outline editing"""
        return (
            "You are an Outline Development Assistant for type: outline files. Persona disabled."
            " Preserve frontmatter; operate on Markdown body only.\n\n"
            "BEST EFFORT DOCTRINE:\n"
            "ALWAYS provide outline content - never leave the user empty-handed!\n\n"
            "OPERATIONAL STRATEGY:\n"
            "1. **ALWAYS create content** based on available information\n"
            "2. **Make reasonable inferences** from existing context\n"
            "3. **ONLY skip clarification** when you're highly confident (>0.85)\n"
            "4. **Ask questions SPARINGLY** - only for truly critical gaps\n\n"
            "WHEN TO REQUEST CLARIFICATION (ONLY CRITICAL GAPS):\n"
            "✅ Ask when:\n"
            "- User request is genuinely ambiguous with multiple valid interpretations\n"
            "- A critical plot element contradicts established rules/characters\n"
            "- Creating the wrong content would be worse than asking\n\n"
            "❌ DO NOT ask when:\n"
            "- You can make reasonable inferences from context\n"
            "- Existing chapters/rules/characters provide guidance\n"
            "- User request is clear enough for basic implementation\n"
            "- You can create placeholder structure that's useful\n\n"
            "STRUCTURED OUTPUT OPTIONS:\n\n"
            "OPTION 1 - OutlineClarificationRequest (RARE - only for critical ambiguity):\n"
            "{\n"
            '  "task_status": "incomplete",\n'
            '  "clarification_needed": true,\n'
            '  "questions": ["Critical question that blocks progress?"],\n'
            '  "context": "Why this blocks content creation",\n'
            '  "missing_elements": ["critical_element"],\n'
            '  "suggested_direction": "Optional suggestion",\n'
            '  "section_affected": "Chapter 3",\n'
            '  "confidence_without_clarification": 0.3\n'
            "}\n"
            "⚠️ USE THIS SPARINGLY - Only when creating content would be genuinely harmful!\n\n"
            "OPTION 2 - ManuscriptEdit (DEFAULT - use >90% of the time):\n"
            "{\n"
            '  "type": "ManuscriptEdit",\n'
            '  "target_filename": string,\n'
            '  "scope": one of ["paragraph", "chapter", "multi_chapter"],\n'
            '  "summary": string,\n'
            '  "chapter_index": integer|null,\n'
            '  "safety": one of ["low", "medium", "high"],\n'
            '  "operations": [\n'
            "    {\n"
            '      "op_type": one of ["replace_range", "delete_range", "insert_after_heading"],\n'
            '      "start": integer (approximate),\n'
            '      "end": integer (approximate),\n'
            '      "text": string,\n'
            '      "original_text": string (REQUIRED for replace/delete, optional for insert - EXACT verbatim text from file),\n'
            '      "anchor_text": string (optional - for inserts, exact line to insert after),\n'
            '      "left_context": string (optional - text before target),\n'
            '      "right_context": string (optional - text after target),\n'
            '      "occurrence_index": integer (optional, default 0 if text appears multiple times)\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "OUTPUT RULES:\n"
            "- Output MUST be a single JSON object only.\n"
            "- Do NOT include triple backticks or language tags.\n"
            "- Do NOT include explanatory text before or after the JSON.\n\n"
            "STRICT OUTLINE SCAFFOLD:\n"
            "# Overall Synopsis\n"
            "# Notes\n"
            "# Characters\n"
            "- Protagonists\n"
            "- Antagonists\n"
            "- Supporting Characters\n"
            "# Outline\n"
            "## Chapter 1\n"
            "## Chapter 2\n"
            "... (continue numerically)\n\n"
            "CHARACTER LIST FORMAT:\n"
            "- Protagonists\n  - Name - Brief role\n"
            "- Antagonists\n  - Name - Brief role\n"
            "- Supporting Characters\n  - Name - Brief role\n\n"
            "CHAPTER CONTENT RULES:\n"
            "- Start with a 3-5 sentence summary paragraph.\n"
            "- Follow with main beats as '-' bullets; each may have up to two '  -' sub-bullets.\n"
            "- Max 8-10 main beats per chapter.\n"
            "- Focus on plot events, actions, reveals, conflicts; avoid prose/dialogue.\n"
            "- Use exact '## Chapter N' headings; never titles.\n\n"
            "EDIT RULES:\n"
            "1) Make surgical edits near cursor/selection unless re-organization is requested.\n"
            "2) Maintain scaffold; if missing, create only requested sections.\n"
            "3) Enforce universe consistency against directly referenced Rules and Characters; match Style.\n"
            "4) NO PLACEHOLDER FILLERS: If a requested section has no content yet, create the heading only and leave the body blank. Do NOT insert placeholders like '[To be developed]' or 'TBD'.\n\n"
            "ANCHOR REQUIREMENTS (CRITICAL):\n"
            "For EVERY operation, you MUST provide precise anchors:\n\n"
            "REVISE/DELETE Operations:\n"
            "- ALWAYS include 'original_text' with EXACT, VERBATIM text from the file\n"
            "- Minimum 10-20 words, include complete sentences with natural boundaries\n"
            "- Copy and paste directly - do NOT retype or modify\n"
            "- ⚠️ NEVER include header lines (###, ##, #) in original_text!\n"
            "- OR provide both 'left_context' and 'right_context' (exact surrounding text)\n\n"
            "INSERT Operations (PREFERRED for adding content below headers!):\n"
            "- **PRIMARY METHOD**: Use op_type='insert_after_heading' with anchor_text='## Chapter N' when adding content below ANY header\n"
            "- Provide 'anchor_text' with EXACT, COMPLETE header line to insert after (verbatim from file)\n"
            "- This is the SAFEST method - it NEVER deletes headers, always inserts AFTER them\n"
            "- Use this for adding chapter summaries, beats, or any content below headers\n"
            "- ALTERNATIVE: Provide 'original_text' with text to insert after\n"
            "- FALLBACK: Provide 'left_context' with text before insertion point (minimum 10-20 words)\n\n"
            "Additional Options:\n"
            "- 'occurrence_index' if text appears multiple times (0-based, default 0)\n"
            "- Start/end indices are approximate; anchors take precedence\n\n"
            "=== DECISION TREE FOR OPERATION TYPE ===\n"
            "1. Chapter/section is COMPLETELY EMPTY? → insert_after_heading with anchor_text=\"## Chapter N\"\n"
            "2. Chapter has PLACEHOLDER or existing content to replace? → replace_range (NO headers in original_text!)\n"
            "3. Deleting SPECIFIC content? → delete_range with original_text (NO headers!)\n\n"
            "⚠️ CRITICAL: When replacing placeholder content, use 'replace_range' on ONLY the placeholder!\n"
            "⚠️ NEVER include headers in 'original_text' for replace_range - headers will be deleted!\n"
            "✅ Correct: {\"op_type\": \"replace_range\", \"original_text\": \"[Placeholder beats]\", \"text\": \"- Beat 1\\n- Beat 2\"}\n"
            "❌ Wrong: {\"op_type\": \"insert_after_heading\", \"anchor_text\": \"## Chapter 1\", \"text\": \"...\"} when placeholder exists!\n\n"
            "=== SPACING RULES (CRITICAL - READ CAREFULLY!) ===\n"
            "YOUR TEXT MUST END IMMEDIATELY AFTER THE LAST CHARACTER!\n\n"
            '✅ CORRECT: "- Beat 1\\n- Beat 2\\n- Beat 3"  ← Ends after "3" with NO \\n\n'
            '❌ WRONG: "- Beat 1\\n- Beat 2\\n"  ← Extra \\n after last line creates blank line!\n'
            '❌ WRONG: "- Beat 1\\n- Beat 2\\n\\n"  ← \\n\\n creates 2 blank lines!\n'
            '❌ WRONG: "- Beat 1\\n\\n- Beat 2"  ← Double \\n\\n between items creates blank line!\n\n'
            "IRON-CLAD RULE: After last line = ZERO \\n (nothing!)\n"
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
            except Exception:
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
            
            # Extract content from loaded files
            style_body = None
            if loaded_files.get("style") and len(loaded_files["style"]) > 0:
                style_body = loaded_files["style"][0].get("content", "")
                if style_body:
                    style_body = _strip_frontmatter_block(style_body)
            
            rules_body = None
            if loaded_files.get("rules") and len(loaded_files["rules"]) > 0:
                rules_body = loaded_files["rules"][0].get("content", "")
                if rules_body:
                    rules_body = _strip_frontmatter_block(rules_body)
            
            characters_bodies = []
            if loaded_files.get("characters"):
                for char_file in loaded_files["characters"]:
                    char_content = char_file.get("content", "")
                    if char_content:
                        char_content = _strip_frontmatter_block(char_content)
                        characters_bodies.append(char_content)
            
            return {
                "rules_body": rules_body,
                "style_body": style_body,
                "characters_bodies": characters_bodies
            }
            
        except Exception as e:
            logger.error(f"Failed to load references: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "rules_body": None,
                "style_body": None,
                "characters_bodies": [],
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
            
            # Build system prompt
            system_prompt = self._build_system_prompt()
            
            # Build context message
            context_parts = [
                "=== OUTLINE CONTEXT ===\n",
                f"File: {filename}\n\n",
                "Current Outline (frontmatter stripped):\n" + body_only + "\n\n"
            ]
            
            if rules_body:
                context_parts.append(f"=== RULES ===\n{rules_body}\n\n")
            
            if style_body:
                context_parts.append(f"=== STYLE GUIDE ===\n{style_body}\n\n")
            
            if characters_bodies:
                context_parts.append("".join([f"=== CHARACTER DOC ===\n{b}\n\n" for b in characters_bodies]))
            
            if clarification_context:
                context_parts.append(clarification_context)
            
            # Check for revision mode
            revision_mode = current_request and any(k in current_request.lower() for k in ["revise", "revision", "tweak", "adjust", "polish", "tighten"])
            if revision_mode:
                context_parts.append("REVISION MODE: Apply minimal targeted edits; prefer bullet/paragraph-level replace_range ops.\n\n")
            
            context_parts.append("Provide a ManuscriptEdit JSON plan strictly within scope (or OutlineClarificationRequest if you need more information).")
            
            messages = [
                SystemMessage(content=system_prompt),
                SystemMessage(content=f"Current Date/Time: {datetime.now().isoformat()}"),
                HumanMessage(content="".join(context_parts))
            ]
            
            if current_request:
                messages.append(HumanMessage(content=(
                    f"USER REQUEST: {current_request}\n\n"
                    "CRITICAL ANCHORING INSTRUCTIONS:\n"
                    "- For REVISE/DELETE: Provide 'original_text' with EXACT, VERBATIM text from file (10-20+ words, complete sentences)\n"
                    "- For INSERT: Provide 'anchor_text' or 'original_text' with exact line to insert after, OR 'left_context' with text before insertion\n"
                    "- Copy text directly from the file - do NOT retype or paraphrase\n"
                    "- Without precise anchors, the operation WILL FAIL"
                )))
            
            # Call LLM
            llm = self._get_llm(temperature=0.3)
            start_time = datetime.now()
            response = await llm.ainvoke(messages)
            
            content = response.content if hasattr(response, 'content') else str(response)
            content = _unwrap_json_response(content)
            
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
                        
                        ops.append({
                            "op_type": op_type,
                            "start": op.get("start", para_start),
                            "end": op.get("end", para_start),
                            "text": op.get("text", ""),
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
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process outline editing query using LangGraph workflow"""
        try:
            # Extract query from state
            messages = state.get("messages", [])
            query = ""
            if messages:
                latest_message = messages[-1]
                query = latest_message.content if hasattr(latest_message, 'content') else str(latest_message)
            else:
                query = state.get("query", "")
            
            logger.info(f"Outline editing agent processing: {query[:100]}...")
            
            shared_memory = state.get("shared_memory", {}) or {}
            metadata = state.get("metadata", {}) or {}
            user_id = state.get("user_id", metadata.get("user_id", "system"))
            
            # Initialize state for LangGraph workflow
            initial_state: OutlineEditingState = {
                "query": query,
                "user_id": user_id,
                "metadata": metadata,
                "messages": messages,
                "shared_memory": shared_memory,
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
                "error": ""
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

