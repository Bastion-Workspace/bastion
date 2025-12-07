"""
Rules Editing Agent - LangGraph Implementation
Gated to rules documents. Consumes active editor buffer, cursor/selection, and
referenced style/characters from frontmatter. Produces EditorOperations suitable
for Prefer Editor HITL application.
"""

import hashlib
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict, Tuple

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


# ============================================
# Utility Functions
# ============================================

def _slice_hash(text: str) -> str:
    """Match frontend sliceHash: 32-bit rolling hash to hex string."""
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


# ============================================
# Simplified Resolver (Progressive Search)
# ============================================

def _resolve_operation_simple(
    rules_text: str,
    op_dict: Dict[str, Any],
    selection: Optional[Dict[str, int]] = None,
    frontmatter_end: int = 0
) -> Tuple[int, int, str, float]:
    """
    Simplified operation resolver using progressive search.
    Returns (start, end, text, confidence)
    """
    op_type = op_dict.get("op_type", "replace_range")
    text = op_dict.get("text", "")
    
    # Try anchor_text first (for insert_after_heading)
    anchor_text = op_dict.get("anchor_text")
    if anchor_text and op_type == "insert_after_heading":
        idx = rules_text.find(anchor_text)
        if idx != -1:
            # Find end of line after anchor
            line_end = rules_text.find("\n", idx)
            if line_end == -1:
                line_end = len(rules_text)
            insert_pos = line_end + 1
            return insert_pos, insert_pos, text, 0.9
    
    # Try original_text (for replace/delete)
    original_text = op_dict.get("original_text")
    if original_text:
        idx = rules_text.find(original_text)
        if idx != -1:
            if op_type == "delete_range":
                return idx, idx + len(original_text), "", 0.9
            else:
                return idx, idx + len(original_text), text, 0.9
    
    # Try left_context + right_context
    left_ctx = op_dict.get("left_context")
    right_ctx = op_dict.get("right_context")
    if left_ctx and right_ctx:
        pattern = re.escape(left_ctx) + r"([\s\S]{0,400}?)" + re.escape(right_ctx)
        m = re.search(pattern, rules_text)
        if m:
            start = m.start(1)
            end = m.end(1)
            if op_type == "delete_range":
                return start, end, "", 0.85
            else:
                return start, end, text, 0.85
    
    # Fallback: use approximate positions from op_dict
    start = op_dict.get("start", 0)
    end = op_dict.get("end", 0)
    return start, end, text, 0.5


# ============================================
# LangGraph State
# ============================================

class RulesEditingState(TypedDict):
    """State for rules editing agent LangGraph workflow"""
    query: str
    user_id: str
    metadata: Dict[str, Any]
    messages: List[Any]
    shared_memory: Dict[str, Any]
    active_editor: Dict[str, Any]
    rules: str
    filename: str
    frontmatter: Dict[str, Any]
    cursor_offset: int
    selection_start: int
    selection_end: int
    body_only: str
    para_start: int
    para_end: int
    style_body: Optional[str]
    characters_bodies: List[str]
    current_request: str
    system_prompt: str
    llm_response: str
    structured_edit: Optional[Dict[str, Any]]
    editor_operations: List[Dict[str, Any]]
    response: Dict[str, Any]
    task_status: str
    error: str


# ============================================
# Rules Editing Agent
# ============================================

class RulesEditingAgent(BaseAgent):
    """
    Rules Editing Agent for worldbuilding and series continuity
    
    Gated to rules documents. Consumes full rules body (frontmatter stripped),
    loads Style and Character references directly from this file's frontmatter,
    and emits editor operations for Prefer Editor HITL.
    Uses LangGraph workflow for explicit state management
    """
    
    def __init__(self):
        super().__init__("rules_editing_agent")
        logger.info("Rules Editing Agent ready!")
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """Build LangGraph workflow for rules editing agent"""
        workflow = StateGraph(RulesEditingState)
        
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
        """Build system prompt for rules editing"""
        return (
            "You are a MASTER UNIVERSE ARCHITECT for RULES documents (worldbuilding, series continuity). "
            "Persona disabled. Adhere strictly to frontmatter, project Rules, and Style.\n\n"
            "STRUCTURED OUTPUT REQUIRED: Return ONLY raw JSON (no prose, no markdown, no code fences) matching this schema:\n"
            "{\n"
            '  "type": "ManuscriptEdit",\n'
            '  "target_filename": string,\n'
            '  "scope": one of ["paragraph", "chapter", "multi_chapter"],\n'
            '  "summary": string,\n'
            '  "chapter_index": integer|null,\n'
            '  "safety": one of ["low", "medium", "high"],\n'
            '  "operations": [ { "op_type": one of ["replace_range", "delete_range", "insert_after_heading"], "start": integer, "end": integer, "text": string } ]\n'
            "}\n\n"
            "OUTPUT RULES:\n"
            "- Output MUST be a single JSON object only.\n"
            "- Do NOT include triple backticks or language tags.\n"
            "- Do NOT include explanatory text before or after the JSON.\n\n"
            "FORMATTING CONTRACT (RULES DOCUMENTS):\n"
            "- Never emit YAML frontmatter in operations[].text. Preserve existing frontmatter as-is.\n"
            "- Use Markdown headings and lists for the body.\n"
            "- When creating or normalizing structure, prefer this scaffold (top-level headings):\n"
            "  ## Background\n"
            "  ## Universe Constraints (physical/magical/technological laws)\n"
            "  ## Systems\n"
            "  ### Magic or Technology Systems\n"
            "  ### Resource & Economy Constraints\n"
            "  ## Social Structures & Culture\n"
            "  ### Institutions & Power Dynamics\n"
            "  ## Geography & Environment\n"
            "  ## Religion & Philosophy\n"
            "  ## Timeline & Continuity\n"
            "  ### Chronology (canonical)\n"
            "  ### Continuity Rules (no-retcon constraints)\n"
            "  ## Series Synopsis\n"
            "  ### Book 1\n"
            "  ### Book 2\n"
            "  ... (as needed)\n"
            "  ## Character References\n"
            "  ### Cast Integration & Constraints\n"
            "  ## Change Log (Rule Evolution)\n\n"
            "RULES FOR EDITS:\n"
            "1) Make focused, surgical edits near the cursor/selection unless the user requests re-organization.\n"
            "2) Maintain the scaffold above; if missing, create only the minimal sections the user asked for.\n"
            "3) Prefer paragraph/sentence-level replacements; avoid large-span rewrites unless asked.\n"
            "4) Enforce consistency: cross-check constraints against Series Synopsis and Characters.\n\n"
            "ANCHOR REQUIREMENTS (CRITICAL):\n"
            "For EVERY operation, you MUST provide precise anchors:\n\n"
            "REVISE/DELETE Operations:\n"
            "- ALWAYS include 'original_text' with EXACT, VERBATIM text from the file\n"
            "- Minimum 10-20 words, include complete sentences with natural boundaries\n"
            "- Copy and paste directly - do NOT retype or modify\n"
            "- ⚠️ NEVER include header lines (###, ##, #) in original_text!\n"
            "- OR provide both 'left_context' and 'right_context' (exact surrounding text)\n\n"
            "INSERT Operations (PREFERRED for adding content below headers!):\n"
            "- **PRIMARY METHOD**: Use op_type='insert_after_heading' with anchor_text='## Section' when adding content below ANY header\n"
            "- Provide 'anchor_text' with EXACT, COMPLETE header line to insert after (verbatim from file)\n"
            "- This is the SAFEST method - it NEVER deletes headers, always inserts AFTER them\n"
            "- Use this for adding rules, constraints, systems, or any worldbuilding content below headers\n"
            "- ALTERNATIVE: Provide 'original_text' with text to insert after\n"
            "- FALLBACK: Provide 'left_context' with text before insertion point (minimum 10-20 words)\n\n"
            "Additional Options:\n"
            "- 'occurrence_index' if text appears multiple times (0-based, default 0)\n"
            "- Start/end indices are approximate; anchors take precedence\n\n"
            "=== DECISION TREE FOR OPERATION TYPE ===\n"
            "1. Section is COMPLETELY EMPTY? → insert_after_heading with anchor_text=\"## Section\"\n"
            "2. Section has PLACEHOLDER or existing rules to replace? → replace_range (NO headers in original_text!)\n"
            "3. Deleting SPECIFIC content? → delete_range with original_text (NO headers!)\n\n"
            "⚠️ CRITICAL: When replacing placeholder content, use 'replace_range' on ONLY the placeholder!\n"
            "⚠️ NEVER include headers in 'original_text' for replace_range - headers will be deleted!\n"
            "✅ Correct: {\"op_type\": \"replace_range\", \"original_text\": \"[Placeholder rules]\", \"text\": \"- Rule 1\\n- Rule 2\"}\n"
            "❌ Wrong: {\"op_type\": \"insert_after_heading\"} when placeholder text exists - it will keep the placeholder!\n\n"
            "=== SPACING RULES (CRITICAL - READ CAREFULLY!) ===\n"
            "YOUR TEXT MUST END IMMEDIATELY AFTER THE LAST CHARACTER!\n\n"
            '✅ CORRECT: "- Rule 1\\n- Rule 2\\n- Rule 3"  ← Ends after "3" with NO \\n\n'
            '❌ WRONG: "- Rule 1\\n- Rule 2\\n"  ← Extra \\n after last line creates blank line!\n'
            '❌ WRONG: "- Rule 1\\n- Rule 2\\n\\n"  ← \\n\\n creates 2 blank lines!\n'
            '❌ WRONG: "- Rule 1\\n\\n- Rule 2"  ← Double \\n\\n between items creates blank line!\n\n'
            "IRON-CLAD RULE: After last line = ZERO \\n (nothing!)\n"
            "5) Headings must be clear; do not duplicate sections. If an equivalent heading exists, update it in place.\n"
            "6) When adding Timeline & Continuity entries, keep a chronological order and explicit constraints (MUST/MUST NOT).\n"
            "7) When adding Series Synopsis entries, keep book-by-book bullets with continuity notes.\n"
            "8) NO PLACEHOLDER FILLERS: If a requested section has no content yet, create the heading only and leave the body blank. Do NOT insert placeholders like '[To be developed]' or 'TBD'.\n"
        )
    
    async def _prepare_context_node(self, state: RulesEditingState) -> Dict[str, Any]:
        """Prepare context: extract active editor, validate rules type"""
        try:
            logger.info("Preparing context for rules editing...")
            
            shared_memory = state.get("shared_memory", {}) or {}
            active_editor = shared_memory.get("active_editor", {}) or {}
            
            rules = active_editor.get("content", "") or ""
            filename = active_editor.get("filename") or "rules.md"
            frontmatter = active_editor.get("frontmatter", {}) or {}
            cursor_offset = int(active_editor.get("cursor_offset", -1))
            selection_start = int(active_editor.get("selection_start", -1))
            selection_end = int(active_editor.get("selection_end", -1))
            
            # STRICT GATE: require explicit frontmatter.type == 'rules'
            doc_type = ""
            if isinstance(frontmatter, dict):
                doc_type = str(frontmatter.get("type") or "").strip().lower()
            if doc_type != "rules":
                logger.info(f"Rules Agent Gate: Detected type='{doc_type}' (expected 'rules'); skipping.")
                return {
                    "error": "Active editor is not a Rules document; rules agent skipping.",
                    "task_status": "error",
                    "response": {
                        "response": "Active editor is not a Rules document; rules agent skipping.",
                        "task_status": "error",
                        "agent_type": "rules_editing_agent"
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
            
            # Get paragraph bounds
            normalized_text = rules.replace("\r\n", "\n")
            body_only = _strip_frontmatter_block(normalized_text)
            para_start, para_end = paragraph_bounds(normalized_text, cursor_offset if cursor_offset >= 0 else 0)
            if selection_start >= 0 and selection_end > selection_start:
                para_start, para_end = selection_start, selection_end
            
            return {
                "active_editor": active_editor,
                "rules": normalized_text,
                "filename": filename,
                "frontmatter": frontmatter,
                "cursor_offset": cursor_offset,
                "selection_start": selection_start,
                "selection_end": selection_end,
                "body_only": body_only,
                "para_start": para_start,
                "para_end": para_end,
                "current_request": current_request.strip()
            }
            
        except Exception as e:
            logger.error(f"Failed to prepare context: {e}")
            return {
                "error": str(e),
                "task_status": "error"
            }
    
    async def _load_references_node(self, state: RulesEditingState) -> Dict[str, Any]:
        """Load referenced context files (style, characters) directly from rules frontmatter"""
        try:
            logger.info("Loading referenced context files from rules frontmatter...")
            
            from orchestrator.tools.reference_file_loader import load_referenced_files
            
            active_editor = state.get("active_editor", {})
            user_id = state.get("user_id", "system")
            
            # Rules reference configuration - load directly from rules' frontmatter (no cascading)
            reference_config = {
                "style": ["style"],
                "characters": ["characters", "character_*"]  # Support both list and individual keys
            }
            
            # Use unified loader (no cascade_config - rules loads directly)
            result = await load_referenced_files(
                active_editor=active_editor,
                user_id=user_id,
                reference_config=reference_config,
                doc_type_filter="rules",
                cascade_config=None  # No cascading for rules
            )
            
            loaded_files = result.get("loaded_files", {})
            
            # Extract content from loaded files
            style_body = None
            if loaded_files.get("style") and len(loaded_files["style"]) > 0:
                style_body = loaded_files["style"][0].get("content", "")
                if style_body:
                    style_body = _strip_frontmatter_block(style_body)
            
            characters_bodies = []
            if loaded_files.get("characters"):
                for char_file in loaded_files["characters"]:
                    char_content = char_file.get("content", "")
                    if char_content:
                        char_content = _strip_frontmatter_block(char_content)
                        characters_bodies.append(char_content)
            
            return {
                "style_body": style_body,
                "characters_bodies": characters_bodies
            }
            
        except Exception as e:
            logger.error(f"Failed to load references: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "style_body": None,
                "characters_bodies": [],
                "error": str(e)
            }
    
    async def _generate_edit_plan_node(self, state: RulesEditingState) -> Dict[str, Any]:
        """Generate edit plan using LLM"""
        try:
            logger.info("Generating rules edit plan...")
            
            rules = state.get("rules", "")
            filename = state.get("filename", "rules.md")
            body_only = state.get("body_only", "")
            current_request = state.get("current_request", "")
            
            style_body = state.get("style_body")
            characters_bodies = state.get("characters_bodies", [])
            
            para_start = state.get("para_start", 0)
            selection_start = state.get("selection_start", -1)
            selection_end = state.get("selection_end", -1)
            
            # Build system prompt
            system_prompt = self._build_system_prompt()
            
            # Build context message
            context_parts = [
                "=== RULES CONTEXT ===\n",
                f"File: {filename}\n\n",
                "Current Buffer (frontmatter stripped):\n" + body_only + "\n\n"
            ]
            
            if style_body:
                context_parts.append(f"=== STYLE GUIDE ===\n{style_body}\n\n")
            
            if characters_bodies:
                context_parts.append("".join([f"=== CHARACTER DOC ===\n{b}\n\n" for b in characters_bodies]))
            
            # Check for revision mode
            revision_mode = current_request and any(k in current_request.lower() for k in ["revise", "revision", "tweak", "adjust", "polish", "tighten"])
            if revision_mode:
                context_parts.append("REVISION MODE: Apply minimal targeted edits per the user's rubric. Prefer paragraph-level replace_range ops.\n\n")
            
            context_parts.append("Maintain the FORMATTING CONTRACT sections where present; if creating new content, adhere to the scaffold. Provide a ManuscriptEdit JSON plan strictly within scope.")
            
            messages = [
                SystemMessage(content=system_prompt),
                SystemMessage(content=f"Current Date/Time: {datetime.now().isoformat()}"),
                HumanMessage(content="".join(context_parts))
            ]
            
            if current_request:
                messages.append(HumanMessage(content=(
                    f"USER REQUEST: {current_request}\n\n"
                    "ANCHORING: For replace/delete, include 'original_text' or both 'left_context' and 'right_context' (<=60 chars each)."
                )))
            
            # Call LLM
            llm = self._get_llm(temperature=0.3, state=state)
            start_time = datetime.now()
            response = await llm.ainvoke(messages)
            
            content = response.content if hasattr(response, 'content') else str(response)
            content = _unwrap_json_response(content)
            
            # Parse structured edit
            structured_edit = None
            try:
                raw = json.loads(content)
                if isinstance(raw, dict) and isinstance(raw.get("operations"), list):
                    raw.setdefault("target_filename", filename)
                    raw.setdefault("scope", "paragraph")
                    raw.setdefault("summary", "Planned rules edit generated from context.")
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
                    "error": "Failed to produce a valid Rules edit plan. Ensure ONLY raw JSON ManuscriptEdit with operations is returned.",
                    "task_status": "error"
                }
            
            # Check edit intent - if no edit intent, suppress operations
            edit_intent = False
            try:
                lr = (current_request or "").lower()
                edit_intent = any(k in lr for k in [
                    "edit", "change", "replace", "update", "revise", "rewrite",
                    "insert", "add ", "delete", "adjust", "fix ", "fix:", "modify"
                ])
            except Exception:
                edit_intent = False
            if not edit_intent:
                structured_edit["operations"] = []
            
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
    
    async def _resolve_operations_node(self, state: RulesEditingState) -> Dict[str, Any]:
        """Resolve editor operations with progressive search"""
        try:
            logger.info("Resolving editor operations...")
            
            rules = state.get("rules", "")
            structured_edit = state.get("structured_edit")
            selection_start = state.get("selection_start", -1)
            selection_end = state.get("selection_end", -1)
            para_start = state.get("para_start", 0)
            para_end = state.get("para_end", 0)
            current_request = state.get("current_request", "")
            
            if not structured_edit or not isinstance(structured_edit.get("operations"), list):
                return {
                    "editor_operations": [],
                    "error": "No operations to resolve",
                    "task_status": "error"
                }
            
            fm_end_idx = _frontmatter_end_index(rules)
            selection = {"start": selection_start, "end": selection_end} if selection_start >= 0 and selection_end >= 0 else None
            
            # Check revision mode
            revision_mode = current_request and any(k in current_request.lower() for k in ["revise", "revision", "tweak", "adjust", "polish", "tighten", "edit only"])
            
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
                        rules,
                        op,
                        selection=selection,
                        frontmatter_end=fm_end_idx
                    )
                    
                    logger.info(f"Resolved {op.get('op_type')} [{resolved_start}:{resolved_end}] confidence={resolved_confidence:.2f}")
                    
                    # Protect YAML frontmatter
                    if resolved_start < fm_end_idx:
                        if op.get("op_type") == "delete_range":
                            continue
                        if resolved_end <= fm_end_idx:
                            resolved_start = fm_end_idx
                            resolved_end = fm_end_idx
                        else:
                            resolved_start = fm_end_idx
                    
                    # Clamp to selection/paragraph in revision mode
                    if revision_mode and op.get("op_type") != "delete_range":
                        if selection_start >= 0 and selection_end > selection_start:
                            resolved_start = max(selection_start, resolved_start)
                            resolved_end = min(selection_end, resolved_end)
                        else:
                            resolved_start = max(para_start, resolved_start)
                            resolved_end = min(para_end, resolved_end)
                    
                    resolved_start = max(0, min(len(rules), resolved_start))
                    resolved_end = max(resolved_start, min(len(rules), resolved_end))
                    
                    # Handle spacing for inserts
                    if resolved_start == resolved_end:
                        left_tail = rules[max(0, resolved_start-2):resolved_start]
                        if left_tail.endswith("\n\n"):
                            needed_prefix = ""
                        elif left_tail.endswith("\n"):
                            needed_prefix = "\n"
                        else:
                            needed_prefix = "\n\n"
                        try:
                            leading_stripped = re.sub(r'^\n+', '', resolved_text)
                            resolved_text = f"{needed_prefix}{leading_stripped}"
                        except Exception:
                            resolved_text = f"{needed_prefix}{resolved_text}"
                    
                    # Calculate pre_hash
                    pre_slice = rules[resolved_start:resolved_end]
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
                        "occurrence_index": op.get("occurrence_index", 0)
                    }
                    
                    editor_operations.append(resolved_op)
                    
                except Exception as e:
                    logger.warning(f"Failed to resolve operation: {e}")
                    continue
            
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
    
    async def _format_response_node(self, state: RulesEditingState) -> Dict[str, Any]:
        """Format final response with editor operations"""
        try:
            logger.info("Formatting rules editing response...")
            
            structured_edit = state.get("structured_edit")
            editor_operations = state.get("editor_operations", [])
            current_request = state.get("current_request", "")
            
            if not structured_edit:
                error = state.get("error", "Unknown error")
                return {
                    "response": {
                        "response": f"Failed to generate rules edit plan: {error}",
                        "task_status": "error",
                        "agent_type": "rules_editing_agent"
                    },
                    "task_status": "error"
                }
            
            # Build preview text
            preview = "\n\n".join([op.get("text", "").strip() for op in editor_operations if op.get("text", "").strip()])
            response_text = preview if preview else (structured_edit.get("summary") or "Edit plan ready.")
            
            # Build response dict
            response_dict = {
                "response": response_text,
                "task_status": "complete",
                "agent_type": "rules_editing_agent"
            }
            
            # Add editor operations if present
            if editor_operations:
                response_dict["editor_operations"] = editor_operations
                response_dict["manuscript_edit"] = {
                    **structured_edit,
                    "operations": editor_operations
                }
                response_dict["content_preview"] = response_text[:2000]
            
            # Add assistant message to state for checkpointing
            updated_state = self._add_assistant_response_to_messages(state, response_text)
            
            return {
                "response": response_dict,
                "task_status": "complete",
                **updated_state
            }
            
        except Exception as e:
            logger.error(f"Failed to format response: {e}")
            return {
                "response": {
                    "response": f"Failed to format response: {str(e)}",
                    "task_status": "error",
                    "agent_type": "rules_editing_agent"
                },
                "task_status": "error",
                "error": str(e)
            }
    
    async def process(self, query: str, metadata: Dict[str, Any] = None, messages: List[Any] = None) -> Dict[str, Any]:
        """Process rules editing query using LangGraph workflow"""
        try:
            logger.info(f"Rules editing agent processing: {query[:100]}...")
            
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
            initial_state: RulesEditingState = {
                "query": query,
                "user_id": user_id,
                "metadata": metadata,
                "messages": conversation_messages,
                "shared_memory": shared_memory_merged,
                "active_editor": {},
                "rules": "",
                "filename": "rules.md",
                "frontmatter": {},
                "cursor_offset": -1,
                "selection_start": -1,
                "selection_end": -1,
                "body_only": "",
                "para_start": 0,
                "para_end": 0,
                "style_body": None,
                "characters_bodies": [],
                "current_request": "",
                "system_prompt": "",
                "llm_response": "",
                "structured_edit": None,
                "editor_operations": [],
                "response": {},
                "task_status": "",
                "error": ""
            }
            
            # Invoke LangGraph workflow with checkpointing
            final_state = await workflow.ainvoke(initial_state, config=config)
            
            # Return response from final state
            return final_state.get("response", {
                "response": "Rules editing failed",
                "task_status": "error",
                "agent_type": "rules_editing_agent"
            })
            
        except Exception as e:
            logger.error(f"Rules Editing Agent failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "response": f"Rules editing failed: {str(e)}",
                "task_status": "error",
                "agent_type": "rules_editing_agent"
            }


# Singleton instance
_rules_editing_agent_instance = None


def get_rules_editing_agent() -> RulesEditingAgent:
    """Get global rules editing agent instance"""
    global _rules_editing_agent_instance
    if _rules_editing_agent_instance is None:
        _rules_editing_agent_instance = RulesEditingAgent()
    return _rules_editing_agent_instance

