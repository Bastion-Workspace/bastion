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
from orchestrator.utils.editor_operation_resolver import resolve_editor_operation

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

# Removed: _resolve_operation_simple - now using centralized resolver from orchestrator.utils.editor_operation_resolver


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
    request_type: str  # "question" or "edit_request"
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
        workflow.add_node("detect_request_type", self._detect_request_type_node)
        workflow.add_node("generate_edit_plan", self._generate_edit_plan_node)
        workflow.add_node("resolve_operations", self._resolve_operations_node)
        workflow.add_node("format_response", self._format_response_node)
        
        # Entry point
        workflow.set_entry_point("prepare_context")
        
        # Flow: prepare_context -> load_references -> detect_request_type -> generate_edit_plan -> resolve_operations -> format_response -> END
        workflow.add_edge("prepare_context", "load_references")
        workflow.add_edge("load_references", "detect_request_type")
        workflow.add_edge("detect_request_type", "generate_edit_plan")
        workflow.add_edge("generate_edit_plan", "resolve_operations")
        workflow.add_edge("resolve_operations", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow.compile(checkpointer=checkpointer)
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for rules editing"""
        return (
            "You are a MASTER UNIVERSE ARCHITECT for RULES documents (worldbuilding, series continuity). "
            "Persona disabled. Adhere strictly to frontmatter, project Rules, and Style.\n\n"
            "**CRITICAL: WORK WITH AVAILABLE INFORMATION FIRST**\n"
            "Always start by working with what you know from the request, existing rules content, and references:\n"
            "- Make edits based on available information - don't wait for clarification\n"
            "- Use context from style guide and character profiles to inform your work\n"
            "- Add or revise content based on reasonable inferences from the request\n"
            "- Only ask questions when critical information is missing that prevents you from making meaningful progress\n"
            "\n"
            "**WHEN TO ASK QUESTIONS (Rarely - Only When Truly Necessary)**:\n"
            "- Only when the request is so vague that you cannot make ANY reasonable edits (e.g., 'improve rules' with no existing content)\n"
            "- Only when there's a critical conflict that requires user decision (e.g., existing rule directly contradicts new request)\n"
            "- When asking, provide operations for what you CAN do, then ask questions in the summary about what you need\n"
            "\n"
            "**HOW TO ASK QUESTIONS**: Include operations for work you CAN do, then add questions/suggestions in the summary field.\n"
            "DO NOT return empty operations array - always provide edits based on available information.\n\n"
            "STRUCTURED OUTPUT REQUIRED: Return ONLY raw JSON (no prose, no markdown, no code fences) matching this schema:\n"
            "{\n"
            '  "type": "ManuscriptEdit",\n'
            '  "target_filename": string,\n'
            '  "scope": one of ["paragraph", "chapter", "multi_chapter"],\n'
            '  "summary": string,\n'
            '  "chapter_index": integer|null,\n'
            '  "safety": one of ["low", "medium", "high"],\n'
            '  "operations": [ { "op_type": one of ["replace_range", "delete_range", "insert_after_heading", "insert_after"], "start": integer, "end": integer, "text": string } ]\n'
            "}\n\n"
            "OUTPUT RULES:\n"
            "- Output MUST be a single JSON object only.\n"
            "- Do NOT include triple backticks or language tags.\n"
            "- Do NOT include explanatory text before or after the JSON.\n"
            "- If asking questions/seeking clarification: Return empty operations array and put questions in summary field\n"
            "- If making edits: Return operations array with edits and brief description in summary field\n\n"
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
            "  ### Cast Integration & Constraints\n\n"
            "RULES FOR EDITS:\n"
            "0) **WORK FIRST, ASK LATER**: Always make edits based on available information. Use context from the request, existing rules content, style guide, and character profiles to inform your work. Only ask questions in the summary if critical information is missing that prevents meaningful progress. Never return empty operations unless the request is completely impossible.\n"
            "1) Make focused, surgical edits near the cursor/selection unless the user requests re-organization.\n"
            "2) Maintain the scaffold above; if missing, create only the minimal sections the user asked for.\n"
            "3) Prefer paragraph/sentence-level replacements; avoid large-span rewrites unless asked.\n"
            "4) Enforce consistency: cross-check constraints against Series Synopsis and Characters.\n"
            "5) **CREATIVE EXPANSION REQUIRED** - When the user provides concepts or ideas, you MUST:\n"
            "   - Expand creatively on the user's concepts - do NOT simply repeat them verbatim\n"
            "   - Add logical implications, consequences, and edge cases\n"
            "   - Elaborate on how concepts interact with existing rules and systems\n"
            "   - Provide examples, specific scenarios, and detailed explanations\n"
            "   - Think through the full implications of the concepts (what does this mean for other systems?)\n"
            "   - Add depth and nuance that makes the rules comprehensive and useful\n"
            "   - Example: User says 'Magic requires components' → Expand to: 'Magic requires physical components that are consumed during casting. Components must be gathered or purchased. Different spells require different components. Components degrade over time if not properly stored. Rare components are expensive and difficult to obtain.'\n"
            "   - Example: User says 'Time period is medieval' → Expand to: 'Time period is medieval (approximately 1000-1500 CE equivalent). Technology level includes basic metallurgy, agriculture, and simple mechanical devices. No gunpowder or advanced machinery. Social structure is feudal with nobility, clergy, and commoners. Transportation is primarily by horse, cart, or foot.'\n"
            "   - The goal is to create rich, detailed rules that go beyond the user's initial concept\n\n"
            "6) **CRITICAL: ORGANIZE AND CONSOLIDATE RULES** - Before adding rules, you MUST:\n"
            "   - **Check for DUPLICATES**: Scan the ENTIRE document to see if this rule already exists somewhere\n"
            "   - **Identify the BEST LOCATION**: Determine which section is most appropriate for each rule\n"
            "   - **CONSOLIDATE duplicates**: If a rule exists in multiple places, keep it in the MOST appropriate section and DELETE it from others\n"
            "   - **MOVE misplaced rules**: If a rule is in the wrong section, DELETE it from the wrong place and ADD it to the right place\n"
            "   - **Avoid redundancy**: Do NOT add the same rule to multiple sections unless there's a specific reason for cross-referencing\n"
            "   - Example: If 'Magic requires components' exists under Systems AND Universe Constraints, keep it in Systems (more specific) and delete from Universe Constraints\n"
            "   - Example: If adding a geography rule but it's already in Background section, MOVE it to Geography section (delete_range from Background, insert_after_heading in Geography)\n"
            "   - Example: If the same timeline event is in both Timeline AND Series Synopsis, consolidate - keep detailed version in Timeline, brief reference in Series Synopsis\n\n"
            "7) **CRITICAL: CROSS-REFERENCE RELATED RULES** - When adding or updating a concept, you MUST:\n"
            "   - Scan the ENTIRE document for related rules that should be updated together\n"
            "   - Identify ALL sections that reference or relate to the concept being added/updated\n"
            "   - Generate MULTIPLE operations if a single concept addition requires updates to multiple related rules\n"
            "   - Example: If adding a magic system rule, check Systems, Universe Constraints, Timeline, and Series Synopsis sections\n"
            "   - Example: If updating a character constraint, check Character References, Series Synopsis, and Continuity Rules\n"
            "   - Example: If adding a timeline event, check Chronology, Continuity Rules, and Series Synopsis for consistency\n"
            "   - NEVER update only one rule when related rules exist that should be updated together\n"
            "   - The operations array can contain MULTIPLE operations - use it to update all related sections in one pass\n\n"
            "ANCHOR REQUIREMENTS (CRITICAL):\n"
            "For EVERY operation, you MUST provide precise anchors:\n\n"
            "REVISE/DELETE Operations:\n"
            "- ALWAYS include 'original_text' with EXACT, VERBATIM text from the file\n"
            "- Minimum 10-20 words, include complete sentences with natural boundaries\n"
            "- Copy and paste directly - do NOT retype or modify\n"
            "- NEVER include header lines (###, ##, #) in original_text!\n"
            "- OR provide both 'left_context' and 'right_context' (exact surrounding text)\n\n"
            "INSERT Operations (ONLY for truly empty sections!):\n"
            "- **insert_after_heading**: Use ONLY when section is completely empty below the header\n"
            "  * op_type='insert_after_heading' with anchor_text='## Section' (exact header line)\n"
            "  * Example: Adding rules after '## Magic System' header when section is completely empty\n"
            "  * ⚠️ CRITICAL WARNING: Before using insert_after_heading, you MUST verify the section is COMPLETELY EMPTY!\n"
            "  * ⚠️ If there is ANY text below the header (even a single line), use replace_range instead!\n"
            "  * ⚠️ Using insert_after_heading when content exists will INSERT BETWEEN the header and existing text, splitting the section!\n"
            "  * ⚠️ This creates duplicate content and breaks the section structure - NEVER do this!\n"
            "  * Example of WRONG behavior: '## Magic\\n[INSERT HERE splits section]\\n- Existing rule' ← WRONG! Use replace_range on existing content!\n"
            "  * Example of CORRECT usage: '## Magic\\n[empty - no text below]' ← OK to use insert_after_heading\n"
            "  * This is the SAFEST method - it NEVER deletes headers, always inserts AFTER them - BUT ONLY FOR EMPTY SECTIONS\n\n"
            "- **insert_after**: Use when continuing text mid-paragraph, mid-sentence, or after specific text\n"
            "  * op_type='insert_after' with anchor_text='last few words before insertion point'\n"
            "  * Example: Continuing a sentence or adding to an existing paragraph\n\n"
            "- **REPLACE Operations (PREFERRED for updating existing content!):\n"
            "- **replace_range**: Use when section exists but needs improvement, completion, or revision\n"
            "  * If section has ANY content (even incomplete or placeholder), use replace_range to update it\n"
            "  * Example: Section has '- Magic requires physical components' but needs more detail → replace_range with original_text='- Magic requires physical components' and expanded text\n"
            "  * Example: Section has '[To be developed]' → replace_range with original_text='[To be developed]' and actual content\n"
            "  * This ensures existing content is replaced/updated, not duplicated\n\n"
            "Additional Options:\n"
            "- 'occurrence_index' if text appears multiple times (0-based, default 0)\n"
            "- Start/end indices are approximate; anchors take precedence\n\n"
            "=== DECISION TREE FOR OPERATION TYPE ===\n"
            "**STEP 1: Read the section content carefully!**\n"
            "- Look at what exists below the header\n"
            "- Is there ANY text at all? Even a single line?\n"
            "\n"
            "**STEP 2: Choose operation based on what exists:**\n"
            "1. Section is COMPLETELY EMPTY below header (no text at all)? → insert_after_heading with anchor_text=\"## Section\"\n"
            "2. Section has ANY content (even incomplete/placeholder/single line)? → replace_range to update it (NO headers in original_text!)\n"
            "3. Adding to existing list/paragraph? → replace_range with original_text matching existing content\n"
            "4. Deleting SPECIFIC content? → delete_range with original_text (NO headers!)\n"
            "5. Continuing mid-sentence? → insert_after\n"
            "6. Rule exists in wrong section? → Two operations: delete_range from wrong section, then insert/replace in correct section\n"
            "7. Same rule exists in multiple sections? → Keep in most appropriate section, delete_range from others\n\n"
            "CRITICAL: When updating existing content (even if incomplete), use 'replace_range' on the existing content!\n"
            "NEVER include headers in 'original_text' for replace_range - headers will be deleted!\n"
            "⚠️ NEVER use insert_after_heading when content exists - it will SPLIT the section and create duplicates!\n"
            "\n"
            "**CORRECT EXAMPLES**:\n"
            "- Updating existing content: {\"op_type\": \"replace_range\", \"original_text\": \"- Magic requires physical components\", \"text\": \"- Magic requires physical components\\n- Components must be consumed during casting\"}\n"
            "- Moving rule to better section: [{\"op_type\": \"delete_range\", \"original_text\": \"- Magic uses verbal components\"}, {\"op_type\": \"insert_after_heading\", \"anchor_text\": \"## Magic Systems\", \"text\": \"- Magic uses verbal components\\n- Verbal components must be spoken clearly\"}]\n"
            "- Consolidating duplicates: [{\"op_type\": \"delete_range\", \"original_text\": \"- Dragons are rare creatures\" (from Background)}, {\"op_type\": \"replace_range\", \"original_text\": \"- Dragons exist\", \"text\": \"- Dragons are rare creatures found in mountain regions\" (in Geography)}]\n"
            "\n"
            "**WRONG EXAMPLES**:\n"
            "- ❌ {\"op_type\": \"insert_after_heading\"} when section has content - will split section!\n"
            "- ❌ Adding same rule to multiple sections without consolidating - creates duplicates!\n"
            "- ❌ Not checking if rule already exists elsewhere - creates redundancy!\n\n"
            "=== SPACING RULES (CRITICAL - READ CAREFULLY!) ===\n"
            "YOUR TEXT MUST END IMMEDIATELY AFTER THE LAST CHARACTER!\n\n"
            'CORRECT: "- Rule 1\\n- Rule 2\\n- Rule 3"  ← Ends after "3" with NO \\n\n'
            'WRONG: "- Rule 1\\n- Rule 2\\n"  ← Extra \\n after last line creates blank line!\n'
            'WRONG: "- Rule 1\\n- Rule 2\\n\\n"  ← \\n\\n creates 2 blank lines!\n'
            'WRONG: "- Rule 1\\n\\n- Rule 2"  ← Double \\n\\n between items creates blank line!\n\n'
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
                    },
                    # ✅ CRITICAL: Preserve state even on error
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", "")
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
                "characters_bodies": characters_bodies,
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
                "style_body": None,
                "characters_bodies": [],
                "error": str(e),
                # ✅ CRITICAL: Preserve state even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    def _route_from_request_type(self, state: RulesEditingState) -> str:
        """Route based on detected request type"""
        request_type = state.get("request_type", "edit_request")
        return request_type if request_type in ("question", "edit_request") else "edit_request"
    
    async def _detect_request_type_node(self, state: RulesEditingState) -> Dict[str, Any]:
        """Detect if request is a question or edit request"""
        try:
            logger.info("Detecting request type...")
            
            current_request = state.get("current_request", "")
            body_only = state.get("body_only", "")
            style_body = state.get("style_body")
            characters_bodies = state.get("characters_bodies", [])
            
            # Build simple prompt for LLM to determine intent
            prompt = f"""Analyze the user's request and determine if it's a QUESTION or an EDIT REQUEST.

**USER REQUEST**: {current_request}

**CONTEXT**:
- Current rules: {body_only[:500] if body_only else "Empty rules"}
- Has style reference: {bool(style_body)}
- Has {len(characters_bodies)} character reference(s)

**INTENT DETECTION**:
- QUESTIONS (including pure questions and conditional edits): User is asking a question - may or may not want edits
  - Pure questions: "What rules are defined?", "Do we have a rule about magic?", "Show me the worldbuilding rules", "What's our time period setting?"
  - Conditional edits: "Do we have magic rules? Add them if not", "What rules? Suggest additions if needed"
  - Questions often start with: "Do you", "What", "Can you", "Are there", "How many", "Show me", "Is", "Does", "Are we", "Suggest"
  - **Key insight**: Questions can be answered, and IF edits are needed based on the answer, they can be made
  - Route ALL questions to edit path - LLM can decide if edits are needed
  
- EDIT REQUESTS: User wants to create, modify, or generate content - NO question asked
  - Examples: "Add magic system rules", "Create worldbuilding rules", "Update the time period section", "Revise the geography rules"
  - Edit requests are action-oriented: "add", "create", "update", "generate", "change", "replace", "revise"
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
            llm = self._get_llm(temperature=0.1, state=state)
            from langchain_core.messages import HumanMessage
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            
            content = response.content if hasattr(response, 'content') else str(response)
            content = _unwrap_json_response(content)
            
            # Parse response
            try:
                result = json.loads(content)
                request_type = result.get("request_type", "edit_request")
                confidence = result.get("confidence", 0.5)
                reasoning = result.get("reasoning", "")
                
                logger.info(f"Request type detected: {request_type} (confidence: {confidence:.2f}) - {reasoning}")
                
                return {
                    "request_type": request_type,
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", "")
                }
            except Exception as e:
                logger.warning(f"Failed to parse request type detection: {e}, defaulting to edit_request")
                return {
                    "request_type": "edit_request",
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", "")
                }
            
        except Exception as e:
            logger.error(f"Failed to detect request type: {e}")
            return {
                "request_type": "edit_request",
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    async def _generate_edit_plan_node(self, state: RulesEditingState) -> Dict[str, Any]:
        """Generate edit plan using LLM"""
        try:
            logger.info("Generating rules edit plan...")
            
            rules = state.get("rules", "")
            filename = state.get("filename", "rules.md")
            body_only = state.get("body_only", "")
            current_request = state.get("current_request", "")
            request_type = state.get("request_type", "edit_request")
            is_question = request_type == "question"
            
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
            
            # Add mode-specific instructions
            if is_question:
                context_parts.append(
                    "\n=== QUESTION REQUEST: ANALYZE AND OPTIONALLY EDIT ===\n"
                    "The user has asked a question about the rules document.\n\n"
                    "**YOUR TASK**:\n"
                    "1. **ANALYZE FIRST**: Answer the user's question by evaluating the current content\n"
                    "   - Pure questions: 'What rules are defined?' → Report current rules\n"
                    "   - Verification questions: 'Do we have a rule about magic?' → Check for rule, report findings\n"
                    "   - Suggestion questions: 'Suggest additions to the rules' → Analyze current content, then suggest additions\n"
                    "   - Conditional questions: 'Do we have magic rules? Add them if not' → Check, then edit if needed\n"
                    "2. **THEN EDIT IF NEEDED**: Based on your analysis, make edits if necessary\n"
                    "   - If question implies a desired state ('Add them if not') → Provide editor operations\n"
                    "   - If question asks for suggestions ('Suggest additions') → Provide editor operations with suggested additions\n"
                    "   - If question is pure information ('What rules?') → No edits needed, just answer\n"
                    "   - Include your analysis in the 'summary' field of your response\n\n"
                    "**RESPONSE FORMAT**:\n"
                    "- In the 'summary' field: Answer the question clearly and explain your analysis\n"
                    "- In the 'operations' array: Provide editor operations ONLY if edits are needed\n"
                    "- If no edits needed: Return empty operations array, but answer the question in summary\n"
                    "- If edits needed: Provide operations AND explain what you found in summary\n\n"
                )
            else:
                # Edit request mode - add "WORK FIRST" guidance (like character agent)
                context_parts.append(
                    "\n=== EDIT REQUEST: WORK WITH AVAILABLE INFORMATION ===\n"
                    "The user wants you to add or revise rules content.\n\n"
                    "**YOUR APPROACH**:\n"
                    "1. **WORK FIRST**: Make edits based on the request and available context (rules file, style guide, characters)\n"
                    "2. **USE INFERENCE**: Make reasonable inferences from the request - don't wait for clarification\n"
                    "3. **ASK ALONG THE WAY**: If you need specific details, include questions in the summary AFTER describing the work you've done\n"
                    "4. **NEVER EMPTY OPERATIONS**: Always provide operations based on what you can determine from the request and context\n\n"
                )
                context_parts.append("Provide a ManuscriptEdit JSON plan for the rules document.")
            
            # Build request with mode-specific instructions
            request_with_instructions = ""
            if current_request:
                if is_question:
                    request_with_instructions = (
                        f"USER REQUEST: {current_request}\n\n"
                        "**QUESTION MODE**: Answer the question first, then provide edits if needed.\n\n"
                        "CRITICAL: CREATIVE EXPANSION AND ELABORATION (if edits are needed)\n"
                        "When the user provides concepts or ideas, you MUST expand creatively:\n"
                        "- Do NOT simply repeat the user's words verbatim - expand on their concepts\n"
                        "- Add logical implications, consequences, and edge cases\n"
                        "- Elaborate on how concepts interact with existing rules and systems\n"
                        "- Provide examples, specific scenarios, and detailed explanations\n"
                        "- Think through the full implications: What does this mean for other systems? How does this affect the world?\n"
                        "- Add depth and nuance that makes the rules comprehensive and useful\n"
                        "- Transform brief concepts into rich, detailed rules that go beyond the initial idea\n"
                        "\n"
                        "CRITICAL: ORGANIZE AND CONSOLIDATE RULES FIRST (if edits are needed)\n"
                        "Before adding ANY new content, you MUST:\n"
                        "1. **CHECK FOR DUPLICATES** - Does this rule already exist somewhere in the document?\n"
                        "2. **IDENTIFY BEST LOCATION** - Which section is most appropriate for this rule?\n"
                        "3. **CONSOLIDATE IF NEEDED** - If rule exists in multiple places, keep it in the MOST appropriate section and DELETE from others\n"
                        "4. **MOVE MISPLACED RULES** - If rule is in wrong section, DELETE from wrong place and ADD to right place\n"
                        "\n"
                        "CRITICAL: CROSS-REFERENCE RELATED RULES (if edits are needed)\n"
                        "After organizing, identify related rules:\n"
                        "1. **SCAN THE ENTIRE DOCUMENT** - Read through ALL sections to identify related rules\n"
                        "2. **IDENTIFY ALL AFFECTED SECTIONS** - When adding/updating a concept, find ALL places it should appear\n"
                        "3. **GENERATE MULTIPLE OPERATIONS** - If a concept affects multiple rules, create operations for EACH affected section\n"
                        "4. **ENSURE CONSISTENCY** - Related rules must be updated together to maintain document coherence\n"
                        "\n"
                        "CRITICAL ANCHORING INSTRUCTIONS (if edits are needed):\n"
                        "- **BEFORE using insert_after_heading**: Verify the section is COMPLETELY EMPTY (no text below header)\n"
                        "- **If section has ANY content**: Use replace_range to update it, NOT insert_after_heading\n"
                        "- **insert_after_heading will SPLIT sections**: If you use it when content exists, it inserts BETWEEN header and existing text!\n"
                        "- **UPDATING EXISTING CONTENT**: If a section exists but needs improvement/completion, use 'replace_range' with 'original_text' matching the EXISTING content\n"
                        "  * Example: Section has '- Magic requires physical components' but needs more → replace_range with original_text='- Magic requires physical components' and expanded text\n"
                        "  * Example: Section has placeholder '[To be developed]' → replace_range with original_text='[To be developed]' and actual content\n"
                        "- **ADDING TO EMPTY SECTIONS**: Only use 'insert_after_heading' when section is completely empty below the header\n"
                        "- For REVISE/DELETE: Provide 'original_text' with EXACT, VERBATIM text from file (10-20+ words, complete sentences)\n"
                        "- For INSERT: Use 'insert_after_heading' with 'anchor_text' ONLY for completely empty sections, or 'insert_after' for mid-paragraph\n"
                        "- NEVER include header lines in original_text for replace_range operations\n"
                        "- Copy text directly from the file - do NOT retype or paraphrase\n"
                        "- Without precise anchors, the operation WILL FAIL\n"
                        "- **KEY RULE**: If content exists (even if incomplete), use replace_range to update it. Only use insert_after_heading for truly empty sections.\n"
                        "- You can return MULTIPLE operations in the operations array - for organizing/consolidating AND for updating related sections"
                    )
                else:
                    request_with_instructions = (
                        f"USER REQUEST: {current_request}\n\n"
                        "**WORK FIRST**: Make edits based on the request and available context. Use reasonable inferences - don't wait for clarification. Only ask questions in the summary if critical information is truly missing.\n\n"
                        "CRITICAL: CREATIVE EXPANSION AND ELABORATION\n"
                        "When the user provides concepts or ideas, you MUST expand creatively:\n"
                        "- Do NOT simply repeat the user's words verbatim - expand on their concepts\n"
                        "- Add logical implications, consequences, and edge cases\n"
                        "- Elaborate on how concepts interact with existing rules and systems\n"
                        "- Provide examples, specific scenarios, and detailed explanations\n"
                        "- Think through the full implications: What does this mean for other systems? How does this affect the world?\n"
                        "- Add depth and nuance that makes the rules comprehensive and useful\n"
                        "- Transform brief concepts into rich, detailed rules that go beyond the initial idea\n"
                        "\n"
                        "Examples of creative expansion:\n"
                        "- User: 'Magic requires components' → You: 'Magic requires physical components that are consumed during casting. Components must be gathered from the environment or purchased from specialized merchants. Different spell types require different component categories (organic, mineral, elemental). Components degrade over time if not properly stored in enchanted containers. Rare components are expensive and difficult to obtain, limiting access to powerful magic. Component quality affects spell effectiveness - using inferior components may cause spell failure or reduced potency.'\n"
                        "- User: 'Time period is medieval' → You: 'Time period is medieval (approximately 1000-1500 CE equivalent). Technology level includes basic metallurgy, agriculture, and simple mechanical devices. No gunpowder or advanced machinery exists. Social structure is feudal with clear hierarchies: nobility (landowners and rulers), clergy (religious authorities), and commoners (peasants, craftsmen, merchants). Transportation is primarily by horse, cart, or foot. Communication is slow and relies on messengers or word-of-mouth. Most people are illiterate, with education limited to the upper classes and clergy.'\n"
                        "\n"
                        "CRITICAL: ORGANIZE AND CONSOLIDATE RULES FIRST\n"
                        "Before adding ANY new content, you MUST:\n"
                        "1. **CHECK FOR DUPLICATES** - Does this rule already exist somewhere in the document?\n"
                        "2. **IDENTIFY BEST LOCATION** - Which section is most appropriate for this rule?\n"
                        "3. **CONSOLIDATE IF NEEDED** - If rule exists in multiple places, keep it in the MOST appropriate section and DELETE from others\n"
                        "4. **MOVE MISPLACED RULES** - If rule is in wrong section, DELETE from wrong place and ADD to right place\n"
                        "\n"
                        "Examples of organization operations:\n"
                        "- User adds 'Magic requires components' but it already exists in Background → DELETE from Background, ADD expanded version to Magic Systems section\n"
                        "- Same timeline event in Timeline AND Series Synopsis → Keep detailed version in Timeline, brief reference in Series Synopsis\n"
                        "- Geography rule in Background section → MOVE to Geography section (delete from Background, insert in Geography)\n"
                        "\n"
                        "CRITICAL: CROSS-REFERENCE RELATED RULES\n"
                        "After organizing, identify related rules:\n"
                        "1. **SCAN THE ENTIRE DOCUMENT** - Read through ALL sections to identify related rules\n"
                        "2. **IDENTIFY ALL AFFECTED SECTIONS** - When adding/updating a concept, find ALL places it should appear\n"
                        "3. **GENERATE MULTIPLE OPERATIONS** - If a concept affects multiple rules, create operations for EACH affected section\n"
                        "4. **ENSURE CONSISTENCY** - Related rules must be updated together to maintain document coherence\n"
                        "\n"
                        "Examples of when to generate multiple operations:\n"
                        "- Adding magic system → Update 'Magic Systems' section AND 'Universe Constraints' section if they reference each other\n"
                        "- Adding character constraint → Update 'Character References' AND 'Series Synopsis' if character appears there\n"
                        "- Adding timeline event → Update 'Chronology' AND 'Continuity Rules' AND 'Series Synopsis' if event affects plot\n"
                        "- Updating a concept → If concept appears in multiple sections, update ALL occurrences, not just one\n"
                        "\n"
                        "CRITICAL ANCHORING INSTRUCTIONS:\n"
                        "- **BEFORE using insert_after_heading**: Verify the section is COMPLETELY EMPTY (no text below header)\n"
                        "- **If section has ANY content**: Use replace_range to update it, NOT insert_after_heading\n"
                        "- **insert_after_heading will SPLIT sections**: If you use it when content exists, it inserts BETWEEN header and existing text!\n"
                        "- For REVISE/DELETE: Provide 'original_text' with EXACT, VERBATIM text from file (10-20+ words, complete sentences)\n"
                        "- For INSERT: Use 'insert_after_heading' with 'anchor_text' ONLY for completely empty sections, or 'insert_after' for mid-paragraph\n"
                        "- NEVER include header lines in original_text for replace_range operations\n"
                        "- Copy text directly from the file - do NOT retype or paraphrase\n"
                        "- Without precise anchors, the operation WILL FAIL\n"
                        "- For granular revisions, use replace_range with exact original_text matching the text to change\n"
                        "- You can return MULTIPLE operations in the operations array - one for organizing/consolidating, others for updates"
                    )
            
            # Use standardized helper for message construction with conversation history
            messages_list = state.get("messages", [])
            messages = self._build_editing_agent_messages(
                system_prompt=system_prompt,
                context_parts=context_parts,
                current_request=request_with_instructions,
                messages_list=messages_list,
                look_back_limit=6
            )
            
            # Call LLM (higher temperature for creative expansion)
            llm = self._get_llm(temperature=0.5, state=state)
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
                        if op_type not in ("replace_range", "delete_range", "insert_after_heading", "insert_after"):
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
            
            # Trust the LLM: It understands semantic intent and will generate operations when appropriate.
            # If the user asks a question, the LLM will return empty operations. If they want edits/generation,
            # the LLM will generate operations. No brittle keyword matching needed.
            
            # Log what we got from the LLM
            ops_count = len(structured_edit.get("operations", [])) if structured_edit else 0
            logger.info(f"LLM generated {ops_count} operation(s)")
            if ops_count > 0:
                for i, op in enumerate(structured_edit.get("operations", [])):
                    op_type = op.get("op_type", "unknown")
                    text_preview = (op.get("text", "") or "")[:100]
                    logger.info(f"  Operation {i+1}: {op_type}, text preview: {text_preview}...")
            
            return {
                "llm_response": content,
                "structured_edit": structured_edit,
                "system_prompt": system_prompt
            }
            
        except Exception as e:
            logger.error(f"Failed to generate edit plan: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Use base agent's error handler to surface OpenRouter errors properly
            return self._handle_node_error(e, state, "Edit plan generation")
    
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
            
            # Check if file is empty (only frontmatter)
            body_only = _strip_frontmatter_block(rules)
            is_empty_file = not body_only.strip()
            
            # Check revision mode
            revision_mode = current_request and any(k in current_request.lower() for k in ["revise", "revision", "tweak", "adjust", "polish", "tighten", "edit only"])
            
            editor_operations = []
            operations = structured_edit.get("operations", [])
            
            logger.info(f"Resolving {len(operations)} operation(s) from structured_edit")
            
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
                        content=rules,
                        op_dict=op,
                        selection=selection,
                        frontmatter_end=fm_end_idx,
                        cursor_offset=cursor_pos
                    )
                    
                    # Special handling for empty files: ensure operations insert after frontmatter
                    if is_empty_file and resolved_start < fm_end_idx:
                        resolved_start = fm_end_idx
                        resolved_end = fm_end_idx
                        resolved_confidence = 0.7
                        logger.info(f"Empty file detected - adjusting operation to insert after frontmatter at {fm_end_idx}")
                    
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
            
            logger.info(f"Successfully resolved {len(editor_operations)} operation(s) out of {len(operations)}")
            
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
            request_type = state.get("request_type", "edit_request")
            
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
            
            # Build preview text from operations (for logging purposes)
            preview = "\n\n".join([op.get("text", "").strip() for op in editor_operations if op.get("text", "").strip()])
            
            # Handle questions - prioritize summary (answer/analysis) over operation text
            if request_type == "question":
                summary = structured_edit.get("summary", "") if structured_edit else ""
                if summary and len(summary.strip()) > 20:
                    # Question with answer - use summary as response text (conversational feedback)
                    logger.info(f"Question request with {len(editor_operations)} operations - using summary as conversational response")
                    response_text = summary
                elif editor_operations:
                    # Question with operations but no summary - create fallback
                    logger.warning("Question request with operations but no summary - using fallback")
                    response_text = f"Analysis complete. Made {len(editor_operations)} edit(s) based on your question."
                else:
                    # Pure question with no operations - use summary or fallback
                    response_text = summary if summary else "Analysis complete."
            else:
                # Edit request - use summary if available, otherwise operation preview
                summary = structured_edit.get("summary", "") if structured_edit else ""
                if summary and len(summary.strip()) > 20:
                    response_text = summary
                else:
                    # Use preview text from operations
                    response_text = preview if preview else "Edit plan ready."
            
            logger.info(f"Response formatting: {len(editor_operations)} operation(s), preview length: {len(preview)}, response_text: {response_text[:200]}...")
            
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
            
            # Note: Messages are handled by LangGraph checkpointing automatically
            # No need to manually add them here (consistent with fiction_editing_agent)
            
            return {
                "response": response_dict,
                "task_status": "complete"
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
                "request_type": "edit_request",
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
            
            # Extract final response
            response = final_state.get("response", {})
            task_status = final_state.get("task_status", "complete")
            
            # Debug logging
            logger.info(f"Final state response type: {type(response)}, keys: {response.keys() if isinstance(response, dict) else 'not a dict'}")
            if isinstance(response, dict):
                logger.info(f"Response dict has 'response' key: {'response' in response}, value type: {type(response.get('response'))}, value preview: {str(response.get('response', ''))[:200]}")
            
            if task_status == "error":
                error_msg = final_state.get("error", "Unknown error")
                logger.error(f"Rules editing agent failed: {error_msg}")
                return {
                    "response": f"Rules editing failed: {error_msg}",
                    "task_status": "error",
                    "agent_results": {}
                }
            
            # Extract response text - handle nested structure
            response_text = response.get("response", "") if isinstance(response, dict) else str(response) if response else ""
            if not response_text:
                response_text = "Rules editing complete"  # Fallback only if truly empty
            
            # Build result dict matching fiction_editing_agent pattern
            result = {
                "response": response_text,
                "task_status": task_status,
                "agent_results": {
                    "editor_operations": response.get("editor_operations", []) if isinstance(response, dict) else [],
                    "manuscript_edit": response.get("manuscript_edit") if isinstance(response, dict) else None
                }
            }
            
            # Add editor operations at top level for compatibility with gRPC service
            editor_ops_from_response = response.get("editor_operations", []) if isinstance(response, dict) else []
            manuscript_edit_from_response = response.get("manuscript_edit") if isinstance(response, dict) else None
            
            logger.info(f"Extracting operations: found {len(editor_ops_from_response)} operation(s) in response dict")
            
            if editor_ops_from_response:
                result["editor_operations"] = editor_ops_from_response
                logger.info(f"✅ Added {len(editor_ops_from_response)} editor operation(s) to result")
            if manuscript_edit_from_response:
                result["manuscript_edit"] = manuscript_edit_from_response
                logger.info(f"✅ Added manuscript_edit to result")
            
            logger.info(f"Rules editing agent completed: {task_status}, result keys: {result.keys()}, has editor_ops: {'editor_operations' in result}")
            return result
            
        except Exception as e:
            logger.error(f"Rules Editing Agent failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Use base agent's error response creator to surface OpenRouter errors properly
            error_response = self._create_error_response(str(e))
            error_response["agent_type"] = "rules_editing_agent"
            return error_response


# Singleton instance
_rules_editing_agent_instance = None


def get_rules_editing_agent() -> RulesEditingAgent:
    """Get global rules editing agent instance"""
    global _rules_editing_agent_instance
    if _rules_editing_agent_instance is None:
        _rules_editing_agent_instance = RulesEditingAgent()
    return _rules_editing_agent_instance

