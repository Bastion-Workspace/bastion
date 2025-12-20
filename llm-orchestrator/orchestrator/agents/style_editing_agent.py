"""
Style Editing Agent - LangGraph Implementation
Gated to style documents. Consumes active editor buffer, cursor/selection, and
referenced rules/characters from frontmatter. Supports two modes:
1. Analysis Mode: Analyze narrative examples to generate style guide
2. Editing Mode: Edit existing style guides using structured operations.
Produces EditorOperations suitable for Prefer Editor HITL application.
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


def _detect_analysis_mode(user_request: str, body_only: str) -> Tuple[bool, str]:
    """
    Detect if user wants to analyze narrative examples.
    Returns (is_analysis_mode, extracted_examples)
    """
    if not user_request:
        return False, ""
    
    request_lower = user_request.lower()
    
    # Check for explicit analysis keywords
    analysis_keywords = [
        "analyze", "analysis", "generate style", "create style guide",
        "extract style", "from these examples", "based on this", "style from"
    ]
    
    has_analysis_keyword = any(kw in request_lower for kw in analysis_keywords)
    
    # Check if request contains substantial narrative prose (not just instructions)
    # Look for indicators: descriptive passages, narrative flow, POV patterns
    narrative_indicators = [
        len(user_request) > 200,  # Substantial text
        '"' in user_request or "'" in user_request,  # Potential dialogue/quotes
        any(word in request_lower for word in ["he", "she", "they", "I", "we", "character", "narrator"]),
        "\n" in user_request,  # Multi-line (likely prose)
    ]
    
    has_narrative_content = sum(narrative_indicators) >= 2
    
    # If file is empty and user provides substantial text, likely analysis mode
    is_empty_file = not body_only.strip()
    
    if (has_analysis_keyword or (has_narrative_content and is_empty_file)):
        # Extract examples - try to separate instructions from examples
        # If user says "analyze this:" or similar, extract what follows
        example_markers = ["analyze", "examples:", "this:", "these:", "from:"]
        extracted = user_request
        
        for marker in example_markers:
            if marker in request_lower:
                idx = request_lower.find(marker)
                extracted = user_request[idx + len(marker):].strip()
                break
        
        return True, extracted
    
    return False, ""


# ============================================
# LangGraph State
# ============================================

class StyleEditingState(TypedDict):
    """State for style editing agent LangGraph workflow"""
    query: str
    user_id: str
    metadata: Dict[str, Any]
    messages: List[Any]
    shared_memory: Dict[str, Any]
    active_editor: Dict[str, Any]
    style: str
    filename: str
    frontmatter: Dict[str, Any]
    cursor_offset: int
    selection_start: int
    selection_end: int
    body_only: str
    para_start: int
    para_end: int
    rules_body: Optional[str]
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
    # Analysis mode tracking
    analysis_mode: bool
    narrative_examples: str


# ============================================
# Style Editing Agent
# ============================================

class StyleEditingAgent(BaseAgent):
    """
    Style Editing Agent for narrative style guide development
    
    Gated to style documents. Consumes full style body (frontmatter stripped),
    loads Rules and Character references directly from this file's frontmatter,
    and emits editor operations for Prefer Editor HITL.
    Supports two modes:
    1. Analysis Mode: Analyze narrative examples to generate style guide
    2. Editing Mode: Edit existing style guides
    Uses LangGraph workflow for explicit state management
    """
    
    def __init__(self):
        super().__init__("style_editing_agent")
        logger.info("Style Editing Agent ready!")
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """Build LangGraph workflow for style editing agent"""
        workflow = StateGraph(StyleEditingState)
        
        # Add nodes
        workflow.add_node("prepare_context", self._prepare_context_node)
        workflow.add_node("load_references", self._load_references_node)
        workflow.add_node("detect_request_type", self._detect_request_type_node)
        workflow.add_node("detect_mode", self._detect_mode_node)
        workflow.add_node("analyze_examples", self._analyze_examples_node)
        workflow.add_node("generate_edit_plan", self._generate_edit_plan_node)
        workflow.add_node("resolve_operations", self._resolve_operations_node)
        workflow.add_node("format_response", self._format_response_node)
        
        # Entry point
        workflow.set_entry_point("prepare_context")
        
        # Flow: prepare_context -> load_references -> detect_request_type -> detect_mode -> (analyze_examples if analysis mode) -> generate_edit_plan -> resolve_operations -> format_response -> END
        workflow.add_edge("prepare_context", "load_references")
        workflow.add_edge("load_references", "detect_request_type")
        workflow.add_edge("detect_request_type", "detect_mode")
        workflow.add_conditional_edges(
            "detect_mode",
            lambda state: "analyze_examples" if state.get("analysis_mode", False) else "generate_edit_plan",
            {
                "analyze_examples": "analyze_examples",
                "generate_edit_plan": "generate_edit_plan"
            }
        )
        workflow.add_edge("analyze_examples", "generate_edit_plan")
        workflow.add_edge("generate_edit_plan", "resolve_operations")
        workflow.add_edge("resolve_operations", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow.compile(checkpointer=checkpointer)
    
    def _build_system_prompt(self, analysis_mode: bool = False) -> str:
        """Build system prompt for style editing"""
        base_prompt = (
            "You are a MASTER STYLE ARCHITECT for STYLE GUIDE documents (narrative voice, technique, craft). "
            "Persona disabled. Adhere strictly to frontmatter, project Rules, and Style.\n\n"
            "**CRITICAL: ASK CLARIFYING QUESTIONS WHEN NEEDED**\n"
            "If the user's request is ambiguous, incomplete, or could benefit from clarification:\n"
            "- Ask specific, targeted questions to understand style preferences\n"
            "- Suggest style considerations the user might not have thought of\n"
            "- Point out potential conflicts with existing style guidelines\n"
            "- Propose style refinements that would make the guide more useful\n"
            "Examples of when to ask questions:\n"
            "- User: 'Add POV guidelines' → Ask: 'Which POV? (first-person, third-person limited, third-person omniscient?) Single POV or multiple? How strict are the boundaries?'\n"
            "- User: 'Add pacing guidelines' → Ask: 'What pacing style? (fast-paced thriller, slow literary, varied?) Scene-level or sentence-level pacing techniques?'\n"
            "- User adds conflicting guideline → Point out: 'This conflicts with existing guideline X. Should I update the old guideline, or are these context-dependent?'\n"
            "\n"
            "**HOW TO ASK QUESTIONS**: Return operations array EMPTY and use summary field for questions/suggestions.\n\n"
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
        )
        
        if analysis_mode:
            base_prompt += (
                "=== ANALYSIS MODE ===\n"
                "You are analyzing narrative examples to extract style characteristics and generate a style guide.\n\n"
                "**CRITICAL**: The Style Guide is HOW to write, not WHAT happens. Focus on:\n"
                "- Narrative voice and tone (not plot or story content)\n"
                "- Point of View (POV) technique\n"
                "- Tense usage\n"
                "- Pacing patterns\n"
                "- Sensory detail level\n"
                "- Sentence structure and rhythm\n"
                "- Descriptive techniques\n"
                "- Show-don't-tell approach\n\n"
                "**DO NOT include**:\n"
                "- Dialogue style (that's character-specific, belongs in character profiles)\n"
                "- Plot elements or story content\n"
                "- Character-specific voice patterns\n\n"
            )
        else:
            base_prompt += (
                "FORMATTING CONTRACT (STYLE DOCUMENTS):\n"
                "- Never emit YAML frontmatter in operations[].text. Preserve existing frontmatter as-is.\n"
                "- Use Markdown headings and lists for the body.\n"
                "- When creating or normalizing structure, prefer this scaffold (top-level headings):\n"
                "  ## Narrative Voice\n"
                "  ## Point of View\n"
                "  ## Tense\n"
                "  ## Pacing\n"
                "  ## Sensory Detail Level\n"
                "  ## Sentence Structure\n"
                "  ## Descriptive Techniques\n"
                "  ## Rhythm and Flow\n"
                "  ## Notes\n\n"
                "**IMPORTANT**: Dialogue style is character-specific and belongs in character profiles, NOT in the overall style guide.\n\n"
            )
        
        base_prompt += (
            "RULES FOR EDITS:\n"
            "0) **ASK QUESTIONS WHEN NEEDED**: If request is ambiguous, lacks style detail, or has implications the user may not have considered, ASK instead of guessing. Return empty operations and use summary for questions.\n"
            "1) Make focused, surgical edits near the cursor/selection unless the user requests re-organization.\n"
            "2) Maintain the scaffold above; if missing, create only the minimal sections the user asked for.\n"
            "3) Prefer paragraph/sentence-level replacements; avoid large-span rewrites unless asked.\n"
            "4) Enforce consistency: cross-check against Rules and Characters for worldbuilding alignment.\n"
            "5) **CRITICAL: CROSS-REFERENCE RELATED SECTIONS** - When adding or updating style guidelines, you MUST:\n"
            "   - Scan the ENTIRE document for related sections that should be updated together\n"
            "   - Identify ALL sections that reference or relate to the style guideline being added/updated\n"
            "   - Generate MULTIPLE operations if a single guideline addition requires updates to multiple related sections\n"
            "   - Example: If adding POV guidelines, check Point of View section AND Narrative Voice section if they reference each other\n"
            "   - Example: If updating pacing techniques, check Pacing section AND Rhythm and Flow section if they overlap\n"
            "   - Example: If adding descriptive techniques, check Descriptive Techniques section AND Sentence Structure section if they relate\n"
            "   - NEVER update only one section when related sections exist that should be updated together\n"
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
            "  * Example: Adding style guidelines after '## Narrative Voice' header when section is completely empty\n"
            "  * ⚠️ CRITICAL WARNING: Before using insert_after_heading, you MUST verify the section is COMPLETELY EMPTY!\n"
            "  * ⚠️ If there is ANY text below the header (even a single line), use replace_range instead!\n"
            "  * ⚠️ Using insert_after_heading when content exists will INSERT BETWEEN the header and existing text, splitting the section!\n"
            "  * ⚠️ This creates duplicate content and breaks the section structure - NEVER do this!\n"
            "  * Example of WRONG behavior: '## Pacing\\n[INSERT HERE splits section]\\n- Existing guideline' ← WRONG! Use replace_range on existing content!\n"
            "  * Example of CORRECT usage: '## Pacing\\n[empty - no text below]' ← OK to use insert_after_heading\n"
            "  * This is the SAFEST method - it NEVER deletes headers, always inserts AFTER them - BUT ONLY FOR EMPTY SECTIONS\n\n"
            "- **insert_after**: Use when continuing text mid-paragraph, mid-sentence, or after specific text\n"
            "  * op_type='insert_after' with anchor_text='last few words before insertion point'\n"
            "  * Example: Continuing a sentence or adding to an existing paragraph\n\n"
            "- **REPLACE Operations (PREFERRED for updating existing content!):\n"
            "- **replace_range**: Use when section exists but needs improvement, completion, or revision\n"
            "  * If section has ANY content (even incomplete or placeholder), use replace_range to update it\n"
            "  * Example: Section has '- Third-person limited' but needs more detail → replace_range with original_text='- Third-person limited' and expanded text\n"
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
            "5. Continuing mid-sentence? → insert_after\n\n"
            "CRITICAL: When updating existing content (even if incomplete), use 'replace_range' on the existing content!\n"
            "NEVER include headers in 'original_text' for replace_range - headers will be deleted!\n"
            "⚠️ NEVER use insert_after_heading when content exists - it will SPLIT the section and create duplicates!\n"
            "\n"
            "**CORRECT EXAMPLE**:\n"
            "- Updating existing content: {\"op_type\": \"replace_range\", \"original_text\": \"- Third-person limited\", \"text\": \"- Third-person limited\\n- Focus on single character's perspective\"}\n"
            "\n"
            "**WRONG EXAMPLE**:\n"
            "- ❌ {\"op_type\": \"insert_after_heading\"} when section has content - will split section!\n\n"
            "=== SPACING RULES (CRITICAL - READ CAREFULLY!) ===\n"
            "YOUR TEXT MUST END IMMEDIATELY AFTER THE LAST CHARACTER!\n\n"
            'CORRECT: "- Item 1\\n- Item 2\\n- Item 3"  ← Ends after "3" with NO \\n\n'
            'WRONG: "- Item 1\\n- Item 2\\n"  ← Extra \\n after last line creates blank line!\n'
            'WRONG: "- Item 1\\n- Item 2\\n\\n"  ← \\n\\n creates 2 blank lines!\n'
            'WRONG: "- Item 1\\n\\n- Item 2"  ← Double \\n\\n between items creates blank line!\n\n'
            "IRON-CLAD RULE: After last line = ZERO \\n (nothing!)\n"
            "5) Headings must be clear; do not duplicate sections. If an equivalent heading exists, update it in place.\n"
            "6) NO PLACEHOLDER FILLERS: If a requested section has no content yet, create the heading only and leave the body blank. Do NOT insert placeholders like '[To be developed]' or 'TBD'.\n"
        )
        
        return base_prompt
    
    async def _prepare_context_node(self, state: StyleEditingState) -> Dict[str, Any]:
        """Prepare context: extract active editor, validate style type"""
        try:
            logger.info("Preparing context for style editing...")
            
            shared_memory = state.get("shared_memory", {}) or {}
            active_editor = shared_memory.get("active_editor", {}) or {}
            
            style = active_editor.get("content", "") or ""
            filename = active_editor.get("filename") or "style.md"
            frontmatter = active_editor.get("frontmatter", {}) or {}
            cursor_offset = int(active_editor.get("cursor_offset", -1))
            selection_start = int(active_editor.get("selection_start", -1))
            selection_end = int(active_editor.get("selection_end", -1))
            
            # STRICT GATE: require explicit frontmatter.type == 'style'
            doc_type = ""
            if isinstance(frontmatter, dict):
                doc_type = str(frontmatter.get("type") or "").strip().lower()
            if doc_type != "style":
                logger.info(f"Style Agent Gate: Detected type='{doc_type}' (expected 'style'); skipping.")
                return {
                    "error": "Active editor is not a Style document; style agent skipping.",
                    "task_status": "error",
                    "response": {
                        "response": "Active editor is not a Style document; style agent skipping.",
                        "task_status": "error",
                        "agent_type": "style_editing_agent"
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
            normalized_text = style.replace("\r\n", "\n")
            body_only = _strip_frontmatter_block(normalized_text)
            para_start, para_end = paragraph_bounds(normalized_text, cursor_offset if cursor_offset >= 0 else 0)
            if selection_start >= 0 and selection_end > selection_start:
                para_start, para_end = selection_start, selection_end
            
            return {
                "active_editor": active_editor,
                "style": normalized_text,
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
    
    async def _load_references_node(self, state: StyleEditingState) -> Dict[str, Any]:
        """Load referenced context files (rules, characters) directly from style frontmatter"""
        try:
            logger.info("Loading referenced context files from style frontmatter...")
            
            from orchestrator.tools.reference_file_loader import load_referenced_files
            
            active_editor = state.get("active_editor", {})
            user_id = state.get("user_id", "system")
            
            # Style reference configuration - load directly from style's frontmatter (no cascading)
            reference_config = {
                "rules": ["rules"],
                "characters": ["characters", "character_*"]  # Support both list and individual keys
            }
            
            # Use unified loader (no cascade_config - style loads directly)
            result = await load_referenced_files(
                active_editor=active_editor,
                user_id=user_id,
                reference_config=reference_config,
                doc_type_filter="style",
                cascade_config=None  # No cascading for style
            )
            
            loaded_files = result.get("loaded_files", {})
            
            # Extract content from loaded files
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
                "rules_body": None,
                "characters_bodies": [],
                "error": str(e),
                # ✅ CRITICAL: Preserve state even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    def _route_from_request_type(self, state: StyleEditingState) -> str:
        """Route based on detected request type"""
        request_type = state.get("request_type", "edit_request")
        return request_type if request_type in ("question", "edit_request") else "edit_request"
    
    async def _detect_request_type_node(self, state: StyleEditingState) -> Dict[str, Any]:
        """Detect if request is a question or edit request"""
        try:
            logger.info("Detecting request type...")
            
            current_request = state.get("current_request", "")
            body_only = state.get("body_only", "")
            rules_body = state.get("rules_body")
            characters_bodies = state.get("characters_bodies", [])
            
            # Build simple prompt for LLM to determine intent
            prompt = f"""Analyze the user's request and determine if it's a QUESTION or an EDIT REQUEST.

**USER REQUEST**: {current_request}

**CONTEXT**:
- Current style guide: {body_only[:500] if body_only else "Empty style guide"}
- Has rules reference: {bool(rules_body)}
- Has {len(characters_bodies)} character reference(s)

**INTENT DETECTION**:
- QUESTIONS (including pure questions and conditional edits): User is asking a question - may or may not want edits
  - Pure questions: "What style rules do we have?", "Does this follow our style guide?", "Show me the narrative voice guidelines", "What POV are we using?"
  - Conditional edits: "Do we have POV guidelines? Add them if not", "What style rules? Suggest improvements if needed"
  - Questions often start with: "Do you", "What", "Can you", "Are there", "How many", "Show me", "Is", "Does", "Are we", "Suggest"
  - **Key insight**: Questions can be answered, and IF edits are needed based on the answer, they can be made
  - Route ALL questions to edit path - LLM can decide if edits are needed
  
- EDIT REQUESTS: User wants to create, modify, or generate content - NO question asked
  - Examples: "Add POV guidelines", "Create a style guide", "Update the narrative voice section", "Revise the dialogue patterns"
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
    
    async def _detect_mode_node(self, state: StyleEditingState) -> Dict[str, Any]:
        """Detect if we're in analysis mode or editing mode"""
        try:
            logger.info("Detecting mode (analysis vs editing)...")
            
            current_request = state.get("current_request", "")
            body_only = state.get("body_only", "")
            
            analysis_mode, narrative_examples = _detect_analysis_mode(current_request, body_only)
            
            logger.info(f"Mode detected: {'ANALYSIS' if analysis_mode else 'EDITING'}")
            if analysis_mode:
                logger.info(f"Extracted narrative examples length: {len(narrative_examples)} chars")
            
            return {
                "analysis_mode": analysis_mode,
                "narrative_examples": narrative_examples,
                # ✅ CRITICAL: Preserve state for subsequent nodes
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.error(f"Failed to detect mode: {e}")
            return {
                "analysis_mode": False,
                "narrative_examples": "",
                # ✅ CRITICAL: Preserve state even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    async def _analyze_examples_node(self, state: StyleEditingState) -> Dict[str, Any]:
        """Analyze narrative examples to extract style characteristics"""
        try:
            logger.info("Analyzing narrative examples to extract style characteristics...")
            
            narrative_examples = state.get("narrative_examples", "")
            current_request = state.get("current_request", "")
            rules_body = state.get("rules_body")
            characters_bodies = state.get("characters_bodies", [])
            
            if not narrative_examples:
                logger.warning("No narrative examples to analyze")
                return {
                    # ✅ CRITICAL: Preserve state even when skipping
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", "")
                }
            
            # Build analysis prompt
            analysis_prompt = (
                "=== NARRATIVE EXAMPLES TO ANALYZE ===\n"
                f"{narrative_examples}\n\n"
                "=== TASK ===\n"
                "Analyze these narrative examples and extract style characteristics. Focus on:\n\n"
                "1. **Narrative Voice**: What is the tone, personality, and voice? (formal, casual, poetic, gritty, etc.)\n"
                "2. **Point of View**: What POV is used? (first-person, third-person limited, third-person omniscient, etc.)\n"
                "3. **Tense**: What tense is used? (past, present, mixed)\n"
                "4. **Pacing**: How is pacing handled? (fast, slow, varied, with specific techniques)\n"
                "5. **Sensory Detail Level**: How much sensory detail? (minimal, moderate, rich)\n"
                "6. **Sentence Structure**: What patterns in sentence length, rhythm, and flow?\n"
                "7. **Descriptive Techniques**: How is description handled? (show-don't-tell, metaphor usage, imagery style)\n"
                "8. **Rhythm and Flow**: How are paragraphs structured? What pacing techniques and transitions are used?\n\n"
                "**CRITICAL**: Do NOT include dialogue style - that's character-specific and belongs in character profiles.\n\n"
                "Generate a style guide following this structure:\n"
                "## Narrative Voice\n"
                "[Voice description, tone, personality]\n\n"
                "## Point of View\n"
                "[POV type and technique]\n\n"
                "## Tense\n"
                "[Tense usage and patterns]\n\n"
                "## Pacing\n"
                "[Pacing characteristics and techniques]\n\n"
                "## Sensory Detail Level\n"
                "[Level of sensory description]\n\n"
                "## Sentence Structure\n"
                "[Sentence length patterns, rhythm, flow]\n\n"
                "## Descriptive Techniques\n"
                "[Show-don't-tell approach, metaphor usage, imagery style]\n\n"
                "## Rhythm and Flow\n"
                "[Paragraph structure, pacing techniques, transitions]\n\n"
                "## Notes\n"
                "[Additional style considerations, techniques, or guidelines]\n\n"
            )
            
            if rules_body:
                analysis_prompt += f"=== UNIVERSE RULES (for context) ===\n{rules_body[:500]}...\n\n"
            
            if characters_bodies:
                analysis_prompt += f"=== CHARACTER PROFILES (for context - dialogue style is character-specific) ===\n{characters_bodies[0][:300]}...\n\n"
            
            analysis_prompt += (
                "Return ONLY the style guide content (markdown format with headings and content). "
                "Do NOT include frontmatter. Do NOT include dialogue style sections."
            )
            
            # Call LLM for analysis
            llm = self._get_llm(temperature=0.7, state=state)  # Higher temperature for creative analysis
            
            datetime_context = self._get_datetime_context()
            messages = [
                SystemMessage(content="You are a style analysis expert. Analyze narrative examples and extract style characteristics to create a comprehensive style guide."),
                SystemMessage(content=datetime_context),
                HumanMessage(content=analysis_prompt)
            ]
            
            response = await llm.ainvoke(messages)
            analyzed_content = response.content if hasattr(response, 'content') else str(response)
            
            logger.info(f"Analysis complete, generated {len(analyzed_content)} chars of style guide content")
            
            # Store analyzed content for use in edit plan generation
            return {
                "narrative_examples": narrative_examples,
                "analyzed_style_content": analyzed_content,
                # ✅ CRITICAL: Preserve state for subsequent nodes
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze examples: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "analyzed_style_content": "",
                "error": str(e),
                # ✅ CRITICAL: Preserve state even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    async def _generate_edit_plan_node(self, state: StyleEditingState) -> Dict[str, Any]:
        """Generate edit plan using LLM"""
        try:
            logger.info("Generating style edit plan...")
            
            style = state.get("style", "")
            filename = state.get("filename", "style.md")
            body_only = state.get("body_only", "")
            current_request = state.get("current_request", "")
            analysis_mode = state.get("analysis_mode", False)
            analyzed_style_content = state.get("analyzed_style_content", "")
            request_type = state.get("request_type", "edit_request")
            is_question = request_type == "question"
            
            rules_body = state.get("rules_body")
            characters_bodies = state.get("characters_bodies", [])
            
            para_start = state.get("para_start", 0)
            selection_start = state.get("selection_start", -1)
            selection_end = state.get("selection_end", -1)
            
            # Build system prompt (analysis mode or editing mode)
            system_prompt = self._build_system_prompt(analysis_mode=analysis_mode)
            
            # Build context message
            context_parts = [
                "=== STYLE GUIDE CONTEXT ===\n",
                f"File: {filename}\n\n",
            ]
            
            if analysis_mode and analyzed_style_content:
                # Analysis mode: use analyzed content
                context_parts.append("=== ANALYZED STYLE GUIDE CONTENT ===\n")
                context_parts.append(f"{analyzed_style_content}\n\n")
                context_parts.append("Provide a ManuscriptEdit JSON plan to insert this style guide content into the document.\n")
                context_parts.append("If the file is empty, insert after frontmatter. If sections exist, update them appropriately.\n")
            else:
                # Editing mode: use current content
                context_parts.append("Current Buffer (frontmatter stripped):\n" + body_only + "\n\n")
                
                # Add mode-specific instructions
                if is_question:
                    context_parts.append(
                        "\n=== QUESTION REQUEST: ANALYZE AND OPTIONALLY EDIT ===\n"
                        "The user has asked a question about the style guide.\n\n"
                        "**YOUR TASK**:\n"
                        "1. **ANALYZE FIRST**: Answer the user's question by evaluating the current content\n"
                        "   - Pure questions: 'What style rules do we have?' → Report current style guidelines\n"
                        "   - Verification questions: 'Does this follow our style guide?' → Check compliance, report findings\n"
                        "   - Suggestion questions: 'Suggest improvements to the style guide' → Analyze current content, then suggest additions\n"
                        "   - Conditional questions: 'Do we have POV guidelines? Add them if not' → Check, then edit if needed\n"
                        "2. **THEN EDIT IF NEEDED**: Based on your analysis, make edits if necessary\n"
                        "   - If question implies a desired state ('Add them if not') → Provide editor operations\n"
                        "   - If question asks for suggestions ('Suggest improvements') → Provide editor operations with suggested additions\n"
                        "   - If question is pure information ('What style rules?') → No edits needed, just answer\n"
                        "   - Include your analysis in the 'summary' field of your response\n\n"
                        "**RESPONSE FORMAT**:\n"
                        "- In the 'summary' field: Answer the question clearly and explain your analysis\n"
                        "- In the 'operations' array: Provide editor operations ONLY if edits are needed\n"
                        "- If no edits needed: Return empty operations array, but answer the question in summary\n"
                        "- If edits needed: Provide operations AND explain what you found in summary\n\n"
                    )
                else:
                    context_parts.append("Provide a ManuscriptEdit JSON plan for the style guide document.\n")
            
            if rules_body:
                context_parts.append(f"=== UNIVERSE RULES (for consistency) ===\n{rules_body[:500]}...\n\n")
            
            if characters_bodies:
                context_parts.append("=== CHARACTER PROFILES (for context) ===\n")
                context_parts.append(f"{characters_bodies[0][:300]}...\n\n")
            
            # Build request with mode-specific instructions
            request_with_instructions = ""
            if current_request:
                if is_question:
                    request_with_instructions = (
                        f"USER REQUEST: {current_request}\n\n"
                        "**QUESTION MODE**: Answer the question first, then provide edits if needed.\n\n"
                        "CRITICAL: CROSS-REFERENCE AND MULTIPLE OPERATIONS (if edits are needed)\n"
                        "Before generating operations, you MUST:\n"
                        "1. **SCAN THE ENTIRE DOCUMENT** - Read through ALL sections to identify related style guidelines\n"
                        "2. **IDENTIFY ALL AFFECTED SECTIONS** - When adding/updating a style guideline, find ALL places it should appear\n"
                        "3. **GENERATE MULTIPLE OPERATIONS** - If a guideline affects multiple sections, create operations for EACH affected section\n"
                        "4. **ENSURE CONSISTENCY** - Related sections must be updated together to maintain style guide coherence\n"
                        "\n"
                        "CRITICAL ANCHORING INSTRUCTIONS (if edits are needed):\n"
                        "- **BEFORE using insert_after_heading**: Verify the section is COMPLETELY EMPTY (no text below header)\n"
                        "- **If section has ANY content**: Use replace_range to update it, NOT insert_after_heading\n"
                        "- **insert_after_heading will SPLIT sections**: If you use it when content exists, it inserts BETWEEN header and existing text!\n"
                        "- **UPDATING EXISTING CONTENT**: If a section exists but needs improvement/completion, use 'replace_range' with 'original_text' matching the EXISTING content\n"
                        "  * Example: Section has '- Third-person limited' but needs more → replace_range with original_text='- Third-person limited' and expanded text\n"
                        "  * Example: Section has placeholder '[To be developed]' → replace_range with original_text='[To be developed]' and actual content\n"
                        "- **ADDING TO EMPTY SECTIONS**: Only use 'insert_after_heading' when section is completely empty below the header\n"
                        "- For REVISE/DELETE: Provide 'original_text' with EXACT, VERBATIM text from file (10-20+ words, complete sentences)\n"
                        "- For INSERT: Use 'insert_after_heading' with 'anchor_text' ONLY for completely empty sections, or 'insert_after' for mid-paragraph\n"
                        "- NEVER include header lines in original_text for replace_range operations\n"
                        "- Copy text directly from the file - do NOT retype or paraphrase\n"
                        "- Without precise anchors, the operation WILL FAIL\n"
                        "- **KEY RULE**: If content exists (even if incomplete), use replace_range to update it. Only use insert_after_heading for truly empty sections.\n"
                        "- You can return MULTIPLE operations in the operations array - one for each related section that needs updating"
                    )
                else:
                    request_with_instructions = (
                        f"USER REQUEST: {current_request}\n\n"
                        "CRITICAL: CROSS-REFERENCE AND MULTIPLE OPERATIONS\n"
                        "Before generating operations, you MUST:\n"
                        "1. **SCAN THE ENTIRE DOCUMENT** - Read through ALL sections to identify related style guidelines\n"
                        "2. **IDENTIFY ALL AFFECTED SECTIONS** - When adding/updating a style guideline, find ALL places it should appear\n"
                        "3. **GENERATE MULTIPLE OPERATIONS** - If a guideline affects multiple sections, create operations for EACH affected section\n"
                        "4. **ENSURE CONSISTENCY** - Related sections must be updated together to maintain style guide coherence\n"
                        "\n"
                        "Examples of when to generate multiple operations:\n"
                        "- Adding POV guidelines → Update 'Point of View' section AND 'Narrative Voice' section if they reference each other\n"
                        "- Adding pacing techniques → Update 'Pacing' section AND 'Rhythm and Flow' section if they overlap\n"
                        "- Adding descriptive techniques → Update 'Descriptive Techniques' section AND 'Sentence Structure' section if they relate\n"
                        "- Updating a style guideline → If guideline appears in multiple sections, update ALL occurrences, not just one\n"
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
                        "- You can return MULTIPLE operations in the operations array - one for each related section that needs updating"
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
                    raw.setdefault("summary", "Planned style guide edit generated from context.")
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
                    "task_status": "error",
                    # ✅ CRITICAL: Preserve state even on error
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", "")
                }
            
            if structured_edit is None:
                return {
                    "llm_response": content,
                    "structured_edit": None,
                    "error": "Failed to produce a valid Style guide edit plan. Ensure ONLY raw JSON ManuscriptEdit with operations is returned.",
                    "task_status": "error",
                    # ✅ CRITICAL: Preserve state even on error
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", "")
                }
            
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
                "system_prompt": system_prompt,
                # ✅ CRITICAL: Preserve state for subsequent nodes
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.error(f"Failed to generate edit plan: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "llm_response": "",
                "structured_edit": None,
                "error": str(e),
                "task_status": "error",
                # ✅ CRITICAL: Preserve state even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    async def _resolve_operations_node(self, state: StyleEditingState) -> Dict[str, Any]:
        """Resolve editor operations with progressive search"""
        try:
            logger.info("Resolving editor operations...")
            
            style = state.get("style", "")
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
                    "task_status": "error",
                    # ✅ CRITICAL: Preserve state even on error
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", "")
                }
            
            fm_end_idx = _frontmatter_end_index(style)
            selection = {"start": selection_start, "end": selection_end} if selection_start >= 0 and selection_end >= 0 else None
            
            # Check if file is empty (only frontmatter)
            body_only = _strip_frontmatter_block(style)
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
                        content=style,
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
                    
                    resolved_start = max(0, min(len(style), resolved_start))
                    resolved_end = max(resolved_start, min(len(style), resolved_end))
                    
                    # Handle spacing for inserts
                    if resolved_start == resolved_end:
                        left_tail = style[max(0, resolved_start-2):resolved_start]
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
                    pre_slice = style[resolved_start:resolved_end]
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
                "editor_operations": editor_operations,
                # ✅ CRITICAL: Preserve state for subsequent nodes
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.error(f"Failed to resolve operations: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "editor_operations": [],
                "error": str(e),
                "task_status": "error",
                # ✅ CRITICAL: Preserve state even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    async def _format_response_node(self, state: StyleEditingState) -> Dict[str, Any]:
        """Format final response with editor operations"""
        try:
            logger.info("Formatting style editing response...")
            
            structured_edit = state.get("structured_edit")
            editor_operations = state.get("editor_operations", [])
            current_request = state.get("current_request", "")
            request_type = state.get("request_type", "edit_request")
            
            if not structured_edit:
                error = state.get("error", "Unknown error")
                return {
                    "response": {
                        "response": f"Failed to generate style guide edit plan: {error}",
                        "task_status": "error",
                        "agent_type": "style_editing_agent"
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
                "agent_type": "style_editing_agent"
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
                    "agent_type": "style_editing_agent"
                },
                "task_status": "error",
                "error": str(e)
            }
    
    async def process(self, query: str, metadata: Dict[str, Any] = None, messages: List[Any] = None) -> Dict[str, Any]:
        """Process style editing query using LangGraph workflow"""
        try:
            logger.info(f"Style editing agent processing: {query[:100]}...")
            
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
            initial_state: StyleEditingState = {
                "query": query,
                "user_id": user_id,
                "metadata": metadata,
                "messages": conversation_messages,
                "shared_memory": shared_memory_merged,
                "active_editor": {},
                "style": "",
                "filename": "style.md",
                "frontmatter": {},
                "cursor_offset": -1,
                "selection_start": -1,
                "selection_end": -1,
                "body_only": "",
                "para_start": 0,
                "para_end": 0,
                "rules_body": None,
                "characters_bodies": [],
                "current_request": "",
                "request_type": "edit_request",
                "system_prompt": "",
                "llm_response": "",
                "structured_edit": None,
                "editor_operations": [],
                "response": {},
                "task_status": "",
                "error": "",
                "analysis_mode": False,
                "narrative_examples": ""
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
                logger.error(f"Style editing agent failed: {error_msg}")
                return {
                    "response": f"Style editing failed: {error_msg}",
                    "task_status": "error",
                    "agent_results": {}
                }
            
            # Extract response text - handle nested structure
            response_text = response.get("response", "") if isinstance(response, dict) else str(response) if response else ""
            if not response_text:
                response_text = "Style editing complete"  # Fallback only if truly empty
            
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
            
            logger.info(f"Style editing agent completed: {task_status}, result keys: {result.keys()}, has editor_ops: {'editor_operations' in result}")
            return result
            
        except Exception as e:
            logger.error(f"Style Editing Agent failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "response": f"Style editing failed: {str(e)}",
                "task_status": "error",
                "agent_type": "style_editing_agent"
            }


# Singleton instance
_style_editing_agent_instance = None


def get_style_editing_agent() -> StyleEditingAgent:
    """Get global style editing agent instance"""
    global _style_editing_agent_instance
    if _style_editing_agent_instance is None:
        _style_editing_agent_instance = StyleEditingAgent()
    return _style_editing_agent_instance

