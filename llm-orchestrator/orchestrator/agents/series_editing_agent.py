"""
Series Editing Agent - LangGraph Implementation
Gated to series documents. Consumes full series body (frontmatter stripped),
loads Rules and Character references directly from this file's frontmatter,
and emits editor operations for Prefer Editor HITL application.
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
# LangGraph State
# ============================================

class SeriesEditingState(TypedDict):
    """State for series editing agent LangGraph workflow"""
    query: str
    user_id: str
    metadata: Dict[str, Any]
    messages: List[Any]
    shared_memory: Dict[str, Any]
    active_editor: Dict[str, Any]
    series: str
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
    failed_operations: List[Dict[str, Any]]
    response: Dict[str, Any]
    task_status: str
    error: str


# ============================================
# Series Editing Agent
# ============================================

class SeriesEditingAgent(BaseAgent):
    """
    Series Editing Agent for series continuity and book tracking
    
    Gated to series documents. Consumes full series body (frontmatter stripped),
    loads Rules and Character references directly from this file's frontmatter,
    and emits editor operations for Prefer Editor HITL.
    Uses LangGraph workflow for explicit state management
    """
    
    def __init__(self):
        super().__init__("series_editing_agent")
        logger.info("Series Editing Agent ready!")
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """Build LangGraph workflow for series editing agent"""
        workflow = StateGraph(SeriesEditingState)
        
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
        """Build system prompt for series editing"""
        return (
            "You are a SERIES ARCHITECT for SERIES documents (book series continuity, synopsis generation, book tracking). "
            "Persona disabled. Adhere strictly to frontmatter, project Rules, and Character profiles.\n\n"
            "**YOUR PRIMARY RESPONSIBILITIES**:\n"
            "- Generate 1-2 paragraph synopses for future books in the series\n"
            "- Track which books have been written vs. planned\n"
            "- Maintain continuity across the entire series timeline\n"
            "- Ensure synopses align with established worldbuilding rules and character arcs\n"
            "- Update book status (Written, In Progress, Planned, Future)\n\n"
            "**CRITICAL: USE RULES AND CHARACTERS FOR CONSISTENCY**\n"
            "- Reference worldbuilding rules to ensure synopses don't violate established constraints\n"
            "- Use character profiles to maintain character consistency and development arcs\n"
            "- Cross-reference series timeline to avoid continuity errors\n"
            "- Example: If rules say 'Magic requires components', synopses should reflect this constraint\n"
            "- Example: If character profile says 'Franklin is cautious', synopses should show this trait\n\n"
            "**WHEN TO ASK QUESTIONS (Ask When Information Is Needed)**:\n"
            "- When the user's request is vague or incomplete and you need specific details to proceed\n"
            "- When multiple interpretations are possible and you need clarification on which direction to take\n"
            "- When you need to know which book number to create a synopsis for\n"
            "- When you need to know the scope or level of detail the user wants\n"
            "- When there's a conflict between existing series content and the new request that requires user decision\n\n"
            "**HOW TO ASK QUESTIONS**:\n"
            "- Provide questions in the summary field of your response\n"
            "- Use EITHER multiple choice options OR free-form questions:\n"
            "  * Multiple choice: 'Which book should this synopsis be for? A) Book 3, B) Book 4, C) Book 5, D) Other (please specify)'\n"
            "  * Free form: 'What specific plot elements should I include in the synopsis? (e.g., main conflict, character arcs, key events?)'\n"
            "- If you can make partial edits based on available information, do so and ask questions about the missing parts\n"
            "- If the request is completely unclear, return empty operations array and ask clarifying questions in the summary\n"
            "- Be specific about what information you need - don't ask vague questions\n\n"
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
            "FORMATTING CONTRACT (SERIES DOCUMENTS):\n"
            "- Never emit YAML frontmatter in operations[].text. Preserve existing frontmatter as-is.\n"
            "- Use Markdown headings and lists for the body.\n"
            "- When creating or normalizing structure, prefer this scaffold (top-level headings):\n"
            "  ## Series Overview\n"
            "  Brief description of the series, main themes, and overall arc.\n\n"
            "  ## Book 1: [Title]\n"
            "  **Status**: Written | In Progress | Planned | Future\n"
            "  **Synopsis**: 1-2 paragraph synopsis of the book.\n\n"
            "  ## Book 2: [Title]\n"
            "  **Status**: Written | In Progress | Planned | Future\n"
            "  **Synopsis**: 1-2 paragraph synopsis of the book.\n\n"
            "  ... (continue for all books in the series)\n\n"
            "  ## Series Timeline\n"
            "  Chronological events across all books (optional, for complex series).\n\n"
            "RULES FOR EDITS:\n"
            "0) **WORK FIRST, ASK LATER**: Always make edits based on available information. Use context from the request, existing series content, rules, and character profiles to inform your work. Only ask questions in the summary if critical information is missing that prevents meaningful progress. Never return empty operations unless the request is completely impossible.\n"
            "1) Make focused, surgical edits near the cursor/selection unless the user requests re-organization.\n"
            "2) Maintain the scaffold above; if missing, create only the minimal sections the user asked for.\n"
            "3) Prefer paragraph/sentence-level replacements; avoid large-span rewrites unless asked.\n"
            "4) Enforce consistency: cross-check synopses against Rules and Character profiles.\n"
            "5) **SYNOPSIS GENERATION**: When generating synopses for future books:\n"
            "   - Generate 1-2 paragraphs that capture the main plot, character arcs, and key events\n"
            "   - Ensure synopses align with established worldbuilding rules\n"
            "   - Maintain character consistency based on character profiles\n"
            "   - Reference previous books' events when relevant for continuity\n"
            "   - Use the rules document to ensure no violations of established constraints\n"
            "   - Example: If rules say 'No time travel', synopsis shouldn't include time travel\n"
            "   - Example: If character profile says 'Sarah is a pacifist', synopsis should reflect this\n\n"
            "6) **BOOK STATUS TRACKING**: When updating book status:\n"
            "   - Use clear status indicators: **Status**: Written | In Progress | Planned | Future\n"
            "   - Update status when user indicates a book is complete or started\n"
            "   - Maintain chronological order of books\n"
            "   - If adding a new book entry, determine appropriate status based on context\n\n"
            "7) **CONTINUITY MANAGEMENT**: When adding or updating content:\n"
            "   - Check existing books for timeline consistency\n"
            "   - Ensure character arcs progress logically across books\n"
            "   - Reference rules document to avoid contradictions\n"
            "   - If conflicts arise, ask the user for clarification\n\n"
            "ANCHOR REQUIREMENTS (CRITICAL):\n"
            "For EVERY operation, you MUST provide precise anchors:\n\n"
            "REVISE/DELETE Operations:\n"
            "- ALWAYS include 'original_text' with EXACT, VERBATIM text from the file\n"
            "- Minimum 10-20 words for small edits, but for paragraph replacements: include the FULL paragraph (could be 50-200+ words)\n"
            "- Copy and paste directly - do NOT retype or modify\n"
            "- NEVER include header lines (###, ##, #) in original_text!\n"
            "- OR provide both 'left_context' and 'right_context' (exact surrounding text)\n\n"
            "INSERT Operations (ONLY for truly empty sections!):\n"
            "- **insert_after_heading**: Use ONLY when section is completely empty below the header\n"
            "  * op_type='insert_after_heading' with anchor_text='## Book X' (exact header line)\n"
            "  * Example: Adding synopsis after '## Book 3: Title' header when section is completely empty\n"
            "  * ⚠️ CRITICAL WARNING: Before using insert_after_heading, you MUST verify the section is COMPLETELY EMPTY!\n"
            "  * ⚠️ If there is ANY text below the header (even a single line), use replace_range instead!\n"
            "- **insert_after**: Use when continuing text mid-paragraph, mid-sentence, or after specific text\n"
            "  * op_type='insert_after' with anchor_text='last few words before insertion point'\n\n"
            "- **REPLACE Operations (PREFERRED for updating existing content!)**:\n"
            "- **replace_range**: Use when section exists but needs improvement, completion, or revision\n"
            "  * If section has ANY content (even incomplete or placeholder), use replace_range to update it\n"
            "  * Example: Section has '**Status**: Planned' but needs synopsis → replace_range with original_text='**Status**: Planned' and expanded text\n"
            "  * Example: Section has placeholder '[Synopsis to be written]' → replace_range with original_text='[Synopsis to be written]' and actual synopsis\n\n"
            "Additional Options:\n"
            "- 'occurrence_index' if text appears multiple times (0-based, default 0)\n"
            "- Start/end indices are approximate; anchors take precedence\n\n"
            "=== DECISION TREE FOR OPERATION TYPE ===\n"
            "**STEP 1: Read the section content carefully!**\n"
            "- Look at what exists below the header\n"
            "- Is there ANY text at all? Even a single line?\n"
            "\n"
            "**STEP 2: Choose operation based on what exists:**\n"
            "1. Section is COMPLETELY EMPTY below header (no text at all)? → insert_after_heading with anchor_text=\"## Book X\"\n"
            "2. Section has ANY content (even incomplete/placeholder/single line)? → replace_range to update it (NO headers in original_text!)\n"
            "3. Adding to existing synopsis? → replace_range with original_text matching existing content\n"
            "4. Deleting SPECIFIC content? → delete_range with original_text (NO headers!)\n"
            "5. Continuing mid-sentence? → insert_after\n\n"
            "CRITICAL: When updating existing content (even if incomplete), use 'replace_range' on the existing content!\n"
            "NEVER include headers in 'original_text' for replace_range - headers will be deleted!\n"
            "⚠️ NEVER use insert_after_heading when content exists - it will SPLIT the section and create duplicates!\n"
        )
    
    async def _prepare_context_node(self, state: SeriesEditingState) -> Dict[str, Any]:
        """Prepare context: extract active editor, validate series type"""
        try:
            logger.info("Preparing context for series editing...")
            
            shared_memory = state.get("shared_memory", {}) or {}
            active_editor = shared_memory.get("active_editor", {}) or {}
            
            series = active_editor.get("content", "") or ""
            filename = active_editor.get("filename") or "series.md"
            frontmatter = active_editor.get("frontmatter", {}) or {}
            cursor_offset = int(active_editor.get("cursor_offset", -1))
            selection_start = int(active_editor.get("selection_start", -1))
            selection_end = int(active_editor.get("selection_end", -1))
            
            # STRICT GATE: require explicit frontmatter.type == 'series'
            doc_type = ""
            if isinstance(frontmatter, dict):
                doc_type = str(frontmatter.get("type") or "").strip().lower()
            if doc_type != "series":
                logger.info(f"Series Agent Gate: Detected type='{doc_type}' (expected 'series'); skipping.")
                return {
                    "error": "Active editor is not a Series document; series agent skipping.",
                    "task_status": "error",
                    "response": {
                        "response": "Active editor is not a Series document; series agent skipping.",
                        "task_status": "error",
                        "agent_type": "series_editing_agent"
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
            normalized_text = series.replace("\r\n", "\n")
            body_only = _strip_frontmatter_block(normalized_text)
            para_start, para_end = paragraph_bounds(normalized_text, cursor_offset if cursor_offset >= 0 else 0)
            if selection_start >= 0 and selection_end > selection_start:
                para_start, para_end = selection_start, selection_end
            
            return {
                "active_editor": active_editor,
                "series": normalized_text,
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
    
    async def _load_references_node(self, state: SeriesEditingState) -> Dict[str, Any]:
        """Load referenced context files (rules, characters) directly from series frontmatter"""
        try:
            logger.info("Loading referenced context files from series frontmatter...")
            
            from orchestrator.tools.reference_file_loader import load_referenced_files
            
            active_editor = state.get("active_editor", {})
            user_id = state.get("user_id", "system")
            
            # Series reference configuration - load directly from series' frontmatter (no cascading)
            reference_config = {
                "rules": ["rules"],
                "characters": ["characters", "character_*"]  # Support both list and individual keys
            }
            
            # Use unified loader (no cascade_config - series loads directly)
            result = await load_referenced_files(
                active_editor=active_editor,
                user_id=user_id,
                reference_config=reference_config,
                doc_type_filter="series",
                cascade_config=None  # No cascading for series
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
            
            logger.info(f"Loaded {len(characters_bodies)} character reference(s), rules: {bool(rules_body)}")
            
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
    
    async def _detect_request_type_node(self, state: SeriesEditingState) -> Dict[str, Any]:
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
- Current series: {body_only[:500] if body_only else "Empty series"}
- Has rules reference: {bool(rules_body)}
- Has {len(characters_bodies)} character reference(s)

**INTENT DETECTION**:
- QUESTIONS: User is asking a question - may or may not want edits
  - Pure questions: "What books are in the series?", "What's the status of Book 3?", "Show me the synopsis for Book 2"
  - Conditional edits: "Do we have a synopsis for Book 4? Add one if not", "What books? Suggest additions if needed"
  - Questions often start with: "Do you", "What", "Can you", "Are there", "How many", "Show me", "Is", "Does", "Are we", "Suggest"
  - **Key insight**: Questions can be answered, and IF edits are needed based on the answer, they can be made
  - Route ALL questions to edit path - LLM can decide if edits are needed
  
- EDIT REQUESTS: User wants to create, modify, or generate content - NO question asked
  - Examples: "Add synopsis for Book 3", "Create synopsis for future book", "Update Book 2 status to Written", "Generate synopsis for Book 5"
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
    
    async def _generate_edit_plan_node(self, state: SeriesEditingState) -> Dict[str, Any]:
        """Generate edit plan using LLM"""
        try:
            logger.info("Generating series edit plan...")
            
            series = state.get("series", "")
            filename = state.get("filename", "series.md")
            body_only = state.get("body_only", "")
            current_request = state.get("current_request", "")
            request_type = state.get("request_type", "edit_request")
            is_question = request_type == "question"
            
            rules_body = state.get("rules_body")
            characters_bodies = state.get("characters_bodies", [])
            
            para_start = state.get("para_start", 0)
            selection_start = state.get("selection_start", -1)
            selection_end = state.get("selection_end", -1)
            
            # Build system prompt
            system_prompt = self._build_system_prompt()
            
            # Build context message
            context_parts = [
                "=== SERIES CONTEXT ===\n",
                f"File: {filename}\n\n",
                "Current Buffer (frontmatter stripped):\n" + body_only + "\n\n"
            ]
            
            if rules_body:
                context_parts.append(f"=== RULES REFERENCE ===\n{rules_body[:2000]}\n\n")
            
            if characters_bodies:
                context_parts.append("".join([f"=== CHARACTER REFERENCE ===\n{b[:1000]}\n\n" for b in characters_bodies]))
            
            # Add mode-specific instructions
            if is_question:
                context_parts.append(
                    "\n=== QUESTION REQUEST: ANALYZE AND OPTIONALLY EDIT ===\n"
                    "The user has asked a question about the series document.\n\n"
                    "**YOUR TASK**:\n"
                    "1. **ANALYZE FIRST**: Answer the user's question by evaluating the current content\n"
                    "2. **THEN EDIT IF NEEDED**: Based on your analysis, make edits if necessary\n"
                    "   - Include your analysis in the 'summary' field of your response\n"
                    "   - Provide editor operations ONLY if edits are needed\n\n"
                )
            else:
                context_parts.append(
                    "\n=== EDIT REQUEST: GENERATE SYNOPSIS OR UPDATE SERIES ===\n"
                    "The user wants you to add or revise series content.\n\n"
                    "**YOUR APPROACH**:\n"
                    "1. **USE RULES AND CHARACTERS**: Reference rules and character profiles to ensure consistency\n"
                    "2. **GENERATE QUALITY SYNOPSES**: Create 1-2 paragraph synopses that align with established worldbuilding\n"
                    "3. **TRACK BOOK STATUS**: Update status indicators when appropriate\n"
                    "4. **MAINTAIN CONTINUITY**: Ensure new content doesn't contradict existing books\n"
                    "5. **ASK WHEN NEEDED**: If you need specific details, ask questions in the summary\n\n"
                )
            
            # Build request with mode-specific instructions
            request_with_instructions = ""
            if current_request:
                if is_question:
                    request_with_instructions = (
                        f"USER REQUEST: {current_request}\n\n"
                        "**QUESTION MODE**: Answer the question first, then provide edits if needed.\n\n"
                        "CRITICAL: Use rules and character profiles to ensure any generated content is consistent.\n"
                    )
                else:
                    request_with_instructions = (
                        f"USER REQUEST: {current_request}\n\n"
                        "**SYNOPSIS GENERATION GUIDELINES**:\n"
                        "- Generate 1-2 paragraphs that capture main plot, character arcs, and key events\n"
                        "- Ensure synopses align with established worldbuilding rules\n"
                        "- Maintain character consistency based on character profiles\n"
                        "- Reference previous books' events when relevant for continuity\n"
                        "- Use the rules document to ensure no violations of established constraints\n\n"
                        "**BOOK STATUS TRACKING**:\n"
                        "- Use clear status indicators: **Status**: Written | In Progress | Planned | Future\n"
                        "- Update status when user indicates a book is complete or started\n"
                        "- Maintain chronological order of books\n\n"
                        "CRITICAL ANCHORING INSTRUCTIONS:\n"
                        "- **BEFORE using insert_after_heading**: Verify the section is COMPLETELY EMPTY (no text below header)\n"
                        "- **If section has ANY content**: Use replace_range to update it, NOT insert_after_heading\n"
                        "- For REVISE/DELETE: Provide 'original_text' with EXACT, VERBATIM text from file\n"
                        "- For INSERT: Use 'insert_after_heading' with 'anchor_text' ONLY for completely empty sections\n"
                        "- NEVER include header lines in original_text for replace_range operations\n"
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
            
            # Call LLM (moderate temperature for accurate documentation)
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
                    raw.setdefault("summary", "Planned series edit generated from context.")
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
                    "error": "Failed to produce a valid Series edit plan. Ensure ONLY raw JSON ManuscriptEdit with operations is returned.",
                    "task_status": "error"
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
                "system_prompt": system_prompt
            }
            
        except Exception as e:
            logger.error(f"Failed to generate edit plan: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Use base agent's error handler to surface OpenRouter errors properly
            return self._handle_node_error(e, state, "Edit plan generation")
    
    async def _resolve_operations_node(self, state: SeriesEditingState) -> Dict[str, Any]:
        """Resolve editor operations with progressive search"""
        try:
            logger.info("Resolving editor operations...")
            
            series = state.get("series", "")
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
            
            fm_end_idx = _frontmatter_end_index(series)
            selection = {"start": selection_start, "end": selection_end} if selection_start >= 0 and selection_end >= 0 else None
            
            # Check if file is empty (only frontmatter)
            body_only = _strip_frontmatter_block(series)
            is_empty_file = not body_only.strip()
            
            editor_operations = []
            failed_operations = []
            operations = structured_edit.get("operations", [])
            
            logger.info(f"Resolving {len(operations)} operation(s) from structured_edit")
            
            for op in operations:
                # Sanitize op text
                op_text = op.get("text", "")
                if isinstance(op_text, str):
                    op_text = _strip_frontmatter_block(op_text)
                    op_text = re.sub(r"\n{3,}", "\n\n", op_text)
                
                # Resolve operation
                try:
                    # Use centralized resolver
                    cursor_pos = state.get("cursor_offset", -1)
                    cursor_pos = cursor_pos if cursor_pos >= 0 else None
                    resolved_start, resolved_end, resolved_text, resolved_confidence = resolve_editor_operation(
                        content=series,
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
                    
                    resolved_start = max(0, min(len(series), resolved_start))
                    resolved_end = max(resolved_start, min(len(series), resolved_end))
                    
                    # Handle spacing for inserts
                    if resolved_start == resolved_end:
                        left_tail = series[max(0, resolved_start-2):resolved_start]
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
                    
                    # Check if resolution failed (-1, -1)
                    if resolved_start == -1 and resolved_end == -1:
                        logger.error(f"❌ Operation resolution FAILED - original_text or anchor_text not found")
                        failed_operations.append({
                            "op_type": op.get("op_type", "edit"),
                            "original_text": op.get("original_text"),
                            "anchor_text": op.get("anchor_text"),
                            "text": op.get("text", ""),
                            "error": "Anchor or original text not found"
                        })
                        continue
                    
                    # Calculate pre_hash
                    pre_slice = series[resolved_start:resolved_end]
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
                    # Collect failed operation
                    failed_operations.append({
                        "op_type": op.get("op_type", "edit"),
                        "original_text": op.get("original_text"),
                        "anchor_text": op.get("anchor_text"),
                        "text": op.get("text", ""),
                        "error": str(e)
                    })
                    continue
            
            logger.info(f"Successfully resolved {len(editor_operations)} operation(s) out of {len(operations)}")
            
            return {
                "editor_operations": editor_operations,
                "failed_operations": failed_operations
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
    
    async def _format_response_node(self, state: SeriesEditingState) -> Dict[str, Any]:
        """Format final response with editor operations"""
        try:
            logger.info("Formatting series editing response...")
            
            structured_edit = state.get("structured_edit")
            editor_operations = state.get("editor_operations", [])
            current_request = state.get("current_request", "")
            request_type = state.get("request_type", "edit_request")
            
            if not structured_edit:
                error = state.get("error", "Unknown error")
                return {
                    "response": {
                        "response": f"Failed to generate series edit plan: {error}",
                        "task_status": "error",
                        "agent_type": "series_editing_agent"
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
            
            # Add failed operations if present
            failed_operations = state.get("failed_operations", [])
            if failed_operations:
                failed_section = "\n\n**⚠️ UNRESOLVED EDITS (Manual Action Required)**\n"
                failed_section += "The following generated content could not be automatically placed in the series. You can copy and paste these sections manually:\n\n"
                
                for i, op in enumerate(failed_operations, 1):
                    op_type = op.get("op_type", "edit")
                    error = op.get("error", "Anchor text not found")
                    text = op.get("text", "")
                    anchor = op.get("anchor_text") or op.get("original_text")
                    
                    failed_section += f"#### Unresolved Edit {i} ({op_type})\n"
                    failed_section += f"- **Reason**: {error}\n"
                    if anchor:
                        failed_section += f"- **Intended near**:\n> {anchor[:200]}...\n"
                    
                    failed_section += "\n**Generated Content** (Scroll-safe):\n"
                    failed_section += f"{text}\n\n"
                    failed_section += "---\n"
                
                response_text = response_text + failed_section
            
            # Build response dict
            response_dict = {
                "response": response_text,
                "task_status": "complete",
                "agent_type": "series_editing_agent"
            }
            
            # Add editor operations if present
            if editor_operations:
                response_dict["editor_operations"] = editor_operations
                response_dict["manuscript_edit"] = {
                    **structured_edit,
                    "operations": editor_operations
                }
                response_dict["content_preview"] = response_text[:2000]
            
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
                    "agent_type": "series_editing_agent"
                },
                "task_status": "error",
                "error": str(e)
            }
    
    async def process(self, query: str, metadata: Dict[str, Any] = None, messages: List[Any] = None) -> Dict[str, Any]:
        """Process series editing query using LangGraph workflow"""
        try:
            logger.info(f"Series editing agent processing: {query[:100]}...")
            
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
            initial_state: SeriesEditingState = {
                "query": query,
                "user_id": user_id,
                "metadata": metadata,
                "messages": conversation_messages,
                "shared_memory": shared_memory_merged,
                "active_editor": {},
                "series": "",
                "filename": "series.md",
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
                "failed_operations": [],
                "response": {},
                "task_status": "",
                "error": ""
            }
            
            # Invoke LangGraph workflow with checkpointing
            final_state = await workflow.ainvoke(initial_state, config=config)
            
            # Extract final response
            response = final_state.get("response", {})
            task_status = final_state.get("task_status", "complete")
            
            if task_status == "error":
                error_msg = final_state.get("error", "Unknown error")
                logger.error(f"Series editing agent failed: {error_msg}")
                return {
                    "response": f"Series editing failed: {error_msg}",
                    "task_status": "error",
                    "agent_results": {}
                }
            
            # Extract response text - handle nested structure
            response_text = response.get("response", "") if isinstance(response, dict) else str(response) if response else ""
            if not response_text:
                response_text = "Series editing complete"  # Fallback only if truly empty
            
            # Build result dict matching other editing agents pattern
            result = {
                "response": response_text,
                "task_status": task_status,
                "agent_type": "series_editing_agent"
            }
            
            # Add editor operations if present
            if isinstance(response, dict) and response.get("editor_operations"):
                result["editor_operations"] = response["editor_operations"]
                result["manuscript_edit"] = response.get("manuscript_edit")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ {self.agent_type} failed: {e}")
            return self._create_error_response(str(e))


# ============================================
# Factory Function
# ============================================

def get_series_editing_agent() -> SeriesEditingAgent:
    """Get or create singleton series editing agent"""
    global _series_editing_agent
    if _series_editing_agent is None:
        _series_editing_agent = SeriesEditingAgent()
    return _series_editing_agent


_series_editing_agent: Optional[SeriesEditingAgent] = None
