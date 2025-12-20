"""
Character Development Agent
LangGraph agent for character document editing

Gated to character documents. Consumes active editor buffer, cursor/selection,
and cascades outline → rules/style/characters where available. Produces
EditorOperations for Prefer Editor HITL application.
"""

import hashlib
import json
import logging
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, TypedDict, Tuple
from langchain_core.messages import AIMessage

from langgraph.graph import StateGraph, END
from orchestrator.agents.base_agent import BaseAgent
from orchestrator.utils.editor_operation_resolver import resolve_editor_operation

logger = logging.getLogger(__name__)


# Utility functions
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
    """Find the index where frontmatter ends (after closing ---)."""
    try:
        match = re.search(r'^---\s*\n[\s\S]*?\n---\s*\n', text, re.MULTILINE)
        if match:
            return match.end()
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


def paragraph_bounds(text: str, cursor_offset: int) -> tuple:
    """Find paragraph boundaries around cursor."""
    if not text:
        return 0, 0
    cursor = max(0, min(len(text), cursor_offset))
    # expand left to previous blank line or start
    left = text.rfind("\n\n", 0, cursor)
    start = left + 2 if left != -1 else 0
    # expand right to next blank line or end
    right = text.find("\n\n", cursor)
    end = right if right != -1 else len(text)
    return start, end


# Removed: _resolve_operation_simple - now using centralized resolver from orchestrator.utils.editor_operation_resolver


class CharacterDevelopmentState(TypedDict):
    """State for character development agent LangGraph workflow"""
    query: str
    user_id: str
    metadata: Dict[str, Any]
    messages: List[Any]
    shared_memory: Dict[str, Any]
    active_editor: Dict[str, Any]
    text: str
    filename: str
    frontmatter: Dict[str, Any]
    cursor_offset: int
    selection_start: int
    selection_end: int
    outline_body: Optional[str]
    rules_body: Optional[str]
    style_text: Optional[str]
    character_bodies: List[str]
    para_start: int
    para_end: int
    current_request: str
    request_type: str  # "question" or "edit_request"
    system_prompt: str
    llm_response: str
    structured_edit: Optional[Dict[str, Any]]
    editor_operations: List[Dict[str, Any]]
    response: Dict[str, Any]
    task_status: str
    error: str


class CharacterDevelopmentAgent(BaseAgent):
    """
    Character Development Agent for character document editing
    
    Gated to character documents. Consumes active editor buffer, cursor/selection,
    and cascades outline → rules/style/characters where available. Produces
    EditorOperations for Prefer Editor HITL application.
    Uses LangGraph workflow for explicit state management
    """
    
    def __init__(self):
        super().__init__("character_development_agent")
        self._grpc_client = None
        logger.info("Character Development Agent ready!")
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """Build LangGraph workflow for character development agent"""
        workflow = StateGraph(CharacterDevelopmentState)
        
        # Add nodes
        workflow.add_node("prepare_context", self._prepare_context_node)
        workflow.add_node("load_references", self._load_references_node)
        workflow.add_node("detect_request_type", self._detect_request_type_node)
        workflow.add_node("generate_edit_plan", self._generate_edit_plan_node)
        workflow.add_node("resolve_operations", self._resolve_operations_node)
        workflow.add_node("format_response", self._format_response_node)
        
        # Entry point
        workflow.set_entry_point("prepare_context")
        
        # Flow: prepare_context -> load_references -> detect_request_type -> (conditional routing)
        workflow.add_edge("prepare_context", "load_references")
        workflow.add_edge("load_references", "detect_request_type")
        
        # Conditional routing based on request type
        workflow.add_conditional_edges(
            "detect_request_type",
            self._route_from_request_type,
            {
                "question": "generate_edit_plan",  # Questions also go to edit path (LLM decides if edits needed)
                "edit_request": "generate_edit_plan"
            }
        )
        
        # Continue with edit processing
        workflow.add_edge("generate_edit_plan", "resolve_operations")
        workflow.add_edge("resolve_operations", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow.compile(checkpointer=checkpointer)
    
    async def _get_grpc_client(self):
        """Get or create gRPC client for backend tools"""
        if self._grpc_client is None:
            from orchestrator.clients.backend_tool_client import get_backend_tool_client
            self._grpc_client = await get_backend_tool_client()
        return self._grpc_client
    
    async def _prepare_context_node(self, state: CharacterDevelopmentState) -> Dict[str, Any]:
        """Prepare context: extract editor info and check document type"""
        try:
            logger.info("Preparing context for character development...")
            
            shared_memory = state.get("shared_memory", {}) or {}
            active_editor = shared_memory.get("active_editor", {}) or {}
            
            text = active_editor.get("content", "") or ""
            filename = active_editor.get("filename") or "character.md"
            frontmatter = active_editor.get("frontmatter", {}) or {}
            cursor_offset = int(active_editor.get("cursor_offset", -1))
            selection_start = int(active_editor.get("selection_start", -1))
            selection_end = int(active_editor.get("selection_end", -1))
            
            # Gate by type: character (strict)
            doc_type = str(frontmatter.get("type", "")).strip().lower()
            if doc_type != "character":
                return {
                    "response": {
                        "messages": [AIMessage(content="Active editor is not a character file; skipping.")],
                        "agent_results": {
                            "agent_type": self.agent_type,
                            "is_complete": True,
                            "skipped": True
                        },
                        "is_complete": True
                    },
                    "task_status": "skipped"
                }
            
            # Scope: prefer selection; else paragraph around cursor
            para_start, para_end = paragraph_bounds(text, cursor_offset if cursor_offset >= 0 else 0)
            if selection_start >= 0 and selection_end > selection_start:
                para_start, para_end = selection_start, selection_end
            
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
                "text": text,
                "filename": filename,
                "frontmatter": frontmatter,
                "cursor_offset": cursor_offset,
                "selection_start": selection_start,
                "selection_end": selection_end,
                "para_start": para_start,
                "para_end": para_end,
                "current_request": current_request.strip()
            }
            
        except Exception as e:
            logger.error(f"Failed to prepare context: {e}")
            return {
                "text": "",
                "filename": "character.md",
                "frontmatter": {},
                "error": str(e),
                "task_status": "error"
            }
    
    async def _load_references_node(self, state: CharacterDevelopmentState) -> Dict[str, Any]:
        """Load referenced context files (characters, style, rules) directly from character frontmatter"""
        try:
            logger.info("Loading referenced context files from character frontmatter...")
            
            from orchestrator.tools.reference_file_loader import load_referenced_files
            
            active_editor = state.get("active_editor", {})
            user_id = state.get("user_id", "system")
            
            # Character reference configuration - load directly from character's frontmatter (no cascading)
            reference_config = {
                "characters": ["characters", "character_*"],  # Other character sheets
                "style": ["style"],                           # Optional: style guide
                "rules": ["rules"]                             # Optional: world rules
            }
            
            # Use unified loader (no cascade_config - character loads directly)
            result = await load_referenced_files(
                active_editor=active_editor,
                user_id=user_id,
                reference_config=reference_config,
                doc_type_filter="character",
                cascade_config=None  # No cascading for character files
            )
            
            loaded_files = result.get("loaded_files", {})
            
            # Extract content from loaded files
            outline_body = None  # Characters don't typically reference outlines directly
            
            rules_body = None
            if loaded_files.get("rules") and len(loaded_files["rules"]) > 0:
                rules_body = loaded_files["rules"][0].get("content", "")
                if rules_body:
                    rules_body = _strip_frontmatter_block(rules_body)
            
            style_text = None
            if loaded_files.get("style") and len(loaded_files["style"]) > 0:
                style_text = loaded_files["style"][0].get("content", "")
                if style_text:
                    style_text = _strip_frontmatter_block(style_text)
            
            character_bodies = []
            if loaded_files.get("characters"):
                for char_file in loaded_files["characters"]:
                    char_content = char_file.get("content", "")
                    if char_content:
                        char_content = _strip_frontmatter_block(char_content)
                        character_bodies.append(char_content)
            
            logger.info(f"Loaded {len(character_bodies)} character reference(s), style: {bool(style_text)}, rules: {bool(rules_body)}")
            
            return {
                "outline_body": outline_body,
                "rules_body": rules_body,
                "style_text": style_text,
                "character_bodies": character_bodies,
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
                "outline_body": None,
                "rules_body": None,
                "style_text": None,
                "character_bodies": [],
                "error": str(e),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    def _route_from_request_type(self, state: CharacterDevelopmentState) -> str:
        """Route based on detected request type"""
        request_type = state.get("request_type", "edit_request")
        return request_type if request_type in ("question", "edit_request") else "edit_request"
    
    async def _detect_request_type_node(self, state: CharacterDevelopmentState) -> Dict[str, Any]:
        """Detect if request is a question or edit request"""
        try:
            logger.info("Detecting request type...")
            
            current_request = state.get("current_request", "")
            text = state.get("text", "")
            outline_body = state.get("outline_body")
            rules_body = state.get("rules_body")
            style_text = state.get("style_text")
            character_bodies = state.get("character_bodies", [])
            
            # Build simple prompt for LLM to determine intent
            body_only = _strip_frontmatter_block(text)
            prompt = f"""Analyze the user's request and determine if it's a QUESTION or an EDIT REQUEST.

**USER REQUEST**: {current_request}

**CONTEXT**:
- Current character: {body_only[:500] if body_only else "Empty character"}
- Has rules reference: {bool(rules_body)}
- Has style reference: {bool(style_text)}
- Has outline reference: {bool(outline_body)}
- Has {len(character_bodies)} character reference(s)

**INTENT DETECTION**:
- QUESTIONS (including pure questions and conditional edits): User is asking a question - may or may not want edits
  - Pure questions: "Does she have blue eyes?", "What traits does this character have?", "Show me the character profile"
  - Conditional edits: "Does she have blue eyes? Revise to ensure", "What traits? Add three more if less than five"
  - Questions often start with: "Do you", "What", "Can you", "Are there", "How many", "Show me", "Is", "Does", "Are we", "Suggest"
  - **Key insight**: Questions can be answered, and IF edits are needed based on the answer, they can be made
  - Route ALL questions to edit path - LLM can decide if edits are needed
  
- EDIT REQUESTS: User wants to create, modify, or generate content - NO question asked
  - Examples: "Add three traits", "Create a character profile", "Update the personality section", "Revise the dialogue patterns"
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
                    "character_bodies": state.get("character_bodies", []),
                    "outline_body": state.get("outline_body"),
                    "rules_body": state.get("rules_body"),
                    "style_text": state.get("style_text"),
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
                    "character_bodies": state.get("character_bodies", []),
                    "outline_body": state.get("outline_body"),
                    "rules_body": state.get("rules_body"),
                    "style_text": state.get("style_text"),
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
                "character_bodies": state.get("character_bodies", []),
                "outline_body": state.get("outline_body"),
                "rules_body": state.get("rules_body"),
                "style_text": state.get("style_text"),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    async def _generate_edit_plan_node(self, state: CharacterDevelopmentState) -> Dict[str, Any]:
        """Generate edit plan using LLM"""
        try:
            logger.info("Generating character edit plan...")
            
            text = state.get("text", "")
            filename = state.get("filename", "character.md")
            frontmatter = state.get("frontmatter", {})
            outline_body = state.get("outline_body")
            rules_body = state.get("rules_body")
            style_text = state.get("style_text")
            character_bodies = state.get("character_bodies", [])
            current_request = state.get("current_request", "")
            
            # Build system prompt
            system_prompt = self._build_system_prompt()
            
            # Determine if this is a question or edit request
            request_type = state.get("request_type", "edit_request")
            is_question = request_type == "question"
            
            # Build user message with context
            context_parts = [
                "=== CHARACTER CONTEXT ===\n",
                f"File: {filename}\n\n",
                "Current Character (frontmatter stripped):\n" + _strip_frontmatter_block(text) + "\n\n"
            ]
            
            if outline_body:
                context_parts.append("=== OUTLINE (if present) ===\n" + outline_body + "\n\n")
            if rules_body:
                context_parts.append("=== RULES (if present) ===\n" + rules_body + "\n\n")
            if style_text:
                context_parts.append("=== STYLE GUIDE (if present) ===\n" + style_text + "\n\n")
            if character_bodies:
                context_parts.append("".join(["=== RELATED CHARACTER DOC ===\n" + b + "\n\n" for b in character_bodies]))
            
            # Add mode-specific instructions
            if is_question:
                context_parts.append(
                    "\n=== QUESTION REQUEST: ANALYZE AND OPTIONALLY EDIT ===\n"
                    "The user has asked a question about the character.\n\n"
                    "**YOUR TASK**:\n"
                    "1. **ANALYZE FIRST**: Answer the user's question by evaluating the current content\n"
                    "   - Pure questions: 'Does she have blue eyes?' → Check and report eye color\n"
                    "   - Suggestion questions: 'Suggest three or four additions' → Analyze current content, then suggest additions\n"
                    "   - Verification questions: 'Does this character have X trait?' → Check for trait, report findings\n"
                    "   - Conditional questions: 'Does she have blue eyes? Revise to ensure' → Check, then edit if needed\n"
                    "2. **THEN EDIT IF NEEDED**: Based on your analysis, make edits if necessary\n"
                    "   - If question implies a desired state ('Revise to ensure blue eyes') → Provide editor operations\n"
                    "   - If question asks for suggestions ('Suggest additions') → Provide editor operations with suggested additions\n"
                    "   - If question is pure information ('Does she have blue eyes?') → No edits needed, just answer\n"
                    "   - Include your analysis in the 'summary' field of your response\n\n"
                    "**RESPONSE FORMAT**:\n"
                    "- In the 'summary' field: Answer the question clearly and explain your analysis\n"
                    "- In the 'operations' array: Provide editor operations ONLY if edits are needed\n"
                    "- If no edits needed: Return empty operations array, but answer the question in summary\n"
                    "- If edits needed: Provide operations AND explain what you found in summary\n\n"
                )
            else:
                # Edit request mode
                context_parts.append(
                    "\n=== EDIT REQUEST: WORK WITH AVAILABLE INFORMATION ===\n"
                    "The user wants you to add or revise character content.\n\n"
                    "**YOUR APPROACH**:\n"
                    "1. **WORK FIRST**: Make edits based on the request and available context (character file, outline, rules, related characters)\n"
                    "2. **USE INFERENCE**: Make reasonable inferences from the request - don't wait for clarification\n"
                    "3. **ASK ALONG THE WAY**: If you need specific details, include questions in the summary AFTER describing the work you've done\n"
                    "4. **NEVER EMPTY OPERATIONS**: Always provide operations based on what you can determine from the request and context\n\n"
                )
                if current_request and any(k in current_request.lower() for k in ["revise", "revision", "tweak", "adjust", "polish", "tighten"]):
                    context_parts.append("REVISION MODE: Apply minimal targeted edits; use paragraph-level replace_range ops.\n\n")
            
            context_parts.append("Provide a ManuscriptEdit JSON plan strictly within scope.")
            
            # Build request with mode-specific instructions
            request_with_instructions = ""
            if current_request:
                if is_question:
                    request_with_instructions = (
                        f"USER REQUEST: {current_request}\n\n"
                        "**QUESTION MODE**: Answer the question first, then provide edits if needed.\n\n"
                        "CRITICAL: CHECK FOR DUPLICATES FIRST (if edits are needed)\n"
                        "Before adding ANY new content:\n"
                        "1. **CHECK FOR SIMILAR INFO** - Does similar character info already exist in related sections?\n"
                        "2. **CONSOLIDATE IF NEEDED** - If trait appears in multiple places, ensure each adds unique perspective\n"
                        "3. **AVOID REDUNDANCY** - Don't add identical information to multiple sections\n"
                        "\n"
                        "CRITICAL: CROSS-REFERENCE RELATED SECTIONS (if edits are needed)\n"
                        "After checking for duplicates:\n"
                        "1. **SCAN THE ENTIRE DOCUMENT** - Read through ALL sections to identify related character information\n"
                        "2. **IDENTIFY ALL AFFECTED SECTIONS** - When adding/updating character info, find ALL places it should appear\n"
                        "3. **GENERATE MULTIPLE OPERATIONS** - If character info affects multiple sections, create operations for EACH affected section\n"
                        "4. **ENSURE CONSISTENCY** - Related sections must be updated together to maintain character coherence\n"
                        "\n"
                        "CRITICAL ANCHORING INSTRUCTIONS (if edits are needed):\n"
                        "- **BEFORE using insert_after_heading**: Verify the section is COMPLETELY EMPTY (no text below header)\n"
                        "- **If section has ANY content**: Use replace_range to update it, NOT insert_after_heading\n"
                        "- **insert_after_heading will SPLIT sections**: If you use it when content exists, it inserts BETWEEN header and existing text!\n"
                        "- **UPDATING EXISTING CONTENT**: If a section exists but needs improvement/completion, use 'replace_range' with 'original_text' matching the EXISTING content\n"
                        "  * Example: Section has '- Analytical thinker' but needs more → replace_range with original_text='- Analytical thinker' and text='- Analytical thinker\\n- Methodical problem-solver\\n- Protective of family'\n"
                        "  * Example: Section has placeholder '[To be developed]' → replace_range with original_text='[To be developed]' and actual content\n"
                        "- **ADDING TO EMPTY SECTIONS**: Only use 'insert_after_heading' when section is completely empty below the header\n"
                        "- For REVISE/DELETE: Provide 'original_text' with EXACT, VERBATIM text from file (10-20+ words, complete sentences)\n"
                        "- For INSERT: Use 'insert_after_heading' with 'anchor_text' ONLY for completely empty sections, or 'insert_after' for mid-paragraph\n"
                        "- NEVER include header lines in original_text for replace_range operations\n"
                        "- Copy text directly from the file - do NOT retype or paraphrase\n"
                        "- Without precise anchors, the operation WILL FAIL\n"
                        "- **KEY RULE**: If content exists (even if incomplete), use replace_range to update it. Only use insert_after_heading for truly empty sections.\n"
                        "- You can return MULTIPLE operations in the operations array - for checking/consolidating duplicates AND updating related sections"
                    )
                else:
                    request_with_instructions = (
                        f"USER REQUEST: {current_request}\n\n"
                        "**WORK FIRST**: Make edits based on the request and available context. Use reasonable inferences - don't wait for clarification. Only ask questions in the summary if critical information is truly missing.\n\n"
                        "CRITICAL: CHECK FOR DUPLICATES FIRST\n"
                        "Before adding ANY new content:\n"
                        "1. **CHECK FOR SIMILAR INFO** - Does similar character info already exist in related sections?\n"
                        "2. **CONSOLIDATE IF NEEDED** - If trait appears in multiple places, ensure each adds unique perspective\n"
                        "3. **AVOID REDUNDANCY** - Don't add identical information to multiple sections\n"
                        "\n"
                        "CRITICAL: CROSS-REFERENCE AND MULTIPLE OPERATIONS\n"
                        "After checking for duplicates:\n"
                        "1. **SCAN THE ENTIRE DOCUMENT** - Read through ALL sections to identify related character information\n"
                        "2. **IDENTIFY ALL AFFECTED SECTIONS** - When adding/updating character info, find ALL places it should appear\n"
                        "3. **GENERATE MULTIPLE OPERATIONS** - If character info affects multiple sections, create operations for EACH affected section\n"
                        "4. **ENSURE CONSISTENCY** - Related sections must be updated together to maintain character coherence\n"
                        "\n"
                        "Examples of when to generate multiple operations:\n"
                        "- Adding personality trait → Update 'Personality' section AND 'Dialogue Patterns' if trait affects speech\n"
                        "- Adding relationship detail → Update 'Relationships' section AND 'Character Arc' if relationship affects development\n"
                        "- Adding backstory → Update 'Basic Information' AND 'Personality' AND 'Character Arc' if backstory shapes character\n"
                        "- Updating character info → If info appears in multiple sections, update ALL occurrences, not just one\n"
                        "\n"
                        "CRITICAL ANCHORING INSTRUCTIONS:\n"
                        "- **BEFORE using insert_after_heading**: Verify the section is COMPLETELY EMPTY (no text below header)\n"
                        "- **If section has ANY content**: Use replace_range to update it, NOT insert_after_heading\n"
                        "- **insert_after_heading will SPLIT sections**: If you use it when content exists, it inserts BETWEEN header and existing text!\n"
                        "- **UPDATING EXISTING CONTENT**: If a section exists but needs improvement/completion, use 'replace_range' with 'original_text' matching the EXISTING content\n"
                        "  * Example: Section has '- Analytical thinker' but needs more → replace_range with original_text='- Analytical thinker' and text='- Analytical thinker\\n- Methodical problem-solver\\n- Protective of family'\n"
                        "  * Example: Section has placeholder '[To be developed]' → replace_range with original_text='[To be developed]' and new content\n"
                        "- **ADDING TO EMPTY SECTIONS**: Only use 'insert_after_heading' when section is completely empty below the header\n"
                        "- For REVISE/DELETE: Provide 'original_text' with EXACT, VERBATIM text from file (10-20+ words, complete sentences)\n"
                        "- For INSERT: Use 'insert_after_heading' with 'anchor_text' ONLY for completely empty sections, or 'insert_after' for mid-paragraph\n"
                        "- NEVER include header lines in original_text for replace_range operations\n"
                        "- Copy text directly from the file - do NOT retype or paraphrase\n"
                        "- Without precise anchors, the operation WILL FAIL\n"
                        "- **KEY RULE**: If content exists (even if incomplete), use replace_range to update it. Only use insert_after_heading for truly empty sections.\n"
                        "- You can return MULTIPLE operations in the operations array - for checking/consolidating duplicates AND updating related sections"
                    )
            
            # Use standardized helper for message construction with conversation history
            messages_list = state.get("messages", [])
            langchain_messages = self._build_editing_agent_messages(
                system_prompt=system_prompt,
                context_parts=context_parts,
                current_request=request_with_instructions,
                messages_list=messages_list,
                look_back_limit=6
            )
            
            # Call LLM using BaseAgent's _get_llm method - pass state to access user's model selection
            llm = self._get_llm(temperature=0.35, state=state)
            start_time = datetime.now()
            
            response = await llm.ainvoke(langchain_messages)
            
            content = response.content if hasattr(response, 'content') else str(response)
            content = _unwrap_json_response(content)
            
            # Parse structured response
            structured_edit = None
            try:
                raw = json.loads(content)
                if isinstance(raw, dict) and isinstance(raw.get("operations"), list):
                    raw.setdefault("target_filename", filename)
                    raw.setdefault("scope", "paragraph")
                    raw.setdefault("summary", "Planned character edit generated from context.")
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
                    "character_bodies": state.get("character_bodies", []),
                    "outline_body": state.get("outline_body"),
                    "rules_body": state.get("rules_body"),
                    "style_text": state.get("style_text"),
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
                    "error": "Failed to produce a valid Character edit plan. Ensure ONLY raw JSON ManuscriptEdit with operations is returned (no code fences or prose).",
                    "task_status": "error",
                    "character_bodies": state.get("character_bodies", []),
                    "outline_body": state.get("outline_body"),
                    "rules_body": state.get("rules_body"),
                    "style_text": state.get("style_text"),
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", "")
                }
            
            return {
                "llm_response": content,
                "structured_edit": structured_edit,
                "system_prompt": system_prompt,
                "character_bodies": state.get("character_bodies", []),
                "outline_body": state.get("outline_body"),
                "rules_body": state.get("rules_body"),
                "style_text": state.get("style_text"),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.error(f"Failed to generate edit plan: {e}")
            return {
                "llm_response": "",
                "structured_edit": None,
                "error": str(e),
                "task_status": "error",
                "character_bodies": state.get("character_bodies", []),
                "outline_body": state.get("outline_body"),
                "rules_body": state.get("rules_body"),
                "style_text": state.get("style_text"),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    async def _resolve_operations_node(self, state: CharacterDevelopmentState) -> Dict[str, Any]:
        """Resolve editor operations with progressive search"""
        try:
            logger.info("Resolving editor operations...")
            
            text = state.get("text", "")
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
            
            fm_end_idx = _frontmatter_end_index(text)
            selection = {"start": selection_start, "end": selection_end} if selection_start >= 0 and selection_end >= 0 else None
            
            editor_operations = []
            operations = structured_edit.get("operations", [])
            
            for op in operations:
                # Resolve operation
                try:
                    # Use centralized resolver
                    cursor_pos = state.get("cursor_offset", -1)
                    cursor_pos = cursor_pos if cursor_pos >= 0 else None
                    resolved_start, resolved_end, resolved_text, resolved_confidence = resolve_editor_operation(
                        content=text,
                        op_dict=op,
                        selection=selection,
                        frontmatter_end=fm_end_idx,
                        cursor_offset=cursor_pos
                    )
                    
                    # CRITICAL: Ensure operations never occur before frontmatter end
                    if resolved_start < fm_end_idx:
                        if op.get("op_type") == "delete_range":
                            # Skip deletions targeting frontmatter
                            logger.warning(f"Skipping delete_range operation that targets frontmatter")
                            continue
                        # For inserts/replaces, clamp to frontmatter end
                        if resolved_end <= fm_end_idx:
                            resolved_start = fm_end_idx
                            resolved_end = fm_end_idx
                        else:
                            # Overlap: clamp start to body start
                            resolved_start = fm_end_idx
                    
                    # Validate resolved positions
                    if resolved_start < 0 or resolved_end < 0:
                        logger.warning(f"Invalid resolved positions [{resolved_start}:{resolved_end}], skipping operation")
                        continue
                    
                    logger.info(f"Resolved {op.get('op_type')} [{resolved_start}:{resolved_end}] confidence={resolved_confidence:.2f}")
                    
                    # Clean text (remove frontmatter if accidentally included)
                    if isinstance(resolved_text, str):
                        resolved_text = _strip_frontmatter_block(resolved_text)
                    
                    # Calculate pre_hash
                    pre_slice = text[resolved_start:resolved_end] if resolved_start < len(text) and resolved_end <= len(text) else ""
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
                    # Fallback positioning - ensure we're after frontmatter
                    fallback_start = max(para_start, fm_end_idx)
                    fallback_end = max(para_end, fallback_start)
                    
                    # Ensure fallback doesn't go before frontmatter
                    if fallback_start < fm_end_idx:
                        fallback_start = fm_end_idx
                        fallback_end = max(fallback_end, fm_end_idx)
                    
                    pre_slice = text[fallback_start:fallback_end] if fallback_start < len(text) else ""
                    resolved_op = {
                        "op_type": op.get("op_type", "replace_range"),
                        "start": fallback_start,
                        "end": fallback_end,
                        "text": _strip_frontmatter_block(op.get("text", "")),
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
    
    async def _format_response_node(self, state: CharacterDevelopmentState) -> Dict[str, Any]:
        """Format final response with editor operations"""
        try:
            logger.info("Formatting response...")
            
            structured_edit = state.get("structured_edit", {})
            editor_operations = state.get("editor_operations", [])
            task_status = state.get("task_status", "complete")
            request_type = state.get("request_type", "edit_request")
            
            if task_status == "error":
                error_msg = state.get("error", "Unknown error")
                return {
                    "response": {
                        "response": f"Character development failed: {error_msg}",
                        "task_status": "error",
                        "agent_type": "character_development_agent"
                    },
                    "task_status": "error"
                }
            
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
                    logger.info(f"📝 Using summary as response text ({len(summary)} chars)")
                else:
                    # Build prose preview from operations
                    generated_preview = "\n\n".join([
                        op.get("text", "").strip()
                        for op in editor_operations
                        if op.get("text", "").strip()
                    ]).strip()
                    logger.info(f"📝 Generated preview from {len(editor_operations)} operations: {len(generated_preview)} chars")
                    if not generated_preview and editor_operations:
                        logger.warning(f"⚠️ Operations have no 'text' field! First op keys: {list(editor_operations[0].keys()) if editor_operations else 'N/A'}")
                    response_text = generated_preview if generated_preview else "Edit plan ready."
            
            # Build response with editor operations
            result = {
                "messages": [AIMessage(content=response_text)],
                "agent_results": {
                    "agent_type": self.agent_type,
                    "is_complete": True,
                    "content_preview": response_text[:2000],
                    "editor_operations": editor_operations,
                    "manuscript_edit": {
                        "target_filename": structured_edit.get("target_filename"),
                        "scope": structured_edit.get("scope"),
                        "summary": structured_edit.get("summary"),
                        "chapter_index": structured_edit.get("chapter_index"),
                        "safety": structured_edit.get("safety"),
                        "operations": editor_operations
                    }
                },
                "is_complete": True
            }
            
            logger.info(f"📤 FORMAT_RESPONSE: Returning {len(editor_operations)} editor operation(s) at state level")
            
            return {
                "response": result,
                "editor_operations": editor_operations,
                "task_status": task_status
            }
            
        except Exception as e:
            logger.error(f"Failed to format response: {e}")
            return {
                "response": {
                    "messages": [AIMessage(content=f"Character development failed: {str(e)}")],
                    "agent_results": {
                        "agent_type": self.agent_type,
                        "is_complete": False
                    },
                    "is_complete": False
                },
                "task_status": "error"
            }
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for character development"""
        return (
            "You are a Character Development Assistant for type: character files. Persona disabled."
            " Preserve frontmatter; write clean Markdown in body.\n\n"
            "**CRITICAL: WORK WITH AVAILABLE INFORMATION FIRST**\n"
            "Always start by working with what you know from the request, existing character content, and references:\n"
            "- Make edits based on available information - don't wait for clarification\n"
            "- Use context from outline, rules, style guide, and related characters to inform your work\n"
            "- Add or revise content based on reasonable inferences from the request\n"
            "- Only ask questions when critical information is missing that prevents you from making meaningful progress\n"
            "\n"
            "**WHEN TO ASK QUESTIONS (Rarely - Only When Truly Necessary)**:\n"
            "- Only when the request is so vague that you cannot make ANY reasonable edits (e.g., 'improve character' with no existing content)\n"
            "- Only when there's a critical conflict that requires user decision (e.g., existing trait directly contradicts new request)\n"
            "- When asking, provide operations for what you CAN do, then ask questions in the summary about what you need\n"
            "\n"
            "**HOW TO ASK QUESTIONS**: Include operations for work you CAN do, then add questions/suggestions in the summary field.\n"
            "DO NOT return empty operations array - always provide edits based on available information.\n\n"
            "STRUCTURED OUTPUT REQUIRED: Return ONLY raw JSON (no prose, no markdown, no code fences) matching this schema:\n"
            "{\n"
            "  \"type\": \"ManuscriptEdit\",\n"
            "  \"target_filename\": string,\n"
            "  \"scope\": one of [\"paragraph\", \"chapter\", \"multi_chapter\"],\n"
            "  \"summary\": string,\n"
            "  \"chapter_index\": integer|null,\n"
            "  \"safety\": one of [\"low\", \"medium\", \"high\"],\n"
            "  \"operations\": [\n"
            "    {\n"
            "      \"op_type\": one of [\"replace_range\", \"delete_range\", \"insert_after_heading\", \"insert_after\"],\n"
            "      \"start\": integer (approximate),\n"
            "      \"end\": integer (approximate),\n"
            "      \"text\": string,\n"
            "      \"original_text\": string (REQUIRED for replace/delete, optional for insert - EXACT verbatim text from file),\n"
            "      \"anchor_text\": string (optional - for inserts, exact line to insert after),\n"
            "      \"left_context\": string (optional - text before target),\n"
            "      \"right_context\": string (optional - text after target),\n"
            "      \"occurrence_index\": integer (optional, default 0 if text appears multiple times)\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            "OUTPUT RULES:\n"
            "- Output MUST be a single JSON object only.\n"
            "- Do NOT include triple backticks or language tags.\n"
            "- Do NOT include explanatory text before or after the JSON.\n"
            "- Always provide operations based on available information - work with what you know\n"
            "- If you need clarification, include it in the summary field AFTER describing the work you've done\n"
            "- Never return empty operations array unless the request is completely impossible to fulfill\n\n"
            "FORMATTING CONTRACT (CHARACTER FILES):\n"
            "- Never emit YAML frontmatter in operations[].text; preserve existing YAML.\n"
            "- **CRITICAL: USE BULLET POINTS, NOT PARAGRAPHS**\n"
            "- Character profiles must be concise and scannable - use bullet lists for ALL content, never write full paragraphs\n"
            "- Each piece of information should be a separate bullet point\n"
            "- Use Markdown headings for section organization, then bullet lists for all content within sections\n"
            "- Example format:\n"
            "  ### Personality\n"
            "  - Trait: Analytical and methodical\n"
            "  - Strength: Excellent problem-solving under pressure\n"
            "  - Flaw: Tends to overthink simple decisions\n"
            "- NOT this format (avoid paragraphs):\n"
            "  ### Personality\n"
            "  The character is analytical and methodical, with excellent problem-solving skills that shine under pressure. However, they tend to overthink simple decisions...\n"
            "- Preferred major-character scaffold: Basic Information, Personality (traits/strengths/flaws), Dialogue Patterns, Internal Monologue, Relationships, Character Arc.\n"
            "- Supporting cast: concise entries (Role, Traits, Speech, Relation to MC, Notes).\n"
            "- Relationships doc: pairs with Relationship Type, Dynamics, Conflict Sources, Interaction Patterns, Evolution.\n\n"
            "**UNIVERSE RULES (if provided)**: Use for universe consistency and worldbuilding constraints\n"
            "- Rules define the world's constraints: magic systems, technology levels, social structures, geography, timeline, etc.\n"
            "- When developing character abilities, powers, or skills, ensure they align with the universe's magic/technology rules\n"
            "- When adding character background, verify it fits within the established timeline, geography, and social structures\n"
            "- When defining character affiliations or organizations, check rules for established groups, hierarchies, and power structures\n"
            "- When adding character traits that involve world elements (e.g., 'knows ancient magic'), verify against rules for magic systems\n"
            "- Use rules to inform character limitations, capabilities, and constraints within the universe\n"
            "- Example: If rules specify 'magic requires physical components', character abilities should reflect this constraint\n"
            "- Example: If rules define 'medieval technology level', character background shouldn't include modern technology\n"
            "- Example: If rules establish 'noble houses', character affiliations should reference these houses, not create new ones\n\n"
            "**CHARACTER REFERENCES (if provided)**: Use for relationship consistency and character harmony\n"
            "- Each referenced character is a DIFFERENT character with distinct traits, dialogue patterns, and behaviors\n"
            "- Check for relationship consistency (A's view of B should match B's view of A)\n"
            "- Verify trait conflicts/complementarity in relationships\n"
            "- Ensure power dynamics and hierarchies are consistent across character sheets\n"
            "- Use for comparison when user asks about character differences or relationships\n"
            "- When adding relationships, cross-reference the other character's sheet to ensure mutual consistency\n"
            "- When adding traits, consider how they contrast or complement referenced characters\n"
            "- When updating character dynamics, verify consistency with how the relationship is described in other character sheets\n\n"
            "EDIT RULES:\n"
            "0) **WORK FIRST, ASK LATER**: Always make edits based on available information. Use context from the request, existing character content, outline, rules, and related characters to inform your work. Only ask questions in the summary if critical information is missing that prevents meaningful progress. Never return empty operations unless the request is completely impossible.\n"
            "1) **BULLET POINTS ONLY**: Always format content as bullet points, never as paragraphs. Each fact, trait, or piece of information should be its own bullet point.\n"
            "   - When reformatting existing paragraph-based content (e.g., 'trim down', 'convert to bullets', 'make concise'), break paragraphs into individual bullet points\n"
            "   - Extract key facts, traits, and information from paragraphs and convert each into a separate bullet point\n"
            "   - Example: 'The character is analytical and methodical, with excellent problem-solving skills that shine under pressure. However, they tend to overthink simple decisions.'\n"
            "     → Convert to: '- Analytical and methodical\\n- Excellent problem-solving under pressure\\n- Tends to overthink simple decisions'\n"
            "2) Make surgical edits near cursor/selection unless re-organization is requested.\n"
            "3) Maintain existing structure; update in place; avoid duplicate headings.\n"
            "4) Enforce universe consistency against Rules and outline-provided character network.\n"
            "5) NO PLACEHOLDER FILLERS: If a requested section has no content yet, create the heading only and leave the body blank. Do NOT insert placeholders like '[To be developed]' or 'TBD'.\n"
            "6) For GRANULAR REVISIONS: Use replace_range with exact original_text matching the specific text to change (e.g., 'blue eyes' → 'green eyes')\n"
            "7) NEVER insert content at the beginning of the file - always use proper anchors after frontmatter\n"
            "8) **CHECK FOR DUPLICATES IN RELATED SECTIONS** - Before adding character information:\n"
            "   - Check if similar information already exists in related sections\n"
            "   - If trait appears in both Personality AND Character Arc, consider consolidating or updating consistently\n"
            "   - If backstory detail is scattered across sections, consolidate to most appropriate location\n"
            "   - Example: 'Protective of family' in both Personality and Relationships → Update Personality (trait definition), reference in Relationships (how it affects dynamics)\n"
            "   - Avoid redundant identical information - each section should add unique perspective\n\n"
            "9) **CRITICAL: CROSS-REFERENCE RELATED SECTIONS** - When adding or updating character information, you MUST:\n"
            "   - Scan the ENTIRE document for related sections that should be updated together\n"
            "   - Identify ALL sections that reference or relate to the information being added/updated\n"
            "   - Generate MULTIPLE operations if a single addition requires updates to multiple related sections\n"
            "   - Example: If adding a personality trait, check Personality, Relationships, Character Arc, and Dialogue Patterns sections\n"
            "   - Example: If updating a relationship, check Relationships section AND any character arc notes that reference it\n"
            "   - Example: If adding a backstory detail, check Basic Information, Personality, and Character Arc sections\n"
            "   - NEVER update only one section when related sections exist that should be updated together\n"
            "   - The operations array can contain MULTIPLE operations - use it to update all related sections in one pass\n\n"
            "ANCHOR REQUIREMENTS (CRITICAL):\n"
            "For EVERY operation, you MUST provide precise anchors:\n\n"
            "REVISE/DELETE Operations:\n"
            "- ALWAYS include 'original_text' with EXACT, VERBATIM text from the file\n"
            "- For granular edits: Match the EXACT text to change (e.g., if changing 'blue' to 'green', original_text='blue')\n"
            "- For bullet point edits: Each bullet should be concise (5-15 words typically), but include enough context to be meaningful\n"
            "- When replacing existing bullet points: Match the exact original_text including the bullet marker (e.g., '- Original text here')\n"
            "- Copy and paste directly - do NOT retype or modify\n"
            "- NEVER include header lines (###, ##) in original_text!\n"
            "- NEVER target frontmatter - all operations must be after frontmatter end\n"
            "- OR provide both 'left_context' and 'right_context' (exact surrounding text)\n\n"
            "INSERT Operations (ONLY for truly empty sections!):\n"
            "- **insert_after_heading**: Use ONLY when section is completely empty below the header\n"
            "  * op_type='insert_after_heading' with anchor_text='### Header' (exact header line)\n"
            "  * Example: Adding traits after '### Traits' header when section is completely empty\n"
            "  * ⚠️ CRITICAL WARNING: Before using insert_after_heading, you MUST verify the section is COMPLETELY EMPTY!\n"
            "  * ⚠️ If there is ANY text below the header (even a single line), use replace_range instead!\n"
            "  * ⚠️ Using insert_after_heading when content exists will INSERT BETWEEN the header and existing text, splitting the section!\n"
            "  * ⚠️ This creates duplicate content and breaks the section structure - NEVER do this!\n"
            "  * Example of WRONG behavior: '### Personality\\n[INSERT HERE splits section]\\n- Existing trait' ← WRONG! Use replace_range on existing content!\n"
            "  * Example of CORRECT usage: '### Personality\\n[empty - no text below]' ← OK to use insert_after_heading\n"
            "  * This is the SAFEST method - it NEVER deletes headers, always inserts AFTER them - BUT ONLY FOR EMPTY SECTIONS\n\n"
            "- **insert_after**: Use when continuing text mid-bullet, mid-sentence, or after specific text\n"
            "  * op_type='insert_after' with anchor_text='last few words before insertion point'\n"
            "  * Example: Continuing a bullet point or adding to an existing bullet list\n"
            "  * anchor_text should be the exact text (last few words) where you want to insert after\n"
            "  * The resolver will find the end of the bullet point containing the anchor and insert there\n"
            "  * **PREFER**: Adding new bullet points rather than extending existing ones - keep bullets concise\n\n"
            "- **REPLACE Operations (PREFERRED for updating existing content!):\n"
            "- **replace_range**: Use when section exists but needs improvement, completion, or revision\n"
            "  * If section has ANY content (even incomplete or placeholder), use replace_range to update it\n"
            "  * Example: Section has '- Analytical thinker' but needs more traits → replace_range with original_text='- Analytical thinker' and expanded text\n"
            "  * Example: Section has '[To be developed]' → replace_range with original_text='[To be developed]' and actual content\n"
            "  * Example: Section has paragraph text that needs conversion to bullets → replace_range with original_text matching the entire paragraph, and new_text as bullet points\n"
            "  * When reformatting paragraphs to bullets: Match the exact paragraph text in original_text, then provide bullet points in new_text\n"
            "  * This ensures existing content is replaced/updated, not duplicated\n\n"
            "- **CRITICAL ANCHORING RULES**:\n"
            "  * Provide 'anchor_text' with EXACT, COMPLETE text to insert after (verbatim from file)\n"
            "  * Provide 'original_text' with EXACT, VERBATIM existing content to replace (verbatim from file)\n"
            "  * NEVER insert at position 0 or before frontmatter end - always use proper anchors\n"
            "  * ALTERNATIVE: Provide 'original_text' with text to insert after\n"
            "  * FALLBACK: Provide 'left_context' with text before insertion point (minimum 10-20 words)\n\n"
            "- **DECISION TREE**:\n"
            "  **STEP 1: Read the section content carefully!**\n"
            "  - Look at what exists below the header\n"
            "  - Is there ANY text at all? Even a single line?\n"
            "  \n"
            "  **STEP 2: Choose operation based on what exists:**\n"
            "  * Section is COMPLETELY EMPTY below header (no text at all)? → insert_after_heading\n"
            "  * Section has ANY content (even incomplete/placeholder/single line)? → replace_range to update it\n"
            "  * Adding to existing bullet list? → replace_range with original_text matching existing content, or add new bullet points\n"
            "  * Continuing mid-bullet or mid-sentence? → insert_after (but prefer adding new bullet points instead)\n"
            "  * Same info in multiple sections? → Update consistently or consolidate\n"
            "  * **CRITICAL**: When improving/completing existing sections, ALWAYS use replace_range to update, not insert_after_heading (which would duplicate content)\n\n"
            "NO PLACEHOLDER TEXT: Leave empty sections blank, do NOT insert '[To be developed]' or 'TBD'.\n"
        )
    
    async def process(self, query: str, metadata: Dict[str, Any] = None, messages: List[Any] = None) -> Dict[str, Any]:
        """
        Process character development request using LangGraph workflow
        
        Args:
            query: User query string
            metadata: Optional metadata dictionary (persona, editor context, etc.)
            messages: Optional conversation history
            
        Returns:
            Dictionary with character edit response and operations
        """
        try:
            logger.info(f"Character Development Agent: Starting character edit: {query[:100]}...")
            
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
            
            # Build initial state for LangGraph workflow
            initial_state: CharacterDevelopmentState = {
                "query": query,
                "user_id": user_id,
                "metadata": metadata,
                "messages": conversation_messages,
                "shared_memory": shared_memory_merged,
                "active_editor": {},
                "text": "",
                "filename": "character.md",
                "frontmatter": {},
                "cursor_offset": -1,
                "selection_start": -1,
                "selection_end": -1,
                "outline_body": None,
                "rules_body": None,
                "style_text": None,
                "character_bodies": [],
                "para_start": 0,
                "para_end": 0,
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
            
            # Invoke LangGraph workflow with checkpointing (workflow and config already created above)
            final_state = await workflow.ainvoke(initial_state, config=config)
            
            # Extract response from final state
            response = final_state.get("response", {
                "messages": [AIMessage(content="Character development failed")],
                "agent_results": {
                    "agent_type": self.agent_type,
                    "is_complete": False
                },
                "is_complete": False
            })
            
            # Extract editor_operations from state (stored at state level by _format_response_node)
            editor_operations = final_state.get("editor_operations", [])
            # Also check nested response for operations (fallback)
            if not editor_operations:
                editor_operations = response.get("agent_results", {}).get("editor_operations", [])
            
            task_status = final_state.get("task_status", "complete")
            
            logger.info(f"📊 STATE EXTRACTION: editor_operations from state={len(final_state.get('editor_operations', []))}, from response={len(response.get('agent_results', {}).get('editor_operations', []))}, final={len(editor_operations)}")
            
            # Build result dict matching Fiction editing agent pattern
            # Response structure from _format_response_node: { "messages": [...], "agent_results": {...}, "is_complete": True }
            result = {
                "messages": response.get("messages", []),
                "agent_results": response.get("agent_results", {}),
                "is_complete": response.get("is_complete", True)
            }
            
            # Add editor_operations at top level for compatibility with gRPC service
            if editor_operations:
                result["editor_operations"] = editor_operations
                # Also ensure they're in agent_results (they should already be there from _format_response_node)
                if "agent_results" not in result:
                    result["agent_results"] = {}
                result["agent_results"]["editor_operations"] = editor_operations
                # Include manuscript_edit if available
                manuscript_edit = response.get("agent_results", {}).get("manuscript_edit")
                if manuscript_edit:
                    result["manuscript_edit"] = manuscript_edit
                    result["agent_results"]["manuscript_edit"] = manuscript_edit
                logger.info(f"✅ Added {len(editor_operations)} editor operation(s) to result")
            else:
                logger.warning(f"⚠️ No editor_operations found in state or response (state keys: {list(final_state.keys())})")
            
            logger.info(f"Character development agent completed: {task_status}, operations: {len(editor_operations)}")
            return result
            
        except Exception as e:
            logger.error(f"Character Development Agent ERROR: {e}")
            return {
                "messages": [AIMessage(content=f"Character development failed: {str(e)}")],
                "agent_results": {
                    "agent_type": self.agent_type,
                    "is_complete": False
                },
                "is_complete": False
            }


def get_character_development_agent() -> CharacterDevelopmentAgent:
    """Get CharacterDevelopmentAgent instance"""
    return CharacterDevelopmentAgent()

