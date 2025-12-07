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
from typing import Dict, Any, List, Optional, TypedDict
from langchain_core.messages import AIMessage

from langgraph.graph import StateGraph, END
from orchestrator.agents.base_agent import BaseAgent

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
        workflow.add_node("generate_edit_plan", self._generate_edit_plan_node)
        workflow.add_node("process_operations", self._process_operations_node)
        
        # Entry point
        workflow.set_entry_point("prepare_context")
        
        # Linear flow: prepare_context -> load_references -> generate_edit_plan -> process_operations -> END
        workflow.add_edge("prepare_context", "load_references")
        workflow.add_edge("load_references", "generate_edit_plan")
        workflow.add_edge("generate_edit_plan", "process_operations")
        workflow.add_edge("process_operations", END)
        
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
        """Load referenced context files (outline, rules, style, characters)"""
        try:
            logger.info("Loading referenced context files...")
            
            active_editor = state.get("active_editor", {})
            filename = state.get("filename", "character.md")
            frontmatter = state.get("frontmatter", {})
            
            # Use backend tool client to load file context
            # For now, we'll load via document search if references exist
            # In a full implementation, this would use a dedicated file context loader tool
            outline_body = None
            rules_body = None
            style_text = None
            character_bodies = []
            
            # If frontmatter has references, we could load them here
            # For now, we'll proceed without loading external files
            # This can be enhanced later with a dedicated backend tool
            
            return {
                "outline_body": outline_body,
                "rules_body": rules_body,
                "style_text": style_text,
                "character_bodies": character_bodies
            }
            
        except Exception as e:
            logger.error(f"Failed to load references: {e}")
            return {
                "outline_body": None,
                "rules_body": None,
                "style_text": None,
                "character_bodies": [],
                "error": str(e)
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
            
            if current_request and any(k in current_request.lower() for k in ["revise", "revision", "tweak", "adjust", "polish", "tighten"]):
                context_parts.append("REVISION MODE: Apply minimal targeted edits; use paragraph-level replace_range ops.\n\n")
            
            context_parts.append("Provide a ManuscriptEdit JSON plan strictly within scope.")
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": f"Current Date/Time: {datetime.now().isoformat()}"},
                {"role": "user", "content": "".join(context_parts)}
            ]
            
            if current_request:
                messages.append({
                    "role": "user",
                    "content": (
                        f"USER REQUEST: {current_request}\n\n"
                        "CRITICAL ANCHORING INSTRUCTIONS:\n"
                        "- For REVISE/DELETE: Provide 'original_text' with EXACT, VERBATIM text from file (10-20+ words, complete sentences)\n"
                        "- For INSERT: Provide 'anchor_text' or 'original_text' with exact line to insert after, OR 'left_context' with text before insertion\n"
                        "- Copy text directly from the file - do NOT retype or paraphrase\n"
                        "- Without precise anchors, the operation WILL FAIL"
                    )
                })
            
            # Call LLM using BaseAgent's _get_llm method - pass state to access user's model selection
            llm = self._get_llm(temperature=0.35, state=state)
            start_time = datetime.now()
            
            # Convert messages to LangChain format
            from langchain_core.messages import SystemMessage, HumanMessage
            langchain_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    langchain_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
            
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
                    "task_status": "error"
                }
            
            if structured_edit is None:
                return {
                    "llm_response": content,
                    "structured_edit": None,
                    "error": "Failed to produce a valid Character edit plan. Ensure ONLY raw JSON ManuscriptEdit with operations is returned (no code fences or prose).",
                    "task_status": "error"
                }
            
            return {
                "llm_response": content,
                "structured_edit": structured_edit,
                "system_prompt": system_prompt
            }
            
        except Exception as e:
            logger.error(f"Failed to generate edit plan: {e}")
            return {
                "llm_response": "",
                "structured_edit": None,
                "error": str(e),
                "task_status": "error"
            }
    
    async def _process_operations_node(self, state: CharacterDevelopmentState) -> Dict[str, Any]:
        """Process and resolve editor operations"""
        try:
            logger.info("Processing editor operations...")
            
            text = state.get("text", "")
            filename = state.get("filename", "character.md")
            structured_edit = state.get("structured_edit")
            para_start = state.get("para_start", 0)
            para_end = state.get("para_end", 0)
            selection_start = state.get("selection_start", -1)
            selection_end = state.get("selection_end", -1)
            current_request = state.get("current_request", "")
            
            if not structured_edit or not structured_edit.get("operations"):
                return {
                    "response": {
                        "messages": [AIMessage(content="No operations to process.")],
                        "agent_results": {
                            "agent_type": self.agent_type,
                            "is_complete": True
                        },
                        "is_complete": True
                    },
                    "task_status": "complete"
                }
            
            # Process operations
            # For now, we'll do basic processing without the full resolver
            # In production, this would call the backend resolver via gRPC
            editor_operations = []
            
            revision_mode = any(k in (current_request or "").lower() for k in ["revise", "revision", "tweak", "adjust", "polish", "tighten", "edit only"])
            
            # Calculate frontmatter end
            try:
                body_only_text = _strip_frontmatter_block(text)
                fm_end_idx = len(text) - len(body_only_text)
            except Exception:
                fm_end_idx = 0
            
            for op in structured_edit.get("operations", []):
                if not isinstance(op, dict):
                    continue
                
                start_ix = int(op.get("start", para_start))
                end_ix = int(op.get("end", para_end))
                op_type = op.get("op_type", "replace_range")
                
                if op_type not in ("replace_range", "delete_range", "insert_after_heading"):
                    op_type = "replace_range"
                
                # Basic processing - in production would use full resolver
                start = max(0, min(len(text), start_ix))
                end = max(0, min(len(text), end_ix))
                
                # Protect frontmatter
                if start < fm_end_idx:
                    if op_type == "delete_range":
                        continue
                    if end <= fm_end_idx:
                        start = fm_end_idx
                        end = fm_end_idx
                    else:
                        start = fm_end_idx
                
                # Clamp for revision mode
                if revision_mode and op_type != "delete_range":
                    if selection_start >= 0 and selection_end > selection_start:
                        start = max(selection_start, start)
                        end = min(selection_end, end)
                
                # Clean text
                op_text = op.get("text", "")
                if isinstance(op_text, str):
                    op_text = _strip_frontmatter_block(op_text)
                
                # Build operation
                processed_op = {
                    "op_type": op_type,
                    "start": start,
                    "end": end,
                    "text": op_text,
                    "original_text": op.get("original_text"),
                    "anchor_text": op.get("anchor_text"),
                    "left_context": op.get("left_context"),
                    "right_context": op.get("right_context"),
                    "occurrence_index": op.get("occurrence_index", 0),
                    "pre_hash": _slice_hash(text[start:end]) if start < end else ""
                }
                
                editor_operations.append(processed_op)
            
            # Build response
            preview = "\n\n".join([op.get("text", "").strip() for op in editor_operations if op.get("text", "").strip()])
            response_text = preview if preview else (structured_edit.get("summary") or "Edit plan ready.")
            
            result = {
                "messages": [AIMessage(content=response_text)],
                "agent_results": {
                    "agent_type": self.agent_type,
                    "is_complete": True,
                    "content_preview": response_text[:2000],
                    "editor_operations": editor_operations,
                    "manuscript_edit": {
                        **structured_edit,
                        "operations": editor_operations
                    }
                },
                "is_complete": True
            }
            
            return {
                "response": result,
                "editor_operations": editor_operations,
                "task_status": "complete"
            }
            
        except Exception as e:
            logger.error(f"Failed to process operations: {e}")
            return {
                "response": {
                    "messages": [AIMessage(content=f"Character agent encountered an error: {str(e)}")],
                    "agent_results": {
                        "agent_type": self.agent_type,
                        "is_complete": False
                    },
                    "is_complete": False
                },
                "task_status": "error",
                "error": str(e)
            }
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for character development"""
        return (
            "You are a Character Development Assistant for type: character files. Persona disabled."
            " Preserve frontmatter; write clean Markdown in body.\n\n"
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
            "      \"op_type\": one of [\"replace_range\", \"delete_range\", \"insert_after_heading\"],\n"
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
            "- Do NOT include explanatory text before or after the JSON.\n\n"
            "FORMATTING CONTRACT (CHARACTER FILES):\n"
            "- Never emit YAML frontmatter in operations[].text; preserve existing YAML.\n"
            "- Use Markdown headings and bullet lists for sections.\n"
            "- Preferred major-character scaffold: Basic Information, Personality (traits/strengths/flaws), Dialogue Patterns, Internal Monologue, Relationships, Character Arc.\n"
            "- Supporting cast: concise entries (Role, Traits, Speech, Relation to MC, Notes).\n"
            "- Relationships doc: pairs with Relationship Type, Dynamics, Conflict Sources, Interaction Patterns, Evolution.\n\n"
            "EDIT RULES:\n"
            "1) Make surgical edits near cursor/selection unless re-organization is requested.\n"
            "2) Maintain existing structure; update in place; avoid duplicate headings.\n"
            "3) Enforce universe consistency against Rules and outline-provided character network.\n"
            "4) NO PLACEHOLDER FILLERS: If a requested section has no content yet, create the heading only and leave the body blank. Do NOT insert placeholders like '[To be developed]' or 'TBD'.\n"
            "ANCHOR REQUIREMENTS (CRITICAL):\n"
            "For EVERY operation, you MUST provide precise anchors:\n\n"
            "REVISE/DELETE Operations:\n"
            "- ALWAYS include 'original_text' with EXACT, VERBATIM text from the file\n"
            "- Minimum 10-20 words, include complete sentences with natural boundaries\n"
            "- Copy and paste directly - do NOT retype or modify\n"
            "- ⚠️ NEVER include header lines (###, ##) in original_text!\n"
            "- OR provide both 'left_context' and 'right_context' (exact surrounding text)\n\n"
            "INSERT Operations (PREFERRED for adding content below headers!):\n"
            "- **PRIMARY METHOD**: Use op_type='insert_after_heading' with anchor_text='### Header' when adding content below ANY header\n"
            "- Provide 'anchor_text' with EXACT, COMPLETE header line to insert after (verbatim from file)\n"
            "- This is the SAFEST method - it NEVER deletes headers, always inserts AFTER them\n"
            "- Use this even when the section has placeholder text - the resolver will position correctly\n"
            "- ALTERNATIVE: Provide 'original_text' with text to insert after\n"
            "- FALLBACK: Provide 'left_context' with text before insertion point (minimum 10-20 words)\n\n"
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
            
            # Return response from final state
            return final_state.get("response", {
                "messages": [AIMessage(content="Character development failed")],
                "agent_results": {
                    "agent_type": self.agent_type,
                    "is_complete": False
                },
                "is_complete": False
            })
            
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

