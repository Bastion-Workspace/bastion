"""
Org Inbox Agent
LangGraph agent for org-mode inbox management through natural language commands
"""

import logging
import re
from typing import Dict, Any, List, Optional, TypedDict
from langchain_core.messages import AIMessage

from langgraph.graph import StateGraph, END
from orchestrator.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class OrgInboxState(TypedDict):
    """State for org inbox agent LangGraph workflow"""
    query: str
    user_id: str
    metadata: Dict[str, Any]
    messages: List[Any]
    shared_memory: Dict[str, Any]
    user_message: str
    operation: str
    payload: Dict[str, Any]
    response: Dict[str, Any]
    task_status: str
    error: str


class OrgInboxAgent(BaseAgent):
    """
    Org Inbox Agent for org-mode inbox.org management
    
    Handles todo/event/contact/checkbox operations through natural language
    with LLM-powered interpretation and org-mode formatting
    Uses LangGraph workflow for explicit state management
    """
    
    def __init__(self):
        super().__init__("org_inbox_agent")
        self._grpc_client = None
        logger.info("🗂️ Org Inbox Agent ready!")
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """Build LangGraph workflow for org inbox agent"""
        workflow = StateGraph(OrgInboxState)
        
        # Add nodes
        workflow.add_node("prepare_context", self._prepare_context_node)
        workflow.add_node("infer_operation", self._infer_operation_node)
        workflow.add_node("execute_operation", self._execute_operation_node)
        
        # Entry point
        workflow.set_entry_point("prepare_context")
        
        # Linear flow: prepare_context -> infer_operation -> execute_operation -> END
        workflow.add_edge("prepare_context", "infer_operation")
        workflow.add_edge("infer_operation", "execute_operation")
        workflow.add_edge("execute_operation", END)
        
        return workflow.compile(checkpointer=checkpointer)
    
    async def _get_grpc_client(self):
        """Get or create gRPC client for backend tools"""
        if self._grpc_client is None:
            from orchestrator.clients.backend_tool_client import get_backend_tool_client
            self._grpc_client = await get_backend_tool_client()
        return self._grpc_client
    
    async def _prepare_context_node(self, state: OrgInboxState) -> Dict[str, Any]:
        """Prepare context: extract message"""
        try:
            logger.info("📋 Preparing context for org inbox management...")
            
            messages = state.get("messages", [])
            if not messages:
                return {
                    "error": "No user message found for org inbox processing",
                    "task_status": "error"
                }
            
            latest_message = messages[-1]
            user_message = latest_message.content if hasattr(latest_message, 'content') else str(latest_message)
            
            logger.info(f"🗂️ Org Inbox Agent: Processing message: {user_message[:100]}...")
            
            return {
                "user_message": user_message
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare context: {e}")
            return {
                "user_message": "",
                "error": str(e),
                "task_status": "error"
            }
    
    async def _infer_operation_node(self, state: OrgInboxState) -> Dict[str, Any]:
        """Infer operation type from message"""
        try:
            logger.info("🔍 Inferring operation type...")
            
            user_message = state.get("user_message", "")
            shared_memory = state.get("shared_memory", {})
            
            # Determine operation type from message
            operation = await self._infer_operation(user_message, shared_memory)
            payload = shared_memory.get("org_inbox_payload", {})
            
            logger.info(f"🗂️ Org Inbox Agent: Inferred operation: {operation}")
            
            return {
                "operation": operation,
                "payload": payload
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to infer operation: {e}")
            return {
                "operation": "list",  # Default to list
                "payload": {},
                "error": str(e)
            }
    
    async def _execute_operation_node(self, state: OrgInboxState) -> Dict[str, Any]:
        """Execute the inferred operation"""
        try:
            logger.info("⚙️ Executing org inbox operation...")
            
            operation = state.get("operation", "list")
            user_message = state.get("user_message", "")
            user_id = state.get("user_id", "")
            payload = state.get("payload", {})
            shared_memory = state.get("shared_memory", {})
            
            # Execute operation
            if operation == "add":
                result = await self._handle_add_operation(user_message, user_id, shared_memory, payload)
            elif operation == "list":
                result = await self._handle_list_operation(user_id)
            elif operation == "toggle":
                result = await self._handle_toggle_operation(user_id, payload)
            elif operation == "update":
                result = await self._handle_update_operation(user_id, payload)
            elif operation == "schedule":
                result = await self._handle_schedule_operation(user_id, payload)
            elif operation == "archive_done":
                result = await self._handle_archive_operation(user_id)
            else:
                # Default to list if unclear
                result = await self._handle_list_operation(user_id)
            
            logger.info("✅ Org Inbox Agent: Completed inbox processing")
            
            return {
                "response": result,
                "task_status": "complete"
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to execute operation: {e}")
            return {
                "response": self._create_error_result(f"Org inbox operation failed: {str(e)}"),
                "task_status": "error",
                "error": str(e)
            }
    
    async def process(self, query: str, metadata: Dict[str, Any] = None, messages: List[Any] = None) -> Dict[str, Any]:
        """
        Process org inbox management commands using LangGraph workflow
        
        Args:
            query: User query string
            metadata: Optional metadata dictionary (persona, editor context, etc.)
            messages: Optional conversation history
            
        Returns:
            Dictionary with org inbox operation response and metadata
        """
        try:
            logger.info(f"🗂️ Org Inbox Agent: Starting inbox processing: {query[:100]}...")
            
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
            
            # Merge with any shared_memory from incoming metadata
            shared_memory_merged = shared_memory.copy()
            shared_memory_merged.update(existing_shared_memory)
            
            # Build initial state for LangGraph workflow
            initial_state: OrgInboxState = {
                "query": query,
                "user_id": user_id,
                "metadata": metadata,
                "messages": conversation_messages,
                "shared_memory": shared_memory_merged,
                "user_message": "",
                "operation": "",
                "payload": {},
                "response": {},
                "task_status": "",
                "error": ""
            }
            
            # Invoke LangGraph workflow with checkpointing
            final_state = await workflow.ainvoke(initial_state, config=config)
            
            # Return response from final state
            return final_state.get("response", {
                "messages": [AIMessage(content="Org inbox operation failed")],
                "agent_results": {
                    "agent_type": self.agent_type,
                    "is_complete": False
                },
                "is_complete": False
            })
            
        except Exception as e:
            logger.error(f"❌ Org Inbox Agent ERROR: {e}")
            return self._create_error_result(f"Org inbox operation failed: {str(e)}")
    
    async def _infer_operation(self, user_message: str, state: Dict[str, Any]) -> str:
        """Infer the operation from user message"""
        lowered = user_message.lower()
        
        # Check for explicit operation in state first
        op = (state.get("org_inbox_operation") or "").lower()
        if op:
            return op
        
        # Pattern matching for operations
        if any(k in lowered for k in ["add ", "capture ", "note ", "todo ", "remember ", "save "]):
            return "add"
        elif any(k in lowered for k in ["list", "show", "review", "inbox", "what's in", "see my"]):
            return "list"
        elif any(k in lowered for k in ["done", "complete", "toggle", "mark as done"]):
            return "toggle"
        elif any(k in lowered for k in ["edit", "update", "change", "modify"]):
            return "update"
        elif any(k in lowered for k in ["schedule", "set schedule", "set date"]):
            return "schedule"
        elif any(k in lowered for k in ["archive", "archive done", "clean up done"]):
            return "archive_done"
        
        # Default to list
        return "list"
    
    async def _handle_add_operation(
        self, 
        user_message: str, 
        user_id: str, 
        state: Dict[str, Any],
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle adding items to org inbox with LLM interpretation"""
        try:
            # Use LLM to interpret the add request
            interpretation = await self._llm_interpret_add(state, user_message, payload)
            
            if interpretation.get("clarification_needed"):
                return self._create_response(
                    success=True,
                    response=interpretation.get("clarification_question", "I need more information."),
                    is_complete=False
                )
            
            grpc_client = await self._get_grpc_client()
            
            title = interpretation.get("title", "").strip()
            entry_kind = interpretation.get("entry_kind", "todo")
            schedule = interpretation.get("schedule")
            repeater = interpretation.get("repeater")
            suggested_tags = interpretation.get("suggested_tags", [])
            contact_properties = interpretation.get("contact_properties")
            
            # Add the item via gRPC
            result = await grpc_client.add_org_inbox_item(
                user_id=user_id,
                text=title,
                kind=entry_kind,
                schedule=schedule,
                repeater=repeater,
                tags=suggested_tags,
                contact_properties=contact_properties
            )
            
            if not result.get("success"):
                return self._create_error_result(result.get("error", "Failed to add item"))
            
            # Build response message
            response_parts = []
            if entry_kind == "contact":
                response_parts.append(f"Added contact '{title}' to inbox.org")
            elif entry_kind == "event":
                response_parts.append(f"Added event '{title}' to inbox.org")
            else:
                response_parts.append(f"Added TODO '{title}' to inbox.org")
            
            if schedule:
                response_parts.append(f"(scheduled {schedule}")
                if repeater:
                    response_parts[-1] += f", repeats {repeater}"
                response_parts[-1] += ")"
            
            if suggested_tags:
                response_parts.append(f"| tags: {':'.join(suggested_tags)}")
            
            # Use LLM's confirmation if available
            if interpretation.get("assistant_confirmation"):
                response_text = interpretation.get("assistant_confirmation")
            else:
                response_text = " ".join(response_parts)
            
            return self._create_response(
                success=True,
                response=response_text,
                org_inbox_result={
                    "action": "add",
                    "line_index": result.get("line_index"),
                    "title": title,
                    "kind": entry_kind
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Org Inbox Agent: Add operation failed: {e}")
            return self._create_error_result(f"Failed to add item: {str(e)}")
    
    async def _handle_list_operation(self, user_id: str) -> Dict[str, Any]:
        """Handle listing org inbox items"""
        try:
            grpc_client = await self._get_grpc_client()
            
            result = await grpc_client.list_org_inbox_items(user_id=user_id)
            
            if not result.get("success"):
                return self._create_error_result(result.get("error", "Failed to list items"))
            
            items = result.get("items", [])
            path = result.get("path", "inbox.org")
            
            if not items:
                response_text = f"Your inbox ({path}) is empty."
            else:
                response_parts = [f"📋 Inbox has {len(items)} items:"]
                for i, item in enumerate(items[:10], 1):  # Show first 10
                    status_icon = "✅" if item.get("is_done") else "⬜"
                    todo_state = item.get("todo_state", "")
                    text = item.get("text", "")
                    tags = item.get("tags", [])
                    
                    item_line = f"{i}. {status_icon} {todo_state} {text}".strip()
                    if tags:
                        item_line += f" :{':'.join(tags)}:"
                    response_parts.append(item_line)
                
                if len(items) > 10:
                    response_parts.append(f"... and {len(items) - 10} more items")
                
                response_text = "\n".join(response_parts)
            
            return self._create_response(
                success=True,
                response=response_text,
                org_inbox_result={
                    "action": "list",
                    "count": len(items),
                    "path": path
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Org Inbox Agent: List operation failed: {e}")
            return self._create_error_result(f"Failed to list items: {str(e)}")
    
    async def _handle_toggle_operation(self, user_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle toggling DONE status"""
        try:
            line_index = int(payload.get("line_index", -1))
            if line_index < 0:
                return self._create_error_result("Line index required for toggle operation")
            
            grpc_client = await self._get_grpc_client()
            
            result = await grpc_client.toggle_org_inbox_item(
                user_id=user_id,
                line_index=line_index
            )
            
            if not result.get("success"):
                return self._create_error_result(result.get("error", "Failed to toggle item"))
            
            response_text = f"Toggled item at line {result.get('updated_index', line_index)}"
            
            return self._create_response(
                success=True,
                response=response_text,
                org_inbox_result={
                    "action": "toggle",
                    "line_index": result.get("updated_index")
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Org Inbox Agent: Toggle operation failed: {e}")
            return self._create_error_result(f"Failed to toggle item: {str(e)}")
    
    async def _handle_update_operation(self, user_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle updating item text"""
        try:
            line_index = int(payload.get("line_index", -1))
            new_text = payload.get("new_text", "")
            
            if line_index < 0 or not new_text:
                return self._create_error_result("Line index and new text required for update operation")
            
            grpc_client = await self._get_grpc_client()
            
            result = await grpc_client.update_org_inbox_item(
                user_id=user_id,
                line_index=line_index,
                new_text=new_text
            )
            
            if not result.get("success"):
                return self._create_error_result(result.get("error", "Failed to update item"))
            
            response_text = f"Updated line {result.get('updated_index', line_index)}: {new_text.strip()}"
            
            return self._create_response(
                success=True,
                response=response_text,
                org_inbox_result={
                    "action": "update",
                    "line_index": result.get("updated_index")
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Org Inbox Agent: Update operation failed: {e}")
            return self._create_error_result(f"Failed to update item: {str(e)}")
    
    async def _handle_schedule_operation(self, user_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle setting schedule and repeater"""
        try:
            line_index = int(payload.get("line_index", -1))
            scheduled = payload.get("scheduled")
            repeater = payload.get("repeater")
            
            if line_index < 0 or not scheduled:
                return self._create_error_result("Line index and schedule required")
            
            grpc_client = await self._get_grpc_client()
            
            result = await grpc_client.set_org_inbox_schedule(
                user_id=user_id,
                line_index=line_index,
                scheduled=scheduled,
                repeater=repeater
            )
            
            if not result.get("success"):
                return self._create_error_result(result.get("error", "Failed to set schedule"))
            
            response_text = "Updated schedule"
            if repeater:
                response_text += f" with repeater {repeater}"
            
            return self._create_response(
                success=True,
                response=response_text,
                org_inbox_result={
                    "action": "schedule",
                    "line_index": result.get("updated_index")
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Org Inbox Agent: Schedule operation failed: {e}")
            return self._create_error_result(f"Failed to set schedule: {str(e)}")
    
    async def _handle_archive_operation(self, user_id: str) -> Dict[str, Any]:
        """Handle archiving DONE items"""
        try:
            grpc_client = await self._get_grpc_client()
            
            result = await grpc_client.archive_org_inbox_done(user_id=user_id)
            
            if not result.get("success"):
                return self._create_error_result(result.get("error", "Failed to archive"))
            
            archived_count = result.get("archived_count", 0)
            response_text = result.get("message", f"Archived {archived_count} DONE items")
            
            return self._create_response(
                success=True,
                response=response_text,
                org_inbox_result={
                    "action": "archive_done",
                    "archived_count": archived_count
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Org Inbox Agent: Archive operation failed: {e}")
            return self._create_error_result(f"Failed to archive: {str(e)}")
    
    async def _llm_interpret_add(
        self, 
        state: Dict[str, Any], 
        user_message: str, 
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use LLM to interpret add requests with conversation context"""
        try:
            from langchain_core.messages import HumanMessage
            
            text = (payload.get("text") or user_message or "").strip()
            
            # Build conversation context
            messages = state.get("messages", [])[-5:]
            context_lines = []
            for m in messages:
                role = getattr(m, "type", None) or getattr(m, "role", "user")
                content = getattr(m, "content", "")
                context_lines.append(f"- {role}: {content}")
            context_block = "\n".join(context_lines) if context_lines else ""
            
            # Get persona style
            persona = (state.get("persona") or {}).get("persona_style") or "professional"
            
            # Build LLM prompt
            prompt = f"""You are an Org-Mode Personal Assistant. Analyze the user's request and produce structured JSON for execution.

USER MESSAGE:
{text}

CONTEXT (recent conversation):
{context_block}

REQUIREMENTS:
- Resolve pronouns like "it/this/that" to actionable phrases from the message
- Choose entry_kind: "todo" for tasks, "event" for appointments/meetings/birthdays, "contact" for people, "checkbox" for quick lists
- For contacts, extract properties into contact_properties object:
  * Name fields: FIRST_NAME, MIDDLE_NAME, LAST_NAME
  * Contact info: EMAIL_HOME, EMAIL_WORK, PHONE_MOBILE, PHONE_WORK
  * Organization: COMPANY, TITLE
  * Location: ADDRESS_HOME, ADDRESS_WORK
  * Personal: BIRTHDAY (YYYY-MM-DD), ANNIVERSARY (YYYY-MM-DD), RELATIONSHIP
  * Notes: Additional info in NOTES
- Extract schedule as org timestamp <YYYY-MM-DD Dow> if date/time present; else null
- If repeating cadence (weekly, daily, monthly), return repeater like +1w, .+1m; else null
- Suggest up to 3 tags as simple lowercase slugs
- If ambiguous, set clarification_needed true and propose clarification_question
- Generate assistant_confirmation in {persona} tone describing what was added

OUTPUT JSON SCHEMA:
{{
  "title": "string",
  "entry_kind": "todo" | "event" | "contact" | "checkbox",
  "schedule": "<YYYY-MM-DD Dow>" | null,
  "repeater": "+1w" | ".+1w" | "+1m" | null,
  "suggested_tags": ["string"],
  "contact_properties": {{"EMAIL": "string", "PHONE": "string"}} | null,
  "clarification_needed": true|false,
  "clarification_question": "string|null",
  "assistant_confirmation": "string|null"
}}

Respond with ONLY the JSON."""
            
            # Use centralized LLM mechanism
            llm = self._get_llm(temperature=0.2, state=state)
            llm_messages = [HumanMessage(content=prompt)]
            response = await llm.ainvoke(llm_messages)
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            import json
            try:
                data = json.loads(response_content)
            except Exception:
                # Fallback to simple structure
                data = {
                    "title": text.strip().rstrip("?.!"),
                    "entry_kind": "todo",
                    "schedule": None,
                    "repeater": None,
                    "suggested_tags": [],
                    "clarification_needed": False,
                    "clarification_question": None
                }
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Org Inbox Agent: LLM interpretation failed: {e}")
            # Fallback
            return {
                "title": text.strip() if text else "New entry",
                "entry_kind": "todo",
                "schedule": None,
                "repeater": None,
                "suggested_tags": [],
                "clarification_needed": False,
                "clarification_question": None
            }
    
    def _create_response(
        self, 
        success: bool, 
        response: str, 
        org_inbox_result: Dict[str, Any] = None,
        is_complete: bool = True
    ) -> Dict[str, Any]:
        """Create standardized response"""
        return {
            "messages": [AIMessage(content=response)],
            "agent_results": {
                "agent_type": "org_inbox_agent",
                "success": success,
                "org_inbox": org_inbox_result or {},
                "is_complete": is_complete
            },
            "is_complete": is_complete
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result"""
        logger.error(f"❌ Org Inbox Agent error: {error_message}")
        return {
            "messages": [AIMessage(content=f"Org inbox operation failed: {error_message}")],
            "agent_results": {
                "agent_type": "org_inbox_agent",
                "success": False,
                "error": error_message,
                "is_complete": True
            },
            "is_complete": True
        }


# Singleton instance
_org_inbox_agent_instance = None


def get_org_inbox_agent() -> OrgInboxAgent:
    """Get global org inbox agent instance"""
    global _org_inbox_agent_instance
    if _org_inbox_agent_instance is None:
        _org_inbox_agent_instance = OrgInboxAgent()
    return _org_inbox_agent_instance

