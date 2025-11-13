"""
LangGraph Native HITL Handler - Roosevelt's "Best Practices" Implementation
Following official LangGraph HITL patterns with proper interrupt/resume flow
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class LangGraphNativeHITLHandler:
    """
    Roosevelt's LangGraph Native HITL Handler
    
    Follows official LangGraph best practices:
    1. Use interrupt_before for clean breakpoints
    2. Leverage state persistence through checkpointers
    3. Use proper resumption patterns
    4. Stream real-time updates during execution
    """
    
    def __init__(self):
        self.pending_permissions = {}  # Track pending permission requests
    
    async def handle_permission_interrupt(
        self, 
        graph, 
        conversation_id: str, 
        permission_type: str = "web_search"
    ) -> Dict[str, Any]:
        """
        Handle LangGraph interrupt for permission requests
        
        Following LangGraph best practices:
        - Use graph.aget_state() to get current state
        - Extract permission details from state
        - Return structured permission data
        """
        try:
            logger.info(f"🛑 LANGGRAPH HITL: Handling {permission_type} permission interrupt")
            
            # Enforce namespaced thread_id for strict per-user isolation
            from services.orchestrator_utils import normalize_thread_id, validate_thread_id
            # NOTE: This handler now expects caller to provide user_id in conversation_id input or pass separately
            # For compatibility, if conversation_id already includes user_id prefix, use as-is
            thread_id = conversation_id
            try:
                # If caller passed user_id in shared_memory, prefer it
                user_id = None
                # We can't access state yet; guard: expect caller to pass a user_id attr on class if needed
                if hasattr(self, 'current_user_id'):
                    user_id = getattr(self, 'current_user_id')
                if user_id:
                    thread_id = normalize_thread_id(user_id, conversation_id)
                    validate_thread_id(user_id, thread_id)
            except Exception:
                # Best-effort fallback; proceed with given conversation_id
                pass
            config = {
                "configurable": {
                    "thread_id": thread_id
                }
            }
            
            # Get current state from LangGraph
            state = await graph.aget_state(config)
            
            if not state or not state.values:
                logger.error("❌ LANGGRAPH HITL: No state found for conversation")
                return {"error": "No conversation state found"}
            
            # Extract permission details from state (structured-first)
            messages = state.values.get("messages", [])
            shared_memory = state.values.get("shared_memory", {})
            agent_results = state.values.get("agent_results", {})

            # Prefer structured permission message
            permission_message = None
            if isinstance(agent_results, dict):
                # 1) Direct agent_results.permission_message
                perm_msg = agent_results.get("permission_message")
                if isinstance(perm_msg, str) and perm_msg.strip():
                    permission_message = perm_msg
                # 2) Build from structured_response.permission_request justification
                if permission_message is None:
                    structured_response = agent_results.get("structured_response", {})
                    if isinstance(structured_response, dict):
                        perm_justification = structured_response.get("permission_request")
                        if isinstance(perm_justification, str) and perm_justification.strip():
                            permission_message = (
                                "🔍 Local research complete but insufficient.\n\n"
                                f"🌐 Web Search Needed: {perm_justification}\n\n"
                                "Would you like me to search the web for additional information? (yes/no)"
                            )

            # Fallback: scan recent AI messages for a likely permission prompt
            if permission_message is None:
                for msg in reversed(messages):
                    if (hasattr(msg, 'content') and hasattr(msg, 'type') and msg.type == 'ai'):
                        content = str(msg.content)
                        if any(ind in content.lower() for ind in [
                            "permission request", "web search permission",
                            "would you like me to proceed", "reply with \"yes\" or \"no\"",
                            "estimated cost", "safety level"
                        ]):
                            permission_message = content
                            break

            # Derive details
            structured_response = agent_results.get("structured_response", {}) if isinstance(agent_results, dict) else {}
            permission_request_details = structured_response.get("permission_request", "") if isinstance(structured_response, dict) else ""
            query = structured_response.get("original_query", "") if isinstance(structured_response, dict) else ""
            
            # Create structured permission response
            permission_data = {
                "type": "permission_request",
                "permission_type": permission_type,
                "conversation_id": conversation_id,
                "thread_id": conversation_id,
                "message": permission_message or self._create_default_permission_message(permission_type),
                "details": {
                    "reasoning": permission_request_details,
                    "query": query,
                    "estimated_cost": "~$0.05",
                    "safety_level": "low"
                },
                "state_info": {
                    "next_node": state.next,
                    "current_task": shared_memory.get("current_task", "research"),
                    "requires_approval": True
                },
                "timestamp": datetime.now().isoformat(),
                "status": "awaiting_approval"
            }
            
            # Store pending permission for tracking
            self.pending_permissions[conversation_id] = permission_data
            
            logger.info(f"✅ LANGGRAPH HITL: Permission request prepared for {conversation_id}")
            return permission_data
            
        except Exception as e:
            logger.error(f"❌ LANGGRAPH HITL ERROR: {e}")
            return {
                "error": f"Failed to handle permission interrupt: {str(e)}",
                "type": "error"
            }
    
    async def handle_permission_response(
        self, 
        graph, 
        conversation_id: str, 
        user_response: str
    ) -> Dict[str, Any]:
        """
        Handle user permission response and resume LangGraph execution
        
        Following LangGraph best practices:
        - Parse user approval/denial
        - Use proper resumption with new input
        - Clear pending permission state
        """
        try:
            logger.info(f"🔄 LANGGRAPH HITL: Processing permission response: {user_response}")
            
            # Check if we have a pending permission
            if conversation_id not in self.pending_permissions:
                logger.warning(f"⚠️ LANGGRAPH HITL: No pending permission for {conversation_id}")
                return {"error": "No pending permission request found"}
            
            # Parse user response
            approval_keywords = ["yes", "y", "ok", "okay", "sure", "proceed", "approved", "approve", "allow"]
            denial_keywords = ["no", "n", "deny", "decline", "cancel", "stop"]
            
            user_response_lower = user_response.lower().strip()
            is_approved = any(keyword in user_response_lower for keyword in approval_keywords)
            is_denied = any(keyword in user_response_lower for keyword in denial_keywords)
            
            if not is_approved and not is_denied:
                # Ambiguous response, ask for clarification
                return {
                    "type": "clarification_needed",
                    "message": "Please respond with 'yes' to approve or 'no' to decline the web search permission.",
                    "conversation_id": conversation_id
                }
            
            config = {
                "configurable": {
                    "thread_id": conversation_id
                }
            }
            
            if is_approved:
                logger.info("✅ LANGGRAPH HITL: Permission approved, resuming execution")
                
                # Create resume input with approval
                resume_input = {
                    "messages": [{
                        "role": "user",
                        "content": user_response,
                        "timestamp": datetime.now().isoformat()
                    }],
                    "permission_granted": True,
                    "approved_operation": {
                        "type": "web_search",
                        "granted_at": datetime.now().isoformat(),
                        "user_response": user_response
                    }
                }
                
                # Clear pending permission
                del self.pending_permissions[conversation_id]
                
                return {
                    "type": "permission_approved",
                    "status": "resuming",
                    "conversation_id": conversation_id,
                    "resume_input": resume_input,
                    "config": config
                }
                
            else:  # is_denied
                logger.info("❌ LANGGRAPH HITL: Permission denied")
                
                # Create denial response
                denial_response = {
                    "type": "permission_denied",
                    "status": "completed",
                    "conversation_id": conversation_id,
                    "message": "I understand. I'll work with the local information I found.",
                    "final_response": "Permission denied. Working with available local resources only."
                }
                
                # Clear pending permission
                del self.pending_permissions[conversation_id]
                
                return denial_response
                
        except Exception as e:
            logger.error(f"❌ LANGGRAPH PERMISSION RESPONSE ERROR: {e}")
            return {
                "error": f"Failed to process permission response: {str(e)}",
                "type": "error"
            }
    
    def _create_default_permission_message(self, permission_type: str) -> str:
        """Create default permission message if none found in state"""
        if permission_type == "web_search":
            return """I need permission to search the web for current information about this topic since local sources contain no relevant data.

**🔍 Web Search Permission Request:**
Local search returned no relevant results.

Would you like me to proceed with web search? Please reply with "yes" or "no"."""
        
        return f"Permission required for {permission_type}. Please respond with 'yes' or 'no'."
    
    def get_pending_permission(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get pending permission request for a conversation"""
        return self.pending_permissions.get(conversation_id)
    
    def has_pending_permission(self, conversation_id: str) -> bool:
        """Check if conversation has pending permission request"""
        return conversation_id in self.pending_permissions


# Global instance
_hitl_handler_instance: Optional[LangGraphNativeHITLHandler] = None

def get_langgraph_hitl_handler() -> LangGraphNativeHITLHandler:
    """Get singleton HITL handler instance"""
    global _hitl_handler_instance
    if _hitl_handler_instance is None:
        _hitl_handler_instance = LangGraphNativeHITLHandler()
    return _hitl_handler_instance
