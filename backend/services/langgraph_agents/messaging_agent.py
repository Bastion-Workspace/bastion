"""
Roosevelt's Messaging Agent
Natural language interface for sending messages to rooms

BULLY! Send messages like giving cavalry orders!
"""

import logging
import re
from datetime import datetime
from typing import Dict, Any, Optional, List

from .base_agent import BaseAgent
from models.agent_response_models import TaskStatus

logger = logging.getLogger(__name__)


class MessagingAgent(BaseAgent):
    """
    Agent for sending messages to chat rooms via natural language
    
    Examples:
    - "Send a message to Linda: Hi there!"
    - "Tell Bob: Meeting at 3pm"
    - "Message the Dev Team room: PR is ready for review"
    """
    
    def __init__(self):
        super().__init__("messaging_agent")
        logger.info("💬 BULLY! Messaging Agent saddled and ready!")
    
    def _build_messaging_prompt(self) -> str:
        """Build system prompt for messaging agent"""
        return """You are a messaging assistant that helps users send messages to chat rooms.

Your job is to:
1. Parse the user's natural language request to extract:
   - The target room/person name
   - The message content
2. Find the correct room ID from the user's available rooms
3. Send the message to that room

AVAILABLE TOOLS:
- **get_user_rooms**: Get list of user's chat rooms
- **send_room_message**: Send a message to a specific room

PARSING EXAMPLES:
"Send a message to Linda: Hi there!" 
  → target: "Linda", message: "Hi there!"

"Tell Bob: Meeting at 3pm"
  → target: "Bob", message: "Meeting at 3pm"

"Message Dev Team room: PR is ready"
  → target: "Dev Team", message: "PR is ready"

STRUCTURED OUTPUT REQUIRED:
You MUST respond with valid JSON matching this schema:
{
    "task_status": "complete|incomplete|error",
    "target_name": "room or person name",
    "message_content": "the message to send",
    "room_found": true|false,
    "room_id": "uuid or null",
    "clarification_needed": "question if ambiguous",
    "room_options": ["list of possible rooms if ambiguous"],
    "response": "natural language response to user"
}

INSTRUCTIONS:
1. First, parse the user's request to extract target and message
2. Get the user's rooms list
3. Match the target name against room names (exact or fuzzy match)
4. If multiple matches, ask for clarification
5. If one match, send the message
6. If no match, inform user the room wasn't found
7. Always respond in a friendly, helpful tone

IMPORTANT:
- Be case-insensitive when matching room names
- For direct messages (person names), match against participant names in the room
- Handle variations like "send to", "message", "tell", etc.
"""
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process messaging request"""
        try:
            logger.info("💬 Messaging agent processing request...")
            
            user_message = state.get("messages", [])[-1].content if state.get("messages") else ""
            user_id = state.get("user_id")
            
            if not user_id:
                return self._create_error_result("User ID not available")
            
            # Build system prompt
            system_prompt = self._build_messaging_prompt()
            
            # Get messaging agent tools
            await self._initialize_tools()
            from services.langgraph_tools.centralized_tool_registry import get_tool_objects_for_agent
            messaging_tools = await get_tool_objects_for_agent(self.agent_type_enum)
            
            # Prepare messages with conversation history
            messages = await self._prepare_messages(state, system_prompt)
            
            # Call LLM
            start_time = datetime.now()
            chat_service = await self._get_chat_service()
            model_name = await self._get_model_name()
            
            response = await chat_service.openai_client.chat.completions.create(
                messages=messages,
                model=model_name,
                tools=messaging_tools,
                tool_choice="auto"
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Execute tool calls and get final answer
            final_answer, tools_used = await self._execute_tool_calls(response, messages, state)
            
            # Create success result
            return self._create_success_result(
                response=final_answer,
                tools_used=tools_used,
                processing_time=processing_time
            )
        
        except Exception as e:
            logger.error(f"❌ Messaging agent error: {e}")
            return self._create_error_result(f"Messaging agent failed: {str(e)}")


# Singleton instance
messaging_agent = MessagingAgent()

