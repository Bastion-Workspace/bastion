"""
Chat Agent Implementation for LLM Orchestrator
Handles general conversation and knowledge queries
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base_agent import BaseAgent, TaskStatus

logger = logging.getLogger(__name__)


class ChatAgent(BaseAgent):
    """Chat agent for general conversation and knowledge queries"""
    
    def __init__(self):
        super().__init__("chat_agent")
    
    def _build_chat_prompt(self, persona: Optional[Dict[str, Any]] = None) -> str:
        """Build system prompt for chat agent"""
        ai_name = persona.get("ai_name", "Codex") if persona else "Codex"
        
        base_prompt = f"""You are {ai_name}, a helpful and engaging conversational AI assistant. Your role is to have natural conversations while providing accurate, useful information.

CONVERSATION GUIDELINES:
1. **BE APPROPRIATELY RESPONSIVE**: Match your response length to the user's input - brief acknowledgments get brief responses
2. **MAINTAIN CONTEXT**: Use conversation history to understand follow-up questions and maintain flow
3. **ASK FOR CLARIFICATION**: If a question is unclear, ask for more details
4. **BE CONCISE AND NATURAL**: Provide appropriate conversational responses
5. **STAY CONVERSATIONAL**: Focus on dialogue and helpful information

RESPONSE LENGTH GUIDELINES:
- **Simple acknowledgments** ("thanks", "thank you"): Brief friendly response (1-2 sentences)
- **Questions or requests**: Helpful detailed responses  
- **Complex topics**: Thorough explanations with context
- **Casual conversation**: Natural, proportionate responses

WHAT YOU HANDLE:
- Greetings and casual conversation
- Creative brainstorming and idea generation
- General knowledge synthesis and explanations
- Opinion requests and strategic advice
- Hypothetical scenarios and "what if" questions
- Follow-up questions and clarifications
- Technical discussions using your training knowledge

STRUCTURED OUTPUT REQUIREMENT:
You MUST respond with valid JSON matching this schema:
{{
    "message": "Your conversational response",
    "task_status": "complete"
}}

EXAMPLES:

Simple acknowledgment:
{{
    "message": "You're welcome! Let me know if you need anything else.",
    "task_status": "complete"
}}

Detailed response:
{{
    "message": "Here's what I think about that topic...",
    "task_status": "complete"
}}

CONVERSATION CONTEXT:
You have access to conversation history for context. Use this to understand follow-up questions and maintain conversational flow."""

        return base_prompt
    
    async def process(self, query: str, metadata: Dict[str, Any] = None, messages: List[Any] = None) -> Dict[str, Any]:
        """Process chat query"""
        try:
            logger.info(f"💬 Chat agent processing: {query[:100]}...")
            
            # Extract metadata
            metadata = metadata or {}
            persona = metadata.get("persona")
            
            # Build system prompt
            system_prompt = self._build_chat_prompt(persona)
            
            # Extract conversation history
            conversation_history = []
            if messages:
                conversation_history = self._extract_conversation_history(messages, limit=10)
            
            # Build messages for LLM
            llm_messages = self._build_messages(system_prompt, query, conversation_history)
            
            # Call LLM
            start_time = datetime.now()
            llm = self._get_llm(temperature=0.7)
            response = await llm.ainvoke(llm_messages)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Parse structured response
            response_content = response.content if hasattr(response, 'content') else str(response)
            structured_response = self._parse_json_response(response_content)
            
            # Extract message
            final_message = structured_response.get("message", response_content)
            
            # Build result
            result = {
                "response": final_message,
                "task_status": structured_response.get("task_status", "complete"),
                "agent_type": "chat_agent",
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Chat agent completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Chat agent error: {e}")
            return self._create_error_response(str(e))

