"""
Base Agent Class for LLM Orchestrator Agents
Provides common functionality for all agents running in the llm-orchestrator microservice
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config.settings import settings

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Agent task completion status"""
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    PERMISSION_REQUIRED = "permission_required"
    ERROR = "error"


class BaseAgent:
    """
    Base class for all LLM Orchestrator agents
    
    Provides:
    - LLM access with proper configuration
    - Message history management
    - Common helper methods
    - Error handling patterns
    """
    
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.llm = None
        logger.info(f"Initializing {agent_type} agent")
    
    def _get_llm(self, temperature: float = 0.7, model: Optional[str] = None) -> ChatOpenAI:
        """Get configured LLM instance"""
        return ChatOpenAI(
            model=model or settings.DEFAULT_MODEL,
            openai_api_key=settings.OPENROUTER_API_KEY,
            openai_api_base=settings.OPENROUTER_BASE_URL,
            temperature=temperature
        )
    
    def _extract_conversation_history(self, messages: List[Any], limit: int = 10) -> List[Dict[str, str]]:
        """Extract conversation history from LangChain messages"""
        try:
            history = []
            for msg in messages[-limit:]:
                if hasattr(msg, 'content'):
                    role = "assistant" if hasattr(msg, 'type') and msg.type == "ai" else "user"
                    history.append({
                        "role": role,
                        "content": msg.content
                    })
            return history
        except Exception as e:
            logger.error(f"Failed to extract conversation history: {e}")
            return []
    
    def _build_messages(self, system_prompt: str, user_query: str, conversation_history: List[Dict[str, str]] = None) -> List[Any]:
        """Build message list for LLM"""
        messages = [SystemMessage(content=system_prompt)]
        
        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))
        
        # Add current query
        messages.append(HumanMessage(content=user_query))
        
        return messages
    
    def _create_error_response(self, error_message: str, task_status: TaskStatus = TaskStatus.ERROR) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "task_status": task_status.value,
            "response": f"Error: {error_message}",
            "error_message": error_message,
            "timestamp": datetime.now().isoformat()
        }
    
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON response from LLM, handling markdown code blocks"""
        import json
        import re
        
        try:
            # Remove markdown code blocks if present
            json_text = content.strip()
            
            if '```json' in json_text:
                match = re.search(r'```json\s*\n(.*?)\n```', json_text, re.DOTALL)
                if match:
                    json_text = match.group(1).strip()
            elif '```' in json_text:
                match = re.search(r'```\s*\n(.*?)\n```', json_text, re.DOTALL)
                if match:
                    json_text = match.group(1).strip()
            
            # Find JSON object if mixed with other text
            if not json_text.startswith('{'):
                match = re.search(r'\{.*\}', json_text, re.DOTALL)
                if match:
                    json_text = match.group(0)
            
            return json.loads(json_text)
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            # Return the raw content wrapped in a simple structure
            return {
                "response": content,
                "task_status": "complete",
                "parsing_fallback": True
            }
    
    async def process(self, query: str, metadata: Dict[str, Any] = None, messages: List[Any] = None) -> Dict[str, Any]:
        """
        Process agent request - to be implemented by subclasses
        
        Args:
            query: User query string
            metadata: Optional metadata dictionary (persona, editor context, etc.)
            messages: Optional conversation history
            
        Returns:
            Dictionary with agent response
        """
        raise NotImplementedError("Subclasses must implement process() method")

