"""
Simple Agent Conversation Sharing - No Complex State Management

This is what we ACTUALLY need: agents that can share conversation data easily.
No complex state, no over-engineering - just simple conversation sharing.
"""

from typing import List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


class SimpleConversationManager:
    """Dead simple conversation manager that agents can use"""
    
    def __init__(self):
        self.conversations: Dict[str, List[BaseMessage]] = {}
        self.context: Dict[str, Dict[str, Any]] = {}
    
    def add_message(self, conversation_id: str, message: BaseMessage):
        """Add a message to conversation history"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        self.conversations[conversation_id].append(message)
    
    def get_conversation(self, conversation_id: str, limit: int = 20) -> List[BaseMessage]:
        """Get recent conversation messages"""
        messages = self.conversations.get(conversation_id, [])
        return messages[-limit:] if messages else []
    
    def set_context(self, conversation_id: str, key: str, value: Any):
        """Set context for conversation"""
        if conversation_id not in self.context:
            self.context[conversation_id] = {}
        self.context[conversation_id][key] = value
    
    def get_context(self, conversation_id: str, key: str = None):
        """Get context for conversation"""
        conv_context = self.context.get(conversation_id, {})
        if key:
            return conv_context.get(key)
        return conv_context


# Global instance for simple sharing
simple_conversation_manager = SimpleConversationManager()


class SimpleAgent:
    """Base class for agents that need conversation sharing"""
    
    def __init__(self, name: str):
        self.name = name
        self.conversation_manager = simple_conversation_manager
    
    def get_conversation_history(self, conversation_id: str) -> List[BaseMessage]:
        """Get conversation history"""
        return self.conversation_manager.get_conversation(conversation_id)
    
    def add_response(self, conversation_id: str, response: str):
        """Add agent response to conversation"""
        message = AIMessage(content=response)
        message.name = self.name  # Track which agent responded
        self.conversation_manager.add_message(conversation_id, message)
    
    def get_original_question(self, conversation_id: str) -> str:
        """Get the original user question from conversation"""
        messages = self.get_conversation_history(conversation_id)
        for msg in messages:
            if isinstance(msg, HumanMessage) and len(msg.content.strip()) > 10:
                # Skip simple responses like "yes", "no"
                simple_responses = ["yes", "no", "ok", "sure", "go ahead", "proceed", "continue"]
                if msg.content.lower().strip() not in simple_responses:
                    return msg.content
        return ""
    
    def share_context(self, conversation_id: str, key: str, value: Any):
        """Share context with other agents"""
        self.conversation_manager.set_context(conversation_id, f"{self.name}_{key}", value)
    
    def get_shared_context(self, conversation_id: str, agent_name: str = None, key: str = None):
        """Get context shared by other agents"""
        if agent_name and key:
            return self.conversation_manager.get_context(conversation_id, f"{agent_name}_{key}")
        return self.conversation_manager.get_context(conversation_id)


# Example usage:
"""
# In ChatAgent:
class ChatAgent(SimpleAgent):
    def __init__(self):
        super().__init__("chat")
    
    async def process(self, conversation_id: str, query: str):
        # Get conversation history automatically
        history = self.get_conversation_history(conversation_id)
        
        # Process with full context
        response = await self.generate_response(query, history)
        
        # If escalating to research agent
        if "need research" in response:
            # Share context for research agent
            self.share_context(conversation_id, "escalation_reason", "insufficient_local_data")
            self.share_context(conversation_id, "original_question", self.get_original_question(conversation_id))
        
        # Add response to conversation
        self.add_response(conversation_id, response)
        return response

# In ResearchAgent:
class ResearchAgent(SimpleAgent):
    def __init__(self):
        super().__init__("research")
    
    async def process(self, conversation_id: str, query: str):
        # Get full conversation history automatically
        history = self.get_conversation_history(conversation_id)
        
        # Get context shared by chat agent
        original_question = self.get_shared_context(conversation_id, "chat", "original_question")
        escalation_reason = self.get_shared_context(conversation_id, "chat", "escalation_reason")
        
        # Now I know exactly what the user originally asked and why I was called
        if original_question:
            query = f"Research: {original_question}"
        
        response = await self.generate_response(query, history)
        self.add_response(conversation_id, response)
        return response
"""
