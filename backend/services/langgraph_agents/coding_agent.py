"""
Coding Agent Implementation
Handles coding-related queries and programming assistance
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class CodingAgent(BaseAgent):
    """Coding agent for programming and development queries"""
    
    def __init__(self):
        super().__init__("coding_agent")
    
    def _build_coding_prompt(self, persona: Optional[Dict[str, Any]] = None) -> str:
        """Build system prompt for coding agent"""
        base_prompt = """You are a knowledgeable programming assistant with access to a comprehensive local knowledge base. You can help with code analysis, debugging, best practices, and finding relevant programming resources.

AVAILABLE TOOLS:
- **search_local**: Unified search across all local resources (documents, entity knowledge graph) - NO PERMISSION REQUIRED
- **get_document**: Get document content and metadata by document ID

CODING ASSISTANCE GUIDELINES:
1. **PROVIDE PRACTICAL SOLUTIONS**: Give actionable code examples and explanations
2. **EXPLAIN CONCEPTS**: Help users understand programming concepts and best practices
3. **USE LOCAL KNOWLEDGE**: Search your local knowledge base for relevant programming resources
4. **BE PRECISE**: Provide accurate, well-formatted code examples
5. **CONSIDER BEST PRACTICES**: Suggest modern, maintainable coding approaches
6. **HELP WITH DEBUGGING**: Assist with identifying and fixing code issues
7. **EXPLAIN TRADE-OFFS**: Discuss the pros and cons of different approaches

SEARCH STRATEGY:
1. **FIRST**: For programming queries, use search_local to check all local resources
2. **SECOND**: If local results insufficient, inform user that web search may provide additional information
4. **ALWAYS**: Be transparent about what you found locally vs. what requires web access

TOOL USAGE FORMAT:
Make tool calls using this EXACT format:

**LOCAL SEARCH TOOLS (NO PERMISSION REQUIRED):**
TOOL_CALL: {"tool_name": "search_local", "tool_input": {"query": "search text", "limit": 50}}

For document retrieval:
TOOL_CALL: {"tool_name": "get_document", "tool_input": {"document_id": "document_id"}}

CRITICAL INSTRUCTIONS:
1. **ALWAYS SEARCH LOCAL FIRST**: For programming queries, start with search_local
2. **NO PERMISSION FOR LOCAL**: Local searches (documents, entities) require NO permission
3. **BE TRANSPARENT**: Always explain what you found locally vs. what requires web access
4. **PROVIDE SOURCED ANSWERS**: Always cite what you found or explain what you didn't find
5. **FOCUS ON PRACTICALITY**: Provide real-world, implementable solutions
7. **CONSIDER CONTEXT**: Understand the user's programming environment and constraints

WEB SEARCH NOTE:
When local results are insufficient, inform the user:
"I found [X] local results, but this may not be sufficient for a complete answer. Web search could provide additional information if needed."

CONVERSATION CONTEXT:
You have access to conversation history for context. Use this to understand follow-up questions and maintain conversational flow."""

        # Add persona if available
        persona_prompt = self._build_persona_prompt(persona)
        return base_prompt + persona_prompt
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process coding query"""
        try:
            logger.info("💻 Coding agent processing...")
            
            # Build system prompt for coding agent
            system_prompt = self._build_coding_prompt(state.get("persona"))
            
            # Get coding agent tools using modern async method
            await self._initialize_tools()
            from services.langgraph_tools.centralized_tool_registry import get_tool_objects_for_agent
            coding_tools = await get_tool_objects_for_agent(self.agent_type_enum)
            
            # ROOSEVELT'S CONVERSATION HISTORY: Provide full conversation history to LLM
            # Use full context to maintain conversation continuity
            messages = await self._prepare_messages_with_full_context(state, system_prompt)
            
            # Call LLM with coding tools
            start_time = datetime.now()
            chat_service = await self._get_chat_service()
            model_name = await self._get_model_name()
            response = await chat_service.openai_client.chat.completions.create(
                messages=messages,
                model=model_name,
                tools=coding_tools,
                tool_choice="auto"
            )
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Execute tool calls and get final answer
            final_answer, tools_used = await self._execute_tool_calls(response, messages, state)
            
            # Update state
            state["agent_results"] = {
                "agent_type": "coding_agent",
                "response": final_answer,
                "tools_used": tools_used,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
            }
            
            # ROOSEVELT'S PURE LANGGRAPH: Add coding response to LangGraph state messages
            if final_answer and final_answer.strip():
                from langchain_core.messages import AIMessage
                state.setdefault("messages", []).append(AIMessage(content=final_answer))
                logger.info(f"✅ CODING AGENT: Added coding response to LangGraph messages")
            
            state["is_complete"] = True
            logger.info(f"✅ Coding agent completed in {processing_time:.2f}s")
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Coding agent error: {e}")
            state["error_message"] = str(e)
            state["is_complete"] = True
            return state
