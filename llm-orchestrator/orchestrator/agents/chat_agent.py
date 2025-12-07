"""
Chat Agent Implementation for LLM Orchestrator
Handles general conversation and knowledge queries
"""

import logging
from typing import Dict, Any, List, Optional, TypedDict
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from .base_agent import BaseAgent, TaskStatus

logger = logging.getLogger(__name__)


class ChatState(TypedDict):
    """State for chat agent LangGraph workflow"""
    query: str
    user_id: str
    metadata: Dict[str, Any]
    messages: List[Any]
    persona: Optional[Dict[str, Any]]
    system_prompt: str
    conversation_history: List[Dict[str, str]]
    llm_messages: List[Any]
    response: Dict[str, Any]
    task_status: str
    error: str
    shared_memory: Dict[str, Any]  # For storing primary_agent_selected and continuity data


class ChatAgent(BaseAgent):
    """Chat agent for general conversation and knowledge queries"""
    
    def __init__(self):
        super().__init__("chat_agent")
        logger.info("💬 Chat Agent ready for conversation!")
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """Build LangGraph workflow for chat agent"""
        workflow = StateGraph(ChatState)
        
        # Add nodes
        workflow.add_node("prepare_context", self._prepare_context_node)
        workflow.add_node("generate_response", self._generate_response_node)
        
        # Entry point
        workflow.set_entry_point("prepare_context")
        
        # Linear flow: prepare context -> generate response -> END
        workflow.add_edge("prepare_context", "generate_response")
        workflow.add_edge("generate_response", END)
        
        # Compile with checkpointer for state persistence
        return workflow.compile(checkpointer=checkpointer)
    
    def _build_chat_prompt(self, persona: Optional[Dict[str, Any]] = None) -> str:
        """Build system prompt for chat agent"""
        ai_name = persona.get("ai_name", "Alex") if persona else "Alex"
        persona_style = persona.get("persona_style", "professional") if persona else "professional"
        
        # Build style instruction based on persona_style
        style_instruction = self._get_style_instruction(persona_style)
        
        base_prompt = f"""You are {ai_name}, a conversational AI assistant. Your role is to have natural conversations while providing accurate, useful information.

{style_instruction}

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

PROJECT GUIDANCE:
- If user asks about electronics/circuits/components without an electronics project open:
  * Suggest: "To work on electronics projects, create one first: Right-click a folder → 'New Project' → select 'Electronics'."
  * Then provide general information if helpful
- If user asks about project-specific work (e.g., "add a component to our system") without a project open:
  * Guide them to create a project first using the same instructions

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
    
    async def _prepare_context_node(self, state: ChatState) -> Dict[str, Any]:
        """Prepare context: extract persona, build prompt, extract conversation history"""
        try:
            logger.info(f"💬 Preparing context for chat query: {state['query'][:100]}...")
            
            # Extract metadata and persona
            metadata = state.get("metadata", {})
            persona = metadata.get("persona")
            
            # Build system prompt
            system_prompt = self._build_chat_prompt(persona)
            
            # Extract conversation history
            conversation_history = []
            messages = state.get("messages", [])
            if messages:
                conversation_history = self._extract_conversation_history(messages, limit=10)
            
            # Build messages for LLM
            llm_messages = self._build_messages(system_prompt, state["query"], conversation_history)
            
            return {
                "persona": persona,
                "system_prompt": system_prompt,
                "conversation_history": conversation_history,
                "llm_messages": llm_messages
            }
            
        except Exception as e:
            logger.error(f"❌ Context preparation failed: {e}")
            return {
                "error": str(e),
                "task_status": "error"
            }
    
    async def _generate_response_node(self, state: ChatState) -> Dict[str, Any]:
        """Generate response: call LLM and parse structured output"""
        try:
            logger.info("💬 Generating chat response...")
            
            llm_messages = state.get("llm_messages", [])
            if not llm_messages:
                return {
                    "error": "No LLM messages prepared",
                    "task_status": "error",
                    "response": {}
                }
            
            # Call LLM - pass state to access user's model selection from metadata
            start_time = datetime.now()
            llm = self._get_llm(temperature=0.7, state=state)
            response = await llm.ainvoke(llm_messages)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Parse structured response
            response_content = response.content if hasattr(response, 'content') else str(response)
            structured_response = self._parse_json_response(response_content)
            
            # Extract message
            final_message = structured_response.get("message", response_content)
            
            # Add assistant response to messages for checkpoint persistence
            state = self._add_assistant_response_to_messages(state, final_message)
            
            # Store primary_agent_selected in shared_memory for conversation continuity
            shared_memory = state.get("shared_memory", {})
            shared_memory["primary_agent_selected"] = "chat_agent"
            shared_memory["last_agent"] = "chat_agent"
            state["shared_memory"] = shared_memory
            
            # Build result
            result = {
                "response": {
                "response": final_message,
                "task_status": structured_response.get("task_status", "complete"),
                "agent_type": "chat_agent",
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
                },
                "task_status": structured_response.get("task_status", "complete"),
                "messages": state.get("messages", []),
                "shared_memory": shared_memory
            }
            
            logger.info(f"✅ Chat response generated in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Response generation failed: {e}")
            return {
                "error": str(e),
                "task_status": "error",
                "response": self._create_error_response(str(e))
            }
    
    async def process(self, query: str, metadata: Dict[str, Any] = None, messages: List[Any] = None) -> Dict[str, Any]:
        """Process chat query using LangGraph workflow"""
        try:
            logger.info(f"💬 Chat agent processing: {query[:100]}...")
            
            # Get workflow (lazy initialization with checkpointer)
            workflow = await self._get_workflow()
            
            # Extract user_id from metadata
            metadata = metadata or {}
            user_id = metadata.get("user_id", "system")
            
            # Get checkpoint config (handles thread_id from conversation_id/user_id)
            config = self._get_checkpoint_config(metadata)
            
            # Prepare new messages (current query)
            new_messages = self._prepare_messages_with_query(messages, query)
            
            # Load and merge checkpointed messages to preserve conversation history
            conversation_messages = await self._load_and_merge_checkpoint_messages(
                workflow, config, new_messages
            )
            
            # Load shared_memory from checkpoint if available
            checkpoint_state = await workflow.aget_state(config)
            existing_shared_memory = {}
            if checkpoint_state and checkpoint_state.values:
                existing_shared_memory = checkpoint_state.values.get("shared_memory", {})
            
            # Merge with any shared_memory from metadata
            shared_memory = metadata.get("shared_memory", {}) or {}
            shared_memory.update(existing_shared_memory)
            
            # Initialize state for LangGraph workflow
            initial_state: ChatState = {
                "query": query,
                "user_id": user_id,
                "metadata": metadata,
                "messages": conversation_messages,
                "persona": None,
                "system_prompt": "",
                "conversation_history": [],
                "llm_messages": [],
                "response": {},
                "task_status": "",
                "error": "",
                "shared_memory": shared_memory
            }
            
            # Run LangGraph workflow with checkpointing
            result_state = await workflow.ainvoke(initial_state, config=config)
            
            # Extract final response
            response = result_state.get("response", {})
            task_status = result_state.get("task_status", "complete")
            
            if task_status == "error":
                error_msg = result_state.get("error", "Unknown error")
                logger.error(f"❌ Chat agent failed: {error_msg}")
                return self._create_error_response(error_msg)
            
            logger.info(f"✅ Chat agent completed: {task_status}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Chat agent failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._create_error_response(str(e))

