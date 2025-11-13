"""
Minimal LangGraph Foundation - Following 2025 Best Practices
Clean, simple, effective agent routing
"""

import logging
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from enum import Enum

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from services.langgraph_agents import ChatAgent, ResearchAgentHITL
from services.langgraph_postgres_checkpointer import get_postgres_checkpointer

logger = logging.getLogger(__name__)


class ExecutionMode(str, Enum):
    """Simple execution modes"""
    CHAT = "chat"
    RESEARCH = "research"


class MinimalState(TypedDict):
    """Minimal state following LangGraph 2025 best practices"""
    # Core LangGraph pattern - messages with proper annotation
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Essential context only
    user_id: str
    conversation_id: Optional[str]
    execution_mode: ExecutionMode
    
    # Persona customization
    persona: Optional[Dict[str, Any]]
    
    # Simple completion tracking
    is_complete: bool


class MinimalLangGraphFoundation:
    """Minimal LangGraph foundation following 2025 best practices"""
    
    def __init__(self):
        self.graph = None
        self.postgres_checkpointer = None
        self.is_initialized = False
        
        # Initialize agents
        self.chat_agent = ChatAgent()
        self.research_agent = ResearchAgentHITL()
        
    async def initialize(self):
        """Initialize the minimal LangGraph"""
        try:
            logger.info("🚀 Initializing Minimal LangGraph Foundation with PostgreSQL persistence...")
            
            # Initialize PostgreSQL checkpointer
            self.postgres_checkpointer = await get_postgres_checkpointer()
            logger.info("✅ PostgreSQL checkpointer initialized")
            
            # Create the state graph
            self.graph = StateGraph(MinimalState)
            
            # Add agent nodes
            self.graph.add_node("chat_agent", self._chat_node)
            self.graph.add_node("research_agent", self._research_node)
            self.graph.add_node("router", self._router_node)
            
            # Set entry point to router
            self.graph.set_entry_point("router")
            
            # Add routing edges
            self.graph.add_conditional_edges(
                "router",
                self._route_decision,
                {
                    "chat": "chat_agent",
                    "research": "research_agent"
                }
            )
            
            # Both agents end the conversation
            self.graph.add_edge("chat_agent", END)
            self.graph.add_edge("research_agent", END)
            
            # Compile with PostgreSQL persistence
            self.graph = self.graph.compile(checkpointer=self.postgres_checkpointer.checkpointer)
            
            self.is_initialized = True
            logger.info("✅ Minimal LangGraph Foundation initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Minimal LangGraph Foundation: {e}")
            raise
    
    async def _router_node(self, state: MinimalState) -> MinimalState:
        """Smart router node that determines which agent to use based on context"""
        try:
            # Get the latest user message
            latest_message = None
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == "human"):
                    latest_message = msg.content
                    break
            
            if not latest_message:
                logger.warning("No user message found, defaulting to chat")
                state["execution_mode"] = ExecutionMode.CHAT
                return state
            
            # INTELLIGENT ROUTING: Context-aware decision making
            query = latest_message.lower().strip()
            
            # Check for web search permission grants in context
            permission_grants = [
                "yes", "y", "okay", "ok", "sure", "proceed", "go ahead", 
                "do it", "please do", "search the web", "search web",
                "look online", "web search", "go online"
            ]
            
            # Check conversation history for web search permission requests
            web_search_permission_context = self._check_web_search_permission_context(state["messages"])
            
            if query in permission_grants and web_search_permission_context:
                state["execution_mode"] = ExecutionMode.RESEARCH
                logger.info(f"🔬 Routing to research agent: web search permission granted")
                return state
            
            # Research keywords for direct research requests
            research_keywords = [
                "research", "find", "search", "who", "what", "when", "where", "why", 
                "how", "analyze", "investigate", "study", "explain", "tell me about", 
                "information about", "details about", "look up", "find out",
                "latest", "current", "recent", "news", "update", "today"
            ]
            
            # Simple greetings stay in chat
            simple_greetings = [
                "hi", "hello", "hey", "good morning", "good afternoon", 
                "good evening", "how are you", "thanks", "thank you", 
                "bye", "goodbye", "how's it going", "what's up"
            ]
            
            if query in simple_greetings or len(query) < 10:
                state["execution_mode"] = ExecutionMode.CHAT
                logger.info(f"🗣️ Routing to chat agent: simple greeting/chat")
            elif any(keyword in query for keyword in research_keywords):
                state["execution_mode"] = ExecutionMode.RESEARCH
                logger.info(f"🔬 Routing to research agent: research keywords detected")
            else:
                state["execution_mode"] = ExecutionMode.CHAT
                logger.info(f"🗣️ Routing to chat agent: default for conversation")
            
            return state
        
        except Exception as e:
            logger.error(f"❌ Router error: {e}")
            state["execution_mode"] = ExecutionMode.CHAT
            return state
    
    def _check_web_search_permission_context(self, messages: List[BaseMessage]) -> bool:
        """Check if recent conversation context suggests web search permission was requested"""
        try:
            # Look at the last few AI messages for web search permission requests
            recent_ai_messages = []
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) or (hasattr(msg, 'type') and msg.type == "ai"):
                    recent_ai_messages.append(msg.content.lower())
                    if len(recent_ai_messages) >= 3:  # Check last 3 AI messages
                        break
            
            # Keywords that indicate web search permission was requested
            permission_request_indicators = [
                "web search", "search the web", "would you like me to proceed",
                "permission to search", "search for current", "web for current",
                "look online", "search online", "internet search"
            ]
            
            for ai_message in recent_ai_messages:
                if any(indicator in ai_message for indicator in permission_request_indicators):
                    logger.info(f"🔍 Web search permission context detected in recent AI message")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking web search permission context: {e}")
            return False
    
    def _route_decision(self, state: MinimalState) -> str:
        """Decide which agent to route to based on execution mode"""
        mode = state.get("execution_mode", ExecutionMode.CHAT)
        if mode == ExecutionMode.RESEARCH:
            return "research"
        else:
            return "chat"
    
    async def _chat_node(self, state: MinimalState) -> MinimalState:
        """Chat agent node"""
        try:
            logger.info("💬 Processing with chat agent...")
            
            # Convert state to agent format
            agent_state = {
                "messages": state["messages"],
                "user_id": state["user_id"],
                "conversation_id": state["conversation_id"],
                "current_query": self._get_latest_user_message(state),
                "persona": state.get("persona"),  # Pass persona to agent
                "agent_insights": {"chat_agent": {}},  # Required by agent
                "shared_memory": {}  # Required by agent
            }
            
            # CONVERSATION INTELLIGENCE: Log conversation context
            msg_count = len(state["messages"])
            logger.info(f"💬 Chat agent receiving {msg_count} messages for context")
            
            # Process with chat agent
            result_state = await self.chat_agent.process(agent_state)
            
            # Extract response and add to messages
            response = result_state.get("agent_results", {}).get("response", "I couldn't generate a response.")
            
            # Add AI response to messages (LangGraph will handle this automatically)
            state["messages"] = state["messages"] + [AIMessage(content=response)]
            state["is_complete"] = True
            
            logger.info("✅ Chat agent completed successfully")
            return state
            
        except Exception as e:
            logger.error(f"❌ Chat agent error: {e}")
            state["messages"] = state["messages"] + [AIMessage(content=f"I encountered an error: {str(e)}")]
            state["is_complete"] = True
            return state
    
    async def _research_node(self, state: MinimalState) -> MinimalState:
        """Research agent node"""
        try:
            logger.info("🔬 Processing with research agent...")
            
            # Convert state to agent format
            agent_state = {
                "messages": state["messages"],
                "user_id": state["user_id"],
                "conversation_id": state["conversation_id"],
                "current_query": self._get_latest_user_message(state),
                "persona": state.get("persona"),  # Pass persona to agent
                "agent_insights": {"research_agent": {}},  # Required by agent
                "shared_memory": {}  # Required by agent
            }
            
            # ROOSEVELT'S CAVALRY INTELLIGENCE: Log conversation context
            msg_count = len(state["messages"])
            logger.info(f"🔬 Research agent receiving {msg_count} messages for context")
            
            # Process with research agent
            result_state = await self.research_agent.process(agent_state)
            
            # Extract response and add to messages
            response = result_state.get("agent_results", {}).get("response", "I couldn't complete the research.")
            
            # Add AI response to messages
            state["messages"] = state["messages"] + [AIMessage(content=response)]
            state["is_complete"] = True
            
            logger.info("✅ Research agent completed successfully")
            return state
            
        except Exception as e:
            logger.error(f"❌ Research agent error: {e}")
            state["messages"] = state["messages"] + [AIMessage(content=f"I encountered a research error: {str(e)}")]
            state["is_complete"] = True
            return state
    
    def _get_latest_user_message(self, state: MinimalState) -> str:
        """Get the latest user message from state"""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == "human"):
                return msg.content
        return ""
    
    async def _load_conversation_history(self, user_id: str, conversation_id: Optional[str]) -> List[BaseMessage]:
        """Load full conversation history from database - ROOSEVELT'S MEMORY CAVALRY!"""
        try:
            if not conversation_id:
                logger.info("🧠 No conversation_id provided, starting fresh conversation")
                return []
            
            # Get conversation service to load messages
            from services.conversation_service import ConversationService
            conversation_service = ConversationService()
            conversation_service.set_current_user(user_id)
            
            # Load conversation messages from database
            messages_result = await conversation_service.get_conversation_messages(conversation_id, user_id, limit=50)
            
            if not messages_result.get("success", False):
                logger.warning(f"⚠️ Could not load conversation {conversation_id}, starting fresh")
                return []
            
            # Convert database messages to LangGraph messages
            conversation_messages = []
            db_messages = messages_result.get("messages", [])
            
            for db_msg in db_messages:
                role = db_msg.get("role", "user")
                content = db_msg.get("content", "")
                
                if role == "user":
                    conversation_messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    conversation_messages.append(AIMessage(content=content))
                # Skip system messages for LangGraph state
            
            logger.info(f"🧠 Loaded {len(conversation_messages)} messages from conversation {conversation_id}")
            return conversation_messages
            
        except Exception as e:
            logger.error(f"❌ Failed to load conversation history: {e}")
            return []  # Start fresh on error
    
    async def process_query(
        self,
        user_id: str,
        conversation_id: Optional[str],
        query: str,
        persona: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process query using minimal LangGraph with best practices"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Create thread_id for conversation persistence
            thread_id = conversation_id or f"thread_{user_id}"
            
            # ROOSEVELT'S MEMORY FIX: Load full conversation history from PostgreSQL!
            conversation_messages = await self.postgres_checkpointer.load_conversation_history(user_id, conversation_id)
            
            # Add the new user query to the conversation
            conversation_messages.append(HumanMessage(content=query))
            
            # Create initial state with FULL conversation history
            initial_state: MinimalState = {
                "messages": conversation_messages,  # ← FIXED: Full conversation history!
                "user_id": user_id,
                "conversation_id": conversation_id,
                "execution_mode": ExecutionMode.CHAT,  # Will be determined by router
                "persona": persona,  # Include persona customization
                "is_complete": False
            }
            
            # Process through LangGraph with checkpointing
            config = {
                "configurable": {
                    "thread_id": thread_id
                }
            }
            
            # Execute the graph
            final_state = await self.graph.ainvoke(initial_state, config=config)
            
            # ROOSEVELT'S DUAL PERSISTENCE: Sync messages to conversation database
            try:
                messages = final_state.get("messages", [])
                if messages and conversation_id:
                    await self.postgres_checkpointer.sync_messages_to_conversation_db(
                        user_id=user_id,
                        conversation_id=conversation_id,
                        messages=messages
                    )
                    logger.info(f"✅ Synced {len(messages)} messages to conversation database")
            except Exception as sync_error:
                logger.error(f"⚠️ Failed to sync messages to conversation DB: {sync_error}")
                # Don't fail the request for sync errors
            
            # Extract the final response
            response = self._extract_final_response(final_state)
            
            return {
                "response": response,
                "success": True,
                "conversation_id": conversation_id,
                "thread_id": thread_id,
                "execution_mode": final_state.get("execution_mode", ExecutionMode.CHAT).value
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to process query: {e}")
            return {
                "response": f"I encountered an error: {str(e)}",
                "success": False,
                "error": str(e),
                "conversation_id": conversation_id
            }
    
    def _extract_final_response(self, final_state: MinimalState) -> str:
        """Extract the final response from the state"""
        messages = final_state.get("messages", [])
        if messages:
            # Get the last AI message
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) or (hasattr(msg, 'type') and msg.type == 'ai'):
                    return msg.content
        
        return "I couldn't generate a response."


# Global instance
minimal_langgraph_foundation = None

def get_minimal_langgraph_foundation():
    """Get the global minimal LangGraph foundation instance"""
    global minimal_langgraph_foundation
    if minimal_langgraph_foundation is None:
        minimal_langgraph_foundation = MinimalLangGraphFoundation()
    return minimal_langgraph_foundation
