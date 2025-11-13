"""
ROOSEVELT'S AGENT TEMPLATE - Copy and Customize for New Agents
This template provides the complete framework for adding new agents to the system

USAGE:
1. Copy this file to new_agent_name.py  
2. Replace ALL instances of "Template" with your agent name
3. Update capabilities, specialties, and tools
4. Follow the integration checklist below
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

from .base_agent import BaseAgent
from models.agent_response_models import TaskStatus

logger = logging.getLogger(__name__)


class TemplateAgent(BaseAgent):
    """
    Roosevelt's [DOMAIN] Intelligence Specialist
    
    CUSTOMIZE: Replace [DOMAIN] with your agent's domain (Gardening, Finance, etc.)
    """
    
    def __init__(self):
        super().__init__("template_agent")  # CUSTOMIZE: Replace with your agent name
        logger.info("🌱 BULLY! Template Agent assembled and ready for [DOMAIN] operations!")
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process [DOMAIN] requests with conversation intelligence"""
        try:
            logger.info("🌱 Template Agent charging forward with [DOMAIN] expertise...")
            
            query = self._extract_current_user_query(state)
            if not query.strip():
                return self._create_error_response("No query provided")
            
            # ROOSEVELT'S CONVERSATION INTELLIGENCE: Check cache first
            cached_context = await self._get_relevant_context(query, state)
            if cached_context and self._should_use_cached_context(query, state):
                logger.info("🏆 CACHE-FIRST: Using conversation intelligence")
                return await self._process_with_cached_context(query, cached_context, state)
            
            # CROSS-AGENT INTELLIGENCE: Get context from other agents
            cross_agent_context = await self._get_cross_agent_context(query, state)
            
            # Process with domain-specific logic
            return await self._execute_domain_processing(query, cross_agent_context, state)
            
        except Exception as e:
            logger.error(f"❌ Template agent failed: {e}")
            return self._create_error_response(f"Template agent error: {str(e)}")
    
    async def _process_with_cached_context(self, query: str, cached_context: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process using cached conversation intelligence"""
        try:
            logger.info("🏆 PROCESSING WITH CACHED CONTEXT: Using conversation intelligence")
            
            # Build response using cached context
            system_prompt = f"""You are Roosevelt's [DOMAIN] Intelligence Agent using CACHED CONVERSATION INTELLIGENCE.

**MISSION**: Answer the user's [DOMAIN] query using cached context from this conversation.

**USER QUERY**: {query}

**CACHED CONTEXT FROM CONVERSATION**:
{cached_context}

**INSTRUCTIONS**:
1. **USE CACHED CONTEXT**: Build your response using the provided conversation context
2. **REFERENCE PREVIOUS WORK**: Acknowledge and build upon previous agent outputs
3. **DOMAIN EXPERTISE**: Apply [DOMAIN] knowledge to the cached information
4. **NO REDUNDANT SEARCHES**: Work with provided context

**STRUCTURED OUTPUT REQUIRED**:
You MUST respond with valid JSON matching this schema:
{{
    "task_status": "complete",
    "response": "Your [DOMAIN] response using cached context",
    "confidence_level": 0.9,
    "collaboration_suggestion": "Optional suggestion for other agents",
    "collaboration_confidence": 0.0,
    "suggested_agent": null
}}

**By George!** Use the conversation intelligence to provide excellent [DOMAIN] advice!
"""

            # Execute with LLM
            response = await self._execute_with_llm(system_prompt, query)
            
            # Create successful result
            state["agent_results"] = {
                "agent_type": "template_agent",
                "response": response,
                "cache_used": True,
                "timestamp": datetime.now().isoformat(),
                "processing_time": 0.5
            }
            
            state["latest_response"] = response
            
            # ROOSEVELT'S PURE LANGGRAPH: Add agent response to LangGraph state messages
            if response and response.strip():
                from langchain_core.messages import AIMessage
                state.setdefault("messages", []).append(AIMessage(content=response))
                logger.info(f"✅ TEMPLATE AGENT: Added cached response to LangGraph messages")
            
            state["is_complete"] = True
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Cached context processing failed: {e}")
            return self._create_error_response(f"Cache processing failed: {str(e)}")
    
    async def _get_cross_agent_context(self, query: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Get relevant context from other agents in this conversation"""
        try:
            conversation_intel = self._get_conversation_intelligence(state)
            cross_agent_context = {}
            
            # Check for weather data (useful for gardening)
            weather_outputs = conversation_intel.get("agent_outputs", {}).get("weather_agent", [])
            if weather_outputs:
                cross_agent_context["weather"] = self._extract_weather_context(weather_outputs, conversation_intel)
            
            # Check for location research (useful for gardening zones)
            research_outputs = conversation_intel.get("agent_outputs", {}).get("research_agent", [])
            if research_outputs:
                cross_agent_context["location_research"] = self._extract_location_context(research_outputs, conversation_intel)
            
            # Check for chat brainstorming (garden ideas, plant preferences)
            chat_outputs = conversation_intel.get("agent_outputs", {}).get("chat_agent", [])
            if chat_outputs:
                cross_agent_context["user_preferences"] = self._extract_preference_context(chat_outputs, conversation_intel)
            
            return cross_agent_context
            
        except Exception as e:
            logger.error(f"❌ Cross-agent context extraction failed: {e}")
            return {}
    
    async def _execute_domain_processing(self, query: str, cross_agent_context: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute domain-specific processing with cross-agent intelligence"""
        try:
            # CUSTOMIZE: Implement your agent's core logic here
            
            # Example for gardening agent:
            location = self._extract_location_from_context(cross_agent_context, query)
            weather_data = cross_agent_context.get("weather", {})
            user_preferences = cross_agent_context.get("user_preferences", {})
            
            # Build domain-specific prompt
            system_prompt = self._build_domain_prompt(query, location, weather_data, user_preferences)
            
            # Execute with LLM
            response = await self._execute_with_llm(system_prompt, query)
            
            # Create successful result
            state["agent_results"] = {
                "agent_type": "template_agent",
                "response": response,
                "cross_agent_context_used": list(cross_agent_context.keys()),
                "timestamp": datetime.now().isoformat()
            }
            
            state["latest_response"] = response
            
            # ROOSEVELT'S PURE LANGGRAPH: Add agent response to LangGraph state messages
            if response and response.strip():
                from langchain_core.messages import AIMessage
                state.setdefault("messages", []).append(AIMessage(content=response))
                logger.info(f"✅ TEMPLATE AGENT: Added cross-agent response to LangGraph messages")
            
            state["is_complete"] = True
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Domain processing failed: {e}")
            return self._create_error_response(f"Domain processing failed: {str(e)}")
    
    def _build_domain_prompt(self, query: str, location: str, weather_data: Dict, preferences: Dict) -> str:
        """Build domain-specific prompt - CUSTOMIZE for your agent"""
        return f"""You are Roosevelt's [DOMAIN] Intelligence Specialist.

**MISSION**: Provide expert [DOMAIN] advice using cross-agent intelligence.

**USER QUERY**: {query}
**LOCATION CONTEXT**: {location}
**WEATHER CONTEXT**: {weather_data}
**USER PREFERENCES**: {preferences}

**EXPERTISE AREAS**:
- CUSTOMIZE: List your agent's expertise areas
- CUSTOMIZE: Add domain-specific capabilities
- CUSTOMIZE: Include specializations

**COLLABORATION OPPORTUNITIES**:
- Suggest Weather Agent for forecasts if needed
- Suggest Research Agent for specific factual lookups
- Suggest Data Formatting Agent for complex schedules/tables

**RESPONSE FORMAT**: Provide helpful [DOMAIN] advice with collaboration suggestions when beneficial.
"""
    
    async def _execute_with_llm(self, system_prompt: str, user_query: str) -> str:
        """Execute LLM call with domain prompt"""
        try:
            chat_service = await self._get_chat_service()
            model_name = await self._get_model_name()
            
            from langchain_openai import ChatOpenAI
            from langchain.schema import SystemMessage, HumanMessage
            
            client = ChatOpenAI(
                model=model_name,
                temperature=0.3,  # CUSTOMIZE: Adjust temperature for your domain
                openai_api_key=chat_service.openai_client.api_key,
                openai_api_base=str(chat_service.openai_client.base_url)
            )
            
            response = await client.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_query)
            ])
            
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            logger.error(f"❌ LLM execution failed: {e}")
            raise
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "agent_results": {
                "agent_type": "template_agent",
                "task_status": "error",
                "error_message": error_message,
                "confidence": 0.0
            },
            "latest_response": f"❌ [DOMAIN] agent error: {error_message}",
            "is_complete": True
        }


"""
INTEGRATION CHECKLIST - Copy this when adding a new agent:

□ 1. AgentCapability enum (agent_intelligence_network.py)
□ 2. AgentType enum (centralized_tool_registry.py)  
□ 3. Agent class implementation (langgraph_agents/new_agent.py)
□ 4. Agent registration (agent_intelligence_network.py)
□ 5. Tool permissions (centralized_tool_registry.py)
□ 6. Type mapping (base_agent.py)
□ 7. Orchestrator instance (langgraph_official_orchestrator.py)
□ 8. Orchestrator routing (langgraph_official_orchestrator.py)
□ 9. Intent type (capability_based_intent_service.py)
□ 10. Intent examples (capability_based_intent_service.py)
□ 11. Custom tools (optional - langgraph_tools/domain_tools.py)
□ 12. Response models (optional - models/agent_response_models.py)

AUTOMATIC FEATURES (No configuration needed):
✅ Conversation intelligence access
✅ Cross-agent context visibility  
✅ LangGraph state management
✅ PostgreSQL persistence
✅ Error handling and recovery
✅ Universal formatting capabilities
✅ Collaboration suggestion framework
"""
