"""
Weather Agent
Weather information and forecasting agent with structured outputs
"""

import logging
import os
import json
from typing import Dict, Any, List, Optional, TypedDict
from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langgraph.graph import StateGraph, END
from orchestrator.agents.base_agent import BaseAgent
from orchestrator.services.weather_request_analyzer import get_weather_request_analyzer
from orchestrator.services.weather_response_formatters import get_weather_response_formatters
from orchestrator.tools.weather_tools import get_weather_tools

logger = logging.getLogger(__name__)


class WeatherState(TypedDict):
    """State for weather agent LangGraph workflow"""
    query: str
    user_id: str
    metadata: Dict[str, Any]
    messages: List[Any]
    shared_memory: Dict[str, Any]
    persona: Optional[Dict[str, Any]]
    user_preferences: Dict[str, Any]
    communication_style: str
    detail_level: str
    weather_request: Dict[str, Any]
    weather_data: Dict[str, Any]
    recommendations: Optional[str]
    collaboration_data: Dict[str, Any]
    response: Dict[str, Any]
    task_status: str
    error: str


class WeatherAgent(BaseAgent):
    """
    Meteorological Intelligence Agent
    Provides weather conditions and forecasts with structured outputs
    Uses LangGraph workflow for explicit state management
    """
    
    def __init__(self):
        super().__init__("weather_agent")
        self._analyzer = None
        self._formatter = None
        self._weather_tools = None
        logger.info("🌤️ Weather Agent ready for meteorological intelligence!")
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """Build LangGraph workflow for weather agent"""
        workflow = StateGraph(WeatherState)
        
        # Add nodes
        workflow.add_node("prepare_context", self._prepare_context_node)
        workflow.add_node("analyze_request", self._analyze_request_node)
        workflow.add_node("get_weather_data", self._get_weather_data_node)
        workflow.add_node("generate_response", self._generate_response_node)
        
        # Entry point
        workflow.set_entry_point("prepare_context")
        
        # Flow: prepare_context -> analyze_request -> (conditional) -> get_weather_data -> generate_response -> END
        workflow.add_edge("prepare_context", "analyze_request")
        
        # Conditional routing from analyze_request
        workflow.add_conditional_edges(
            "analyze_request",
            self._route_from_analysis,
            {
                "location_needed": "generate_response",  # Generate clarification response
                "get_data": "get_weather_data"
            }
        )
        
        workflow.add_edge("get_weather_data", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow.compile(checkpointer=checkpointer)
    
    async def _get_analyzer(self):
        """Get weather request analyzer instance"""
        if self._analyzer is None:
            self._analyzer = get_weather_request_analyzer()
        return self._analyzer
    
    async def _get_formatter(self):
        """Get weather response formatter instance"""
        if self._formatter is None:
            self._formatter = get_weather_response_formatters()
        return self._formatter
    
    async def _get_weather_tools(self):
        """Get weather tools instance"""
        if self._weather_tools is None:
            self._weather_tools = await get_weather_tools()
        return self._weather_tools
    
    async def _prepare_context_node(self, state: WeatherState) -> Dict[str, Any]:
        """Prepare context: extract messages, preferences, and communication style"""
        try:
            logger.info("📋 Preparing context for weather agent...")
            
            messages = state.get("messages", [])
            shared_memory = state.get("shared_memory", {})
            persona = state.get("persona", {})
            
            # Extract user message
            user_message = ""
            if messages:
                last_message = messages[-1]
                if hasattr(last_message, 'content'):
                    user_message = last_message.content
                elif isinstance(last_message, dict):
                    user_message = last_message.get("content", "")
            
            # Get user preferences from shared memory and state persona  
            user_preferences = shared_memory.get("user_preferences", {})
            
            # Extract persona_style using centralized approach
            persona_style = "professional"  # Default
            if persona:
                if isinstance(persona, str):
                    persona_style = persona
                elif isinstance(persona, dict):
                    persona_style = persona.get("persona_style", persona.get("style", "professional"))
            
            # Use user preferences if available, otherwise use persona_style
            communication_style = user_preferences.get("communication_style", persona_style)
            detail_level = user_preferences.get("preferred_detail_level", "moderate")
            
            return {
                "query": user_message,
                "user_preferences": user_preferences,
                "communication_style": communication_style,
                "detail_level": detail_level,
                # ✅ CRITICAL: Preserve critical state keys
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "persona": state.get("persona")
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare context: {e}")
            return {
                "query": "",
                "user_preferences": {},
                "communication_style": "casual",
                "detail_level": "moderate",
                "error": str(e),
                # ✅ CRITICAL: Preserve critical state keys even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "persona": state.get("persona")
            }
    
    async def _analyze_request_node(self, state: WeatherState) -> Dict[str, Any]:
        """Analyze weather request using intelligent analysis"""
        try:
            logger.info("🔍 Analyzing weather request...")
            
            query = state.get("query", "")
            shared_memory = state.get("shared_memory", {})
            
            # Get analyzer instance
            analyzer = await self._get_analyzer()
            
            # Analyze the weather request
            weather_request = await analyzer.analyze_weather_request(query, shared_memory)
            
            return {
                "weather_request": weather_request,
                # ✅ CRITICAL: Preserve critical state keys
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "persona": state.get("persona"),
                "user_preferences": state.get("user_preferences", {}),
                "communication_style": state.get("communication_style", "professional"),
                "detail_level": state.get("detail_level", "moderate")
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze request: {e}")
            return {
                "weather_request": {
                    "success": False,
                    "error": str(e)
                },
                "error": str(e),
                # ✅ CRITICAL: Preserve critical state keys even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "persona": state.get("persona"),
                "user_preferences": state.get("user_preferences", {}),
                "communication_style": state.get("communication_style", "professional"),
                "detail_level": state.get("detail_level", "moderate")
            }
    
    def _route_from_analysis(self, state: WeatherState) -> str:
        """Route from analysis: check if location is needed or proceed to get data"""
        weather_request = state.get("weather_request", {})
        
        if not weather_request.get("success", False):
            return "location_needed"  # Error case - will be handled in response
        
        if weather_request.get("location") == "LOCATION_NEEDED":
            return "location_needed"
        
        return "get_data"
    
    async def _get_weather_data_node(self, state: WeatherState) -> Dict[str, Any]:
        """Get weather data using weather tools"""
        try:
            logger.info("🌤️ Fetching weather data...")
            
            weather_request = state.get("weather_request", {})
            user_id = state.get("user_id", "system")
            
            # Get weather tools instance
            weather_tools = await self._get_weather_tools()
            
            # Get weather data
            weather_data = await self._get_weather_data(weather_request, weather_tools, user_id)
            
            # Store results in shared memory if successful
            if weather_data.get("success", False):
                shared_memory = state.get("shared_memory", {})
                analyzer = await self._get_analyzer()
                analyzer.update_shared_memory(shared_memory, weather_request, weather_data, self.agent_type)
            
            return {
                "weather_data": weather_data,
                # ✅ CRITICAL: Preserve critical state keys
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "persona": state.get("persona"),
                "user_preferences": state.get("user_preferences", {}),
                "communication_style": state.get("communication_style", "professional"),
                "detail_level": state.get("detail_level", "moderate"),
                "weather_request": state.get("weather_request", {})
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get weather data: {e}")
            return {
                "weather_data": {
                    "success": False,
                    "error": str(e)
                },
                "error": str(e),
                # ✅ CRITICAL: Preserve critical state keys even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "persona": state.get("persona"),
                "user_preferences": state.get("user_preferences", {}),
                "communication_style": state.get("communication_style", "professional"),
                "detail_level": state.get("detail_level", "moderate"),
                "weather_request": state.get("weather_request", {})
            }
    
    async def _generate_response_node(self, state: WeatherState) -> Dict[str, Any]:
        """Generate formatted response with recommendations and collaboration suggestions"""
        try:
            logger.info("📝 Generating weather response...")
            
            weather_request = state.get("weather_request", {})
            weather_data = state.get("weather_data", {})
            query = state.get("query", "")
            communication_style = state.get("communication_style", "casual")
            detail_level = state.get("detail_level", "moderate")
            shared_memory = state.get("shared_memory", {})
            
            # Get formatter instance
            formatter = await self._get_formatter()
            
            # Check for errors
            if not weather_request.get("success", False):
                error_response = await formatter.format_error_response(
                    weather_request.get("error", "Unknown error"), 
                    communication_style
                )
                return {
                    "response": self._create_response(error_response, is_complete=False),
                    "task_status": "error",
                    # ✅ CRITICAL: Preserve critical state keys even on error
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", ""),
                    "persona": state.get("persona")
                }
            
            # Check if location clarification is needed
            if weather_request.get("location") == "LOCATION_NEEDED":
                clarification_response = "Where would you like weather for? Please provide a city name, state, or ZIP code."
                return {
                    "response": self._create_response(clarification_response, is_complete=False),
                    "task_status": "incomplete",
                    # ✅ CRITICAL: Preserve critical state keys
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", ""),
                    "persona": state.get("persona")
                }
            
            # Check for weather data errors
            if not weather_data.get("success", False):
                error_response = await formatter.format_error_response(
                    weather_data.get("error", "Unknown error"),
                    communication_style
                )
                return {
                    "response": self._create_response(error_response, is_complete=False),
                    "task_status": "error",
                    # ✅ CRITICAL: Preserve critical state keys even on error
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", ""),
                    "persona": state.get("persona")
                }
            
            # Get LLM-enhanced intelligent recommendations
            recommendations = await self._get_llm_recommendations(
                weather_request, 
                weather_data, 
                communication_style,
                state
            )
            
            # Detect research opportunities
            collaboration_data = await self._detect_collaboration_opportunities(
                query, 
                weather_request, 
                weather_data,
                state
            )
            
            # Format response based on user preferences
            formatted_response = await formatter.format_weather_response(
                weather_request, 
                weather_data, 
                communication_style, 
                detail_level,
                shared_memory
            )
            
            # Add collaboration suggestion to response if detected
            if collaboration_data.get("should_collaborate", False):
                collaboration_text = f"\n\n💡 **Additional Research**: {collaboration_data.get('suggested_research', '')}"
                formatted_response += collaboration_text
            
            # Add recommendations if available
            if recommendations:
                formatted_response += f"\n\n💡 {recommendations}"
            
            location = weather_request.get("location", "unknown")
            logger.info(f"✅ Weather response generated for {location}")
            
            # Persist agent selection for conversation continuity
            shared_memory = state.get("shared_memory", {})
            shared_memory["primary_agent_selected"] = "weather_agent"
            shared_memory["last_agent"] = "weather_agent"
            
            return {
                "response": self._create_response(formatted_response, is_complete=True),
                "recommendations": recommendations,
                "collaboration_data": collaboration_data,
                "task_status": "complete",
                "shared_memory": shared_memory,
                # ✅ CRITICAL: Preserve critical state keys
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "persona": state.get("persona")
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to generate response: {e}")
            communication_style = state.get("communication_style", "casual")
            formatter = await self._get_formatter()
            error_response = await formatter.format_error_response(str(e), communication_style)
            return {
                "response": self._create_response(error_response, is_complete=False),
                "task_status": "error",
                "error": str(e),
                # ✅ CRITICAL: Preserve critical state keys even on exception
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "persona": state.get("persona")
            }
    
    async def process(self, query: str, metadata: Dict[str, Any] = None, messages: List[Any] = None) -> Dict[str, Any]:
        """
        Process weather requests using LangGraph workflow
        
        Args:
            query: User's weather query
            metadata: Metadata dict with user_id, conversation_id, shared_memory, persona, etc.
            messages: Optional conversation messages
            
        Returns:
            Dictionary with weather response and metadata
        """
        try:
            logger.info(f"🌤️ Weather Agent: Processing weather query: {query[:100]}...")
            
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
            
            # Extract persona (can be string or dict)
            persona = metadata.get("persona", {})
            if isinstance(persona, str):
                # Convert string persona to dict format
                persona = {"persona_style": persona, "style": persona}
            
            # Extract persona_style for initial state
            persona_style = "professional"  # Default
            if persona:
                if isinstance(persona, dict):
                    persona_style = persona.get("persona_style", persona.get("style", "professional"))
                elif isinstance(persona, str):
                    persona_style = persona
            
            # Build initial state for LangGraph workflow
            initial_state: WeatherState = {
                "query": query,
                "user_id": user_id,
                "metadata": metadata,
                "messages": conversation_messages,
                "shared_memory": shared_memory,
                "persona": persona if isinstance(persona, dict) else {"persona_style": persona_style},
                "user_preferences": {},
                "communication_style": persona_style,
                "detail_level": "moderate",
                "weather_request": {},
                "weather_data": {},
                "recommendations": None,
                "collaboration_data": {},
                "response": {},
                "task_status": "",
                "error": ""
            }
            
            # Invoke LangGraph workflow with checkpointing
            final_state = await workflow.ainvoke(initial_state, config=config)
            
            # Extract final response (weather agent returns response dict with _create_response result)
            response_dict = final_state.get("response", {})
            task_status = final_state.get("task_status", "complete")
            
            if task_status == "error":
                error_msg = final_state.get("error", "Unknown error")
                logger.error(f"❌ Weather agent failed: {error_msg}")
                return self._create_error_response(error_msg)
            
            # Extract shared_memory if it was updated
            updated_shared_memory = final_state.get("shared_memory", {})
            if updated_shared_memory:
                # Merge back into response for persistence
                if isinstance(response_dict, dict):
                    response_dict["shared_memory"] = updated_shared_memory
            
            logger.info(f"✅ Weather agent completed: {task_status}")
            return response_dict
            
        except Exception as e:
            logger.error(f"❌ Weather Agent: Processing failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._create_error_response(str(e))
    
    async def _get_weather_data(self, weather_request: Dict[str, Any], weather_tools, user_id: str) -> Dict[str, Any]:
        """Get weather data using weather tools"""
        try:
            if weather_request["request_type"] == "current":
                # Get current conditions
                result = await weather_tools.get_weather_conditions(
                    location=weather_request["location"],
                    units=weather_request["units"],
                    user_id=user_id
                )
            else:
                # Get forecast
                result = await weather_tools.get_weather_forecast(
                    location=weather_request["location"],
                    days=weather_request["forecast_days"],
                    units=weather_request["units"],
                    user_id=user_id
                )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error getting weather data: {e}")
            return {
                "success": False,
                "error": f"Could not retrieve weather data: {str(e)}"
            }
    
    async def _get_llm_recommendations(self, request: Dict[str, Any], weather_data: Dict[str, Any], style: str, state: Optional[Dict[str, Any]] = None) -> str:
        """Get intelligent LLM-based weather recommendations and insights"""
        try:
            # Use centralized LLM access from BaseAgent
            fast_model = self._get_fast_model(state)
            llm = self._get_llm(temperature=0.3, model=fast_model, state=state)
            
            # Get centralized persona style instructions from BaseAgent
            style_instruction = self._get_style_instruction(style)
            
            # Build prompt for weather intelligence
            weather_info = self._extract_weather_summary(request, weather_data)
            
            prompt = f"""Based on the weather data provided, provide intelligent recommendations and ONE fun weather feature.

WEATHER DATA:
{weather_info}

{style_instruction}

REQUIREMENTS:
1. Provide 1-2 sentences of practical recommendations focusing on:
   - Optimal activities for these conditions
   - What to watch out for
   - Any interesting meteorological insights

2. Choose ONE fun feature to include (pick the most appropriate for these conditions):
   - **Activity Matchmaking**: Suggest a specific activity that's perfect for this weather (e.g., "Perfect day for a picnic in the park!" or "Great weather for indoor reading with hot cocoa")
   - **Weather-Based Food/Drink**: Suggest a food or drink that matches the weather (e.g., "Time for a refreshing iced tea!" or "Perfect weather for a warm bowl of soup")
   - **Weather Challenge**: Offer a fun weather-related challenge or observation task (e.g., "Can you spot a rainbow today?" or "Perfect day for cloud watching - see if you can identify different cloud types!")

Format your response as:
[Practical recommendations]

🎯 [Fun Feature Type]: [Your fun suggestion]

Be brief, engaging, and match the communication style. Make it feel natural and fun!
"""
            
            # Get LLM recommendations using LangChain interface
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            recommendation = response.content if hasattr(response, 'content') else str(response)
            recommendation = recommendation.strip()
            
            # Return recommendation (no truncation - max_tokens handled centrally by OpenRouter)
            return recommendation if recommendation else None
            
        except Exception as e:
            logger.error(f"❌ Error getting LLM recommendations: {e}")
            return None
    
    def _extract_weather_summary(self, request: Dict[str, Any], weather_data: Dict[str, Any]) -> str:
        """Extract key weather information for LLM analysis"""
        try:
            summary = f"Location: {request['location']}\n"
            summary += f"Request Type: {request['request_type']}\n"
            summary += f"Units: {request['units']}\n"
            
            if request["request_type"] == "current":
                current = weather_data.get("current", {})
                summary += f"Temperature: {current.get('temperature', 'N/A')}\n"
                summary += f"Conditions: {current.get('conditions', 'N/A')}\n"
                summary += f"Humidity: {current.get('humidity', 'N/A')}%\n"
                summary += f"Wind Speed: {current.get('wind_speed', 'N/A')}\n"
            else:
                forecast = weather_data.get("forecast", [])
                if forecast:
                    day1 = forecast[0]
                    summary += f"Today's High/Low: {day1.get('temperature', {}).get('high', 'N/A')}/{day1.get('temperature', {}).get('low', 'N/A')}\n"
                    summary += f"Conditions: {day1.get('conditions', 'N/A')}\n"
                    summary += f"Precipitation Chance: {day1.get('precipitation_probability', 'N/A')}%\n"
            
            return summary
                
        except Exception as e:
            logger.error(f"❌ Error extracting weather summary: {e}")
            return "Weather data available but summary extraction failed."
    
    async def _detect_collaboration_opportunities(self, user_message: str, weather_request: Dict[str, Any], weather_data: Dict[str, Any], state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Detect opportunities for weather→research collaboration using LLM intelligence"""
        try:
            # Use centralized LLM access from BaseAgent
            fast_model = self._get_fast_model(state)
            llm = self._get_llm(temperature=0.2, model=fast_model, state=state)
            
            # Extract weather conditions for context
            weather_summary = self._extract_weather_summary(weather_request, weather_data)
            
            collaboration_prompt = f"""Analyze if this weather query should trigger additional research collaboration.

USER MESSAGE: "{user_message}"
LOCATION: "{weather_request['location']}"
WEATHER CONDITIONS: {weather_summary}

COLLABORATION SCENARIOS TO DETECT:

1. TRAVEL WEATHER → TRAVEL RESEARCH
   - Keywords: trip, vacation, travel, visiting, flight, hotel
   - Suggestion: Research destination activities, travel advisories, local attractions

2. OUTDOOR ACTIVITY WEATHER → ACTIVITY RESEARCH  
   - Keywords: hiking, camping, skiing, beach, festival, concert, wedding
   - Suggestion: Research activity conditions, safety, equipment, permits

3. SEVERE WEATHER → EMERGENCY RESEARCH
   - Conditions: storms, extreme temperatures, weather warnings
   - Suggestion: Research safety protocols, emergency procedures, closures

4. EVENT WEATHER → EVENT RESEARCH
   - Keywords: event, party, wedding, graduation, sports, concert
   - Suggestion: Research venue alternatives, indoor options, logistics

IMPORTANT: Only suggest collaboration if it would provide SIGNIFICANT additional value beyond the weather report.

RESPONSE FORMAT (JSON only):
{{
    "should_collaborate": true/false,
    "collaboration_type": "travel"|"activity"|"emergency"|"event"|"none",
    "suggested_research": "specific research question that would help the user",
    "confidence": 0.0-1.0,
    "reasoning": "why this collaboration makes sense"
}}

Example:
User: "Weather for my hiking trip to Yosemite this weekend"
→ {{"should_collaborate": true, "collaboration_type": "activity", "suggested_research": "Research current Yosemite trail conditions, permits, and safety recommendations for weekend hiking", "confidence": 0.8, "reasoning": "User mentioned specific outdoor activity at specific location"}}

Return ONLY valid JSON, nothing else."""

            # Use LangChain interface with system message for JSON output
            messages = [
                SystemMessage(content="You are a collaboration analyzer. Always respond with valid JSON only."),
                HumanMessage(content=collaboration_prompt)
            ]
            response = await llm.ainvoke(messages)
            
            # Parse LLM response
            try:
                response_content = response.content if hasattr(response, 'content') else str(response)
                collaboration_data = json.loads(response_content)
                logger.info(f"🤝 Collaboration Analysis: {collaboration_data.get('collaboration_type', 'none')} (confidence: {collaboration_data.get('confidence', 0.0)})")
                return collaboration_data
                
            except json.JSONDecodeError as e:
                logger.warning(f"❌ Failed to parse collaboration JSON: {e}")
                return {"should_collaborate": False, "collaboration_type": "none", "confidence": 0.0}
            
        except Exception as e:
            logger.error(f"❌ Collaboration detection failed: {e}")
            return {"should_collaborate": False, "collaboration_type": "none", "confidence": 0.0}
    
    def _create_response(self, response_text: str, is_complete: bool = True) -> Dict[str, Any]:
        """Create standardized response"""
        return {
            "messages": [AIMessage(content=response_text)],
            "agent_results": {
                "agent_type": self.agent_type,
                "is_complete": is_complete,
                "response": response_text
            },
            "is_complete": is_complete
        }


# Singleton instance
_weather_agent_instance = None


def get_weather_agent() -> WeatherAgent:
    """Get global weather agent instance"""
    global _weather_agent_instance
    if _weather_agent_instance is None:
        _weather_agent_instance = WeatherAgent()
    return _weather_agent_instance

