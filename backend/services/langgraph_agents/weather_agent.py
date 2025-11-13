"""
Weather Agent - Roosevelt's "Meteorological Intelligence" Service
Weather information and forecasting agent with structured outputs and LLM intelligence
"""

import logging
from typing import Dict, Any
from datetime import datetime
from pydantic import ValidationError

from services.langgraph_agents.base_agent import BaseAgent
from services.langgraph_agents.weather_request_analyzer import WeatherRequestAnalyzer
from services.langgraph_agents.weather_response_formatters import WeatherResponseFormatters
from models.agent_response_models import WeatherResponse, TaskStatus
from models.shared_memory_models import SharedMemory, validate_shared_memory

logger = logging.getLogger(__name__)


class WeatherAgent(BaseAgent):
    """
    Roosevelt's Meteorological Intelligence Agent
    Provides weather conditions and forecasts with structured outputs and LLM intelligence
    """
    
    def __init__(self):
        super().__init__("weather_agent")
        self.analyzer = WeatherRequestAnalyzer()
        self.formatter = WeatherResponseFormatters()
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process weather requests with structured outputs following Roosevelt's best practices"""
        try:
            logger.info("🌤️ WEATHER AGENT: Starting meteorological intelligence operation...")
            
            # Extract user message and context
            messages = state.get("messages", [])
            user_message = messages[-1].content if messages else ""
            shared_memory = state.get("shared_memory", {})
            user_id = state.get("user_id")
            
            # Get user preferences from shared memory and state persona  
            user_preferences = shared_memory.get("user_preferences", {})
            configured_persona = state.get("persona", "casual")  # Get configured persona from state
            communication_style = user_preferences.get("communication_style", configured_persona)  # Use configured persona as default
            detail_level = user_preferences.get("preferred_detail_level", "moderate")
            
            # Analyze the weather request using intelligent analysis
            weather_request = await self.analyzer.analyze_weather_request(user_message, shared_memory)
            
            if not weather_request["success"]:
                error_response = await self.formatter.format_error_response(weather_request["error"], communication_style)
                return await self._create_structured_response(
                    weather_data=error_response,
                    location=weather_request.get("location", "unknown"),
                    request_type="current",
                    units="imperial",
                    task_status=TaskStatus.ERROR,
                    confidence=0.0
                )
            
            # ROOSEVELT'S POLITE INQUIRY: Check if location clarification is needed
            if weather_request.get("location") == "LOCATION_NEEDED":
                clarification_response = "Where would you like weather for? Please provide a city name, state, or ZIP code."
                return await self._create_structured_response(
                    weather_data=clarification_response,
                    location="location_needed",
                    request_type="location_request",
                    units=weather_request.get("units", "imperial"),
                    task_status=TaskStatus.INCOMPLETE,
                    confidence=1.0
                )
            
            # Get weather data using centralized tool registry
            weather_data = await self._get_weather_data_via_tools(weather_request, state)
            
            if not weather_data["success"]:
                error_response = await self.formatter.format_error_response(weather_data["error"], communication_style)
                return await self._create_structured_response(
                    weather_data=error_response,
                    location=weather_request["location"],
                    request_type=weather_request["request_type"],
                    units=weather_request["units"],
                    task_status=TaskStatus.ERROR,
                    confidence=0.0
                )
            
            # Store results in shared memory using analyzer
            self.analyzer.update_shared_memory(shared_memory, weather_request, weather_data, self.agent_type)
            
            # Get LLM-enhanced intelligent recommendations
            recommendations = await self._get_llm_recommendations(weather_request, weather_data, communication_style)
            
            # ROOSEVELT'S COLLABORATION INTELLIGENCE: Detect research opportunities
            collaboration_data = await self._detect_collaboration_opportunities(user_message, weather_request, weather_data)
            
            # Format response based on user preferences
            formatted_response = await self.formatter.format_weather_response(
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
            
            # ROOSEVELT'S UNIVERSAL FORMATTING: Apply intelligent formatting if beneficial
            formatted_response = await self._apply_universal_formatting(
                user_message, formatted_response, state
            )
            
            # Create structured response with Pydantic validation
            structured_result = await self._create_structured_response(
                weather_data=formatted_response,
                location=weather_request["location"],
                request_type=weather_request["request_type"],
                units=weather_request["units"],
                task_status=TaskStatus.COMPLETE,
                confidence=0.95,
                cached_data=weather_data.get("cached", False),
                recommendations=recommendations,
                collaboration_suggestion=collaboration_data.get("suggested_research"),
                collaboration_confidence=collaboration_data.get("confidence", 0.0)
            )
            
            logger.info(f"✅ WEATHER AGENT: Successfully provided weather intelligence for {weather_request['location']}")
            return structured_result
            
        except Exception as e:
            logger.error(f"❌ WEATHER AGENT: Processing failed: {e}")
            # Use configured persona for error response
            configured_persona = state.get("persona", "casual")
            error_response = await self.formatter.format_error_response(str(e), configured_persona)
            return await self._create_structured_response(
                weather_data=error_response,
                location="unknown",
                request_type="current",
                units="imperial",
                task_status=TaskStatus.ERROR,
                confidence=0.0
            )
    
    async def _get_weather_data_via_tools(self, weather_request: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Get weather data using centralized tool registry following Roosevelt's patterns"""
        try:
            # Use centralized tool registry for proper tool access
            from services.langgraph_tools.centralized_tool_registry import get_tool_registry, AgentType
            
            tool_registry = await get_tool_registry()
            
            if weather_request["request_type"] == "current":
                # Get current conditions
                weather_tool = tool_registry.get_tool_function("weather_conditions", AgentType.WEATHER_AGENT)
                if not weather_tool:
                    raise ValueError("Weather conditions tool not available")
                result = await weather_tool(
                    location=weather_request["location"],
                    units=weather_request["units"],
                    user_id=state.get("user_id")
                )
            else:
                # Get forecast
                weather_tool = tool_registry.get_tool_function("weather_forecast", AgentType.WEATHER_AGENT)
                if not weather_tool:
                    raise ValueError("Weather forecast tool not available")
                result = await weather_tool(
                    location=weather_request["location"],
                    days=weather_request["forecast_days"],
                    units=weather_request["units"],
                    user_id=state.get("user_id")
                )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error getting weather data via tools: {e}")
            return {
                "success": False,
                "error": f"Could not retrieve weather data: {str(e)}"
            }
    
    async def _get_llm_recommendations(self, request: Dict[str, Any], weather_data: Dict[str, Any], style: str) -> str:
        """Get intelligent LLM-based weather recommendations and insights"""
        try:
            # Use OpenRouter for LLM recommendations
            from openai import AsyncOpenAI
            from config import settings
            
            client = AsyncOpenAI(
                api_key=settings.OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1"
            )
            
            # Build prompt for weather intelligence
            weather_info = self._extract_weather_summary(request, weather_data)
            
            prompt = f"""Based on the weather data provided, give intelligent recommendations and insights.

WEATHER DATA:
{weather_info}

COMMUNICATION STYLE: {style}

Provide 1-2 sentences of intelligent recommendations focusing on:
- Optimal activities for these conditions
- What to watch out for
- Any interesting meteorological insights

Keep recommendations practical and match the {style} communication style.
Be brief but insightful - this is additional context, not the main response.
"""
            
            # Get LLM recommendations
            response = await client.chat.completions.create(
                model=settings.FAST_MODEL,  # Use configured fast model for recommendations
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3  # Lower temperature for consistent recommendations
            )
            
            recommendation = response.choices[0].message.content.strip()
            
            # Validate and clean up recommendation
            if len(recommendation) > 200:  # Keep recommendations concise
                recommendation = recommendation[:197] + "..."
            
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
    
    async def _detect_collaboration_opportunities(self, user_message: str, weather_request: Dict[str, Any], weather_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect opportunities for weather→research collaboration using LLM intelligence"""
        try:
            # Use OpenRouter for LLM collaboration analysis
            from openai import AsyncOpenAI
            from config import settings
            
            client = AsyncOpenAI(
                api_key=settings.OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1"
            )
            
            # ROOSEVELT'S PERMISSION-AWARE COLLABORATION: Get research agent permission
            from services.agent_intelligence_network import get_agent_network, CollaborationPermission
            agent_network = get_agent_network()
            research_agent_info = agent_network.get_agent_info("research_agent")
            
            if research_agent_info:
                research_permission = research_agent_info.collaboration_permission
                research_auto_execute = research_permission == CollaborationPermission.AUTO_USE
                logger.info(f"🔐 WEATHER COLLABORATION CHECK: research_agent permission = {research_permission.value} (auto: {research_auto_execute})")
            else:
                research_permission = CollaborationPermission.SUGGEST_ONLY
                research_auto_execute = False
            
            # Extract weather conditions for context
            weather_summary = self._extract_weather_summary(weather_request, weather_data)
            
            collaboration_prompt = f"""Analyze if this weather query should trigger additional research collaboration.

USER MESSAGE: "{user_message}"
LOCATION: "{weather_request['location']}"
WEATHER CONDITIONS: {weather_summary}
RESEARCH AGENT PERMISSION: {research_permission.value} {"(AUTO-EXECUTE)" if research_auto_execute else "(SUGGEST-ONLY)"}

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

PERMISSION LEVELS:
- AUTO_USE: Agent will be executed automatically when beneficial (e.g., Data Formatting)
- SUGGEST_ONLY: User will be asked for approval before agent execution (e.g., Weather, Research)

IMPORTANT: Only suggest collaboration if it would provide SIGNIFICANT additional value beyond the weather report.

RESPONSE FORMAT (JSON only):
{{
    "should_collaborate": true/false,
    "collaboration_type": "travel"|"activity"|"emergency"|"event"|"none",
    "suggested_research": "specific research question that would help the user",
    "confidence": 0.0-1.0,
    "reasoning": "why this collaboration makes sense",
    "execution_type": "{research_permission.value}"
}}

Example:
User: "Weather for my hiking trip to Yosemite this weekend"
→ {{"should_collaborate": true, "collaboration_type": "activity", "suggested_research": "Research current Yosemite trail conditions, permits, and safety recommendations for weekend hiking", "confidence": 0.8, "reasoning": "User mentioned specific outdoor activity at specific location", "execution_type": "suggest_only"}}

Return ONLY valid JSON, nothing else."""

            response = await client.chat.completions.create(
                model=settings.FAST_MODEL,  # Use configured fast model for collaboration analysis
                messages=[{"role": "user", "content": collaboration_prompt}],
                temperature=0.2  # Low temperature for consistent analysis
            )
            
            # Parse LLM response
            import json
            try:
                collaboration_data = json.loads(response.choices[0].message.content)
                logger.info(f"🤝 COLLABORATION ANALYSIS: {collaboration_data.get('collaboration_type', 'none')} (confidence: {collaboration_data.get('confidence', 0.0)})")
                return collaboration_data
                
            except json.JSONDecodeError as e:
                logger.warning(f"❌ Failed to parse collaboration JSON: {e}")
                return {"should_collaborate": False, "collaboration_type": "none", "confidence": 0.0}
            
        except Exception as e:
            logger.error(f"❌ Collaboration detection failed: {e}")
            return {"should_collaborate": False, "collaboration_type": "none", "confidence": 0.0}
    
    async def _apply_universal_formatting(self, user_query: str, weather_response: str, state: Dict[str, Any] = None) -> str:
        """ROOSEVELT'S UNIVERSAL FORMATTING: Apply intelligent formatting to weather responses"""
        try:
            from services.universal_formatting_service import get_universal_formatting_service
            
            formatting_service = get_universal_formatting_service()
            
            # Detect if weather response would benefit from formatting
            formatting_analysis = await formatting_service.detect_formatting_need(
                agent_type="weather_agent",
                user_query=user_query,
                agent_response=weather_response,
                confidence_threshold=0.7  # Medium threshold for weather data
            )
            
            # Apply formatting if recommended and confidence is high enough
            if (formatting_analysis and 
                formatting_service.should_auto_format("weather_agent", formatting_analysis)):
                
                logger.info(f"📊 WEATHER FORMATTING: Applying {formatting_analysis['formatting_type']} formatting")
                # ROOSEVELT'S AUTO-EXECUTION: Pass state to enable Data Formatting Agent execution
                return await formatting_service.apply_formatting(weather_response, formatting_analysis, state)
            
            return weather_response
            
        except Exception as e:
            logger.error(f"❌ Universal formatting failed: {e}")
            return weather_response  # Return original on error
    
    async def _create_structured_response(
        self, 
        weather_data: str, 
        location: str, 
        request_type: str, 
        units: str, 
        task_status: TaskStatus,
        confidence: float,
        cached_data: bool = False,
        recommendations: str = None,
        collaboration_suggestion: str = None,
        collaboration_confidence: float = 0.0
    ) -> Dict[str, Any]:
        """Create structured response using WeatherResponse Pydantic model"""
        try:
            # Create structured response with Pydantic validation
            weather_response = WeatherResponse(
                weather_data=weather_data,
                location=location,
                request_type=request_type,
                units=units,
                task_status=task_status,
                confidence=confidence,
                cached_data=cached_data,
                recommendations=recommendations,
                collaboration_suggestion=collaboration_suggestion,
                collaboration_confidence=collaboration_confidence
            )
            
            # Return in LangGraph-compatible format
            return {
                "agent_results": {
                    "structured_response": weather_response.dict(),
                    "task_status": task_status.value,
                    "is_complete": task_status == TaskStatus.COMPLETE,
                    "collaboration_suggestion": collaboration_suggestion,  # PHASE 1: Track collaboration
                    "collaboration_confidence": collaboration_confidence
                },
                "response": weather_data,  # Natural language response for user (includes collaboration text)
                "latest_response": weather_data  # PHASE 1: Ensure collaboration suggestions appear
            }
            
        except ValidationError as e:
            logger.error(f"❌ Pydantic validation error in weather response: {e}")
            # Fallback to basic response structure
            return {
                "agent_results": {
                    "task_status": TaskStatus.ERROR.value,
                    "is_complete": False,
                    "error": f"Response validation failed: {str(e)}"
                },
                "response": f"**By George!** I encountered a challenge creating the structured weather response: {str(e)}"
            }
        except Exception as e:
            logger.error(f"❌ Error creating structured response: {e}")
            return {
                "agent_results": {
                    "task_status": TaskStatus.ERROR.value,
                    "is_complete": False,
                    "error": str(e)
                },
                "response": f"**BULLY!** Weather operation encountered an unexpected challenge: {str(e)}"
            }