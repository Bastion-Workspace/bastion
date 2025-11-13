"""
Weather Agent
Weather information and forecasting agent with structured outputs
"""

import logging
import os
import json
from typing import Dict, Any
from datetime import datetime
from openai import AsyncOpenAI
from langchain_core.messages import AIMessage

from orchestrator.agents.base_agent import BaseAgent
from orchestrator.services.weather_request_analyzer import get_weather_request_analyzer
from orchestrator.services.weather_response_formatters import get_weather_response_formatters
from orchestrator.tools.weather_tools import get_weather_tools

logger = logging.getLogger(__name__)


class WeatherAgent(BaseAgent):
    """
    Meteorological Intelligence Agent
    Provides weather conditions and forecasts with structured outputs
    """
    
    def __init__(self):
        super().__init__("weather_agent")
        self._analyzer = None
        self._formatter = None
        self._weather_tools = None
        self._openai_client = None
    
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
    
    async def _get_openai_client(self):
        """Get or create OpenAI client"""
        if self._openai_client is None:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable not set")
            
            self._openai_client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
        return self._openai_client
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process weather requests with structured outputs"""
        try:
            logger.info("🌤️ Weather Agent: Starting meteorological intelligence operation...")
            
            # Extract user message and context
            messages = state.get("messages", [])
            user_message = messages[-1].content if messages else ""
            shared_memory = state.get("shared_memory", {})
            user_id = state.get("user_id")
            
            # Get instances
            analyzer = await self._get_analyzer()
            formatter = await self._get_formatter()
            weather_tools = await self._get_weather_tools()
            
            # Get user preferences from shared memory and state persona  
            user_preferences = shared_memory.get("user_preferences", {})
            configured_persona = state.get("persona", "casual")
            communication_style = user_preferences.get("communication_style", configured_persona)
            detail_level = user_preferences.get("preferred_detail_level", "moderate")
            
            # Analyze the weather request using intelligent analysis
            weather_request = await analyzer.analyze_weather_request(user_message, shared_memory)
            
            if not weather_request["success"]:
                error_response = await formatter.format_error_response(weather_request["error"], communication_style)
                return self._create_response(error_response, is_complete=False)
            
            # Check if location clarification is needed
            if weather_request.get("location") == "LOCATION_NEEDED":
                clarification_response = "Where would you like weather for? Please provide a city name, state, or ZIP code."
                return self._create_response(clarification_response, is_complete=False)
            
            # Get weather data
            weather_data = await self._get_weather_data(weather_request, weather_tools, user_id)
            
            if not weather_data["success"]:
                error_response = await formatter.format_error_response(weather_data["error"], communication_style)
                return self._create_response(error_response, is_complete=False)
            
            # Store results in shared memory
            analyzer.update_shared_memory(shared_memory, weather_request, weather_data, self.agent_type)
            
            # Get LLM-enhanced intelligent recommendations
            recommendations = await self._get_llm_recommendations(weather_request, weather_data, communication_style)
            
            # Detect research opportunities
            collaboration_data = await self._detect_collaboration_opportunities(user_message, weather_request, weather_data)
            
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
            
            logger.info(f"✅ Weather Agent: Successfully provided weather intelligence for {weather_request['location']}")
            return self._create_response(formatted_response, is_complete=True)
            
        except Exception as e:
            logger.error(f"❌ Weather Agent: Processing failed: {e}")
            configured_persona = state.get("persona", "casual")
            formatter = await self._get_formatter()
            error_response = await formatter.format_error_response(str(e), configured_persona)
            return self._create_response(error_response, is_complete=False)
    
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
    
    async def _get_llm_recommendations(self, request: Dict[str, Any], weather_data: Dict[str, Any], style: str) -> str:
        """Get intelligent LLM-based weather recommendations and insights"""
        try:
            client = await self._get_openai_client()
            fast_model = os.getenv("FAST_MODEL", "anthropic/claude-haiku-4.5")
            
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
                model=fast_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            recommendation = response.choices[0].message.content.strip()
            
            # Validate and clean up recommendation
            if len(recommendation) > 200:
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
            client = await self._get_openai_client()
            fast_model = os.getenv("FAST_MODEL", "anthropic/claude-haiku-4.5")
            
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

            response = await client.chat.completions.create(
                model=fast_model,
                messages=[{"role": "user", "content": collaboration_prompt}],
                temperature=0.2
            )
            
            # Parse LLM response
            try:
                collaboration_data = json.loads(response.choices[0].message.content)
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

