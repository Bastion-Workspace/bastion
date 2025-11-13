"""
Weather Request Analyzer
Analyzes user weather requests to determine intent, location, and preferences
"""

import logging
import re
import os
from typing import Dict, Any
from datetime import datetime
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class WeatherRequestAnalyzer:
    """Analyzes and extracts intent from weather-related user requests"""
    
    def __init__(self):
        self._openai_client = None
    
    async def _get_openai_client(self) -> AsyncOpenAI:
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
    
    async def analyze_weather_request(self, user_message: str, shared_memory: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user message to determine weather request type and location"""
        try:
            message_lower = user_message.lower()
            
            # Check for forecast vs current conditions
            is_forecast = any(word in message_lower for word in [
                "forecast", "tomorrow", "this week", "next", "upcoming", 
                "will it", "going to", "later", "tonight", "weekend"
            ])
            
            # Extract location from message or use defaults
            location = await self._extract_location(user_message, shared_memory)
            
            # Determine forecast length if requesting forecast
            forecast_days = 1
            if is_forecast:
                if any(word in message_lower for word in ["week", "7 days", "seven days"]):
                    forecast_days = 5
                elif any(word in message_lower for word in ["3 days", "three days"]):
                    forecast_days = 3
                elif any(word in message_lower for word in ["tomorrow", "next day"]):
                    forecast_days = 1
                else:
                    forecast_days = 3  # Default forecast length
            
            # Determine units preference
            units = "imperial"  # Default
            if "celsius" in message_lower or "°c" in message_lower:
                units = "metric"
            elif "fahrenheit" in message_lower or "°f" in message_lower:
                units = "imperial"
            else:
                # Check user preferences from shared memory
                weather_prefs = shared_memory.get("weather_data", {}).get("user_preferences", {})
                units = weather_prefs.get("units", "imperial")
            
            return {
                "success": True,
                "request_type": "forecast" if is_forecast else "current",
                "location": location,
                "forecast_days": forecast_days,
                "units": units,
                "original_message": user_message
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing weather request: {e}")
            return {
                "success": False,
                "error": f"Could not understand weather request: {str(e)}"
            }
    
    async def _extract_location(self, user_message: str, shared_memory: Dict[str, Any]) -> str:
        """Extract location using LLM intelligence instead of crude regex patterns"""
        try:
            # LLM-powered location extraction
            location_result = await self._extract_location_with_llm(user_message, shared_memory)
            return location_result["location"]
            
        except Exception as e:
            logger.error(f"❌ LLM location extraction failed: {e}")
            # Graceful fallback to basic patterns only as last resort
            return await self._extract_location_fallback(user_message, shared_memory)
    
    async def _extract_location_with_llm(self, user_message: str, shared_memory: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM intelligence for contextual location extraction"""
        try:
            client = await self._get_openai_client()
            
            # Get comprehensive conversation context
            conversation_context = self._build_conversation_context(shared_memory)
            
            # Get recent location context from shared memory
            weather_data = shared_memory.get("weather_data", {})
            recent_locations = list(weather_data.get("recent_queries", {}).keys())[-3:]  # Last 3 locations
            
            # Get fast model from environment
            fast_model = os.getenv("FAST_MODEL", "anthropic/claude-haiku-4.5")
            
            location_prompt = f"""Extract the location from this weather request. Return ONLY the location in a format suitable for weather APIs.

USER MESSAGE: "{user_message}"

RECENT WEATHER LOCATIONS: {recent_locations if recent_locations else "None"}

CONVERSATION CONTEXT: {conversation_context}

EXTRACTION RULES (PRIORITIZE CONVERSATION CONTEXT):
- If explicit location mentioned: return that location
- **CRITICAL: If no explicit location but conversation context contains research about a specific city/location, USE THAT LOCATION**
- If recent weather locations exist: return most recent weather location  
- For ZIP codes: return the ZIP code exactly
- For cities: return "City, State" or "City, Country" format
- For ambiguous locations: return most likely interpretation based on conversation context
- **If absolutely no location clues exist: return "LOCATION_NEEDED" (user will be asked to clarify)**

EXAMPLES:
"What's the weather in NYC?" → "New York, NY"
"How's the weather today?" (recent: ["Chicago, IL"]) → "Chicago, IL"  
"Weather for 90210?" → "90210"
"Is it raining?" (recent: ["Boston, MA"]) → "Boston, MA"
"Weather there?" (recent: ["Miami, FL"]) → "Miami, FL"
"Check on the weather for the trip" (context: research about Baltimore sightseeing) → "Baltimore, MD"
"What's the weather like?" (context: research about Dallas attractions) → "Dallas, TX"
"Weather forecast?" (no context) → "LOCATION_NEEDED"

Return ONLY the location string, nothing else."""

            response = await client.chat.completions.create(
                model=fast_model,
                messages=[{"role": "user", "content": location_prompt}],
                temperature=0.1  # Low temperature for consistent extraction
            )
            
            extracted_location = response.choices[0].message.content.strip()
            
            # Handle special cases
            if extracted_location == "DEFAULT_LOCATION":
                # Use fallback logic
                extracted_location = await self._get_fallback_location(shared_memory)
                confidence = 0.3
            else:
                confidence = 0.8
            
            logger.info(f"🧠 LLM extracted location: '{extracted_location}' (confidence: {confidence})")
            
            return {
                "location": extracted_location,
                "confidence": confidence,
                "extraction_method": "llm"
            }
            
        except Exception as e:
            logger.error(f"❌ LLM location extraction failed: {e}")
            raise
    
    async def _extract_location_fallback(self, user_message: str, shared_memory: Dict[str, Any]) -> str:
        """Fallback to basic patterns if LLM fails"""
        logger.warning("🚨 Using fallback location extraction - LLM failed")
        
        # Try to find ZIP code in message
        zip_match = re.search(r'\b\d{5}\b', user_message)
        if zip_match:
            return zip_match.group()
        
        # Try to find city name patterns (only as emergency fallback)
        city_patterns = [
            r'\bin\s+([A-Za-z\s,]+?)(?:\s|$|[.!?])',
            r'\bfor\s+([A-Za-z\s,]+?)(?:\s|$|[.!?])',
            r'\bat\s+([A-Za-z\s,]+?)(?:\s|$|[.!?])',
        ]
        
        for pattern in city_patterns:
            match = re.search(pattern, user_message, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                if len(location) > 2 and not location.lower() in ["it", "the", "a", "an"]:
                    return location
        
        # Use fallback location from preferences/config
        return await self._get_fallback_location(shared_memory)
    
    async def _get_fallback_location(self, shared_memory: Dict[str, Any]) -> str:
        """Get fallback location from preferences or config"""
        # Check shared memory for user's default location
        weather_data = shared_memory.get("weather_data", {})
        location_prefs = weather_data.get("location_preferences", {})
        
        if "user_home_zip" in location_prefs:
            return location_prefs["user_home_zip"]
        
        # Check recent queries for location context
        recent_queries = weather_data.get("recent_queries", {})
        if recent_queries:
            # Use the most recent location
            latest_location = max(recent_queries.keys(), 
                                key=lambda k: recent_queries[k].get("timestamp", ""))
            if latest_location:
                return latest_location
        
        # Check environment for default location
        default_location = os.getenv("WEATHER_DEFAULT_LOCATION")
        if default_location:
            return default_location
        
        # Ask user instead of guessing
        return "LOCATION_NEEDED"
    
    def update_shared_memory(self, shared_memory: Dict[str, Any], request: Dict[str, Any], weather_data: Dict[str, Any], agent_name: str) -> None:
        """Update shared memory with weather results"""
        try:
            # Update weather data in shared memory
            weather_memory = shared_memory.setdefault("weather_data", {
                "recent_queries": {},
                "location_preferences": {},
                "user_preferences": {}
            })
            
            # Store the query result
            location = request["location"]
            weather_memory["recent_queries"][location] = {
                "request_type": request["request_type"],
                "data": weather_data,
                "timestamp": datetime.utcnow().isoformat(),
                "units": request["units"]
            }
            
            # Update location preferences
            if location not in weather_memory["location_preferences"].get("frequent_locations", []):
                frequent_locations = weather_memory["location_preferences"].setdefault("frequent_locations", [])
                frequent_locations.append(location)
                # Keep only last 5 locations
                weather_memory["location_preferences"]["frequent_locations"] = frequent_locations[-5:]
            
            # Update user preferences
            weather_memory["user_preferences"]["units"] = request["units"]
            
            # Store weather query metadata in shared memory for other agents
            shared_memory.setdefault("weather_queries", []).append({
                "location": location,
                "request_type": request["request_type"],
                "success": weather_data["success"],
                "timestamp": datetime.utcnow().isoformat(),
                "agent": agent_name
            })
            
        except Exception as e:
            logger.error(f"❌ Error updating shared memory: {e}")
    
    def _build_conversation_context(self, shared_memory: Dict[str, Any]) -> str:
        """Build comprehensive conversation context for location extraction"""
        try:
            context_parts = []
            
            # 1. Research findings context (most important for location detection)
            research_findings = shared_memory.get("research_findings", {})
            if research_findings:
                for key, research_data in list(research_findings.items())[-2:]:  # Last 2 research topics
                    if isinstance(research_data, dict):
                        findings = research_data.get("findings", "")
                        # Extract location mentions from research findings
                        if findings:
                            context_parts.append(f"Recent research topic: '{key}' - {findings[:150]}...")
            
            # 2. Search results context
            search_results = shared_memory.get("search_results", {})
            if search_results:
                for mode, results in search_results.items():
                    if isinstance(results, dict) and results.get("findings"):
                        context_parts.append(f"Recent {mode} search results available")
            
            # 3. Agent results context
            if shared_memory.get("agent_results"):
                context_parts.append("Recent agent results available")
            
            # 4. Recent topics or conversation intelligence
            conversation_intel = shared_memory.get("conversation_intelligence", {})
            if conversation_intel:
                topics = conversation_intel.get("topics_discovered", [])
                if topics:
                    context_parts.append(f"Conversation topics: {', '.join(topics[-3:])}")  # Last 3 topics
            
            if context_parts:
                return " | ".join(context_parts)
            else:
                return "No recent conversation context"
                
        except Exception as e:
            logger.error(f"❌ Error building conversation context: {e}")
            return "Context unavailable"


# Global instance
_analyzer_instance = None


def get_weather_request_analyzer() -> WeatherRequestAnalyzer:
    """Get global weather request analyzer instance"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = WeatherRequestAnalyzer()
    return _analyzer_instance

