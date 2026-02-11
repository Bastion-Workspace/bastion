"""
Weather Request Analyzer
Analyzes user weather requests to determine intent, location, and preferences
"""

import logging
import re
import os
from typing import Dict, Any, Optional
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from config.settings import settings

logger = logging.getLogger(__name__)


class WeatherRequestAnalyzer:
    """Analyzes and extracts intent from weather-related user requests"""
    
    def __init__(self):
        self._llm = None
    
    def _get_llm(self, temperature: float = 0.1, model: str = None) -> ChatOpenAI:
        """Get configured LLM instance using centralized settings"""
        if model is None:
            model = settings.FAST_MODEL
        
        return ChatOpenAI(
            model=model,
            openai_api_key=settings.OPENROUTER_API_KEY,
            openai_api_base=settings.OPENROUTER_BASE_URL,
            temperature=temperature
        )
    
    async def analyze_weather_request(self, user_message: str, shared_memory: Dict[str, Any], user_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze user message to determine weather request type and location"""
        try:
            message_lower = user_message.lower()
            
            # Check for historical weather requests first
            is_historical, date_str = await self._detect_historical_intent(user_message, shared_memory)
            
            if is_historical and date_str:
                # Extract location from message or use defaults
                location = await self._extract_location(user_message, shared_memory, user_id)
                
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
                
                # Check if this is a year-only query (needs multi-month expansion)
                is_year_only = len(date_str.split("-")) == 1  # Just YYYY format
                
                return {
                    "success": True,
                    "request_type": "history",
                    "location": location,
                    "date_str": date_str,
                    "is_year_only": is_year_only,
                    "units": units,
                    "original_message": user_message
                }
            
            # Check for forecast vs current conditions
            is_forecast = any(word in message_lower for word in [
                "forecast", "tomorrow", "this week", "next", "upcoming", 
                "will it", "going to", "later", "tonight", "weekend"
            ])
            
            # Extract location from message or use defaults
            location = await self._extract_location(user_message, shared_memory, user_id)
            
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
    
    async def _detect_historical_intent(self, user_message: str, shared_memory: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Detect if user is requesting historical weather and extract date string"""
        try:
            message_lower = user_message.lower()
            
            # Check for historical keywords
            historical_keywords = [
                "historical", "history", "past", "previous", "last year", "last month",
                "in december", "in january", "in february", "in march", "in april",
                "in may", "in june", "in july", "in august", "in september",
                "in october", "in november", "december", "january", "february",
                "march", "april", "may", "june", "july", "august", "september",
                "october", "november", "average", "was", "were", "back in"
            ]
            
            has_historical_keyword = any(keyword in message_lower for keyword in historical_keywords)
            
            # Use LLM to extract date if historical intent detected
            if has_historical_keyword:
                date_str = await self._extract_date_with_llm(user_message, shared_memory)
                if date_str:
                    return (True, date_str)
            
            # Also check for explicit date patterns (YYYY-MM-DD or YYYY-MM)
            import re
            date_patterns = [
                r'\b(\d{4}-\d{2}-\d{2})\b',  # YYYY-MM-DD
                r'\b(\d{4}-\d{2})\b',        # YYYY-MM
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, user_message)
                if match:
                    return (True, match.group(1))
            
            return (False, None)
            
        except Exception as e:
            logger.error(f"❌ Error detecting historical intent: {e}")
            return (False, None)
    
    async def _extract_date_with_llm(self, user_message: str, shared_memory: Dict[str, Any]) -> Optional[str]:
        """Use LLM to extract date string from historical weather request"""
        try:
            llm = self._get_llm(temperature=0.1, model=settings.FAST_MODEL)
            
            date_extraction_prompt = f"""Extract the date from this historical weather request. Return ONLY the date in one of these formats:
- YYYY-MM-DD for a specific day (e.g., "2022-12-15")
- YYYY-MM for a monthly average (e.g., "2022-12")
- YYYY for a full year (e.g., "2022") - use when user asks for "all of 2022", "throughout 2022", "average for 2022", etc.

USER MESSAGE: "{user_message}"

EXTRACTION RULES:
- If specific day mentioned: return YYYY-MM-DD
- If only month/year mentioned: return YYYY-MM
- If user asks for "all of [year]", "throughout [year]", "average for [year]", "monthly temps in [year]": return YYYY (year only)
- If relative date (e.g., "last December"): infer the year (assume recent past if not specified)
- If no clear date: return "NONE"

EXAMPLES:
"weather in December 2022" → "2022-12"
"temperature on December 15, 2022" → "2022-12-15"
"average temperature last December" → "2023-12" (if current year is 2024)
"average monthly temps in 2022" → "2022" (year only - indicates all months needed)
"all of 2022" → "2022" (year only)
"throughout 2022" → "2022" (year only)
"weather in 2022" (ambiguous) → "2022" (year only - safer to assume all months)

Return ONLY the date string or "NONE", nothing else."""

            response = await llm.ainvoke([HumanMessage(content=date_extraction_prompt)])
            extracted_date = response.content if hasattr(response, 'content') else str(response)
            extracted_date = extracted_date.strip()
            
            if extracted_date.upper() == "NONE" or not extracted_date:
                return None
            
            # Validate date format (now supports YYYY, YYYY-MM, or YYYY-MM-DD)
            import re
            if re.match(r'^\d{4}(-\d{2}(-\d{2})?)?$', extracted_date):
                logger.info(f"🧠 LLM extracted date: '{extracted_date}'")
                return extracted_date
            else:
                logger.warning(f"⚠️ Invalid date format extracted: '{extracted_date}'")
                return None
                
        except Exception as e:
            logger.error(f"❌ LLM date extraction failed: {e}")
            return None
    
    async def _extract_location(self, user_message: str, shared_memory: Dict[str, Any], user_id: Optional[str] = None) -> str:
        """Extract location using LLM intelligence instead of crude regex patterns"""
        try:
            # LLM-powered location extraction
            location_result = await self._extract_location_with_llm(user_message, shared_memory, user_id)
            return location_result["location"]
            
        except Exception as e:
            logger.error(f"❌ LLM location extraction failed: {e}")
            # Graceful fallback to basic patterns only as last resort
            return await self._extract_location_fallback(user_message, shared_memory, user_id)
    
    async def _extract_location_with_llm(self, user_message: str, shared_memory: Dict[str, Any], user_id: Optional[str] = None) -> Dict[str, Any]:
        """Use LLM intelligence for contextual location extraction"""
        try:
            # Use centralized LLM access
            llm = self._get_llm(temperature=0.1, model=settings.FAST_MODEL)
            
            # Get comprehensive conversation context
            conversation_context = self._build_conversation_context(shared_memory)
            
            # Get recent location context from shared memory
            weather_data = shared_memory.get("weather_data", {})
            recent_locations = list(weather_data.get("recent_queries", {}).keys())[-3:]  # Last 3 locations
            location_prefs = weather_data.get("location_preferences", {})
            user_home_zip = location_prefs.get("user_home_zip", "")
            
            # Note: User ZIP code should be loaded into shared_memory by the weather agent
            # from metadata when available. If not in shared_memory, we'll use LOCATION_NEEDED
            
            location_prompt = f"""Extract the location from this weather request. Return ONLY the location in a format suitable for weather APIs.

USER MESSAGE: "{user_message}"

RECENT WEATHER LOCATIONS: {recent_locations if recent_locations else "None"}
USER DEFAULT LOCATION: {user_home_zip if user_home_zip else "None"}

CONVERSATION CONTEXT: {conversation_context}

EXTRACTION RULES (PRIORITIZE IN THIS ORDER):
1. If explicit location mentioned: return that location
2. **CRITICAL: If no explicit location but conversation context contains research about a specific city/location, USE THAT LOCATION**
3. If recent weather locations exist: return most recent weather location
4. **If user has a default location (user_home_zip): use that as fallback before asking for clarification**
5. For ZIP codes: return the ZIP code exactly
6. For cities: return "City, State" or "City, Country" format
7. For ambiguous locations: return most likely interpretation based on conversation context
8. **ONLY if absolutely no location clues exist AND no default location: return "LOCATION_NEEDED"**

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

            # Use LangChain interface
            response = await llm.ainvoke([HumanMessage(content=location_prompt)])
            extracted_location = response.content if hasattr(response, 'content') else str(response)
            extracted_location = extracted_location.strip()
            # Strip common LLM preambles that slip through (e.g. "Let me correct that: 14532" -> "14532")
            extracted_location = self._strip_location_preamble(extracted_location, user_message)
            
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
    
    async def _extract_location_fallback(self, user_message: str, shared_memory: Dict[str, Any], user_id: Optional[str] = None) -> str:
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
        return await self._get_fallback_location(shared_memory, user_id)
    
    def _strip_location_preamble(self, extracted_location: str, user_message: str) -> str:
        """Strip common LLM preambles from location so we get only the actual place (e.g. ZIP or city)."""
        if not extracted_location or extracted_location == "LOCATION_NEEDED":
            return extracted_location
        # Common preambles that some models prepend despite "Return ONLY the location"
        preambles = [
            r"^Let me correct that:\s*",
            r"^Actually,?\s*",
            r"^I think you mean\s*",
            r"^You probably meant\s*",
            r"^I believe you mean\s*",
            r"^Did you mean\s*",
            r"^Perhaps you meant\s*",
        ]
        cleaned = extracted_location
        for pattern in preambles:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
        # If nothing left or only preamble (no real location), try ZIP from user message
        if not cleaned or len(cleaned) < 2:
            zip_match = re.search(r"\b\d{5}\b", user_message)
            if zip_match:
                return zip_match.group()
        return cleaned if cleaned else extracted_location
    
    async def _get_fallback_location(self, shared_memory: Dict[str, Any], user_id: Optional[str] = None) -> str:
        """Get fallback location from preferences or config"""
        # Check shared memory for user's default location
        weather_data = shared_memory.get("weather_data", {})
        location_prefs = weather_data.get("location_preferences", {})
        
        if "user_home_zip" in location_prefs:
            return location_prefs["user_home_zip"]
        
        # Note: User ZIP code should be loaded into shared_memory by the weather agent
        # from metadata when available. If not in shared_memory, we'll use LOCATION_NEEDED
        
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

