"""
Weather Tools Module
Weather operations and forecasting using OpenWeatherMap API
"""

import logging
import aiohttp
import asyncio
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class WeatherTools:
    """Weather tools for agents using OpenWeatherMap API"""
    
    def __init__(self):
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.geocoding_url = "https://api.openweathermap.org/geo/1.0"
        self.cache = {}  # Simple in-memory cache
        self.cache_duration = timedelta(minutes=10)
        
    def get_tools(self) -> Dict[str, Any]:
        """Get all weather tools"""
        return {
            "weather_conditions": self.get_weather_conditions,
            "weather_forecast": self.get_weather_forecast,
        }
    
    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all weather tools"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "weather_conditions",
                    "description": "Get current weather conditions for a specific location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string", 
                                "description": "Location (ZIP code, city name, or 'city,country' format). Examples: '90210', 'Los Angeles', 'London,UK'"
                            },
                            "units": {
                                "type": "string", 
                                "description": "Temperature units: 'imperial' (Fahrenheit), 'metric' (Celsius), or 'kelvin'",
                                "enum": ["imperial", "metric", "kelvin"],
                                "default": "imperial"
                            }
                        },
                        "required": ["location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "weather_forecast",
                    "description": "Get weather forecast for a specific location (up to 5 days)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "Location (ZIP code, city name, or 'city,country' format). Examples: '90210', 'Los Angeles', 'London,UK'"
                            },
                            "days": {
                                "type": "integer",
                                "description": "Number of days to forecast (1-5)",
                                "minimum": 1,
                                "maximum": 5,
                                "default": 3
                            },
                            "units": {
                                "type": "string",
                                "description": "Temperature units: 'imperial' (Fahrenheit), 'metric' (Celsius), or 'kelvin'",
                                "enum": ["imperial", "metric", "kelvin"],
                                "default": "imperial"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
    
    async def get_weather_conditions(self, location: str, units: str = "imperial", user_id: str = None) -> Dict[str, Any]:
        """Get current weather conditions for a location"""
        try:
            logger.info(f"🌤️ Getting current weather for: {location}")
            
            # Get API key from environment
            api_key = os.getenv("OPENWEATHERMAP_API_KEY")
            
            if not api_key:
                return {
                    "success": False,
                    "error": "OpenWeatherMap API key not configured",
                    "location": location
                }
            
            # Check cache first
            cache_key = f"current_{location}_{units}"
            if self._is_cached(cache_key):
                logger.info(f"🎯 Using cached weather data for {location}")
                return self.cache[cache_key]["data"]
            
            # Get coordinates for the location
            coords = await self._get_coordinates(location, api_key)
            if not coords["success"]:
                return coords
            
            lat, lon = coords["lat"], coords["lon"]
            
            # Get current weather
            url = f"{self.base_url}/weather"
            params = {
                "lat": lat,
                "lon": lon,
                "appid": api_key,
                "units": units
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"OpenWeatherMap API error: {response.status} - {error_text}",
                            "location": location
                        }
                    
                    data = await response.json()
            
            # Format the response
            result = self._format_current_weather(data, location, units)
            
            # Cache the result
            self._cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Weather conditions request failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "location": location
            }
    
    async def get_weather_forecast(self, location: str, days: int = 3, units: str = "imperial", user_id: str = None) -> Dict[str, Any]:
        """Get weather forecast for a location"""
        try:
            logger.info(f"🌦️ Getting {days}-day forecast for: {location}")
            
            # Get API key from environment
            api_key = os.getenv("OPENWEATHERMAP_API_KEY")
            
            if not api_key:
                return {
                    "success": False,
                    "error": "OpenWeatherMap API key not configured",
                    "location": location
                }
            
            # Validate days parameter
            if days < 1 or days > 5:
                return {
                    "success": False,
                    "error": "Days must be between 1 and 5",
                    "location": location,
                    "days": days
                }
            
            # Check cache first
            cache_key = f"forecast_{location}_{days}_{units}"
            if self._is_cached(cache_key):
                logger.info(f"🎯 Using cached forecast data for {location}")
                return self.cache[cache_key]["data"]
            
            # Get coordinates for the location
            coords = await self._get_coordinates(location, api_key)
            if not coords["success"]:
                return coords
            
            lat, lon = coords["lat"], coords["lon"]
            
            # Get forecast
            url = f"{self.base_url}/forecast"
            params = {
                "lat": lat,
                "lon": lon,
                "appid": api_key,
                "units": units
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"OpenWeatherMap API error: {response.status} - {error_text}",
                            "location": location
                        }
                    
                    data = await response.json()
            
            # Format the response
            result = self._format_forecast(data, location, days, units)
            
            # Cache the result
            self._cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Weather forecast request failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "location": location,
                "days": days
            }
    
    async def _get_coordinates(self, location: str, api_key: str) -> Dict[str, Any]:
        """Get latitude and longitude for a location with enhanced geocoding"""
        try:
            # Check if location is a ZIP code (US format)
            if location.isdigit() and len(location) == 5:
                # Use ZIP code geocoding
                url = f"{self.geocoding_url}/zip"
                params = {
                    "zip": f"{location},US",
                    "appid": api_key
                }
            else:
                # Enhanced geocoding: Try multiple location formats
                location_variants = self._generate_location_variants(location)
                
                for variant in location_variants:
                    logger.info(f"🎯 Trying location variant: '{variant}'")
                    url = f"{self.geocoding_url}/direct"
                    params = {
                        "q": variant,
                        "limit": 1,
                        "appid": api_key
                    }
                    
                    # Try this variant
                    coords = await self._try_geocoding_request(url, params, variant)
                    if coords["success"]:
                        logger.info(f"✅ Successfully geocoded: '{location}' → '{variant}'")
                        return coords
                    else:
                        logger.info(f"❌ Failed variant: '{variant}' - {coords.get('error', 'Unknown error')}")
                
                # If all variants failed, return the last error
                return {
                    "success": False,
                    "error": f"Location not found after trying {len(location_variants)} variants: {location}",
                    "location": location,
                    "variants_tried": location_variants
                }
            
            # Handle ZIP code request (simple case)
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        return {
                            "success": False,
                            "error": f"Could not find ZIP code: {location}",
                            "location": location
                        }
                    
                    data = await response.json()
            
            # ZIP code response format
            if not data:
                return {
                    "success": False,
                    "error": f"Invalid ZIP code: {location}",
                    "location": location
                }
            return {
                "success": True,
                "lat": data["lat"],
                "lon": data["lon"],
                "name": data.get("name", location),
                "country": data.get("country", "US")
            }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Geocoding error: {str(e)}",
                "location": location
            }
    
    def _generate_location_variants(self, location: str) -> List[str]:
        """Enhanced geocoding: Generate multiple location format variants"""
        variants = [location]  # Always try original first
        
        # State abbreviation expansion mapping
        state_abbreviations = {
            "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
            "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
            "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
            "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
            "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
            "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
            "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
            "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
            "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
            "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming"
        }
        
        # Try different formats for "City, State" patterns
        if "," in location:
            parts = [part.strip() for part in location.split(",")]
            if len(parts) == 2:
                city, state = parts
                
                # If state is abbreviation, try full name
                if state.upper() in state_abbreviations:
                    full_state = state_abbreviations[state.upper()]
                    variants.append(f"{city}, {full_state}")
                    variants.append(f"{city},{full_state}")  # No space after comma
                    variants.append(f"{city} {full_state}")   # No comma
                
                # Try without spaces around comma
                variants.append(f"{city},{state}")
                
                # Try with "US" country code
                variants.append(f"{city}, {state}, US")
                variants.append(f"{city},{state},US")
                
                # Try just the city name
                variants.append(city)
        
        # Try with "US" suffix if not already present
        if ",US" not in location.upper() and ",United States" not in location:
            variants.append(f"{location}, US")
            variants.append(f"{location},US")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variants = []
        for variant in variants:
            if variant not in seen:
                seen.add(variant)
                unique_variants.append(variant)
        
        return unique_variants
    
    async def _try_geocoding_request(self, url: str, params: Dict[str, Any], location_variant: str) -> Dict[str, Any]:
        """Try a single geocoding request"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        return {
                            "success": False,
                            "error": f"API error {response.status}",
                            "location": location_variant
                        }
                    
                    data = await response.json()
                    
                    # Handle empty response
                    if not data or len(data) == 0:
                        return {
                            "success": False,
                            "error": "No results found",
                            "location": location_variant
                        }
                    
                    # Extract coordinates from first result
                    result = data[0]
                    return {
                        "success": True,
                        "lat": result["lat"],
                        "lon": result["lon"],
                        "name": result.get("name", location_variant),
                        "country": result.get("country", "Unknown"),
                        "state": result.get("state", ""),
                        "original_query": location_variant
                    }
                    
        except Exception as e:
            return {
                "success": False,
                "error": f"Request error: {str(e)}",
                "location": location_variant
            }
    
    def _format_current_weather(self, data: Dict[str, Any], location: str, units: str) -> Dict[str, Any]:
        """Format current weather data into a user-friendly response"""
        try:
            # Determine temperature unit symbol
            temp_unit = "°F" if units == "imperial" else "°C" if units == "metric" else "K"
            wind_unit = "mph" if units == "imperial" else "m/s"
            
            # Extract key information
            main = data["main"]
            weather = data["weather"][0]
            wind = data.get("wind", {})
            
            return {
                "success": True,
                "location": {
                    "name": data["name"],
                    "country": data["sys"]["country"],
                    "query": location
                },
                "current": {
                    "temperature": main["temp"],
                    "feels_like": main["feels_like"],
                    "humidity": main["humidity"],
                    "pressure": main["pressure"],
                    "conditions": weather["description"].title(),
                    "conditions_code": weather["main"],
                    "wind_speed": wind.get("speed", 0),
                    "wind_direction": wind.get("deg", 0),
                    "visibility": data.get("visibility", 0) / 1000 if data.get("visibility") else None,  # Convert to km
                    "cloudiness": data["clouds"]["all"] if "clouds" in data else 0
                },
                "units": {
                    "temperature": temp_unit,
                    "wind_speed": wind_unit,
                    "pressure": "hPa",
                    "visibility": "km"
                },
                "timestamp": datetime.utcnow().isoformat(),
                "data_source": "OpenWeatherMap"
            }
            
        except Exception as e:
            logger.error(f"❌ Error formatting current weather data: {e}")
            return {
                "success": False,
                "error": f"Error formatting weather data: {str(e)}",
                "location": location
            }
    
    def _format_forecast(self, data: Dict[str, Any], location: str, days: int, units: str) -> Dict[str, Any]:
        """Format forecast data into a user-friendly response"""
        try:
            # Determine temperature unit symbol
            temp_unit = "°F" if units == "imperial" else "°C" if units == "metric" else "K"
            wind_unit = "mph" if units == "imperial" else "m/s"
            
            # Group forecast data by day
            daily_forecasts = {}
            
            for item in data["list"]:
                dt = datetime.fromtimestamp(item["dt"])
                date_key = dt.strftime("%Y-%m-%d")
                
                if date_key not in daily_forecasts:
                    daily_forecasts[date_key] = {
                        "date": date_key,
                        "day_name": dt.strftime("%A"),
                        "temperatures": [],
                        "conditions": [],
                        "humidity": [],
                        "wind_speeds": [],
                        "precipitation": []
                    }
                
                daily_forecasts[date_key]["temperatures"].append(item["main"]["temp"])
                daily_forecasts[date_key]["conditions"].append(item["weather"][0]["description"])
                daily_forecasts[date_key]["humidity"].append(item["main"]["humidity"])
                daily_forecasts[date_key]["wind_speeds"].append(item["wind"].get("speed", 0))
                
                # Check for precipitation
                rain = item.get("rain", {}).get("3h", 0)
                snow = item.get("snow", {}).get("3h", 0)
                daily_forecasts[date_key]["precipitation"].append(rain + snow)
            
            # Process and limit to requested days
            forecast_days = []
            for date_key in sorted(daily_forecasts.keys())[:days]:
                day_data = daily_forecasts[date_key]
                
                forecast_days.append({
                    "date": day_data["date"],
                    "day_name": day_data["day_name"],
                    "temperature": {
                        "high": max(day_data["temperatures"]),
                        "low": min(day_data["temperatures"]),
                        "average": sum(day_data["temperatures"]) / len(day_data["temperatures"])
                    },
                    "conditions": max(set(day_data["conditions"]), key=day_data["conditions"].count),  # Most common condition
                    "humidity": sum(day_data["humidity"]) / len(day_data["humidity"]),
                    "wind_speed": sum(day_data["wind_speeds"]) / len(day_data["wind_speeds"]),
                    "precipitation_total": sum(day_data["precipitation"]),
                    "precipitation_probability": min(100, len([p for p in day_data["precipitation"] if p > 0]) / len(day_data["precipitation"]) * 100)
                })
            
            return {
                "success": True,
                "location": {
                    "name": data["city"]["name"],
                    "country": data["city"]["country"],
                    "query": location
                },
                "forecast": forecast_days,
                "units": {
                    "temperature": temp_unit,
                    "wind_speed": wind_unit,
                    "precipitation": "mm"
                },
                "days_requested": days,
                "timestamp": datetime.utcnow().isoformat(),
                "data_source": "OpenWeatherMap"
            }
            
        except Exception as e:
            logger.error(f"❌ Error formatting forecast data: {e}")
            return {
                "success": False,
                "error": f"Error formatting forecast data: {str(e)}",
                "location": location,
                "days": days
            }
    
    def _is_cached(self, cache_key: str) -> bool:
        """Check if data is in cache and still valid"""
        if cache_key not in self.cache:
            return False
        
        cached_time = self.cache[cache_key]["timestamp"]
        return datetime.utcnow() - cached_time < self.cache_duration
    
    def _cache_result(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Cache result with timestamp"""
        self.cache[cache_key] = {
            "data": data,
            "timestamp": datetime.utcnow()
        }
        
        # Simple cache cleanup - remove old entries
        if len(self.cache) > 100:  # Limit cache size
            # Remove oldest entries
            oldest_keys = sorted(self.cache.keys(), 
                               key=lambda k: self.cache[k]["timestamp"])[:20]
            for key in oldest_keys:
                del self.cache[key]


# Global instance for use by agents
_weather_tools_instance = None


async def get_weather_tools() -> WeatherTools:
    """Get global weather tools instance"""
    global _weather_tools_instance
    if _weather_tools_instance is None:
        _weather_tools_instance = WeatherTools()
    return _weather_tools_instance


async def weather_conditions(location: str, units: str = "imperial", user_id: str = None) -> Dict[str, Any]:
    """Tool function: Get current weather conditions"""
    tools_instance = await get_weather_tools()
    return await tools_instance.get_weather_conditions(location, units, user_id)


async def weather_forecast(location: str, days: int = 3, units: str = "imperial", user_id: str = None) -> Dict[str, Any]:
    """Tool function: Get weather forecast"""
    tools_instance = await get_weather_tools()
    return await tools_instance.get_weather_forecast(location, days, units, user_id)

