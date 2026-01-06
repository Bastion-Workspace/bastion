"""
Weather Tools Module
Weather operations and forecasting for LangGraph agents using OpenWeatherMap API
"""

import logging
import aiohttp
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import math

# Import weather models using explicit tools_service path
# This avoids conflicts with backend's 'from models.xxx' imports
from tools_service.models.weather_models import (
    WeatherConditionsRequest,
    WeatherForecastRequest,
    WeatherHistoryRequest
)

logger = logging.getLogger(__name__)


class WeatherTools:
    """Weather tools for LangGraph agents using OpenWeatherMap API"""
    
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
            "weather_history": self.get_weather_history,
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
            },
            {
                "type": "function",
                "function": {
                    "name": "weather_history",
                    "description": "Get historical weather data for a specific location and date. Supports specific days (YYYY-MM-DD) or monthly averages (YYYY-MM)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "Location (ZIP code, city name, or 'city,country' format). Examples: '90210', 'Los Angeles', 'London,UK'"
                            },
                            "date_str": {
                                "type": "string",
                                "description": "Date string: 'YYYY-MM-DD' for specific day, 'YYYY-MM' for monthly average. Example: '2022-12-15' or '2022-12'"
                            },
                            "units": {
                                "type": "string",
                                "description": "Temperature units: 'imperial' (Fahrenheit), 'metric' (Celsius), or 'kelvin'",
                                "enum": ["imperial", "metric", "kelvin"],
                                "default": "imperial"
                            }
                        },
                        "required": ["location", "date_str"]
                    }
                }
            }
        ]
    
    async def get_weather_conditions(self, location: str, units: str = "imperial", user_id: str = None) -> Dict[str, Any]:
        """Get current weather conditions for a location"""
        try:
            logger.info(f"🌤️ Getting current weather for: {location}")
            
            # Import config here to avoid circular imports
            from config import settings
            
            if not settings.OPENWEATHERMAP_API_KEY:
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
            coords = await self._get_coordinates(location, settings.OPENWEATHERMAP_API_KEY, user_id)
            if not coords["success"]:
                return coords
            
            lat, lon = coords["lat"], coords["lon"]
            
            # Get current weather
            url = f"{self.base_url}/weather"
            params = {
                "lat": lat,
                "lon": lon,
                "appid": settings.OPENWEATHERMAP_API_KEY,
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
            
            # Import config here to avoid circular imports
            from config import settings
            
            if not settings.OPENWEATHERMAP_API_KEY:
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
            coords = await self._get_coordinates(location, settings.OPENWEATHERMAP_API_KEY, user_id)
            if not coords["success"]:
                return coords
            
            lat, lon = coords["lat"], coords["lon"]
            
            # Get forecast
            url = f"{self.base_url}/forecast"
            params = {
                "lat": lat,
                "lon": lon,
                "appid": settings.OPENWEATHERMAP_API_KEY,
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
    
    async def get_weather_history(self, location: str, date_str: str, units: str = "imperial", user_id: str = None) -> Dict[str, Any]:
        """Get historical weather data for a location and date"""
        try:
            logger.info(f"📅 Getting historical weather for: {location} on {date_str}")
            
            # Import config here to avoid circular imports
            from config import settings
            
            if not settings.OPENWEATHERMAP_API_KEY:
                return {
                    "success": False,
                    "error": "OpenWeatherMap API key not configured",
                    "location": location
                }
            
            # ROOSEVELT'S DATE RANGE HANDLER: Detect and expand date ranges
            if " to " in date_str or " - " in date_str:
                # This is a date range - expand into monthly queries
                return await self._get_date_range_history(location, date_str, units, user_id)
            
            # Parse date string to determine if it's a specific day or month
            is_monthly = len(date_str.split("-")) == 2  # YYYY-MM format
            
            # Check cache first
            cache_key = f"history_{location}_{date_str}_{units}"
            if self._is_cached(cache_key):
                logger.info(f"🎯 Using cached historical weather data for {location} on {date_str}")
                return self.cache[cache_key]["data"]
            
            # Get coordinates for the location
            coords = await self._get_coordinates(location, settings.OPENWEATHERMAP_API_KEY, user_id)
            if not coords["success"]:
                return coords
            
            lat, lon = coords["lat"], coords["lon"]
            
            if is_monthly:
                # Monthly average - use One Call API 3.0 statistical aggregation
                # For monthly data, we'll sample multiple days and calculate averages
                result = await self._get_monthly_average(lat, lon, date_str, units, settings.OPENWEATHERMAP_API_KEY)
            else:
                # Specific day - use One Call API 3.0 timemachine
                result = await self._get_daily_history(lat, lon, date_str, units, settings.OPENWEATHERMAP_API_KEY)
            
            if not result.get("success"):
                return result
            
            # Format the response
            formatted_result = self._format_historical_weather(result, location, date_str, units, is_monthly)
            
            # Cache the result (longer cache for historical data - 1 hour)
            self._cache_result(cache_key, formatted_result)
            
            return formatted_result
            
        except Exception as e:
            logger.error(f"❌ Historical weather request failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "location": location,
                "date_str": date_str
            }
    
    async def _get_daily_history(self, lat: float, lon: float, date_str: str, units: str, api_key: str) -> Dict[str, Any]:
        """Get historical weather for a specific day using One Call API 3.0 timemachine"""
        try:
            # Parse date to Unix timestamp
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            timestamp = int(date_obj.timestamp())
            
            # Use One Call API 3.0 timemachine endpoint
            url = f"{self.base_url}/onecall/timemachine"
            params = {
                "lat": lat,
                "lon": lon,
                "dt": timestamp,
                "appid": api_key,
                "units": units
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        # Check for subscription-related errors
                        if response.status == 401:
                            return {
                                "success": False,
                                "error": "Historical weather data requires an OpenWeatherMap One Call API 3.0 subscription. Current and forecast weather are still available with a free API key.",
                                "location": f"{lat},{lon}",
                                "date_str": date_str,
                                "subscription_required": True
                            }
                        elif response.status == 403:
                            return {
                                "success": False,
                                "error": "Access denied. Please verify your OpenWeatherMap API key has One Call API 3.0 subscription enabled.",
                                "location": f"{lat},{lon}",
                                "date_str": date_str,
                                "subscription_required": True
                            }
                        return {
                            "success": False,
                            "error": f"OpenWeatherMap API error: {response.status} - {error_text}",
                            "location": f"{lat},{lon}",
                            "date_str": date_str
                        }
                    
                    data = await response.json()
            
            return {
                "success": True,
                "data": data,
                "date_str": date_str,
                "timestamp": timestamp
            }
            
        except ValueError as e:
            return {
                "success": False,
                "error": f"Invalid date format: {date_str}. Use YYYY-MM-DD format.",
                "date_str": date_str
            }
        except Exception as e:
            logger.error(f"❌ Error getting daily history: {e}")
            return {
                "success": False,
                "error": str(e),
                "date_str": date_str
            }
    
    async def _get_monthly_average(self, lat: float, lon: float, date_str: str, units: str, api_key: str) -> Dict[str, Any]:
        """Get monthly average weather by sampling multiple days"""
        try:
            # Parse month
            year, month = map(int, date_str.split("-"))
            
            # Sample days throughout the month (1st, 8th, 15th, 22nd, last day)
            from calendar import monthrange
            last_day = monthrange(year, month)[1]
            sample_days = [1, 8, 15, 22, last_day]
            
            daily_results = []
            subscription_error = None
            
            for day in sample_days:
                day_str = f"{year}-{month:02d}-{day:02d}"
                daily_result = await self._get_daily_history(lat, lon, day_str, units, api_key)
                if daily_result.get("success"):
                    daily_results.append(daily_result)
                else:
                    # Check if this is a subscription error (401)
                    error_msg = daily_result.get("error", "")
                    if "One Call API subscription required" in error_msg or "subscription required" in error_msg.lower():
                        subscription_error = error_msg
                        # Stop trying if we hit subscription error - all will fail
                        break
            
            # If we got a subscription error, return it immediately
            if subscription_error:
                return {
                    "success": False,
                    "error": subscription_error,
                    "date_str": date_str
                }
            
            if not daily_results:
                return {
                    "success": False,
                    "error": "Could not retrieve any historical data for the specified month. Historical weather data requires an OpenWeatherMap One Call API 3.0 subscription.",
                    "date_str": date_str
                }
            
            # Calculate averages from sampled days
            return {
                "success": True,
                "data": daily_results,
                "date_str": date_str,
                "is_monthly": True
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting monthly average: {e}")
            return {
                "success": False,
                "error": str(e),
                "date_str": date_str
            }
    
    async def _get_date_range_history(self, location: str, date_range_str: str, units: str, user_id: str) -> Dict[str, Any]:
        """Get historical weather for a date range by expanding into monthly queries"""
        try:
            # Parse date range (e.g., "2022-10 to 2024-02" or "2022-10 - 2024-02")
            separator = " to " if " to " in date_range_str else " - "
            parts = date_range_str.split(separator)
            if len(parts) != 2:
                return {
                    "success": False,
                    "error": f"Invalid date range format: {date_range_str}. Use 'YYYY-MM to YYYY-MM' or 'YYYY-MM - YYYY-MM'"
                }
            
            start_str = parts[0].strip()
            end_str = parts[1].strip()
            
            # Validate format (should be YYYY-MM)
            if len(start_str.split("-")) != 2 or len(end_str.split("-")) != 2:
                return {
                    "success": False,
                    "error": f"Date range must use YYYY-MM format. Got: {date_range_str}"
                }
            
            # Parse start and end dates
            start_year, start_month = map(int, start_str.split("-"))
            end_year, end_month = map(int, end_str.split("-"))
            
            # Validate range
            if start_year > end_year or (start_year == end_year and start_month > end_month):
                return {
                    "success": False,
                    "error": f"Invalid date range: start date must be before end date. Got: {date_range_str}"
                }
            
            # Calculate months difference
            months_diff = (end_year - start_year) * 12 + (end_month - start_month) + 1
            
            # Limit range to prevent excessive API calls (max 24 months = 2 years)
            if months_diff > 24:
                return {
                    "success": False,
                    "error": f"Date range too large: {months_diff} months. Maximum supported range is 24 months (2 years)."
                }
            
            logger.info(f"📅 Expanding date range {date_range_str} into {months_diff} monthly queries")
            
            # Get coordinates once
            from config import settings
            coords = await self._get_coordinates(location, settings.OPENWEATHERMAP_API_KEY, user_id)
            if not coords["success"]:
                return coords
            
            lat, lon = coords["lat"], coords["lon"]
            
            # Query each month in the range
            monthly_results = []
            current_year = start_year
            current_month = start_month
            
            while (current_year < end_year) or (current_year == end_year and current_month <= end_month):
                month_str = f"{current_year}-{current_month:02d}"
                logger.info(f"📅 Fetching monthly data for {month_str}...")
                
                month_result = await self._get_monthly_average(lat, lon, month_str, units, settings.OPENWEATHERMAP_API_KEY)
                
                if month_result.get("success"):
                    monthly_results.append({
                        "month": month_str,
                        "year": current_year,
                        "month_num": current_month,
                        "data": month_result
                    })
                else:
                    # Log but continue - some months may fail due to subscription limits
                    logger.warning(f"⚠️ Failed to get data for {month_str}: {month_result.get('error', 'Unknown error')}")
                
                # Move to next month
                current_month += 1
                if current_month > 12:
                    current_month = 1
                    current_year += 1
            
            if not monthly_results:
                return {
                    "success": False,
                    "error": f"Could not retrieve any historical data for the range {date_range_str}. Historical weather data requires an OpenWeatherMap One Call API 3.0 subscription."
                }
            
            # Aggregate results across all months
            all_temps = []
            all_mins = []
            all_maxs = []
            all_humidities = []
            all_wind_speeds = []
            all_conditions = []
            
            for month_data in monthly_results:
                daily_results = month_data["data"]["data"]  # List of daily_result dicts
                for daily_result in daily_results:
                    if daily_result.get("success") and "data" in daily_result:
                        data = daily_result["data"]
                        if "current" in data:
                            current = data["current"]
                            all_temps.append(current.get("temp", 0))
                            all_conditions.append(current.get("weather", [{}])[0].get("description", ""))
                            all_humidities.append(current.get("humidity", 0))
                            all_wind_speeds.append(current.get("wind_speed", 0))
            
            # Calculate range-wide averages
            if all_temps:
                range_avg_temp = sum(all_temps) / len(all_temps)
                range_min_temp = min(all_temps)
                range_max_temp = max(all_temps)
            else:
                range_avg_temp = range_min_temp = range_max_temp = 0
            
            range_avg_humidity = sum(all_humidities) / len(all_humidities) if all_humidities else 0
            range_avg_wind = sum(all_wind_speeds) / len(all_wind_speeds) if all_wind_speeds else 0
            most_common_condition = max(set(all_conditions), key=all_conditions.count) if all_conditions else "N/A"
            
            # Format as range summary
            temp_unit = "°F" if units == "imperial" else "°C" if units == "metric" else "K"
            wind_unit = "mph" if units == "imperial" else "m/s"
            
            return {
                "success": True,
                "location": {
                    "name": location,
                    "query": location
                },
                "period": {
                    "type": "date_range",
                    "date_str": date_range_str,
                    "start_date": start_str,
                    "end_date": end_str,
                    "months_in_range": months_diff,
                    "months_retrieved": len(monthly_results)
                },
                "historical": {
                    "average_temperature": range_avg_temp,
                    "min_temperature": range_min_temp,
                    "max_temperature": range_max_temp,
                    "average_humidity": range_avg_humidity,
                    "average_wind_speed": range_avg_wind,
                    "most_common_conditions": most_common_condition,
                    "monthly_data": [
                        {
                            "month": m["month"],
                            "year": m["year"],
                            "month_num": m["month_num"],
                            "average_temperature": self._calculate_month_avg_temp(m["data"]["data"]) if m["data"].get("data") else 0
                        }
                        for m in monthly_results
                    ]
                },
                "units": {
                    "temperature": temp_unit,
                    "wind_speed": wind_unit,
                    "humidity": "%"
                },
                "timestamp": datetime.utcnow().isoformat(),
                "data_source": "OpenWeatherMap"
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting date range history: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": f"Error processing date range: {str(e)}",
                "date_str": date_range_str
            }
    
    def _calculate_month_avg_temp(self, daily_results: List[Dict[str, Any]]) -> float:
        """Calculate average temperature from a list of daily result dicts"""
        temps = []
        for daily_result in daily_results:
            if daily_result.get("success") and "data" in daily_result:
                data = daily_result["data"]
                if "current" in data:
                    temp = data["current"].get("temp", 0)
                    if temp > 0:  # Only count valid temperatures
                        temps.append(temp)
        return sum(temps) / len(temps) if temps else 0.0
    
    def _format_historical_weather(self, result: Dict[str, Any], location: str, date_str: str, units: str, is_monthly: bool) -> Dict[str, Any]:
        """Format historical weather data into a user-friendly response"""
        try:
            temp_unit = "°F" if units == "imperial" else "°C" if units == "metric" else "K"
            wind_unit = "mph" if units == "imperial" else "m/s"
            
            if is_monthly:
                # Monthly average formatting
                daily_results = result["data"]
                all_temps = []
                all_conditions = []
                all_humidities = []
                all_wind_speeds = []
                
                for daily_result in daily_results:
                    data = daily_result["data"]
                    if "current" in data:
                        current = data["current"]
                        all_temps.append(current.get("temp", 0))
                        all_conditions.append(current.get("weather", [{}])[0].get("description", ""))
                        all_humidities.append(current.get("humidity", 0))
                        all_wind_speeds.append(current.get("wind_speed", 0))
                
                if not all_temps:
                    return {
                        "success": False,
                        "error": "No temperature data available for the specified month",
                        "location": location,
                        "date_str": date_str
                    }
                
                return {
                    "success": True,
                    "location": {
                        "name": location,
                        "query": location
                    },
                    "period": {
                        "type": "monthly_average",
                        "date_str": date_str,
                        "year": int(date_str.split("-")[0]),
                        "month": int(date_str.split("-")[1])
                    },
                    "historical": {
                        "average_temperature": sum(all_temps) / len(all_temps),
                        "min_temperature": min(all_temps),
                        "max_temperature": max(all_temps),
                        "average_humidity": sum(all_humidities) / len(all_humidities),
                        "average_wind_speed": sum(all_wind_speeds) / len(all_wind_speeds),
                        "most_common_conditions": max(set(all_conditions), key=all_conditions.count) if all_conditions else "N/A",
                        "sample_days": len(daily_results)
                    },
                    "units": {
                        "temperature": temp_unit,
                        "wind_speed": wind_unit,
                        "humidity": "%"
                    },
                    "timestamp": datetime.utcnow().isoformat(),
                    "data_source": "OpenWeatherMap"
                }
            else:
                # Specific day formatting
                data = result["data"]
                if "current" not in data:
                    return {
                        "success": False,
                        "error": "No current weather data in historical response",
                        "location": location,
                        "date_str": date_str
                    }
                
                current = data["current"]
                weather = current.get("weather", [{}])[0]
                
                return {
                    "success": True,
                    "location": {
                        "name": location,
                        "query": location
                    },
                    "period": {
                        "type": "daily",
                        "date_str": date_str,
                        "date": datetime.strptime(date_str, "%Y-%m-%d").strftime("%B %d, %Y")
                    },
                    "historical": {
                        "temperature": current.get("temp", 0),
                        "feels_like": current.get("feels_like", 0),
                        "humidity": current.get("humidity", 0),
                        "pressure": current.get("pressure", 0),
                        "conditions": weather.get("description", "").title(),
                        "conditions_code": weather.get("main", ""),
                        "wind_speed": current.get("wind_speed", 0),
                        "wind_direction": current.get("wind_deg", 0),
                        "cloudiness": current.get("clouds", 0)
                    },
                    "units": {
                        "temperature": temp_unit,
                        "wind_speed": wind_unit,
                        "pressure": "hPa",
                        "humidity": "%"
                    },
                    "timestamp": datetime.utcnow().isoformat(),
                    "data_source": "OpenWeatherMap"
                }
            
        except Exception as e:
            logger.error(f"❌ Error formatting historical weather data: {e}")
            return {
                "success": False,
                "error": f"Error formatting historical weather data: {str(e)}",
                "location": location,
                "date_str": date_str
            }
    
    async def _get_coordinates(self, location: str, api_key: str, user_id: str = None) -> Dict[str, Any]:
        """Get latitude and longitude for a location via geocoding
        
        Note: Location resolution (vague location → user ZIP code) is handled
        by WeatherLocationRequest.resolve_location() before this method is called.
        This method only performs geocoding of the resolved location.
        """
        try:
            # Location should already be resolved by WeatherLocationRequest
            # This method only performs geocoding

            # Check if location is a ZIP code (US format)
            if str(location).isdigit() and len(str(location)) == 5:
                # Use ZIP code geocoding
                url = f"{self.geocoding_url}/zip"
                params = {
                    "zip": f"{location},US",
                    "appid": api_key
                }
            else:
                # ROOSEVELT'S ENHANCED GEOCODING: Try multiple location formats
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
        """ROOSEVELT'S ENHANCED GEOCODING: Generate multiple location format variants"""
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
        """ROOSEVELT'S GEOCODING REQUEST: Try a single geocoding request"""
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
    
    def _calculate_moon_phase(self, date: datetime = None) -> Dict[str, Any]:
        """Calculate moon phase for a given date"""
        if date is None:
            date = datetime.utcnow()
        
        # Known new moon date (January 6, 2000 18:14 UTC)
        known_new_moon = datetime(2000, 1, 6, 18, 14, 0)
        
        # Calculate days since known new moon
        days_since_new_moon = (date - known_new_moon).total_seconds() / 86400.0
        
        # Lunar cycle is approximately 29.53 days
        lunar_cycle = 29.53058867
        phase = (days_since_new_moon % lunar_cycle) / lunar_cycle
        
        # Determine phase name and icon
        if phase < 0.0625:
            phase_name = "New Moon"
            phase_icon = "🌑"
            phase_value = 0
        elif phase < 0.1875:
            phase_name = "Waxing Crescent"
            phase_icon = "🌒"
            phase_value = 1
        elif phase < 0.3125:
            phase_name = "First Quarter"
            phase_icon = "🌓"
            phase_value = 2
        elif phase < 0.4375:
            phase_name = "Waxing Gibbous"
            phase_icon = "🌔"
            phase_value = 3
        elif phase < 0.5625:
            phase_name = "Full Moon"
            phase_icon = "🌕"
            phase_value = 4
        elif phase < 0.6875:
            phase_name = "Waning Gibbous"
            phase_icon = "🌖"
            phase_value = 5
        elif phase < 0.8125:
            phase_name = "Last Quarter"
            phase_icon = "🌗"
            phase_value = 6
        else:
            phase_name = "Waning Crescent"
            phase_icon = "🌘"
            phase_value = 7
        
        return {
            "phase_name": phase_name,
            "phase_icon": phase_icon,
            "phase_value": phase_value,
            "illumination": round((1 - abs(phase - 0.5) * 2) * 100, 1)
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
            
            # Calculate moon phase
            moon_phase = self._calculate_moon_phase()
            
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
                "moon_phase": moon_phase,
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


# Global instance for use by tool registry
_weather_tools_instance = None


async def _get_weather_tools():
    """Get global weather tools instance"""
    global _weather_tools_instance
    if _weather_tools_instance is None:
        _weather_tools_instance = WeatherTools()
    return _weather_tools_instance


async def weather_conditions(location: Optional[str] = None, units: str = "imperial", user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    LangGraph tool function: Get current weather conditions
    
    Args:
        location: Optional location (ZIP code, city name, or 'city,country' format).
                  If None/empty/vague and user_id provided, uses user's ZIP from profile.
        units: Temperature units (default: "imperial")
        user_id: User ID for automatic location fallback to profile ZIP code
    
    Returns:
        Dict with weather data or error message
    """
    # Resolve location using Pydantic model
    request = WeatherConditionsRequest(location=location, user_id=user_id, units=units)
    resolved_location, error = await request.resolve_location()
    
    if error:
        return {
            "success": False,
            "error": error,
            "location": location
        }
    
    tools_instance = await _get_weather_tools()
    return await tools_instance.get_weather_conditions(resolved_location, units, user_id)


async def weather_forecast(location: Optional[str] = None, days: int = 3, units: str = "imperial", user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    LangGraph tool function: Get weather forecast
    
    Args:
        location: Optional location (ZIP code, city name, or 'city,country' format).
                  If None/empty/vague and user_id provided, uses user's ZIP from profile.
        days: Number of days to forecast (1-5, default: 3)
        units: Temperature units (default: "imperial")
        user_id: User ID for automatic location fallback to profile ZIP code
    
    Returns:
        Dict with forecast data or error message
    """
    # Resolve location using Pydantic model
    request = WeatherForecastRequest(location=location, user_id=user_id, days=days, units=units)
    resolved_location, error = await request.resolve_location()
    
    if error:
        return {
            "success": False,
            "error": error,
            "location": location,
            "days": days
        }
    
    tools_instance = await _get_weather_tools()
    return await tools_instance.get_weather_forecast(resolved_location, days, units, user_id)


async def weather_history(location: Optional[str] = None, date_str: str = "", units: str = "imperial", user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    LangGraph tool function: Get historical weather data
    
    Args:
        location: Optional location (ZIP code, city name, or 'city,country' format).
                  If None/empty/vague and user_id provided, uses user's ZIP from profile.
        date_str: Date string - 'YYYY-MM-DD' (specific day), 'YYYY-MM' (monthly average),
                  or 'YYYY-MM to YYYY-MM' (date range, max 24 months)
        units: Temperature units (default: "imperial")
        user_id: User ID for automatic location fallback to profile ZIP code
    
    Returns:
        Dict with historical weather data or error message
    """
    # Resolve location using Pydantic model
    request = WeatherHistoryRequest(location=location, user_id=user_id, date_str=date_str, units=units)
    resolved_location, error = await request.resolve_location()
    
    if error:
        return {
            "success": False,
            "error": error,
            "location": location,
            "date_str": date_str
        }
    
    tools_instance = await _get_weather_tools()
    return await tools_instance.get_weather_history(resolved_location, date_str, units, user_id)
