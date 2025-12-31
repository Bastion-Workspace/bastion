"""
Weather Tools Module - gRPC Wrapper
Lightweight wrapper that calls the Tools Service via gRPC for weather operations
"""

import logging
from typing import Dict, Any, Optional

from clients.tool_service_client import get_tool_service_client

logger = logging.getLogger(__name__)


async def weather_conditions(location: Optional[str] = None, units: str = "imperial", user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    LangGraph tool function: Get current weather conditions via gRPC
    
    Args:
        location: Optional location (ZIP code, city name, or 'city,country' format).
                  If None/empty/vague and user_id provided, uses user's ZIP from profile.
        units: Temperature units (default: "imperial")
        user_id: User ID for automatic location fallback to profile ZIP code
    
    Returns:
        Dict with weather data or error message
    """
    try:
        tool_client = await get_tool_service_client()
        
        # Call Tools Service via gRPC
        weather_data = await tool_client.get_weather_data(
            location=location or "",
            user_id=user_id or "system",
            data_types=["current"]
        )
        
        if not weather_data:
            return {
                "success": False,
                "error": "Weather service unavailable",
                "location": location
            }
        
        # Extract data from gRPC response and format to match expected structure
        metadata = weather_data.get("metadata", {})
        
        return {
            "success": True,
            "location": {
                "name": weather_data.get("location", location or "Unknown"),
                "query": location or ""
            },
            "current": {
                "temperature": float(metadata.get("temperature", 0)),
                "conditions": metadata.get("conditions", ""),
                "humidity": float(metadata.get("humidity", 0)),
                "wind_speed": float(metadata.get("wind_speed", 0)),
                "feels_like": float(metadata.get("feels_like", 0))
            },
            "moon_phase": {
                "phase_name": metadata.get("moon_phase_name", ""),
                "phase_icon": metadata.get("moon_phase_icon", ""),
                "phase_value": int(metadata.get("moon_phase_value", 0))
            },
            "units": {
                "temperature": "°F" if units == "imperial" else "°C" if units == "metric" else "K",
                "wind_speed": "mph" if units == "imperial" else "m/s"
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Weather conditions request failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "location": location
        }


async def weather_forecast(location: Optional[str] = None, days: int = 3, units: str = "imperial", user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    LangGraph tool function: Get weather forecast via gRPC
    
    Args:
        location: Optional location (ZIP code, city name, or 'city,country' format).
                  If None/empty/vague and user_id provided, uses user's ZIP from profile.
        days: Number of days to forecast (1-5, default: 3)
        units: Temperature units (default: "imperial")
        user_id: User ID for automatic location fallback to profile ZIP code
    
    Returns:
        Dict with forecast data or error message
    """
    try:
        tool_client = await get_tool_service_client()
        
        # Call Tools Service via gRPC
        weather_data = await tool_client.get_weather_data(
            location=location or "",
            user_id=user_id or "system",
            data_types=["forecast"]
        )
        
        if not weather_data:
            return {
                "success": False,
                "error": "Weather service unavailable",
                "location": location,
                "days": days
            }
        
        # Extract forecast data from gRPC response
        metadata = weather_data.get("metadata", {})
        forecast_data_json = metadata.get("forecast_data", "[]")
        
        import json
        forecast_days = json.loads(forecast_data_json) if forecast_data_json else []
        
        # Limit to requested days
        forecast_days = forecast_days[:days]
        
        return {
            "success": True,
            "location": {
                "name": weather_data.get("location", location or "Unknown"),
                "query": location or ""
            },
            "forecast": forecast_days,
            "units": {
                "temperature": "°F" if units == "imperial" else "°C" if units == "metric" else "K",
                "wind_speed": "mph" if units == "imperial" else "m/s"
            },
            "days_requested": days
        }
        
    except Exception as e:
        logger.error(f"❌ Weather forecast request failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "location": location,
            "days": days
        }


async def weather_history(location: Optional[str] = None, date_str: str = "", units: str = "imperial", user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    LangGraph tool function: Get historical weather data via gRPC
    
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
    try:
        tool_client = await get_tool_service_client()
        
        # Call Tools Service via gRPC
        weather_data = await tool_client.get_weather_data(
            location=location or "",
            user_id=user_id or "system",
            data_types=["history"],
            date_str=date_str
        )
        
        if not weather_data:
            return {
                "success": False,
                "error": "Weather service unavailable",
                "location": location,
                "date_str": date_str
            }
        
        # Extract historical data from gRPC response
        metadata = weather_data.get("metadata", {})
        period_type = metadata.get("period_type", "daily")
        
        # Reconstruct expected format for historical data
        if period_type == "monthly_average":
            formatted = {
                "success": True,
                "location": {
                    "name": weather_data.get("location", location or "Unknown"),
                    "query": location or ""
                },
                "period": {
                    "type": "monthly_average",
                    "date_str": date_str
                },
                "historical": {
                    "average_temperature": float(metadata.get("average_temperature", 0)),
                    "min_temperature": float(metadata.get("min_temperature", 0)),
                    "max_temperature": float(metadata.get("max_temperature", 0)),
                    "average_humidity": float(metadata.get("humidity", 0)),
                    "average_wind_speed": float(metadata.get("wind_speed", 0)),
                    "most_common_conditions": metadata.get("conditions", ""),
                    "sample_days": int(metadata.get("sample_days", 0))
                },
                "units": {
                    "temperature": "°F" if units == "imperial" else "°C" if units == "metric" else "K",
                    "wind_speed": "mph" if units == "imperial" else "m/s"
                }
            }
        elif period_type == "date_range":
            formatted = {
                "success": True,
                "location": {
                    "name": weather_data.get("location", location or "Unknown"),
                    "query": location or ""
                },
                "period": {
                    "type": "date_range",
                    "date_str": date_str,
                    "start_date": metadata.get("start_date", ""),
                    "end_date": metadata.get("end_date", ""),
                    "months_in_range": int(metadata.get("months_in_range", 0)),
                    "months_retrieved": int(metadata.get("months_retrieved", 0))
                },
                "historical": {
                    "average_temperature": float(metadata.get("average_temperature", 0)),
                    "min_temperature": float(metadata.get("min_temperature", 0)),
                    "max_temperature": float(metadata.get("max_temperature", 0)),
                    "average_humidity": float(metadata.get("humidity", 0)),
                    "average_wind_speed": float(metadata.get("wind_speed", 0)),
                    "most_common_conditions": metadata.get("conditions", "")
                },
                "units": {
                    "temperature": "°F" if units == "imperial" else "°C" if units == "metric" else "K",
                    "wind_speed": "mph" if units == "imperial" else "m/s"
                }
            }
        else:
            # Daily historical data
            formatted = {
                "success": True,
                "location": {
                    "name": weather_data.get("location", location or "Unknown"),
                    "query": location or ""
                },
                "period": {
                    "type": "daily",
                    "date_str": date_str
                },
                "historical": {
                    "temperature": float(metadata.get("temperature", 0)),
                    "conditions": metadata.get("conditions", ""),
                    "humidity": float(metadata.get("humidity", 0)),
                    "wind_speed": float(metadata.get("wind_speed", 0))
                },
                "units": {
                    "temperature": "°F" if units == "imperial" else "°C" if units == "metric" else "K",
                    "wind_speed": "mph" if units == "imperial" else "m/s"
                }
            }
        
        return formatted
        
    except Exception as e:
        logger.error(f"❌ Historical weather request failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "location": location,
            "date_str": date_str
        }
