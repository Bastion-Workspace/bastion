"""
Weather Tools - Weather data via backend gRPC
"""

import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)


class GetWeatherInputs(BaseModel):
    location: str = Field(default="", description="Location (ZIP, city, or empty to use user's profile ZIP)")


class GetWeatherParams(BaseModel):
    data_types: str = Field(
        default="current",
        description="What to fetch: 'current' (conditions now), 'forecast' (multi-day), or 'current,forecast' (both)",
    )


class GetWeatherOutputs(BaseModel):
    formatted: str = Field(description="Human-readable weather summary")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Raw weather data")


async def get_weather_tool(
    location: str = "",
    user_id: str = "system",
    data_types: Optional[str] = None,
    date_str: Optional[str] = None,
) -> Dict[str, Any]:
    """Get weather data for a location. Returns dict with formatted and data.
    Use data_types='current', 'forecast', or 'current,forecast' to get one or both.
    If location is empty, backend uses user's ZIP from profile (same as status bar)."""
    try:
        location = (location or "").strip()
        logger.info("Weather request: %s", (location or "(user profile)")[:80])
        client = await get_backend_tool_client()
        types_list = [s.strip().lower() for s in (data_types or "current").split(",") if s.strip()]
        if not types_list:
            types_list = ["current"]
        wants_current = "current" in types_list
        wants_forecast = "forecast" in types_list
        if wants_current and wants_forecast:
            result_current = await client.get_weather(
                location=location,
                user_id=user_id,
                data_types=["current"],
                date_str=date_str,
            )
            result_forecast = await client.get_weather(
                location=location,
                user_id=user_id,
                data_types=["forecast"],
                date_str=date_str,
            )
            formatted_parts = []
            merged_data: Dict[str, Any] = {"location": location, "metadata": {}}
            if result_current and isinstance(result_current, dict):
                formatted_parts.append(_format_weather_result(result_current, location))
                merged_data["current_conditions"] = result_current.get("current_conditions")
                merged_data["metadata"].update(result_current.get("metadata") or {})
            if result_forecast and isinstance(result_forecast, dict):
                formatted_parts.append(_format_weather_result(result_forecast, location))
                merged_data["forecast"] = result_forecast.get("forecast")
                merged_data["metadata"].update(result_forecast.get("metadata") or {})
            formatted = "\n\n".join(formatted_parts) if formatted_parts else "No weather data available for that location."
            return {"formatted": formatted, "data": merged_data}
        result = await client.get_weather(
            location=location,
            user_id=user_id,
            data_types=types_list,
            date_str=date_str,
        )
        if not result:
            return {"formatted": "No weather data available for that location.", "data": None}
        if isinstance(result, dict):
            return {"formatted": _format_weather_result(result, location), "data": result}
        return {"formatted": str(result), "data": None}
    except Exception as e:
        logger.error("Weather tool error: %s", e)
        err = str(e)
        return {"formatted": f"Error fetching weather: {err}", "data": None}


register_action(
    name="get_weather",
    category="weather",
    description="Get weather data for a location (current conditions, forecast, or both)",
    inputs_model=GetWeatherInputs,
    params_model=GetWeatherParams,
    outputs_model=GetWeatherOutputs,
    tool_function=get_weather_tool,
)


def _format_weather_result(data: dict, location: str) -> str:
    """Format weather dict into a readable string for the LLM."""
    parts = [f"Weather for {data.get('location', location)}:"]
    meta = data.get("metadata") or {}
    if isinstance(meta, dict):
        for k, v in meta.items():
            if v is not None and str(v).strip():
                parts.append(f"  {k}: {v}")
    current = data.get("current_conditions")
    if current:
        if isinstance(current, str):
            parts.append(f"  Current: {current}")
        elif isinstance(current, dict):
            for k, v in current.items():
                if v is not None:
                    parts.append(f"  {k}: {v}")
    forecast = data.get("forecast")
    if forecast and isinstance(forecast, (list, tuple)):
        parts.append("  Forecast:")
        for i, item in enumerate(forecast[:7], 1):
            if isinstance(item, dict):
                parts.append(f"    {i}. {item}")
            else:
                parts.append(f"    {i}. {item}")
    alerts = data.get("alerts")
    if alerts and isinstance(alerts, (list, tuple)):
        parts.append("  Alerts:")
        for a in alerts:
            parts.append(f"    - {a}")
    if len(parts) == 1 and meta:
        parts.append("  (See metadata above)")
    return "\n".join(parts)
