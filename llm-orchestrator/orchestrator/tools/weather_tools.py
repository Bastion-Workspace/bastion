"""
Weather Tools - Weather data via backend gRPC
"""

import json
import logging
from typing import List, Optional

from orchestrator.backend_tool_client import get_backend_tool_client

logger = logging.getLogger(__name__)


async def get_weather_tool(
    location: str,
    user_id: str = "system",
    data_types: Optional[str] = None,
    date_str: Optional[str] = None,
) -> str:
    """
    Get weather data for a location.

    Args:
        location: Location name (e.g., city, "New York", "London").
        user_id: User ID for access (injected by engine if omitted).
        data_types: Comma-separated types: "current", "forecast", "history". Default "current".
        date_str: For history only: "YYYY-MM-DD" for a specific day, "YYYY-MM" for monthly average,
                 or "YYYY-MM to YYYY-MM" for date ranges (max 24 months, e.g., "2024-01 to 2024-12").

    Returns:
        Formatted weather summary.
    """
    try:
        logger.info("Weather request: %s", location[:80])
        client = await get_backend_tool_client()
        types_list = [s.strip() for s in (data_types or "current").split(",") if s.strip()]
        if not types_list:
            types_list = ["current"]
        result = await client.get_weather(
            location=location,
            user_id=user_id,
            data_types=types_list,
            date_str=date_str,
        )
        if not result:
            return "No weather data available for that location."
        return _format_weather_result(result, location)
    except Exception as e:
        logger.error("Weather tool error: %s", e)
        return f"Error fetching weather: {str(e)}"


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
