"""
OSRM routing service.
Calls self-hosted OSRM HTTP API for road-based route computation.
"""

import logging
from typing import Any

import aiohttp

from config import settings

logger = logging.getLogger(__name__)


class OSRMError(Exception):
    """OSRM API returned an error code."""

    def __init__(self, code: str, message: str = ""):
        self.code = code
        self.message = message
        super().__init__(f"OSRM {code}: {message}")


async def get_route(
    coordinates: list[tuple[float, float]],
    profile: str = "driving",
) -> dict[str, Any]:
    """
    Get road route between coordinates from OSRM.

    Args:
        coordinates: List of (latitude, longitude) pairs.
        profile: OSRM profile (driving, car, bike, foot).

    Returns:
        Dict with geometry (GeoJSON LineString), legs (with steps),
        distance (meters), duration (seconds), waypoints (snapped coords).

    Raises:
        OSRMError: When OSRM returns NoRoute, NoSegment, or other error.
    """
    if len(coordinates) < 2:
        raise ValueError("At least two coordinates required")

    base = (settings.OSRM_BASE_URL or "").rstrip("/")
    if not base:
        raise OSRMError("InvalidValue", "OSRM_BASE_URL not configured")

    coords_str = ";".join(f"{lon},{lat}" for lat, lon in coordinates)
    url = f"{base}/route/v1/{profile}/{coords_str}"
    params = {"steps": "true", "geometries": "geojson", "overview": "full"}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.warning("OSRM request failed: status=%s body=%s", resp.status, text[:200])
                    raise OSRMError("InvalidUrl", f"HTTP {resp.status}: {text[:200]}")

                data = await resp.json()
    except aiohttp.ClientError as e:
        logger.warning("OSRM connection error: %s", e)
        raise OSRMError("InvalidUrl", str(e)) from e

    code = data.get("code", "")
    if code != "Ok":
        msg = data.get("message", "")
        logger.info("OSRM error: code=%s message=%s", code, msg)
        raise OSRMError(code, msg)

    routes = data.get("routes") or []
    if not routes:
        raise OSRMError("NoRoute", "No route found")

    route = routes[0]
    legs = route.get("legs") or []
    geometry = route.get("geometry") or {}
    waypoints = data.get("waypoints") or []

    return {
        "geometry": geometry,
        "legs": legs,
        "distance": route.get("distance", 0),
        "duration": route.get("duration", 0),
        "waypoints": [
            {"location": wp.get("location"), "name": wp.get("name")}
            for wp in waypoints
        ],
    }
