"""
Routing service abstraction.
Dispatches to Valhalla or OSRM based on ROUTING_PROVIDER setting.
Returns a unified contract: geometry, legs, distance, duration, waypoints.
"""

import logging
from typing import Any

from config import settings

logger = logging.getLogger(__name__)


async def get_route(
    coordinates: list[tuple[float, float]],
    profile: str = "driving",
) -> dict[str, Any]:
    """
    Compute route between coordinates using the configured routing provider.
    Returns dict with geometry (GeoJSON LineString), legs, distance (meters), duration (seconds), waypoints.
    """
    provider = (settings.ROUTING_PROVIDER or "valhalla").strip().lower()
    if provider == "osrm":
        from services.osrm_service import get_route as osrm_get_route
        return await osrm_get_route(coordinates, profile=profile)
    from services.valhalla_adapter import get_route as valhalla_get_route
    return await valhalla_get_route(coordinates, profile=profile)


async def get_isochrone(
    origin: tuple[float, float],
    costing: str = "auto",
    contours: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Get isochrone polygons (Valhalla only). Returns GeoJSON FeatureCollection.
    No-op if provider is OSRM (returns empty FeatureCollection).
    """
    provider = (settings.ROUTING_PROVIDER or "valhalla").strip().lower()
    if provider != "valhalla":
        return {"type": "FeatureCollection", "features": []}
    from services.valhalla_adapter import get_isochrone as valhalla_isochrone
    return await valhalla_isochrone(origin, costing=costing, contours=contours)
