"""
Navigation Tools - Locations and routes via backend gRPC
"""

import json
import logging
from typing import Any, Dict, List, Optional

from orchestrator.backend_tool_client import get_backend_tool_client

logger = logging.getLogger(__name__)


def _format_dict_result(data: Dict[str, Any], default_msg: str = "Done.") -> str:
    """Format a dict result into a readable string for the LLM."""
    if not data:
        return default_msg
    if not data.get("success", True):
        return data.get("error", "Operation failed.")
    parts = []
    if "location_id" in data:
        parts.append(f"Location ID: {data['location_id']}")
    if "locations" in data:
        locs = data["locations"]
        total = data.get("total", len(locs))
        parts.append(f"Locations ({total}):")
        for i, loc in enumerate(locs[:20], 1):
            name = loc.get("name", "?")
            addr = loc.get("address", "")
            lid = loc.get("location_id", "")
            parts.append(f"  {i}. {name} | {addr} | ID: {lid}")
    if "routes" in data:
        routes = data["routes"]
        total = data.get("total", len(routes))
        parts.append(f"Saved routes ({total}):")
        for i, r in enumerate(routes[:20], 1):
            name = r.get("name", "?")
            rid = r.get("route_id", "")
            dist = r.get("distance_meters")
            dur = r.get("duration_seconds")
            parts.append(f"  {i}. {name} | ID: {rid} | {dist}m, {dur}s")
    if "geometry" in data and "waypoints" in data:
        dist = data.get("distance") or data.get("distance_meters")
        dur = data.get("duration") or data.get("duration_seconds")
        parts.append(f"Route: distance={dist}, duration={dur}")
        legs = data.get("legs") or data.get("steps") or []
        for i, leg in enumerate(legs[:15], 1):
            if isinstance(leg, dict):
                parts.append(f"  Step {i}: {leg.get('instruction', leg)}")
            else:
                parts.append(f"  Step {i}: {leg}")
    if "route_id" in data:
        parts.append(f"Route saved. Route ID: {data['route_id']}")
    if "message" in data:
        parts.append(data["message"])
    return "\n".join(parts) if parts else default_msg


async def create_location_tool(
    user_id: str = "system",
    name: str = "",
    address: str = "",
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    notes: Optional[str] = None,
) -> str:
    """
    Create a saved location (geocodes address if needed).

    Args:
        user_id: User ID (injected by engine if omitted).
        name: Short name for the location.
        address: Full address (required for geocoding).
        latitude: Optional latitude (skip geocoding if provided).
        longitude: Optional longitude.
        notes: Optional notes.

    Returns:
        Formatted result with location_id.
    """
    try:
        if not name or not address:
            return "Error: name and address are required."
        logger.info("create_location: name=%s address=%s", name, address[:80])
        client = await get_backend_tool_client()
        result = await client.create_location(
            user_id=user_id,
            name=name,
            address=address,
            latitude=latitude,
            longitude=longitude,
            notes=notes,
            is_global=False,
            metadata=None,
            user_role="user",
        )
        return _format_dict_result(result, "Location created.")
    except Exception as e:
        logger.error("create_location_tool error: %s", e)
        return f"Error: {str(e)}"


async def list_locations_tool(user_id: str = "system") -> str:
    """
    List all saved locations for the user.

    Args:
        user_id: User ID (injected by engine if omitted).

    Returns:
        Formatted list of locations with IDs.
    """
    try:
        logger.info("list_locations")
        client = await get_backend_tool_client()
        result = await client.list_locations(user_id=user_id, user_role="user")
        return _format_dict_result(result, "No locations saved.")
    except Exception as e:
        logger.error("list_locations_tool error: %s", e)
        return f"Error: {str(e)}"


async def delete_location_tool(
    user_id: str = "system",
    location_id: str = "",
) -> str:
    """
    Delete a saved location by ID.

    Args:
        user_id: User ID (injected by engine if omitted).
        location_id: ID of the location to delete (from list_locations).

    Returns:
        Success or error message.
    """
    try:
        if not location_id:
            return "Error: location_id is required."
        logger.info("delete_location: location_id=%s", location_id[:50])
        client = await get_backend_tool_client()
        result = await client.delete_location(
            user_id=user_id,
            location_id=location_id,
            user_role="user",
        )
        return _format_dict_result(result, "Location deleted.")
    except Exception as e:
        logger.error("delete_location_tool error: %s", e)
        return f"Error: {str(e)}"


async def compute_route_tool(
    user_id: str = "system",
    from_location_id: Optional[str] = None,
    to_location_id: Optional[str] = None,
    coordinates: Optional[str] = None,
    profile: str = "driving",
) -> str:
    """
    Compute a route between two points. Use location IDs from list_locations, or a coordinates string.

    Args:
        user_id: User ID (injected by engine if omitted).
        from_location_id: Start location ID (optional if coordinates given).
        to_location_id: End location ID (optional if coordinates given).
        coordinates: Alternative: JSON array of [lat,lon] pairs, e.g. "[[-122.4,37.8],[-122.5,37.9]]".
        profile: driving, walking, cycling, etc.

    Returns:
        Formatted route with steps and distance/duration.
    """
    try:
        logger.info("compute_route: from=%s to=%s profile=%s", from_location_id, to_location_id, profile)
        client = await get_backend_tool_client()
        result = await client.compute_route(
            user_id=user_id,
            from_location_id=from_location_id,
            to_location_id=to_location_id,
            coordinates=coordinates,
            profile=profile,
            user_role="user",
        )
        return _format_dict_result(result, "No route computed.")
    except Exception as e:
        logger.error("compute_route_tool error: %s", e)
        return f"Error: {str(e)}"


async def save_route_tool(
    user_id: str = "system",
    name: str = "",
    waypoints: Optional[str] = None,
    geometry: Optional[str] = None,
    steps: Optional[str] = None,
    distance_meters: float = 0,
    duration_seconds: float = 0,
    profile: str = "driving",
) -> str:
    """
    Save a previously computed route. Call compute_route first, then pass its waypoints/geometry/steps here.

    Args:
        user_id: User ID (injected by engine if omitted).
        name: Name for the saved route.
        waypoints: JSON array of waypoints from compute_route result.
        geometry: JSON geometry from compute_route result.
        steps: JSON array of step instructions from compute_route legs.
        distance_meters: Distance from compute_route.
        duration_seconds: Duration from compute_route.
        profile: driving, walking, etc.

    Returns:
        Success message with route_id.
    """
    try:
        if not name:
            return "Error: name is required."
        waypoints_list = json.loads(waypoints) if isinstance(waypoints, str) and waypoints else (waypoints or [])
        geometry_dict = json.loads(geometry) if isinstance(geometry, str) and geometry else (geometry or {})
        steps_list = json.loads(steps) if isinstance(steps, str) and steps else (steps or [])
        if not isinstance(waypoints_list, list):
            waypoints_list = []
        if not isinstance(geometry_dict, dict):
            geometry_dict = {}
        if not isinstance(steps_list, list):
            steps_list = []
        logger.info("save_route: name=%s", name)
        client = await get_backend_tool_client()
        result = await client.save_route(
            user_id=user_id,
            name=name,
            waypoints=waypoints_list,
            geometry=geometry_dict,
            steps=steps_list,
            distance_meters=float(distance_meters),
            duration_seconds=float(duration_seconds),
            profile=profile,
            user_role="user",
        )
        return _format_dict_result(result, "Route saved.")
    except Exception as e:
        logger.error("save_route_tool error: %s", e)
        return f"Error: {str(e)}"


async def list_saved_routes_tool(user_id: str = "system") -> str:
    """
    List saved routes for the user.

    Args:
        user_id: User ID (injected by engine if omitted).

    Returns:
        Formatted list of saved routes with IDs.
    """
    try:
        logger.info("list_saved_routes")
        client = await get_backend_tool_client()
        result = await client.list_saved_routes(user_id=user_id, user_role="user")
        return _format_dict_result(result, "No saved routes.")
    except Exception as e:
        logger.error("list_saved_routes_tool error: %s", e)
        return f"Error: {str(e)}"
