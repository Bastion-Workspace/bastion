"""
Route Planning Subgraph

Reusable subgraph for computing routes and formatting turn-by-turn directions.
Used by Navigation Agent for route computation and visualization.

State: from_location (name or ID), to_location (name or ID), route_profile, etc.
Outputs: route_data, turn_by_turn (markdown), map_snippet (URL), saved_route_id (if saved).
"""

import logging
import urllib.parse
from typing import Dict, Any, List

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)

RoutePlanningState = Dict[str, Any]


def _format_turn_by_turn(legs: List[dict], distance: float, duration: float, from_name: str = "", to_name: str = "") -> str:
    """Format OSRM legs as markdown turn-by-turn list."""
    lines = []
    miles = distance / 1609.34 if distance else 0
    mins = duration / 60.0 if duration else 0
    header = f"## Route: {from_name or 'Start'} → {to_name or 'End'} ({miles:.1f} mi, {mins:.0f} min)\n\n"
    lines.append(header)

    step_num = 1
    for leg in legs:
        steps = leg.get("steps", [])
        for step in steps:
            maneuver = step.get("maneuver", {})
            mod = maneuver.get("modifier", "")
            m_type = maneuver.get("type", "")
            name = (step.get("name") or "").strip() or "unnamed"
            dist = step.get("distance", 0) or 0
            dist_mi = dist / 1609.34
            instruction = maneuver.get("instruction", "")
            if instruction:
                lines.append(f"{step_num}. {instruction} ({dist_mi:.2f} mi)")
            else:
                lines.append(f"{step_num}. Continue on {name} ({dist_mi:.2f} mi)")
            step_num += 1

    if not lines or lines == [header]:
        lines.append("1. Follow the route.")
    return "\n".join(lines)


def _google_maps_route_url(waypoints: List[dict], from_name: str = "", to_name: str = "") -> str:
    """Build a Google Maps URL with origin/destination from waypoints."""
    if not waypoints or len(waypoints) < 2:
        return ""
    try:
        first = waypoints[0]
        last = waypoints[-1]
        origin = first.get("location") if isinstance(first, dict) else first
        dest = last.get("location") if isinstance(last, dict) else last
        if isinstance(origin, list) and len(origin) >= 2:
            lat1, lon1 = origin[1], origin[0]
            origin_str = f"{lat1},{lon1}"
        else:
            origin_str = str(origin)
        if isinstance(dest, list) and len(dest) >= 2:
            lat2, lon2 = dest[1], dest[0]
            dest_str = f"{lat2},{lon2}"
        else:
            dest_str = str(dest)
        params = {"api": "1", "origin": origin_str, "destination": dest_str}
        return "https://www.google.com/maps/dir/?" + urllib.parse.urlencode(params)
    except Exception:
        return ""


async def resolve_route_endpoints_node(state: RoutePlanningState) -> Dict[str, Any]:
    """Resolve from/to location names to location IDs if needed."""
    from_id = (state.get("from_location_id") or "").strip()
    to_id = (state.get("to_location_id") or "").strip()
    from_name = (state.get("from_location_name") or "").strip()
    to_name = (state.get("to_location_name") or "").strip()
    user_id = state.get("user_id", "system")
    user_role = state.get("user_role", "user")

    if from_id and to_id:
        return {
            "from_location_id": from_id,
            "to_location_id": to_id,
            "from_location_name": from_name,
            "to_location_name": to_name,
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }

    try:
        from orchestrator.backend_tool_client import get_backend_tool_client
        client = await get_backend_tool_client()
        list_result = await client.list_locations(user_id=user_id, user_role=user_role)
        if not list_result.get("success"):
            return {
                "route_resolve_error": list_result.get("error", "Failed to list locations"),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
            }
        locations = list_result.get("locations", [])
        name_lower = lambda n: (n or "").strip().lower()

        if not from_id and from_name:
            for loc in locations:
                if name_lower(loc.get("name")) == name_lower(from_name):
                    from_id = loc.get("location_id", "")
                    break
        if not to_id and to_name:
            for loc in locations:
                if name_lower(loc.get("name")) == name_lower(to_name):
                    to_id = loc.get("location_id", "")
                    break

        return {
            "from_location_id": from_id or state.get("from_location_id", ""),
            "to_location_id": to_id or state.get("to_location_id", ""),
            "from_location_name": from_name or state.get("from_location_name", ""),
            "to_location_name": to_name or state.get("to_location_name", ""),
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }
    except Exception as e:
        logger.error(f"resolve_route_endpoints_node failed: {e}")
        return {
            "route_resolve_error": str(e),
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }


async def compute_route_node(state: RoutePlanningState) -> Dict[str, Any]:
    """Compute route via gRPC (OSRM)."""
    if state.get("route_resolve_error"):
        return {
            "route_data": None,
            "route_error": state.get("route_resolve_error"),
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }

    from_id = state.get("from_location_id") or ""
    to_id = state.get("to_location_id") or ""
    coordinates = state.get("coordinates", "").strip()
    profile = state.get("route_profile", "driving")
    user_id = state.get("user_id", "system")
    user_role = state.get("user_role", "user")

    if not coordinates and not (from_id and to_id):
        return {
            "route_data": None,
            "route_error": "Provide from/to location IDs or a coordinates string.",
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }

    try:
        from orchestrator.backend_tool_client import get_backend_tool_client
        client = await get_backend_tool_client()
        result = await client.compute_route(
            user_id=user_id,
            from_location_id=from_id or None,
            to_location_id=to_id or None,
            coordinates=coordinates or None,
            profile=profile,
            user_role=user_role,
        )
        if not result.get("success"):
            return {
                "route_data": None,
                "route_error": result.get("error", "Route computation failed"),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
            }
        return {
            "route_data": result,
            "route_error": None,
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }
    except Exception as e:
        logger.error(f"compute_route_node failed: {e}")
        return {
            "route_data": None,
            "route_error": str(e),
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }


async def format_route_visualization_node(state: RoutePlanningState) -> Dict[str, Any]:
    """Format turn-by-turn and map URL from route_data."""
    route_data = state.get("route_data")
    route_error = state.get("route_error")
    from_name = state.get("from_location_name", "")
    to_name = state.get("to_location_name", "")

    if route_error or not route_data:
        turn_by_turn = f"Route could not be computed: {route_error or 'Unknown error'}."
        map_snippet = ""
    else:
        legs = route_data.get("legs", [])
        distance = route_data.get("distance", 0)
        duration = route_data.get("duration", 0)
        waypoints = route_data.get("waypoints", [])
        turn_by_turn = _format_turn_by_turn(legs, distance, duration, from_name, to_name)
        map_snippet = _google_maps_route_url(waypoints, from_name, to_name)

    return {
        "turn_by_turn": turn_by_turn,
        "map_snippet": map_snippet,
        "route_data": route_data,
        "metadata": state.get("metadata", {}),
        "user_id": state.get("user_id", "system"),
        "shared_memory": state.get("shared_memory", {}),
        "messages": state.get("messages", []),
        "query": state.get("query", ""),
    }


async def save_route_optional_node(state: RoutePlanningState) -> Dict[str, Any]:
    """Save the route if should_save_route and route_name are set."""
    route_data = state.get("route_data")
    route_name = (state.get("route_name") or "").strip()
    should_save = state.get("should_save_route", False)
    user_id = state.get("user_id", "system")
    user_role = state.get("user_role", "user")

    if not should_save or not route_name or not route_data:
        return {
            "saved_route_id": None,
            "turn_by_turn": state.get("turn_by_turn", ""),
            "map_snippet": state.get("map_snippet", ""),
            "route_data": state.get("route_data"),
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }

    try:
        from orchestrator.backend_tool_client import get_backend_tool_client
        client = await get_backend_tool_client()
        waypoints = route_data.get("waypoints", [])
        geometry = route_data.get("geometry", {})
        legs = route_data.get("legs", [])
        steps = []
        for leg in legs:
            steps.extend(leg.get("steps", []))
        distance_meters = route_data.get("distance", 0)
        duration_seconds = route_data.get("duration", 0)
        profile = state.get("route_profile", "driving")

        result = await client.save_route(
            user_id=user_id,
            name=route_name,
            waypoints=waypoints,
            geometry=geometry,
            steps=steps,
            distance_meters=distance_meters,
            duration_seconds=duration_seconds,
            profile=profile,
            user_role=user_role,
        )
        if result.get("success"):
            return {
                "saved_route_id": result.get("route_id"),
                "turn_by_turn": state.get("turn_by_turn", ""),
                "map_snippet": state.get("map_snippet", ""),
                "route_data": state.get("route_data"),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
            }
    except Exception as e:
        logger.error(f"save_route_optional_node failed: {e}")

    return {
        "saved_route_id": None,
        "turn_by_turn": state.get("turn_by_turn", ""),
        "map_snippet": state.get("map_snippet", ""),
        "route_data": state.get("route_data"),
        "metadata": state.get("metadata", {}),
        "user_id": state.get("user_id", "system"),
        "shared_memory": state.get("shared_memory", {}),
        "messages": state.get("messages", []),
        "query": state.get("query", ""),
    }


def build_route_planning_subgraph(checkpointer=None):
    """Build the route planning subgraph (resolve → compute → format → optional save)."""
    workflow = StateGraph(RoutePlanningState)

    workflow.add_node("resolve_route_endpoints", resolve_route_endpoints_node)
    workflow.add_node("compute_route", compute_route_node)
    workflow.add_node("format_route_visualization", format_route_visualization_node)
    workflow.add_node("save_route_optional", save_route_optional_node)

    workflow.set_entry_point("resolve_route_endpoints")
    workflow.add_edge("resolve_route_endpoints", "compute_route")
    workflow.add_edge("compute_route", "format_route_visualization")
    workflow.add_edge("format_route_visualization", "save_route_optional")
    workflow.add_edge("save_route_optional", END)

    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()
