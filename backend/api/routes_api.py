"""
Routes API: compute road routes via OSRM and CRUD for saved routes.
"""

import json
import logging
from fastapi import APIRouter, Depends, HTTPException, Query

from models.api_models import AuthenticatedUserResponse
from models.route_models import (
    RouteResponse,
    SaveRouteRequest,
    SavedRouteListResponse,
    SavedRouteResponse,
)
from services.database_manager.database_helpers import execute, fetch_all, fetch_one
from services.osrm_service import OSRMError, get_route

from api.location_api import require_maps_access

logger = logging.getLogger(__name__)


def _normalize_saved_route_row(row: dict) -> dict:
    """Convert DB row to SavedRouteResponse-friendly dict (UUID -> str, JSON str -> parsed)."""
    out = dict(row)
    if "route_id" in out and out["route_id"] is not None:
        out["route_id"] = str(out["route_id"])
    if "user_id" in out and out["user_id"] is not None:
        out["user_id"] = str(out["user_id"])
    for key in ("waypoints", "geometry", "steps"):
        val = out.get(key)
        if isinstance(val, str):
            try:
                out[key] = json.loads(val)
            except json.JSONDecodeError:
                out[key] = [] if key != "geometry" else {}
        elif val is None:
            out[key] = [] if key != "geometry" else {}
    return out

router = APIRouter(tags=["routes"])


@router.get("/api/routes/route", response_model=RouteResponse)
async def compute_route(
    current_user: AuthenticatedUserResponse = Depends(require_maps_access),
    coordinates: str | None = Query(None, description="lat1,lng1;lat2,lng2;..."),
    from_location_id: str | None = Query(None, description="Start POI location_id"),
    to_location_id: str | None = Query(None, description="End POI location_id"),
    profile: str = Query("driving", description="OSRM profile: driving, car, bike, foot"),
):
    """Compute road route between two or more points. Use either coordinates or from/to location IDs."""
    rls_context = {"user_id": current_user.user_id, "user_role": current_user.role}

    if coordinates:
        try:
            pairs = []
            for part in coordinates.split(";"):
                part = part.strip()
                if not part:
                    continue
                lat_str, lng_str = part.split(",", 1)
                pairs.append((float(lat_str.strip()), float(lng_str.strip())))
            if len(pairs) < 2:
                raise HTTPException(status_code=400, detail="At least two coordinates required")
        except (ValueError, AttributeError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid coordinates: {e}") from e
    elif from_location_id and to_location_id:
        query = """
            SELECT location_id, latitude, longitude
            FROM user_locations
            WHERE location_id = ANY($1::uuid[])
            ORDER BY array_position($1::uuid[], location_id)
        """
        ids = [from_location_id, to_location_id]
        rows = await fetch_all(query, ids, rls_context=rls_context)
        if len(rows) != 2:
            raise HTTPException(status_code=404, detail="One or both locations not found or not accessible")
        pairs = [(float(r["latitude"]), float(r["longitude"])) for r in rows]
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either coordinates=lat1,lng1;lat2,lng2 or from_location_id and to_location_id",
        )

    try:
        result = await get_route(pairs, profile=profile)
        return RouteResponse(**result)
    except OSRMError as e:
        if e.code in ("NoRoute", "NoSegment"):
            raise HTTPException(status_code=404, detail=e.message or "No road route found") from e
        raise HTTPException(status_code=503, detail=e.message or "Routing service error") from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/api/routes", response_model=SavedRouteResponse)
async def save_route(
    body: SaveRouteRequest,
    current_user: AuthenticatedUserResponse = Depends(require_maps_access),
):
    """Save a computed route for later."""
    rls_context = {"user_id": current_user.user_id, "user_role": current_user.role}

    geometry_json = json.dumps(body.geometry)
    waypoints_json = json.dumps(body.waypoints)
    steps_json = json.dumps(body.steps)

    insert_sql = """
        INSERT INTO saved_routes (user_id, name, waypoints, geometry, steps, distance_meters, duration_seconds, profile)
        VALUES ($1, $2, $3::jsonb, $4::jsonb, $5::jsonb, $6, $7, $8)
        RETURNING route_id, user_id, name, waypoints, geometry, steps, distance_meters, duration_seconds, profile, created_at, updated_at
    """
    row = await fetch_one(
        insert_sql,
        current_user.user_id,
        body.name,
        waypoints_json,
        geometry_json,
        steps_json,
        body.distance_meters,
        body.duration_seconds,
        body.profile,
        rls_context=rls_context,
    )
    if not row:
        raise HTTPException(status_code=500, detail="Failed to save route")
    return SavedRouteResponse(**_normalize_saved_route_row(row))


@router.get("/api/routes", response_model=SavedRouteListResponse)
async def list_saved_routes(
    current_user: AuthenticatedUserResponse = Depends(require_maps_access),
):
    """List saved routes for the current user."""
    rls_context = {"user_id": current_user.user_id, "user_role": current_user.role}
    query = """
        SELECT route_id, user_id, name, waypoints, geometry, steps, distance_meters, duration_seconds, profile, created_at, updated_at
        FROM saved_routes
        ORDER BY created_at DESC
    """
    rows = await fetch_all(query, rls_context=rls_context)
    return SavedRouteListResponse(
        routes=[SavedRouteResponse(**_normalize_saved_route_row(r)) for r in rows],
        total=len(rows),
    )


@router.get("/api/routes/{route_id}", response_model=SavedRouteResponse)
async def get_saved_route(
    route_id: str,
    current_user: AuthenticatedUserResponse = Depends(require_maps_access),
):
    """Get a single saved route by ID (RLS ensures ownership)."""
    rls_context = {"user_id": current_user.user_id, "user_role": current_user.role}
    query = """
        SELECT route_id, user_id, name, waypoints, geometry, steps, distance_meters, duration_seconds, profile, created_at, updated_at
        FROM saved_routes
        WHERE route_id = $1::uuid
    """
    row = await fetch_one(query, route_id, rls_context=rls_context)
    if not row:
        raise HTTPException(status_code=404, detail="Route not found")
    return SavedRouteResponse(**_normalize_saved_route_row(row))


@router.delete("/api/routes/{route_id}")
async def delete_saved_route(
    route_id: str,
    current_user: AuthenticatedUserResponse = Depends(require_maps_access),
):
    """Delete a saved route (RLS ensures ownership)."""
    rls_context = {"user_id": current_user.user_id, "user_role": current_user.role}
    query = "DELETE FROM saved_routes WHERE route_id = $1::uuid"
    result = await execute(query, route_id, rls_context=rls_context)
    if result and "DELETE 0" in str(result):
        raise HTTPException(status_code=404, detail="Route not found")
    return {"message": "Route deleted successfully"}
