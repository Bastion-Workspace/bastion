"""
Pydantic models for route computation and saved routes.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class RouteWaypoint(BaseModel):
    """Single waypoint in a route (from POI or custom lat/lng)."""
    location_id: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    name: Optional[str] = None


class SaveRouteRequest(BaseModel):
    """Request body for saving a route."""
    name: str = Field(..., max_length=255, description="Route name")
    waypoints: list[dict[str, Any]] = Field(..., description="Array of waypoint objects")
    geometry: dict[str, Any] = Field(..., description="GeoJSON LineString")
    steps: list[dict[str, Any]] = Field(default_factory=list, description="Turn-by-turn steps from OSRM")
    distance_meters: float = Field(..., ge=0, description="Total distance in meters")
    duration_seconds: float = Field(..., ge=0, description="Total duration in seconds")
    profile: str = Field(default="driving", description="OSRM profile used")


class RouteResponse(BaseModel):
    """Response for computed route (GET /api/routes/route)."""
    geometry: dict[str, Any]
    legs: list[dict[str, Any]]
    distance: float
    duration: float
    waypoints: list[dict[str, Any]]


class SavedRouteResponse(BaseModel):
    """Response for a saved route."""
    route_id: str
    user_id: str
    name: str
    waypoints: list[dict[str, Any]]
    geometry: dict[str, Any]
    steps: list[dict[str, Any]]
    distance_meters: float
    duration_seconds: float
    profile: str
    created_at: datetime
    updated_at: datetime


class SavedRouteListResponse(BaseModel):
    """Response for list of saved routes."""
    routes: list[SavedRouteResponse]
    total: int
