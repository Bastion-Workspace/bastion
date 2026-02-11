"""
Navigation tool models.
Pydantic models for location and route tool requests/responses.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

from pydantic import BaseModel, Field


class LocationCreateInput(BaseModel):
    """Input for creating a location."""
    name: str = Field(..., max_length=255)
    address: str = Field(...)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    notes: Optional[str] = None
    is_global: bool = False
    metadata: Optional[dict] = None


class LocationResponseModel(BaseModel):
    """Location as returned by tools."""
    location_id: str
    user_id: str
    name: str
    address: Optional[str] = None
    latitude: float
    longitude: float
    notes: Optional[str] = None
    is_global: bool
    metadata: Optional[dict] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class RouteComputeInput(BaseModel):
    """Input for computing a route."""
    from_location_id: Optional[str] = None
    to_location_id: Optional[str] = None
    coordinates: Optional[str] = None  # "lat1,lng1;lat2,lng2"
    profile: str = "driving"


class RouteResponseModel(BaseModel):
    """Computed route response."""
    geometry: dict[str, Any]
    legs: list[dict[str, Any]]
    distance: float
    duration: float
    waypoints: list[dict[str, Any]]


class SaveRouteInput(BaseModel):
    """Input for saving a route."""
    name: str = Field(..., max_length=255)
    waypoints: list[dict[str, Any]] = Field(default_factory=list)
    geometry: dict[str, Any] = Field(...)
    steps: list[dict[str, Any]] = Field(default_factory=list)
    distance_meters: float = Field(..., ge=0)
    duration_seconds: float = Field(..., ge=0)
    profile: str = "driving"


class SavedRouteResponseModel(BaseModel):
    """Saved route as returned by tools."""
    route_id: str
    user_id: str
    name: str
    waypoints: list[dict[str, Any]]
    geometry: dict[str, Any]
    steps: list[dict[str, Any]]
    distance_meters: float
    duration_seconds: float
    profile: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
