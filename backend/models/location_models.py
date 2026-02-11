"""
Location Models for Map Feature
Pydantic models for location CRUD operations
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from decimal import Decimal


class LocationCreate(BaseModel):
    """Request model for creating a new location"""
    name: str = Field(..., max_length=255, description="Location name (e.g., Home, Work)")
    address: str = Field(..., description="Full address string")
    latitude: Optional[Decimal] = Field(None, description="Latitude coordinate (auto-geocoded if not provided)")
    longitude: Optional[Decimal] = Field(None, description="Longitude coordinate (auto-geocoded if not provided)")
    notes: Optional[str] = Field(None, description="Optional notes about the location")
    is_global: bool = Field(False, description="If true, location is visible to all users (admin only)")
    metadata: Optional[dict] = Field(None, description="Optional JSON metadata")


class LocationUpdate(BaseModel):
    """Request model for updating an existing location"""
    name: Optional[str] = Field(None, max_length=255, description="Location name")
    address: Optional[str] = Field(None, description="Full address string")
    latitude: Optional[Decimal] = Field(None, description="Latitude coordinate")
    longitude: Optional[Decimal] = Field(None, description="Longitude coordinate")
    notes: Optional[str] = Field(None, description="Optional notes about the location")
    metadata: Optional[dict] = Field(None, description="Optional JSON metadata")


class LocationResponse(BaseModel):
    """Response model for location data"""
    location_id: str
    user_id: str
    name: str
    address: Optional[str]
    latitude: Decimal
    longitude: Decimal
    notes: Optional[str]
    is_global: bool
    metadata: Optional[dict]
    created_at: datetime
    updated_at: datetime


class LocationListResponse(BaseModel):
    """Response model for list of locations"""
    locations: list[LocationResponse]
    total: int
