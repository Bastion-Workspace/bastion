"""
Location API endpoints for map feature
CRUD operations for user locations with geocoding support
"""

import logging
import json
from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
from decimal import Decimal

from models.location_models import LocationCreate, LocationUpdate, LocationResponse, LocationListResponse
from models.api_models import AuthenticatedUserResponse
from utils.auth_middleware import get_current_user
from services.capabilities_service import capabilities_service
from services.database_manager.database_helpers import fetch_one, fetch_all, execute

logger = logging.getLogger(__name__)

router = APIRouter(tags=["locations"])


def _row_to_location_response(row: dict) -> LocationResponse:
    """Convert a DB row (UUID/Decimal) to LocationResponse (str/Decimal)."""
    d = dict(row)
    if d.get("location_id") is not None:
        d["location_id"] = str(d["location_id"])
    if d.get("user_id") is not None:
        d["user_id"] = str(d["user_id"])
    return LocationResponse(**d)


async def require_maps_access(current_user: AuthenticatedUserResponse = Depends(get_current_user)):
    """Dependency to check if user has map access"""
    if current_user.role == "admin":
        return current_user
    
    has_access = await capabilities_service.user_has_feature(
        {"user_id": current_user.user_id, "role": current_user.role},
        "feature.maps.view"
    )
    
    if not has_access:
        raise HTTPException(status_code=403, detail="Map access not enabled for this user")
    
    return current_user


@router.post("/api/locations", response_model=LocationResponse)
async def create_location(
    location: LocationCreate,
    current_user: AuthenticatedUserResponse = Depends(require_maps_access)
):
    """Create a new location (private or global)"""
    try:
        logger.info(f"Creating location '{location.name}' for user {current_user.username}")
        
        # Geocode if lat/lon not provided
        latitude = location.latitude
        longitude = location.longitude
        
        if latitude is None or longitude is None:
            # Use OpenWeatherMap geocoding API directly
            try:
                import aiohttp
                from config import settings
                
                if not settings.OPENWEATHERMAP_API_KEY:
                    raise HTTPException(status_code=500, detail="OpenWeatherMap API key not configured")
                
                # Check if location is a ZIP code (US format)
                if location.address.isdigit() and len(location.address) == 5:
                    url = f"https://api.openweathermap.org/geo/1.0/zip"
                    params = {
                        "zip": f"{location.address},US",
                        "appid": settings.OPENWEATHERMAP_API_KEY
                    }
                else:
                    url = f"https://api.openweathermap.org/geo/1.0/direct"
                    params = {
                        "q": location.address,
                        "limit": 1,
                        "appid": settings.OPENWEATHERMAP_API_KEY
                    }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise HTTPException(status_code=400, detail=f"Failed to geocode address: {error_text}")
                        
                        data = await response.json()
                
                # Handle ZIP code response (single object) or direct response (array)
                if isinstance(data, list):
                    if not data or len(data) == 0:
                        raise HTTPException(status_code=400, detail="Address not found")
                    result = data[0]
                    latitude = Decimal(str(result["lat"]))
                    longitude = Decimal(str(result["lon"]))
                else:
                    # ZIP code response
                    latitude = Decimal(str(data["lat"]))
                    longitude = Decimal(str(data["lon"]))
                
                logger.info(f"Geocoded '{location.address}' to ({latitude}, {longitude})")
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Geocoding failed: {e}")
                raise HTTPException(status_code=400, detail=f"Failed to geocode address: {str(e)}")
        
        # Only admins can create global locations
        if location.is_global and current_user.role != "admin":
            raise HTTPException(status_code=403, detail="Only admins can create global locations")
        
        # Insert into database with RLS context
        rls_context = {
            'user_id': current_user.user_id,
            'user_role': current_user.role
        }
        
        query = """
            INSERT INTO user_locations (user_id, name, address, latitude, longitude, notes, is_global, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
            RETURNING location_id, user_id, name, address, latitude, longitude, notes, is_global, metadata, created_at, updated_at
        """
        
        # Convert metadata dict to JSON string for JSONB column
        metadata_value = json.dumps(location.metadata) if location.metadata else None
        
        result = await fetch_one(
            query,
            current_user.user_id,
            location.name,
            location.address,
            float(latitude),
            float(longitude),
            location.notes,
            location.is_global,
            metadata_value,
            rls_context=rls_context
        )
        
        if not result:
            raise HTTPException(status_code=500, detail="Failed to create location")
        
        logger.info(f"Successfully created location {result['location_id']}")
        return _row_to_location_response(result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create location: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create location: {str(e)}")


@router.get("/api/locations", response_model=LocationListResponse)
async def get_locations(
    current_user: AuthenticatedUserResponse = Depends(require_maps_access)
):
    """Get all locations accessible to the current user"""
    try:
        rls_context = {
            'user_id': current_user.user_id,
            'user_role': current_user.role
        }
        
        query = """
            SELECT location_id, user_id, name, address, latitude, longitude, notes, is_global, metadata, created_at, updated_at
            FROM user_locations
            ORDER BY is_global DESC, name ASC
        """
        
        results = await fetch_all(query, rls_context=rls_context)
        
        return LocationListResponse(
            locations=[_row_to_location_response(r) for r in results],
            total=len(results)
        )
        
    except Exception as e:
        logger.error(f"Failed to get locations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve locations: {str(e)}")


@router.put("/api/locations/{location_id}", response_model=LocationResponse)
async def update_location(
    location_id: str,
    location: LocationUpdate,
    current_user: AuthenticatedUserResponse = Depends(require_maps_access)
):
    """Update an existing location"""
    try:
        rls_context = {
            'user_id': current_user.user_id,
            'user_role': current_user.role
        }
        
        # Build dynamic update query
        updates = []
        params = []
        param_idx = 1
        
        if location.name is not None:
            updates.append(f"name = ${param_idx}")
            params.append(location.name)
            param_idx += 1
        
        if location.address is not None:
            updates.append(f"address = ${param_idx}")
            params.append(location.address)
            param_idx += 1
            
            # Re-geocode if address changed and coordinates not provided
            if location.latitude is None or location.longitude is None:
                try:
                    import aiohttp
                    from config import settings
                    
                    if settings.OPENWEATHERMAP_API_KEY:
                        # Check if location is a ZIP code
                        if location.address.isdigit() and len(location.address) == 5:
                            url = f"https://api.openweathermap.org/geo/1.0/zip"
                            geocode_params = {
                                "zip": f"{location.address},US",
                                "appid": settings.OPENWEATHERMAP_API_KEY
                            }
                        else:
                            url = f"https://api.openweathermap.org/geo/1.0/direct"
                            geocode_params = {
                                "q": location.address,
                                "limit": 1,
                                "appid": settings.OPENWEATHERMAP_API_KEY
                            }
                        
                        async with aiohttp.ClientSession() as session:
                            async with session.get(url, params=geocode_params) as response:
                                if response.status == 200:
                                    data = await response.json()
                                    if isinstance(data, list) and len(data) > 0:
                                        location.latitude = Decimal(str(data[0]["lat"]))
                                        location.longitude = Decimal(str(data[0]["lon"]))
                                    elif isinstance(data, dict) and "lat" in data:
                                        location.latitude = Decimal(str(data["lat"]))
                                        location.longitude = Decimal(str(data["lon"]))
                                    logger.info(f"Re-geocoded '{location.address}' to ({location.latitude}, {location.longitude})")
                except Exception as e:
                    logger.warning(f"Re-geocoding failed: {e}, continuing with existing coordinates")
        
        if location.latitude is not None:
            updates.append(f"latitude = ${param_idx}")
            params.append(float(location.latitude))
            param_idx += 1
        
        if location.longitude is not None:
            updates.append(f"longitude = ${param_idx}")
            params.append(float(location.longitude))
            param_idx += 1
        
        if location.notes is not None:
            updates.append(f"notes = ${param_idx}")
            params.append(location.notes)
            param_idx += 1
        
        if location.metadata is not None:
            updates.append(f"metadata = ${param_idx}::jsonb")
            params.append(json.dumps(location.metadata))
            param_idx += 1
        
        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")
        
        updates.append("updated_at = NOW()")
        
        query = f"""
            UPDATE user_locations
            SET {', '.join(updates)}
            WHERE location_id = ${param_idx}::uuid
            RETURNING location_id, user_id, name, address, latitude, longitude, notes, is_global, metadata, created_at, updated_at
        """
        params.append(location_id)
        
        result = await fetch_one(query, *params, rls_context=rls_context)
        
        if not result:
            raise HTTPException(status_code=404, detail="Location not found or access denied")
        
        logger.info(f"Successfully updated location {location_id}")
        return _row_to_location_response(result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update location: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update location: {str(e)}")


@router.delete("/api/locations/{location_id}")
async def delete_location(
    location_id: str,
    current_user: AuthenticatedUserResponse = Depends(require_maps_access)
):
    """Delete a location"""
    try:
        rls_context = {
            'user_id': current_user.user_id,
            'user_role': current_user.role
        }
        
        query = "DELETE FROM user_locations WHERE location_id = $1::uuid"
        result = await execute(query, location_id, rls_context=rls_context)
        
        # Check if any rows were affected
        if "DELETE 0" in str(result) or result == "DELETE 0":
            raise HTTPException(status_code=404, detail="Location not found or access denied")
        
        logger.info(f"Successfully deleted location {location_id}")
        return {"message": "Location deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete location: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete location: {str(e)}")
