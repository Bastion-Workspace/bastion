"""
GeoJSON API: return map-ready GeoJSON FeatureCollections.
GET /api/geojson/locations - user locations as GeoJSON. Future: /api/geojson/articles, /api/geojson/events.
"""

import logging
from fastapi import APIRouter, Depends

from models.api_models import AuthenticatedUserResponse
from services.database_manager.database_helpers import fetch_all
from api.location_api import require_maps_access

logger = logging.getLogger(__name__)

router = APIRouter(tags=["geojson"])


@router.get("/api/geojson/locations")
async def get_locations_geojson(
    current_user: AuthenticatedUserResponse = Depends(require_maps_access),
):
    """Return all locations accessible to the current user as a GeoJSON FeatureCollection."""
    rls_context = {"user_id": current_user.user_id, "user_role": current_user.role}
    query = """
        SELECT location_id, user_id, name, address, latitude, longitude, notes, is_global, metadata, created_at, updated_at
        FROM user_locations
        ORDER BY is_global DESC, name ASC
    """
    rows = await fetch_all(query, rls_context=rls_context)
    features = []
    for r in rows:
        lat = r.get("latitude")
        lon = r.get("longitude")
        if lat is None or lon is None:
            continue
        try:
            lat_f = float(lat)
            lon_f = float(lon)
        except (TypeError, ValueError):
            continue
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon_f, lat_f]},
            "properties": {
                "id": str(r.get("location_id", "")),
                "name": r.get("name") or "",
                "address": r.get("address") or "",
                "notes": r.get("notes") or "",
                "is_global": bool(r.get("is_global", False)),
            },
        })
    return {"type": "FeatureCollection", "features": features}
