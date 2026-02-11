"""
Navigation tools for location and route operations.
Uses backend database helpers and OSRM when running in tools-service (backend on path).
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_navigation_tools_instance: Optional["NavigationTools"] = None


async def _get_navigation_tools() -> "NavigationTools":
    global _navigation_tools_instance
    if _navigation_tools_instance is None:
        _navigation_tools_instance = NavigationTools()
    return _navigation_tools_instance


class NavigationTools:
    """Location and route operations for agents."""

    async def create_location(
        self,
        user_id: str,
        name: str,
        address: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        notes: Optional[str] = None,
        is_global: bool = False,
        metadata: Optional[dict] = None,
        user_role: str = "user",
    ) -> Dict[str, Any]:
        """Create a saved location. Geocodes address if lat/lon not provided."""
        try:
            from services.database_manager.database_helpers import fetch_one
            from config import settings
        except ImportError as e:
            logger.error(f"Navigation tools backend import failed: {e}")
            return {"success": False, "error": "Backend services not available"}

        lat, lon = latitude, longitude
        if (lat is None or lon is None) and address:
            geocoded = await self._geocode_address(address, settings)
            if not geocoded.get("success"):
                return geocoded
            lat = geocoded["latitude"]
            lon = geocoded["longitude"]
        if lat is None or lon is None:
            return {"success": False, "error": "Address and coordinates are required"}

        if is_global and user_role != "admin":
            return {"success": False, "error": "Only admins can create global locations"}

        rls_context = {"user_id": user_id, "user_role": user_role}
        metadata_value = json.dumps(metadata) if metadata else None
        query = """
            INSERT INTO user_locations (user_id, name, address, latitude, longitude, notes, is_global, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
            RETURNING location_id, user_id, name, address, latitude, longitude, notes, is_global, metadata, created_at, updated_at
        """
        try:
            row = await fetch_one(
                query,
                user_id,
                name,
                address,
                float(lat),
                float(lon),
                notes,
                is_global,
                metadata_value,
                rls_context=rls_context,
            )
        except Exception as e:
            logger.error(f"create_location failed: {e}")
            return {"success": False, "error": str(e)}

        if not row:
            return {"success": False, "error": "Failed to create location"}

        return {
            "success": True,
            "location_id": str(row["location_id"]),
            "user_id": str(row["user_id"]),
            "name": row["name"],
            "address": row.get("address"),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "notes": row.get("notes"),
            "is_global": row.get("is_global", False),
            "metadata": row.get("metadata"),
            "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
            "updated_at": row["updated_at"].isoformat() if row.get("updated_at") else None,
        }

    async def _geocode_address(self, address: str, settings: Any) -> Dict[str, Any]:
        import aiohttp

        if not getattr(settings, "OPENWEATHERMAP_API_KEY", None):
            return {"success": False, "error": "OpenWeatherMap API key not configured"}

        if address.isdigit() and len(address) == 5:
            url = "https://api.openweathermap.org/geo/1.0/zip"
            params = {"zip": f"{address},US", "appid": settings.OPENWEATHERMAP_API_KEY}
        else:
            url = "https://api.openweathermap.org/geo/1.0/direct"
            params = {"q": address, "limit": 1, "appid": settings.OPENWEATHERMAP_API_KEY}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        text = await response.text()
                        return {"success": False, "error": f"Geocoding failed: {text[:200]}"}
                    data = await response.json()
        except Exception as e:
            logger.error(f"Geocoding request failed: {e}")
            return {"success": False, "error": str(e)}

        if isinstance(data, list):
            if not data:
                return {"success": False, "error": "Address not found"}
            result = data[0]
            return {"success": True, "latitude": float(result["lat"]), "longitude": float(result["lon"])}
        if isinstance(data, dict) and "lat" in data:
            return {"success": True, "latitude": float(data["lat"]), "longitude": float(data["lon"])}
        return {"success": False, "error": "Address not found"}

    async def list_locations(self, user_id: str, user_role: str = "user") -> Dict[str, Any]:
        """List all locations accessible to the user."""
        try:
            from services.database_manager.database_helpers import fetch_all
        except ImportError as e:
            logger.error(f"Navigation tools backend import failed: {e}")
            return {"success": False, "error": "Backend services not available"}

        rls_context = {"user_id": user_id, "user_role": user_role}
        query = """
            SELECT location_id, user_id, name, address, latitude, longitude, notes, is_global, metadata, created_at, updated_at
            FROM user_locations
            ORDER BY is_global DESC, name ASC
        """
        try:
            rows = await fetch_all(query, rls_context=rls_context)
        except Exception as e:
            logger.error(f"list_locations failed: {e}")
            return {"success": False, "error": str(e)}

        locations = []
        for r in rows:
            locations.append({
                "location_id": str(r["location_id"]),
                "user_id": str(r["user_id"]),
                "name": r["name"],
                "address": r.get("address"),
                "latitude": float(r["latitude"]),
                "longitude": float(r["longitude"]),
                "notes": r.get("notes"),
                "is_global": r.get("is_global", False),
                "metadata": r.get("metadata"),
                "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
                "updated_at": r["updated_at"].isoformat() if r.get("updated_at") else None,
            })
        return {"success": True, "locations": locations, "total": len(locations)}

    async def get_location_by_name(
        self, user_id: str, name: str, user_role: str = "user"
    ) -> Dict[str, Any]:
        """Get a location by name (case-insensitive match)."""
        result = await self.list_locations(user_id, user_role)
        if not result.get("success"):
            return result
        name_lower = name.strip().lower()
        for loc in result.get("locations", []):
            if (loc.get("name") or "").strip().lower() == name_lower:
                return {"success": True, "location": loc}
        return {"success": False, "error": f"Location '{name}' not found"}

    async def delete_location(
        self, user_id: str, location_id: str, user_role: str = "user"
    ) -> Dict[str, Any]:
        """Delete a location by ID."""
        try:
            from services.database_manager.database_helpers import execute
        except ImportError as e:
            logger.error(f"Navigation tools backend import failed: {e}")
            return {"success": False, "error": "Backend services not available"}

        rls_context = {"user_id": user_id, "user_role": user_role}
        query = "DELETE FROM user_locations WHERE location_id = $1::uuid"
        try:
            result = await execute(query, location_id, rls_context=rls_context)
        except Exception as e:
            logger.error(f"delete_location failed: {e}")
            return {"success": False, "error": str(e)}

        if result and "DELETE 0" in str(result):
            return {"success": False, "error": "Location not found or access denied"}
        return {"success": True, "message": "Location deleted successfully"}

    async def compute_route(
        self,
        user_id: str,
        from_location_id: Optional[str] = None,
        to_location_id: Optional[str] = None,
        coordinates: Optional[str] = None,
        profile: str = "driving",
        user_role: str = "user",
    ) -> Dict[str, Any]:
        """Compute route between two points (location IDs or coordinate string)."""
        try:
            from services.database_manager.database_helpers import fetch_all
            from services.osrm_service import get_route, OSRMError
        except ImportError as e:
            logger.error(f"Navigation tools backend import failed: {e}")
            return {"success": False, "error": "Backend services not available"}

        pairs: List[tuple] = []
        if coordinates:
            try:
                for part in coordinates.split(";"):
                    part = part.strip()
                    if not part:
                        continue
                    lat_str, lng_str = part.split(",", 1)
                    pairs.append((float(lat_str.strip()), float(lng_str.strip())))
                if len(pairs) < 2:
                    return {"success": False, "error": "At least two coordinates required"}
            except (ValueError, AttributeError) as e:
                return {"success": False, "error": f"Invalid coordinates: {e}"}
        elif from_location_id and to_location_id:
            rls_context = {"user_id": user_id, "user_role": user_role}
            query = """
                SELECT location_id, latitude, longitude
                FROM user_locations
                WHERE location_id = ANY($1::uuid[])
                ORDER BY array_position($1::uuid[], location_id)
            """
            ids = [from_location_id, to_location_id]
            rows = await fetch_all(query, ids, rls_context=rls_context)
            if len(rows) != 2:
                return {"success": False, "error": "One or both locations not found or not accessible"}
            pairs = [(float(r["latitude"]), float(r["longitude"])) for r in rows]
        else:
            return {
                "success": False,
                "error": "Provide either coordinates=lat1,lng1;lat2,lng2 or from_location_id and to_location_id",
            }

        try:
            result = await get_route(pairs, profile=profile)
        except OSRMError as e:
            if e.code in ("NoRoute", "NoSegment"):
                return {"success": False, "error": e.message or "No road route found"}
            return {"success": False, "error": e.message or "Routing service error"}
        except ValueError as e:
            return {"success": False, "error": str(e)}

        return {
            "success": True,
            "geometry": result.get("geometry", {}),
            "legs": result.get("legs", []),
            "distance": result.get("distance", 0),
            "duration": result.get("duration", 0),
            "waypoints": result.get("waypoints", []),
        }

    async def save_route(
        self,
        user_id: str,
        name: str,
        waypoints: List[dict],
        geometry: dict,
        steps: List[dict],
        distance_meters: float,
        duration_seconds: float,
        profile: str = "driving",
        user_role: str = "user",
    ) -> Dict[str, Any]:
        """Save a computed route."""
        try:
            from services.database_manager.database_helpers import fetch_one
        except ImportError as e:
            logger.error(f"Navigation tools backend import failed: {e}")
            return {"success": False, "error": "Backend services not available"}

        rls_context = {"user_id": user_id, "user_role": user_role}
        geometry_json = json.dumps(geometry)
        waypoints_json = json.dumps(waypoints)
        steps_json = json.dumps(steps)
        query = """
            INSERT INTO saved_routes (user_id, name, waypoints, geometry, steps, distance_meters, duration_seconds, profile)
            VALUES ($1, $2, $3::jsonb, $4::jsonb, $5::jsonb, $6, $7, $8)
            RETURNING route_id, user_id, name, waypoints, geometry, steps, distance_meters, duration_seconds, profile, created_at, updated_at
        """
        try:
            row = await fetch_one(
                query,
                user_id,
                name,
                waypoints_json,
                geometry_json,
                steps_json,
                distance_meters,
                duration_seconds,
                profile,
                rls_context=rls_context,
            )
        except Exception as e:
            logger.error(f"save_route failed: {e}")
            return {"success": False, "error": str(e)}

        if not row:
            return {"success": False, "error": "Failed to save route"}

        waypoints_out = row.get("waypoints")
        if isinstance(waypoints_out, str):
            try:
                waypoints_out = json.loads(waypoints_out)
            except json.JSONDecodeError:
                waypoints_out = []
        geometry_out = row.get("geometry")
        if isinstance(geometry_out, str):
            try:
                geometry_out = json.loads(geometry_out)
            except json.JSONDecodeError:
                geometry_out = {}
        steps_out = row.get("steps")
        if isinstance(steps_out, str):
            try:
                steps_out = json.loads(steps_out)
            except json.JSONDecodeError:
                steps_out = []

        return {
            "success": True,
            "route_id": str(row["route_id"]),
            "user_id": str(row["user_id"]),
            "name": row["name"],
            "waypoints": waypoints_out or [],
            "geometry": geometry_out or {},
            "steps": steps_out or [],
            "distance_meters": float(row["distance_meters"]),
            "duration_seconds": float(row["duration_seconds"]),
            "profile": row.get("profile", "driving"),
            "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
            "updated_at": row["updated_at"].isoformat() if row.get("updated_at") else None,
        }

    async def list_saved_routes(self, user_id: str, user_role: str = "user") -> Dict[str, Any]:
        """List saved routes for the user."""
        try:
            from services.database_manager.database_helpers import fetch_all
        except ImportError as e:
            logger.error(f"Navigation tools backend import failed: {e}")
            return {"success": False, "error": "Backend services not available"}

        rls_context = {"user_id": user_id, "user_role": user_role}
        query = """
            SELECT route_id, user_id, name, waypoints, geometry, steps, distance_meters, duration_seconds, profile, created_at, updated_at
            FROM saved_routes
            ORDER BY created_at DESC
        """
        try:
            rows = await fetch_all(query, rls_context=rls_context)
        except Exception as e:
            logger.error(f"list_saved_routes failed: {e}")
            return {"success": False, "error": str(e)}

        routes = []
        for r in rows:
            waypoints = r.get("waypoints")
            geometry = r.get("geometry")
            steps = r.get("steps")
            if isinstance(waypoints, str):
                try:
                    waypoints = json.loads(waypoints)
                except json.JSONDecodeError:
                    waypoints = []
            if isinstance(geometry, str):
                try:
                    geometry = json.loads(geometry)
                except json.JSONDecodeError:
                    geometry = {}
            if isinstance(steps, str):
                try:
                    steps = json.loads(steps)
                except json.JSONDecodeError:
                    steps = []
            routes.append({
                "route_id": str(r["route_id"]),
                "user_id": str(r["user_id"]),
                "name": r["name"],
                "waypoints": waypoints or [],
                "geometry": geometry or {},
                "steps": steps or [],
                "distance_meters": float(r["distance_meters"]),
                "duration_seconds": float(r["duration_seconds"]),
                "profile": r.get("profile", "driving"),
                "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
                "updated_at": r["updated_at"].isoformat() if r.get("updated_at") else None,
            })
        return {"success": True, "routes": routes, "total": len(routes)}

    async def get_saved_route(
        self, user_id: str, route_id: str, user_role: str = "user"
    ) -> Dict[str, Any]:
        """Get a single saved route by ID."""
        try:
            from services.database_manager.database_helpers import fetch_one
        except ImportError as e:
            logger.error(f"Navigation tools backend import failed: {e}")
            return {"success": False, "error": "Backend services not available"}

        rls_context = {"user_id": user_id, "user_role": user_role}
        query = """
            SELECT route_id, user_id, name, waypoints, geometry, steps, distance_meters, duration_seconds, profile, created_at, updated_at
            FROM saved_routes
            WHERE route_id = $1::uuid
        """
        try:
            row = await fetch_one(query, route_id, rls_context=rls_context)
        except Exception as e:
            logger.error(f"get_saved_route failed: {e}")
            return {"success": False, "error": str(e)}

        if not row:
            return {"success": False, "error": "Route not found"}

        waypoints = row.get("waypoints")
        geometry = row.get("geometry")
        steps = row.get("steps")
        if isinstance(waypoints, str):
            waypoints = json.loads(waypoints) if waypoints else []
        if isinstance(geometry, str):
            geometry = json.loads(geometry) if geometry else {}
        if isinstance(steps, str):
            steps = json.loads(steps) if steps else []

        return {
            "success": True,
            "route_id": str(row["route_id"]),
            "user_id": str(row["user_id"]),
            "name": row["name"],
            "waypoints": waypoints or [],
            "geometry": geometry or {},
            "steps": steps or [],
            "distance_meters": float(row["distance_meters"]),
            "duration_seconds": float(row["duration_seconds"]),
            "profile": row.get("profile", "driving"),
            "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
            "updated_at": row["updated_at"].isoformat() if row.get("updated_at") else None,
        }

    async def delete_saved_route(
        self, user_id: str, route_id: str, user_role: str = "user"
    ) -> Dict[str, Any]:
        """Delete a saved route."""
        try:
            from services.database_manager.database_helpers import execute
        except ImportError as e:
            logger.error(f"Navigation tools backend import failed: {e}")
            return {"success": False, "error": "Backend services not available"}

        rls_context = {"user_id": user_id, "user_role": user_role}
        query = "DELETE FROM saved_routes WHERE route_id = $1::uuid"
        try:
            result = await execute(query, route_id, rls_context=rls_context)
        except Exception as e:
            logger.error(f"delete_saved_route failed: {e}")
            return {"success": False, "error": str(e)}

        if result and "DELETE 0" in str(result):
            return {"success": False, "error": "Route not found"}
        return {"success": True, "message": "Route deleted successfully"}


async def create_location(
    user_id: str,
    name: str,
    address: str,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    notes: Optional[str] = None,
    is_global: bool = False,
    metadata: Optional[dict] = None,
    user_role: str = "user",
) -> Dict[str, Any]:
    tools = await _get_navigation_tools()
    return await tools.create_location(
        user_id=user_id,
        name=name,
        address=address,
        latitude=latitude,
        longitude=longitude,
        notes=notes,
        is_global=is_global,
        metadata=metadata,
        user_role=user_role,
    )


async def list_locations(user_id: str, user_role: str = "user") -> Dict[str, Any]:
    tools = await _get_navigation_tools()
    return await tools.list_locations(user_id=user_id, user_role=user_role)


async def get_location_by_name(
    user_id: str, name: str, user_role: str = "user"
) -> Dict[str, Any]:
    tools = await _get_navigation_tools()
    return await tools.get_location_by_name(user_id=user_id, name=name, user_role=user_role)


async def delete_location(
    user_id: str, location_id: str, user_role: str = "user"
) -> Dict[str, Any]:
    tools = await _get_navigation_tools()
    return await tools.delete_location(user_id=user_id, location_id=location_id, user_role=user_role)


async def compute_route(
    user_id: str,
    from_location_id: Optional[str] = None,
    to_location_id: Optional[str] = None,
    coordinates: Optional[str] = None,
    profile: str = "driving",
    user_role: str = "user",
) -> Dict[str, Any]:
    tools = await _get_navigation_tools()
    return await tools.compute_route(
        user_id=user_id,
        from_location_id=from_location_id,
        to_location_id=to_location_id,
        coordinates=coordinates,
        profile=profile,
        user_role=user_role,
    )


async def save_route(
    user_id: str,
    name: str,
    waypoints: List[dict],
    geometry: dict,
    steps: List[dict],
    distance_meters: float,
    duration_seconds: float,
    profile: str = "driving",
    user_role: str = "user",
) -> Dict[str, Any]:
    tools = await _get_navigation_tools()
    return await tools.save_route(
        user_id=user_id,
        name=name,
        waypoints=waypoints,
        geometry=geometry,
        steps=steps,
        distance_meters=distance_meters,
        duration_seconds=duration_seconds,
        profile=profile,
        user_role=user_role,
    )


async def list_saved_routes(user_id: str, user_role: str = "user") -> Dict[str, Any]:
    tools = await _get_navigation_tools()
    return await tools.list_saved_routes(user_id=user_id, user_role=user_role)


async def get_saved_route(
    user_id: str, route_id: str, user_role: str = "user"
) -> Dict[str, Any]:
    tools = await _get_navigation_tools()
    return await tools.get_saved_route(user_id=user_id, route_id=route_id, user_role=user_role)


async def delete_saved_route(
    user_id: str, route_id: str, user_role: str = "user"
) -> Dict[str, Any]:
    tools = await _get_navigation_tools()
    return await tools.delete_saved_route(user_id=user_id, route_id=route_id, user_role=user_role)
