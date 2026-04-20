"""gRPC handlers for Navigation operations (locations and routes)."""

import json
import logging
from datetime import datetime, timezone

import grpc
from protos import tool_service_pb2

logger = logging.getLogger(__name__)


class NavigationHandlersMixin:
    """Mixin providing Navigation gRPC handlers.

    Mixed into ToolServiceImplementation; accesses self._get_search_service(),
    self._get_document_repo(), etc. via standard Python MRO.
    """

    # ===== Navigation Operations (locations and routes) =====

    async def CreateLocation(
        self,
        request: tool_service_pb2.CreateLocationRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.CreateLocationResponse:
        """Create a saved location (geocodes address if needed)."""
        try:
            from tools_service.services.navigation_tools import create_location as nav_create_location

            result = await nav_create_location(
                user_id=request.user_id,
                name=request.name,
                address=request.address,
                latitude=request.latitude if request.HasField("latitude") else None,
                longitude=request.longitude if request.HasField("longitude") else None,
                notes=request.notes if request.notes else None,
                is_global=request.is_global,
                metadata=json.loads(request.metadata_json) if request.HasField("metadata_json") and request.metadata_json else None,
                user_role=request.user_role or "user",
            )
            if not result.get("success"):
                return tool_service_pb2.CreateLocationResponse(success=False, error=result.get("error", "Unknown error"))
            return tool_service_pb2.CreateLocationResponse(
                success=True,
                location_id=result.get("location_id", ""),
                user_id=result.get("user_id", ""),
                name=result.get("name", ""),
                address=result.get("address") or "",
                latitude=result.get("latitude", 0),
                longitude=result.get("longitude", 0),
                notes=result.get("notes") or "",
                is_global=result.get("is_global", False),
                created_at=result.get("created_at") or "",
                updated_at=result.get("updated_at") or "",
            )
        except Exception as e:
            logger.error(f"CreateLocation failed: {e}")
            return tool_service_pb2.CreateLocationResponse(success=False, error=str(e))

    async def ListLocations(
        self,
        request: tool_service_pb2.ListLocationsRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ListLocationsResponse:
        """List all locations accessible to the user."""
        try:
            from tools_service.services.navigation_tools import list_locations

            result = await list_locations(
                user_id=request.user_id,
                user_role=request.user_role or "user",
            )
            if not result.get("success"):
                return tool_service_pb2.ListLocationsResponse(success=False, error=result.get("error", "Unknown error"))
            locations = []
            for loc in result.get("locations", []):
                locations.append(tool_service_pb2.LocationProto(
                    location_id=loc.get("location_id", ""),
                    user_id=loc.get("user_id", ""),
                    name=loc.get("name", ""),
                    address=loc.get("address") or "",
                    latitude=loc.get("latitude", 0),
                    longitude=loc.get("longitude", 0),
                    notes=loc.get("notes") or "",
                    is_global=loc.get("is_global", False),
                    created_at=loc.get("created_at") or "",
                    updated_at=loc.get("updated_at") or "",
                ))
            return tool_service_pb2.ListLocationsResponse(success=True, locations=locations, total=result.get("total", 0))
        except Exception as e:
            logger.error(f"ListLocations failed: {e}")
            return tool_service_pb2.ListLocationsResponse(success=False, error=str(e))

    async def DeleteLocation(
        self,
        request: tool_service_pb2.DeleteLocationRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.DeleteLocationResponse:
        """Delete a location by ID."""
        try:
            from tools_service.services.navigation_tools import delete_location

            result = await delete_location(
                user_id=request.user_id,
                location_id=request.location_id,
                user_role=request.user_role or "user",
            )
            if not result.get("success"):
                return tool_service_pb2.DeleteLocationResponse(success=False, error=result.get("error", "Unknown error"))
            return tool_service_pb2.DeleteLocationResponse(success=True, message=result.get("message", "Location deleted"))
        except Exception as e:
            logger.error(f"DeleteLocation failed: {e}")
            return tool_service_pb2.DeleteLocationResponse(success=False, error=str(e))

    async def ComputeRoute(
        self,
        request: tool_service_pb2.ComputeRouteRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ComputeRouteResponse:
        """Compute route between two points (location IDs or coordinates)."""
        try:
            from tools_service.services.navigation_tools import compute_route

            result = await compute_route(
                user_id=request.user_id,
                from_location_id=request.from_location_id if request.from_location_id else None,
                to_location_id=request.to_location_id if request.to_location_id else None,
                coordinates=request.coordinates if request.coordinates else None,
                profile=request.profile or "driving",
                user_role=request.user_role or "user",
            )
            if not result.get("success"):
                return tool_service_pb2.ComputeRouteResponse(success=False, error=result.get("error", "Unknown error"))
            return tool_service_pb2.ComputeRouteResponse(
                success=True,
                geometry_json=json.dumps(result.get("geometry", {})),
                legs_json=json.dumps(result.get("legs", [])),
                distance=result.get("distance", 0),
                duration=result.get("duration", 0),
                waypoints_json=json.dumps(result.get("waypoints", [])),
            )
        except Exception as e:
            logger.error(f"ComputeRoute failed: {e}")
            return tool_service_pb2.ComputeRouteResponse(success=False, error=str(e))

    async def SaveRoute(
        self,
        request: tool_service_pb2.NavSaveRouteRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.NavSaveRouteResponse:
        """Save a computed route."""
        try:
            from tools_service.services.navigation_tools import save_route

            waypoints = json.loads(request.waypoints_json) if request.waypoints_json else []
            geometry = json.loads(request.geometry_json) if request.geometry_json else {}
            steps = json.loads(request.steps_json) if request.steps_json else []

            result = await save_route(
                user_id=request.user_id,
                name=request.name,
                waypoints=waypoints,
                geometry=geometry,
                steps=steps,
                distance_meters=request.distance_meters,
                duration_seconds=request.duration_seconds,
                profile=request.profile or "driving",
                user_role=request.user_role or "user",
            )
            if not result.get("success"):
                return tool_service_pb2.NavSaveRouteResponse(success=False, error=result.get("error", "Unknown error"))
            return tool_service_pb2.NavSaveRouteResponse(
                success=True,
                route_id=result.get("route_id", ""),
                user_id=result.get("user_id", ""),
                name=result.get("name", ""),
                waypoints_json=json.dumps(result.get("waypoints", [])),
                geometry_json=json.dumps(result.get("geometry", {})),
                steps_json=json.dumps(result.get("steps", [])),
                distance_meters=result.get("distance_meters", 0),
                duration_seconds=result.get("duration_seconds", 0),
                profile=result.get("profile", "driving"),
                created_at=result.get("created_at") or "",
                updated_at=result.get("updated_at") or "",
            )
        except Exception as e:
            logger.error(f"SaveRoute failed: {e}")
            return tool_service_pb2.NavSaveRouteResponse(success=False, error=str(e))

    async def ListSavedRoutes(
        self,
        request: tool_service_pb2.ListSavedRoutesRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ListSavedRoutesResponse:
        """List saved routes for the user."""
        try:
            from tools_service.services.navigation_tools import list_saved_routes

            result = await list_saved_routes(
                user_id=request.user_id,
                user_role=request.user_role or "user",
            )
            if not result.get("success"):
                return tool_service_pb2.ListSavedRoutesResponse(success=False, error=result.get("error", "Unknown error"))
            routes = []
            for r in result.get("routes", []):
                routes.append(tool_service_pb2.SavedRouteProto(
                    route_id=r.get("route_id", ""),
                    user_id=r.get("user_id", ""),
                    name=r.get("name", ""),
                    waypoints_json=json.dumps(r.get("waypoints", [])),
                    geometry_json=json.dumps(r.get("geometry", {})),
                    steps_json=json.dumps(r.get("steps", [])),
                    distance_meters=r.get("distance_meters", 0),
                    duration_seconds=r.get("duration_seconds", 0),
                    profile=r.get("profile", "driving"),
                    created_at=r.get("created_at") or "",
                    updated_at=r.get("updated_at") or "",
                ))
            return tool_service_pb2.ListSavedRoutesResponse(success=True, routes=routes, total=result.get("total", 0))
        except Exception as e:
            logger.error(f"ListSavedRoutes failed: {e}")
            return tool_service_pb2.ListSavedRoutesResponse(success=False, error=str(e))

    async def DeleteSavedRoute(
        self,
        request: tool_service_pb2.DeleteSavedRouteRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.DeleteSavedRouteResponse:
        """Delete a saved route."""
        try:
            from tools_service.services.navigation_tools import delete_saved_route

            result = await delete_saved_route(
                user_id=request.user_id,
                route_id=request.route_id,
                user_role=request.user_role or "user",
            )
            if not result.get("success"):
                return tool_service_pb2.DeleteSavedRouteResponse(success=False, error=result.get("error", "Unknown error"))
            return tool_service_pb2.DeleteSavedRouteResponse(success=True, message=result.get("message", "Route deleted"))
        except Exception as e:
            logger.error(f"DeleteSavedRoute failed: {e}")
            return tool_service_pb2.DeleteSavedRouteResponse(success=False, error=str(e))

    async def AnalyzeWebsiteSecurity(
        self,
        request: tool_service_pb2.SecurityAnalysisRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.SecurityAnalysisResponse:
        """Perform passive security analysis of a website."""
        try:
            from tools_service.services.security_analysis_service import analyze_website

            result = await analyze_website(
                url=request.target_url,
                user_id=request.user_id or "system",
                scan_depth=request.scan_depth or "comprehensive",
            )
            if not result.get("success"):
                return tool_service_pb2.SecurityAnalysisResponse(
                    success=False,
                    target_url=result.get("target_url", request.target_url),
                    scan_timestamp=result.get("scan_timestamp", ""),
                    disclaimer=result.get("disclaimer", ""),
                    error=result.get("error", "Unknown error"),
                )
            findings_proto = []
            for f in result.get("findings", []):
                findings_proto.append(tool_service_pb2.SecurityFinding(
                    category=f.get("category", ""),
                    severity=f.get("severity", ""),
                    title=f.get("title", ""),
                    description=f.get("description", ""),
                    url=f.get("url") or None,
                    evidence=f.get("evidence") or None,
                    remediation=f.get("remediation", ""),
                ))
            tech_stack = result.get("technology_stack") or {}
            sec_headers = result.get("security_headers") or {}
            headers_map = {}
            if sec_headers.get("present"):
                headers_map["present"] = ",".join(sec_headers["present"]) if isinstance(sec_headers["present"], list) else str(sec_headers["present"])
            if sec_headers.get("missing"):
                headers_map["missing"] = ",".join(sec_headers["missing"]) if isinstance(sec_headers["missing"], list) else str(sec_headers["missing"])
            return tool_service_pb2.SecurityAnalysisResponse(
                success=True,
                target_url=result.get("target_url", ""),
                scan_timestamp=result.get("scan_timestamp", ""),
                findings=findings_proto,
                technology_stack=tech_stack,
                security_headers=headers_map,
                risk_score=float(result.get("risk_score", 0.0)),
                summary=result.get("summary", ""),
                disclaimer=result.get("disclaimer", ""),
            )
        except Exception as e:
            logger.error(f"AnalyzeWebsiteSecurity failed: {e}")
            return tool_service_pb2.SecurityAnalysisResponse(
                success=False,
                target_url=request.target_url,
                scan_timestamp=datetime.now(timezone.utc).isoformat(),
                disclaimer="This security scan performs passive reconnaissance only. Use only on sites you own or have permission to test.",
                error=str(e),
            )


