"""
Valhalla routing adapter.
Calls Valhalla HTTP API and maps response to the same contract as OSRM (geometry, legs, distance, duration, waypoints).
"""

import logging
from typing import Any

import aiohttp

from config import settings

logger = logging.getLogger(__name__)


def _decode_polyline(encoded: str, precision: int = 6) -> list[tuple[float, float]]:
    """Decode a Google-style encoded polyline (lon,lat order in Valhalla) to list of (lon, lat)."""
    if not encoded:
        return []
    inv = 1.0 / (10 ** precision)
    coords = []
    x = 0
    y = 0
    i = 0
    n = len(encoded)
    while i < n:
        b = 0
        shift = 0
        result = 0
        while True:
            if i >= n:
                break
            ch = ord(encoded[i]) - 63
            i += 1
            result |= (ch & 0x1F) << shift
            shift += 5
            if ch < 0x20:
                break
        dlat = ~(result >> 1) if result & 1 else result >> 1
        y += dlat
        shift = 0
        result = 0
        while True:
            if i >= n:
                break
            ch = ord(encoded[i]) - 63
            i += 1
            result |= (ch & 0x1F) << shift
            shift += 5
            if ch < 0x20:
                break
        dlng = ~(result >> 1) if result & 1 else result >> 1
        x += dlng
        coords.append((x * inv, y * inv))
    return coords


def _valhalla_costing(profile: str) -> str:
    """Map OSRM profile names to Valhalla costing."""
    p = (profile or "driving").lower()
    if p in ("driving", "car"):
        return "auto"
    if p in ("bike", "bicycle", "cycling"):
        return "bicycle"
    if p in ("foot", "walk", "walking"):
        return "pedestrian"
    if p == "truck":
        return "truck"
    return "auto"


async def get_route(
    coordinates: list[tuple[float, float]],
    profile: str = "driving",
) -> dict[str, Any]:
    """
    Get road route between coordinates from Valhalla.
    Returns the same contract as OSRM: geometry (GeoJSON), legs, distance (m), duration (s), waypoints.
    """
    if len(coordinates) < 2:
        raise ValueError("At least two coordinates required")

    base = (settings.VALHALLA_BASE_URL or "").rstrip("/")
    if not base:
        from services.osrm_service import OSRMError
        raise OSRMError("InvalidValue", "VALHALLA_BASE_URL not configured")

    costing = _valhalla_costing(profile)
    locations = [{"lat": lat, "lon": lon} for lat, lon in coordinates]
    body = {
        "locations": locations,
        "costing": costing,
        "units": "kilometers",
        "directions_options": {"units": "kilometers"},
    }
    url = f"{base}/route"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.warning("Valhalla request failed: status=%s body=%s", resp.status, text[:200])
                    from services.osrm_service import OSRMError
                    raise OSRMError("InvalidUrl", f"HTTP {resp.status}: {text[:200]}")

                data = await resp.json()
    except aiohttp.ClientError as e:
        logger.warning("Valhalla connection error: %s", e)
        from services.osrm_service import OSRMError
        raise OSRMError("InvalidUrl", str(e)) from e

    error = data.get("error_code") or data.get("error")
    if error:
        msg = data.get("error_message") or data.get("message") or str(error)
        logger.info("Valhalla error: code=%s message=%s", error, msg)
        from services.osrm_service import OSRMError
        if error in (441, 442, 443):
            raise OSRMError("NoRoute", msg)
        raise OSRMError("InvalidUrl", msg)

    trip = data.get("trip") or {}
    legs_data = trip.get("legs") or []
    summary = trip.get("summary") or {}
    units = (trip.get("units") or "kilometers").lower()
    length_km = float(summary.get("length", 0))
    duration_s = float(summary.get("time", 0))
    if "miles" in units:
        length_m = length_km * 1609.34
    else:
        length_m = length_km * 1000.0

    all_coords = []
    legs = []
    waypoints = []

    for idx, leg in enumerate(legs_data):
        shape_encoded = leg.get("shape", "")
        leg_coords = _decode_polyline(shape_encoded)
        if leg_coords:
            all_coords.extend(leg_coords)
        maneuvers = leg.get("maneuvers") or []
        leg_steps = []
        for m in maneuvers:
            instruction = m.get("instruction", "")
            dist = float(m.get("length", 0))
            if "miles" in units:
                dist_m = dist * 1609.34
            else:
                dist_m = dist * 1000.0
            leg_steps.append({
                "distance": dist_m,
                "duration": float(m.get("time", 0)),
                "maneuver": {
                    "type": m.get("type", 0),
                    "modifier": m.get("modifier", ""),
                    "instruction": instruction,
                },
                "name": (m.get("street_names") or [""])[0],
            })
        legs.append({"steps": leg_steps})
        if idx == 0 and leg_coords:
            waypoints.append({"location": leg_coords[0], "name": None})
        if leg_coords:
            waypoints.append({"location": leg_coords[-1], "name": None})

    if not all_coords and legs_data and legs_data[0].get("shape"):
        all_coords = _decode_polyline(legs_data[0].get("shape", ""))
        for leg in legs_data[1:]:
            all_coords.extend(_decode_polyline(leg.get("shape", "")))

    if not all_coords:
        from services.osrm_service import OSRMError
        raise OSRMError("NoRoute", "No route geometry returned")

    geometry = {
        "type": "LineString",
        "coordinates": [[lon, lat] for lon, lat in all_coords],
    }

    waypoints_out = []
    for i, wp in enumerate(waypoints):
        if i < len(coordinates):
            loc = wp.get("location")
            if isinstance(loc, (list, tuple)) and len(loc) >= 2:
                waypoints_out.append({"location": list(loc), "name": wp.get("name")})
            else:
                waypoints_out.append({"location": coordinates[i][::-1], "name": None})
        else:
            waypoints_out.append(wp)

    if not waypoints_out and len(coordinates) >= 2:
        waypoints_out = [
            {"location": [coordinates[0][1], coordinates[0][0]], "name": None},
            {"location": [coordinates[-1][1], coordinates[-1][0]], "name": None},
        ]

    return {
        "geometry": geometry,
        "legs": legs,
        "distance": length_m,
        "duration": duration_s,
        "waypoints": waypoints_out[: max(2, len(coordinates))],
    }


async def get_isochrone(
    origin: tuple[float, float],
    costing: str = "auto",
    contours: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Get isochrone polygons (areas reachable within given time/distance).
    contours: e.g. [{"time": 15}, {"time": 30}] for 15 and 30 minute drive times.
    """
    base = (settings.VALHALLA_BASE_URL or "").rstrip("/")
    if not base:
        return {"type": "FeatureCollection", "features": []}

    contours = contours or [{"time": 15}, {"time": 30}]
    body = {
        "locations": [{"lat": origin[0], "lon": origin[1]}],
        "costing": costing,
        "contours": contours,
    }
    url = f"{base}/isochrone"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status != 200:
                    return {"type": "FeatureCollection", "features": []}
                return await resp.json()
    except Exception as e:
        logger.warning("Valhalla isochrone error: %s", e)
        return {"type": "FeatureCollection", "features": []}
