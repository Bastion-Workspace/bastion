"""
Calendar API - REST endpoints for Agenda view (O365, future CalDAV).
Exposes list connections, list calendars, and get events for the frontend.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query

from utils.auth_middleware import get_current_user, AuthenticatedUserResponse
from services.external_connections_service import external_connections_service
from clients.connections_service_client import get_connections_service_client

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Calendar"])


@router.get("/api/calendar/connections")
async def list_calendar_connections(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """
    List current user's calendar-capable connections (Microsoft O365 and CalDAV).
    Returns connection id, provider, and display name for the Agenda source selector.
    """
    try:
        out = []
        microsoft = await external_connections_service.get_user_connections(
            current_user.user_id,
            provider="microsoft",
            connection_type="email",
            active_only=True,
        )
        for c in microsoft:
            out.append({
                "id": c["id"],
                "provider": c.get("provider", "microsoft"),
                "display_name": c.get("display_name") or c.get("account_identifier") or "Microsoft",
            })
        caldav = await external_connections_service.get_user_connections(
            current_user.user_id,
            provider="caldav",
            connection_type="calendar",
            active_only=True,
        )
        for c in caldav:
            out.append({
                "id": c["id"],
                "provider": c.get("provider", "caldav"),
                "display_name": c.get("display_name") or c.get("account_identifier") or "CalDAV",
            })
        return {"connections": out, "error": None}
    except Exception as e:
        logger.exception("list_calendar_connections failed: %s", e)
        return {"connections": [], "error": str(e)}


@router.get("/api/calendar/calendars")
async def list_calendars(
    connection_id: Optional[int] = Query(None, description="Connection ID (default: first Microsoft)"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """
    List calendars for the given connection (or first Microsoft connection).
    """
    try:
        client = await get_connections_service_client()
        result = await client.list_calendars(
            user_id=current_user.user_id,
            connection_id=connection_id,
            rls_context={"user_id": current_user.user_id},
        )
        if result.get("error") and not result.get("calendars"):
            return {
                "calendars": [],
                "error": result.get("error", "No calendar connection or token"),
            }
        return {
            "calendars": result.get("calendars", []),
            "error": result.get("error"),
        }
    except Exception as e:
        logger.exception("list_calendars failed: %s", e)
        return {"calendars": [], "error": str(e)}


@router.get("/api/calendar/events")
async def get_calendar_events(
    start: str = Query(..., description="Start datetime ISO 8601 (e.g. 2026-02-20T00:00:00)"),
    end: str = Query(..., description="End datetime ISO 8601 (e.g. 2026-02-27T23:59:59)"),
    connection_id: Optional[int] = Query(None, description="Connection ID"),
    calendar_id: Optional[str] = Query("", description="Calendar ID (empty = default)"),
    top: int = Query(50, ge=1, le=500, description="Max events"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """
    Get calendar events in a date range for the Agenda view.
    """
    try:
        client = await get_connections_service_client()
        result = await client.get_calendar_events(
            user_id=current_user.user_id,
            start_datetime=start,
            end_datetime=end,
            connection_id=connection_id,
            calendar_id=calendar_id or "",
            top=top,
            rls_context={"user_id": current_user.user_id},
        )
        if result.get("error") and not result.get("events"):
            return {
                "events": [],
                "total_count": 0,
                "error": result.get("error", "Failed to load events"),
            }
        return {
            "events": result.get("events", []),
            "total_count": result.get("total_count", 0),
            "error": result.get("error"),
        }
    except Exception as e:
        logger.exception("get_calendar_events failed: %s", e)
        return {"events": [], "total_count": 0, "error": str(e)}
