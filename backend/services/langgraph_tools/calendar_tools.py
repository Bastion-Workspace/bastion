"""
Calendar Tools - LangGraph tools for calendar operations via connections-service.
Returns formatted strings for LLM consumption.
"""

import logging
from typing import Any, Dict, List, Optional

from clients.connections_service_client import get_connections_service_client

logger = logging.getLogger(__name__)


def _format_calendars(calendars: List[Dict[str, Any]]) -> str:
    if not calendars:
        return "No calendars found."
    lines = []
    for i, c in enumerate(calendars, 1):
        default = " (default)" if c.get("is_default") else ""
        lines.append(f"{i}. id: {c.get('id', '')} | name: {c.get('name', '')}{default}")
    return "\n".join(lines)


def _format_events(events: List[Dict[str, Any]], max_preview: int = 150) -> str:
    if not events:
        return "No events in this range."
    lines = []
    for i, e in enumerate(events, 1):
        start = e.get("start_datetime", "")[:19] if e.get("start_datetime") else ""
        end = e.get("end_datetime", "")[:19] if e.get("end_datetime") else ""
        loc = (e.get("location") or "").strip()
        loc_str = f" | {loc}" if loc else ""
        preview = (e.get("body_preview") or "")[:max_preview]
        if (e.get("body_preview") or "").strip() and len((e.get("body_preview") or "")) > max_preview:
            preview += "..."
        block = (
            f"{i}. event_id: {e.get('id', '')}\n"
            f"   Subject: {e.get('subject', '(No subject)')}\n"
            f"   Start: {start} | End: {end}{loc_str}\n"
        )
        if preview:
            block += f"   Preview: {preview}\n"
        lines.append(block)
    lines.append("")
    lines.append("To get full details for an event, call get_event_by_id with the event_id above.")
    return "\n".join(lines)


def _format_event_detail(e: Dict[str, Any]) -> str:
    if not e:
        return "Event not found."
    start = e.get("start_datetime", "")
    end = e.get("end_datetime", "")
    loc = (e.get("location") or "").strip()
    body = (e.get("body_content") or e.get("body_preview") or "").strip()
    attendees = e.get("attendees") or []
    att_str = ", ".join([a.get("email") or a.get("name") or "" for a in attendees if a.get("email") or a.get("name")])
    lines = [
        f"event_id: {e.get('id', '')}",
        f"Subject: {e.get('subject', '(No subject)')}",
        f"Start: {start}",
        f"End: {end}",
        f"All-day: {e.get('is_all_day', False)}",
    ]
    if loc:
        lines.append(f"Location: {loc}")
    if att_str:
        lines.append(f"Attendees: {att_str}")
    if e.get("recurrence"):
        lines.append(f"Recurrence: {e.get('recurrence')}")
    if body:
        lines.append(f"Body: {body[:2000]}{'...' if len(body) > 2000 else ''}")
    if e.get("web_link"):
        lines.append(f"Link: {e.get('web_link')}")
    return "\n".join(lines)


async def list_calendars(
    user_id: str,
    connection_id: Optional[int] = None,
) -> str:
    """List the user's calendars. Returns a formatted summary for the LLM."""
    try:
        client = await get_connections_service_client()
        result = await client.list_calendars(
            user_id=user_id,
            connection_id=connection_id,
        )
        if result.get("error") and not result.get("calendars"):
            return f"Error: {result.get('error', 'Failed to list calendars')}. Ensure a calendar-capable connection (e.g. Microsoft) is configured in Settings."
        calendars = result.get("calendars", [])
        return _format_calendars(calendars)
    except Exception as e:
        logger.exception("list_calendars failed: %s", e)
        return f"Error listing calendars: {e}"


async def get_calendar_events(
    user_id: str,
    start_datetime: str,
    end_datetime: str,
    calendar_id: str = "",
    top: int = 50,
    connection_id: Optional[int] = None,
) -> str:
    """Get calendar events in a date range. start_datetime and end_datetime must be ISO 8601 (e.g. 2026-02-19T00:00:00, 2026-02-20T23:59:59). Returns formatted summary for the LLM."""
    try:
        client = await get_connections_service_client()
        result = await client.get_calendar_events(
            user_id=user_id,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            calendar_id=calendar_id,
            top=top,
            connection_id=connection_id,
        )
        if result.get("error") and not result.get("events"):
            return f"Error: {result.get('error', 'Failed to get events')}. Ensure a calendar connection is configured."
        events = result.get("events", [])
        return _format_events(events)
    except Exception as e:
        logger.exception("get_calendar_events failed: %s", e)
        return f"Error fetching calendar events: {e}"


async def get_event_by_id(
    user_id: str,
    event_id: str,
    connection_id: Optional[int] = None,
) -> str:
    """Get a single calendar event by event_id. Returns full event details for the LLM."""
    try:
        client = await get_connections_service_client()
        result = await client.get_event_by_id(
            user_id=user_id,
            event_id=event_id,
            connection_id=connection_id,
        )
        if result.get("error") and not result.get("event"):
            return f"Error: {result.get('error', 'Event not found')}."
        return _format_event_detail(result.get("event"))
    except Exception as e:
        logger.exception("get_event_by_id failed: %s", e)
        return f"Error fetching event: {e}"


async def create_event(
    user_id: str,
    subject: str,
    start_datetime: str,
    end_datetime: str,
    connection_id: Optional[int] = None,
    calendar_id: str = "",
    location: str = "",
    body: str = "",
    body_is_html: bool = False,
    attendee_emails: Optional[List[str]] = None,
    is_all_day: bool = False,
) -> str:
    """Create a calendar event. start_datetime and end_datetime must be ISO 8601. Returns success or error message."""
    try:
        client = await get_connections_service_client()
        result = await client.create_event(
            user_id=user_id,
            subject=subject or "(No subject)",
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            connection_id=connection_id,
            calendar_id=calendar_id,
            location=location,
            body=body,
            body_is_html=body_is_html,
            attendee_emails=attendee_emails,
            is_all_day=is_all_day,
        )
        if result.get("success"):
            eid = result.get("event_id", "")
            return f"Event created successfully. event_id: {eid}"
        return f"Failed to create event: {result.get('error', 'Unknown error')}"
    except Exception as e:
        logger.exception("create_event failed: %s", e)
        return f"Error creating event: {e}"


async def update_event(
    user_id: str,
    event_id: str,
    connection_id: Optional[int] = None,
    subject: Optional[str] = None,
    start_datetime: Optional[str] = None,
    end_datetime: Optional[str] = None,
    location: Optional[str] = None,
    body: Optional[str] = None,
    body_is_html: bool = False,
    attendee_emails: Optional[List[str]] = None,
    is_all_day: Optional[bool] = None,
) -> str:
    """Update a calendar event. Only provided fields are updated. Returns success or error message."""
    try:
        client = await get_connections_service_client()
        result = await client.update_event(
            user_id=user_id,
            event_id=event_id,
            connection_id=connection_id,
            subject=subject,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            location=location,
            body=body,
            body_is_html=body_is_html,
            attendee_emails=attendee_emails,
            is_all_day=is_all_day,
        )
        if result.get("success"):
            return "Event updated successfully."
        return f"Failed to update event: {result.get('error', 'Unknown error')}"
    except Exception as e:
        logger.exception("update_event failed: %s", e)
        return f"Error updating event: {e}"


async def delete_event(
    user_id: str,
    event_id: str,
    connection_id: Optional[int] = None,
) -> str:
    """Delete a calendar event. Returns success or error message."""
    try:
        client = await get_connections_service_client()
        result = await client.delete_event(
            user_id=user_id,
            event_id=event_id,
            connection_id=connection_id,
        )
        if result.get("success"):
            return "Event deleted successfully."
        return f"Failed to delete event: {result.get('error', 'Unknown error')}"
    except Exception as e:
        logger.exception("delete_event failed: %s", e)
        return f"Error deleting event: {e}"
