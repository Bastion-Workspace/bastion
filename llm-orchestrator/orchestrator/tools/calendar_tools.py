"""
Calendar Tools - Calendar operations via backend gRPC (O365 / Microsoft Graph).
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action
from orchestrator.utils.tool_type_models import CalendarEventRef, CalendarRef

logger = logging.getLogger(__name__)


def _resolve_calendar_date_placeholders(
    start_datetime: str,
    end_datetime: str,
    timezone: str = "UTC",
) -> tuple[str, str]:
    """Replace {today} and {today_end} with ISO 8601 datetimes in the given timezone. Empty/None treated as today range."""
    try:
        import pytz
        tz = pytz.timezone(timezone) if timezone else pytz.UTC
    except Exception:
        import pytz
        tz = pytz.UTC
    now = datetime.now(tz)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_end_dt = now.replace(hour=23, minute=59, second=59, microsecond=999999)
    start_iso = today_start.isoformat()
    end_iso = today_end_dt.isoformat()

    s = (start_datetime or "").strip()
    e = (end_datetime or "").strip()
    if s in ("{today}", "today", ""):
        s = start_iso
    if e in ("{today_end}", "today_end", ""):
        e = end_iso
    return s, e


def _parse_calendars_from_result(result: Any) -> tuple[List[Dict[str, Any]], str]:
    """Parse backend result (formatted string or dict) into list of calendar dicts and display text."""
    if isinstance(result, str):
        text = result
        calendars = []
        for line in result.strip().split("\n"):
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
            parts = line.split("|", 2)
            if len(parts) >= 2:
                id_part = parts[0].strip()
                name_part = parts[1].strip()
                id_match = re.search(r"id:\s*(\S+)", id_part)
                name_match = re.search(r"name:\s*(.+)", name_part)
                cal_id = id_match.group(1) if id_match else ""
                name = name_match.group(1).strip() if name_match else name_part
                calendars.append({"id": cal_id, "name": name, "color": "", "is_default": "(default)" in line, "can_edit": True})
        return calendars, text
    calendars = result.get("calendars") or result.get("items") or []
    text = result.get("formatted") or result.get("content") or str(result)
    return calendars, text


def _parse_events_from_result(result: Any) -> tuple[List[Dict[str, Any]], str]:
    """Parse backend result into list of event dicts and display text."""
    if isinstance(result, str):
        return [], result
    events = result.get("events") or result.get("items") or []
    text = result.get("formatted") or result.get("content") or str(result)
    out = []
    for e in events:
        out.append({
            "event_id": e.get("id") or e.get("event_id") or "",
            "subject": e.get("subject") or "",
            "start_datetime": e.get("start_datetime") or "",
            "end_datetime": e.get("end_datetime") or "",
            "location": e.get("location") or "",
            "body_preview": e.get("body_preview") or "",
            "organizer_email": e.get("organizer_email") or "",
            "organizer_name": e.get("organizer_name") or "",
            "is_all_day": e.get("is_all_day", False),
            "recurrence": e.get("recurrence") or "",
            "calendar_id": e.get("calendar_id") or "",
            "web_link": e.get("web_link") or "",
        })
    return out, text


async def list_calendars_tool(
    user_id: str = "system",
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    """List the user's calendars. Returns dict with calendars, count, formatted, content."""
    try:
        logger.info("list_calendars")
        client = await get_backend_tool_client()
        result = await client.list_calendars(user_id=user_id, connection_id=connection_id)
        if isinstance(result, str):
            calendars_list, text = _parse_calendars_from_result(result)
        else:
            calendars_list = result.get("calendars", [])
            text = result.get("formatted") or str(result)
        refs = [
            CalendarRef(
                id=c.get("id", ""),
                name=c.get("name", ""),
                color=c.get("color", ""),
                is_default=c.get("is_default", False),
                can_edit=c.get("can_edit", True),
            )
            for c in calendars_list
        ]
        return {
            "calendars": [r.model_dump() for r in refs],
            "count": len(refs),
            "formatted": text,
            "content": text,
        }
    except Exception as e:
        logger.error("list_calendars_tool error: %s", e)
        err = str(e)
        return {"calendars": [], "count": 0, "formatted": f"Error: {err}", "content": err}


async def get_calendar_events_tool(
    user_id: str = "system",
    start_datetime: str = "",
    end_datetime: str = "",
    calendar_id: str = "",
    top: int = 50,
    connection_id: Optional[int] = None,
    timezone: str = "UTC",
) -> Dict[str, Any]:
    """Get calendar events in a date range. start_datetime and end_datetime must be ISO 8601, or use {today}/{today_end} for today's range (resolved in given timezone). Returns dict with events, count, formatted, content."""
    try:
        start_datetime, end_datetime = _resolve_calendar_date_placeholders(
            start_datetime or "", end_datetime or "", timezone or "UTC"
        )
        if not start_datetime or not end_datetime:
            msg = "Error: start_datetime and end_datetime are required (ISO 8601, e.g. 2026-02-19T00:00:00, 2026-02-20T23:59:59), or use {today} and {today_end})."
            return {"events": [], "count": 0, "formatted": msg, "content": msg}
        logger.info("get_calendar_events: %s to %s", start_datetime[:19] if len(start_datetime) >= 19 else start_datetime, end_datetime[:19] if len(end_datetime) >= 19 else end_datetime)
        client = await get_backend_tool_client()
        result = await client.get_calendar_events(
            user_id=user_id,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            calendar_id=calendar_id,
            top=top,
            connection_id=connection_id,
        )
        events_list, text = _parse_events_from_result(result)
        refs = []
        for e in events_list:
            if isinstance(e, dict):
                refs.append(CalendarEventRef(
                    event_id=e.get("event_id", e.get("id", "")),
                    subject=e.get("subject", ""),
                    start_datetime=e.get("start_datetime", ""),
                    end_datetime=e.get("end_datetime", ""),
                    location=e.get("location", ""),
                    body_preview=e.get("body_preview", ""),
                    organizer_email=e.get("organizer_email", ""),
                    organizer_name=e.get("organizer_name", ""),
                    is_all_day=e.get("is_all_day", False),
                    recurrence=e.get("recurrence", ""),
                    calendar_id=e.get("calendar_id", ""),
                    web_link=e.get("web_link", ""),
                ))
        return {
            "events": [r.model_dump() for r in refs],
            "count": len(refs),
            "formatted": text,
            "content": text,
        }
    except Exception as e:
        logger.error("get_calendar_events_tool error: %s", e)
        err = str(e)
        return {"events": [], "count": 0, "formatted": f"Error: {err}", "content": err}


async def get_event_by_id_tool(
    user_id: str = "system",
    event_id: str = "",
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Get a single calendar event by event_id. Returns dict with event_id, formatted, content."""
    try:
        if not event_id:
            msg = "Error: event_id is required."
            return {"event_id": "", "formatted": msg, "content": msg}
        logger.info("get_event_by_id: %s", event_id[:50] if event_id else "")
        client = await get_backend_tool_client()
        result = await client.get_calendar_event_by_id(
            user_id=user_id,
            event_id=event_id,
            connection_id=connection_id,
        )
        text = result if isinstance(result, str) else str(result)
        return {"event_id": event_id, "formatted": text, "content": text}
    except Exception as e:
        logger.error("get_event_by_id_tool error: %s", e)
        err = str(e)
        return {"event_id": event_id, "formatted": f"Error: {err}", "content": err}


async def create_event_tool(
    user_id: str = "system",
    subject: str = "",
    start_datetime: str = "",
    end_datetime: str = "",
    confirmed: bool = False,
    connection_id: Optional[int] = None,
    calendar_id: str = "",
    location: str = "",
    body: str = "",
    attendee_emails: Optional[Union[List[str], str]] = None,
    is_all_day: bool = False,
) -> Dict[str, Any]:
    """Create a calendar event. Use confirmed=False first to show a preview, then confirmed=True after user approves. Returns dict with success, event_id, formatted, content."""
    try:
        if not start_datetime or not end_datetime:
            msg = "Error: start_datetime and end_datetime are required (ISO 8601)."
            return {"success": False, "event_id": None, "formatted": msg, "content": msg}
        attendees = attendee_emails if isinstance(attendee_emails, list) else ([s.strip() for s in str(attendee_emails or "").split(",") if s.strip()] if attendee_emails else [])
        if not confirmed:
            preview = (
                "[DRAFT - not created yet]\n"
                f"Subject: {subject or '(No subject)'}\n"
                f"Start: {start_datetime} | End: {end_datetime}\n"
                f"All-day: {is_all_day}\n"
            )
            if location:
                preview += f"Location: {location}\n"
            if attendees:
                preview += f"Attendees: {', '.join(attendees)}\n"
            if body:
                preview += f"Body: {body[:500]}{'...' if len(body) > 500 else ''}\n"
            preview += "\nReply with 'yes', 'create', or 'approve' to create this event. Otherwise say no to cancel."
            return {"success": False, "event_id": None, "formatted": preview, "content": preview}
        logger.info("create_event: subject=%s", (subject or "")[:50])
        client = await get_backend_tool_client()
        result = await client.create_calendar_event(
            user_id=user_id,
            subject=subject or "(No subject)",
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            connection_id=connection_id,
            calendar_id=calendar_id,
            location=location,
            body=body,
            body_is_html=False,
            attendee_emails=attendees if attendees else None,
            is_all_day=is_all_day,
        )
        text = result if isinstance(result, str) else str(result)
        success = "successfully" in text and "Error" not in text
        event_id = None
        if success and "event_id:" in text:
            m = re.search(r"event_id:\s*([A-Za-z0-9_-]+)", text)
            if m:
                event_id = m.group(1)
        return {"success": success, "event_id": event_id, "formatted": text, "content": text}
    except Exception as e:
        logger.error("create_event_tool error: %s", e)
        err = str(e)
        return {"success": False, "event_id": None, "formatted": f"Error: {err}", "content": err}


async def update_event_tool(
    user_id: str = "system",
    event_id: str = "",
    confirmed: bool = False,
    connection_id: Optional[int] = None,
    subject: Optional[str] = None,
    start_datetime: Optional[str] = None,
    end_datetime: Optional[str] = None,
    location: Optional[str] = None,
    body: Optional[str] = None,
    attendee_emails: Optional[Union[List[str], str]] = None,
    is_all_day: Optional[bool] = None,
) -> Dict[str, Any]:
    """Update a calendar event. Use confirmed=False first to show a preview, then confirmed=True after user approves. Returns dict with success, formatted, content."""
    try:
        if not event_id:
            msg = "Error: event_id is required."
            return {"success": False, "formatted": msg, "content": msg}
        attendees = None
        if attendee_emails is not None:
            attendees = attendee_emails if isinstance(attendee_emails, list) else [s.strip() for s in str(attendee_emails).split(",") if s.strip()]
        if not confirmed:
            preview = (
                "[DRAFT - not updated yet]\n"
                f"Updating event_id: {event_id[:50]}...\n"
            )
            if subject is not None:
                preview += f"New subject: {subject}\n"
            if start_datetime is not None:
                preview += f"New start: {start_datetime}\n"
            if end_datetime is not None:
                preview += f"New end: {end_datetime}\n"
            if location is not None:
                preview += f"New location: {location}\n"
            if body is not None:
                preview += f"New body: {body[:500]}{'...' if len(body) > 500 else ''}\n"
            if attendees is not None:
                preview += f"New attendees: {', '.join(attendees)}\n"
            if is_all_day is not None:
                preview += f"All-day: {is_all_day}\n"
            preview += "\nReply with 'yes', 'update', or 'approve' to apply. Otherwise say no to cancel."
            return {"success": False, "formatted": preview, "content": preview}
        logger.info("update_event: event_id=%s", event_id[:50])
        client = await get_backend_tool_client()
        result = await client.update_calendar_event(
            user_id=user_id,
            event_id=event_id,
            connection_id=connection_id,
            subject=subject or "",
            start_datetime=start_datetime or "",
            end_datetime=end_datetime or "",
            location=location or "",
            body=body or "",
            body_is_html=False,
            attendee_emails=attendees,
            is_all_day=is_all_day,
        )
        text = result if isinstance(result, str) else str(result)
        success = text == "Event updated successfully." or ("updated successfully" in text and "Error" not in text)
        return {"success": success, "formatted": text, "content": text}
    except Exception as e:
        logger.error("update_event_tool error: %s", e)
        err = str(e)
        return {"success": False, "formatted": f"Error: {err}", "content": err}


async def delete_event_tool(
    user_id: str = "system",
    event_id: str = "",
    confirmed: bool = False,
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Delete a calendar event. Use confirmed=False first to show a preview, then confirmed=True after user approves. Returns dict with success, formatted, content."""
    try:
        if not event_id:
            msg = "Error: event_id is required."
            return {"success": False, "formatted": msg, "content": msg}
        if not confirmed:
            preview = (
                "[DRAFT - not deleted yet]\n"
                f"Delete event_id: {event_id[:50]}...\n\n"
                "Reply with 'yes', 'delete', or 'approve' to delete this event. Otherwise say no to cancel."
            )
            return {"success": False, "formatted": preview, "content": preview}
        logger.info("delete_event: event_id=%s", event_id[:50])
        client = await get_backend_tool_client()
        result = await client.delete_calendar_event(
            user_id=user_id,
            event_id=event_id,
            connection_id=connection_id,
        )
        text = result if isinstance(result, str) else str(result)
        success = text == "Event deleted successfully." or ("deleted successfully" in text and "Error" not in text)
        return {"success": success, "formatted": text, "content": text}
    except Exception as e:
        logger.error("delete_event_tool error: %s", e)
        err = str(e)
        return {"success": False, "formatted": f"Error: {err}", "content": err}


# ----- I/O models and Action I/O Registry -----

class ListCalendarsOutputs(BaseModel):
    calendars: List[CalendarRef] = Field(default_factory=list, description="List of calendars")
    count: int = Field(description="Number of calendars")
    formatted: str = Field(description="Human-readable result")
    content: str = Field(default="", description="Raw or structured content")


class GetCalendarEventsOutputs(BaseModel):
    events: List[CalendarEventRef] = Field(default_factory=list, description="Events in range")
    count: int = Field(description="Number of events")
    formatted: str = Field(description="Human-readable result")
    content: str = Field(default="", description="Raw or structured content")


class GetEventByIdOutputs(BaseModel):
    event_id: str = Field(description="Event ID that was read")
    formatted: str = Field(description="Human-readable full event content")
    content: str = Field(default="", description="Raw or structured content")


class CreateEventOutputs(BaseModel):
    success: bool = Field(description="Whether create succeeded")
    event_id: Optional[str] = Field(default=None, description="New event ID if created")
    formatted: str = Field(description="Human-readable result")
    content: str = Field(default="", description="Raw or structured content")


class UpdateEventOutputs(BaseModel):
    success: bool = Field(description="Whether update succeeded")
    formatted: str = Field(description="Human-readable result")
    content: str = Field(default="", description="Raw or structured content")


class DeleteEventOutputs(BaseModel):
    success: bool = Field(description="Whether delete succeeded")
    formatted: str = Field(description="Human-readable result")
    content: str = Field(default="", description="Raw or structured content")


class ListCalendarsInputs(BaseModel):
    pass


class GetCalendarEventsInputs(BaseModel):
    start_datetime: str = Field(default="", description="Start of range (ISO 8601). Use {today} for today's midnight.")
    end_datetime: str = Field(default="", description="End of range (ISO 8601). Use {today_end} for tonight at 23:59:59.")
    calendar_id: str = Field(default="", description="Calendar ID; empty = default calendar")
    top: int = Field(default=50, description="Max events to return")
    timezone: str = Field(default="UTC", description="IANA timezone for resolving {today}/{today_end}; e.g. America/New_York.")


class GetEventByIdInputs(BaseModel):
    event_id: str = Field(description="Event ID from get_calendar_events or list")


class CreateEventInputs(BaseModel):
    subject: str = Field(description="Event subject")
    start_datetime: str = Field(description="Start (ISO 8601)")
    end_datetime: str = Field(description="End (ISO 8601)")
    confirmed: bool = Field(default=False, description="True to create, False for draft preview")
    calendar_id: str = Field(default="", description="Calendar ID; empty = default")
    location: str = Field(default="", description="Location")
    body: str = Field(default="", description="Body text")
    attendee_emails: Optional[str] = Field(default=None, description="Comma-separated emails")
    is_all_day: bool = Field(default=False, description="All-day event")


class UpdateEventInputs(BaseModel):
    event_id: str = Field(description="Event ID to update")
    confirmed: bool = Field(default=False, description="True to apply update")
    subject: Optional[str] = Field(default=None, description="New subject")
    start_datetime: Optional[str] = Field(default=None, description="New start (ISO 8601)")
    end_datetime: Optional[str] = Field(default=None, description="New end (ISO 8601)")
    location: Optional[str] = Field(default=None, description="New location")
    body: Optional[str] = Field(default=None, description="New body")
    attendee_emails: Optional[str] = Field(default=None, description="Comma-separated emails")
    is_all_day: Optional[bool] = Field(default=None, description="All-day event")


class DeleteEventInputs(BaseModel):
    event_id: str = Field(description="Event ID to delete")
    confirmed: bool = Field(default=False, description="True to delete")


register_action(name="list_calendars", category="calendar", description="List user's calendars", inputs_model=ListCalendarsInputs, outputs_model=ListCalendarsOutputs, tool_function=list_calendars_tool)
register_action(name="get_calendar_events", category="calendar", description="Get calendar events in a date range", inputs_model=GetCalendarEventsInputs, outputs_model=GetCalendarEventsOutputs, tool_function=get_calendar_events_tool)
register_action(name="get_event_by_id", category="calendar", description="Get a single calendar event by ID", inputs_model=GetEventByIdInputs, outputs_model=GetEventByIdOutputs, tool_function=get_event_by_id_tool)
register_action(name="create_event", category="calendar", description="Create a calendar event", inputs_model=CreateEventInputs, outputs_model=CreateEventOutputs, tool_function=create_event_tool)
register_action(name="update_event", category="calendar", description="Update a calendar event", inputs_model=UpdateEventInputs, outputs_model=UpdateEventOutputs, tool_function=update_event_tool)
register_action(name="delete_event", category="calendar", description="Delete a calendar event", inputs_model=DeleteEventInputs, outputs_model=DeleteEventOutputs, tool_function=delete_event_tool)
