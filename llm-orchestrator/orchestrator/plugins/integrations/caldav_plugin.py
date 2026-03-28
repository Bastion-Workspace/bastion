"""
CalDAV plugin - calendar read/write for Agent Factory (Zone 4).

Provides tools to list calendars and events, and create events. Works with
Google Calendar (via CalDAV URL), Nextcloud, iCloud, and other CalDAV servers.
Requires calendar URL, username, and password (or app password) in connection config.
Uses python-caldav (optional dependency).
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.plugins.base_plugin import BasePlugin, PluginToolSpec


def _parse_date(s: str) -> Optional[datetime]:
    """Parse ISO 8601 date or datetime string to datetime."""
    if not s:
        return None
    s = s.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s.replace("Z", "").split("+")[0], fmt.replace("Z", "").rstrip("Z"))
        except ValueError:
            continue
    return None


class ListCalendarsInputs(BaseModel):
    """Inputs for listing CalDAV calendars."""

    pass


class CalendarRef(BaseModel):
    """Reference to a CalDAV calendar."""

    id: str = Field(description="Calendar ID/URL")
    name: str = Field(description="Display name")
    url: str = Field(description="Calendar URL")


class ListCalendarsOutputs(BaseModel):
    """Outputs for list CalDAV calendars tool."""

    calendars: List[CalendarRef] = Field(description="List of calendars")
    count: int = Field(description="Number of calendars")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class ListEventsInputs(BaseModel):
    """Inputs for listing CalDAV events."""

    start_date: str = Field(description="Start date ISO 8601 (e.g. 2026-02-16)")
    end_date: str = Field(description="End date ISO 8601 (e.g. 2026-02-17)")
    calendar_id: Optional[str] = Field(default=None, description="Calendar ID; empty = default calendar")


class EventRef(BaseModel):
    """Reference to a calendar event."""

    uid: str = Field(description="Event UID")
    title: str = Field(description="Event title")
    start: str = Field(description="Start datetime ISO 8601")
    end: str = Field(description="End datetime ISO 8601")
    description: Optional[str] = Field(default=None)
    location: Optional[str] = Field(default=None)


class ListEventsOutputs(BaseModel):
    """Outputs for list CalDAV events tool."""

    events: List[EventRef] = Field(description="List of events")
    count: int = Field(description="Number of events")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class CreateEventInputs(BaseModel):
    """Inputs for creating a CalDAV event."""

    calendar_id: str = Field(description="Calendar ID to add the event to")
    title: str = Field(description="Event title")
    start_dt: str = Field(description="Start datetime ISO 8601 (e.g. 2026-02-16T09:00:00)")
    end_dt: str = Field(description="End datetime ISO 8601")
    description: str = Field(default="", description="Event description")


class CreateEventOutputs(BaseModel):
    """Outputs for create CalDAV event tool."""

    event_uid: str = Field(description="Created event UID")
    success: bool = Field(description="Whether creation succeeded")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class CalDAVPlugin(BasePlugin):
    """CalDAV integration - list calendars, list events, create events."""

    @property
    def plugin_name(self) -> str:
        return "caldav"

    @property
    def plugin_version(self) -> str:
        return "0.1.0"

    def get_connection_requirements(self) -> Dict[str, str]:
        return {
            "calendar_url": "CalDAV URL (e.g. https://caldav.calendar.google.com or Nextcloud/iCloud CalDAV URL)",
            "username": "Username or email",
            "password": "Password or app password",
        }

    def get_tools(self) -> List[PluginToolSpec]:
        return [
            PluginToolSpec(
                name="caldav_list_calendars",
                category="plugin:caldav",
                description="List CalDAV calendars for the connected account",
                inputs_model=ListCalendarsInputs,
                outputs_model=ListCalendarsOutputs,
                tool_function=self._list_calendars,
            ),
            PluginToolSpec(
                name="caldav_list_events",
                category="plugin:caldav",
                description="List calendar events in a date range",
                inputs_model=ListEventsInputs,
                outputs_model=ListEventsOutputs,
                tool_function=self._list_events,
            ),
            PluginToolSpec(
                name="caldav_create_event",
                category="plugin:caldav",
                description="Create a calendar event",
                inputs_model=CreateEventInputs,
                outputs_model=CreateEventOutputs,
                tool_function=self._create_event,
            ),
        ]

    async def _list_calendars(self) -> Dict[str, Any]:
        """List CalDAV calendars."""
        config = getattr(self, "_config", None) or {}
        url = config.get("calendar_url")
        username = config.get("username")
        password = config.get("password")
        if not url or not username or not password:
            return {
                "calendars": [],
                "count": 0,
                "formatted": "CalDAV plugin: configure calendar_url, username, and password to list calendars.",
            }
        try:
            import caldav
            client = caldav.DAVClient(url=url, username=username, password=password)

            def _list():
                principal = client.principal()
                return principal.get_calendars()

            calendars = await asyncio.to_thread(_list)
            out = []
            for cal in calendars:
                name = getattr(cal, "name", None) or (cal.url.split("/")[-1] if cal.url else "Calendar") or "Calendar"
                out.append(CalendarRef(id=cal.url, name=name, url=cal.url))
            formatted = f"Found {len(out)} calendar(s): " + ", ".join(c.name for c in out) if out else "No calendars found."
            return {"calendars": [c.model_dump() for c in out], "count": len(out), "formatted": formatted}
        except ImportError:
            return {
                "calendars": [],
                "count": 0,
                "formatted": "CalDAV plugin: install caldav package to use (pip install caldav).",
            }
        except Exception as e:
            return {"calendars": [], "count": 0, "formatted": f"CalDAV list calendars failed: {e}"}

    async def _list_events(
        self,
        start_date: str,
        end_date: str,
        calendar_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List events in date range."""
        config = getattr(self, "_config", None) or {}
        url = config.get("calendar_url")
        username = config.get("username")
        password = config.get("password")
        if not url or not username or not password:
            return {
                "events": [],
                "count": 0,
                "formatted": "CalDAV plugin: configure calendar_url, username, and password.",
            }
        try:
            import caldav
            client = caldav.DAVClient(url=url, username=username, password=password)
            start_dt = _parse_date(start_date) or datetime.now()
            end_dt = _parse_date(end_date) or datetime.now()

            def _list_evts():
                principal = client.principal()
                calendars = principal.get_calendars()
                cal = None
                if calendar_id:
                    for c in calendars:
                        if c.url == calendar_id or calendar_id in c.url:
                            cal = c
                            break
                if not cal and calendars:
                    cal = calendars[0]
                if not cal:
                    return []
                return cal.search(start=start_dt, end=end_dt, event=True, expand=True)

            events = await asyncio.to_thread(_list_evts)
            out = []
            for ev in events:
                ical = getattr(ev, "get_icalendar_instance", None) and ev.get_icalendar_instance() or getattr(ev, "icalendar_component", None)
                if not ical:
                    continue
                comp = None
                if hasattr(ical, "subcomponents"):
                    for c in ical.subcomponents:
                        if getattr(c, "name", None) == "VEVENT":
                            comp = c
                            break
                if not comp:
                    comp = ical.get("VEVENT") or (ical if getattr(ical, "name", None) == "VEVENT" else None)
                if not comp:
                    continue
                uid = str(comp.get("UID", "")) if comp.get("UID") else ""
                title = str(comp.get("SUMMARY", "")) if comp.get("SUMMARY") else ""
                dt_start = comp.get("DTSTART")
                dt_end = comp.get("DTEND")
                start_str = str(dt_start.dt) if dt_start and hasattr(dt_start, "dt") else ""
                end_str = str(dt_end.dt) if dt_end and hasattr(dt_end, "dt") else ""
                desc = str(comp.get("DESCRIPTION", "")) if comp.get("DESCRIPTION") else None
                loc = str(comp.get("LOCATION", "")) if comp.get("LOCATION") else None
                out.append(EventRef(uid=uid, title=title, start=start_str, end=end_str, description=desc, location=loc))
            formatted = f"Found {len(out)} event(s) between {start_date} and {end_date}." if out else "No events in range."
            return {"events": [e.model_dump() for e in out], "count": len(out), "formatted": formatted}
        except ImportError:
            return {"events": [], "count": 0, "formatted": "CalDAV plugin: install caldav package (pip install caldav)."}
        except Exception as e:
            return {"events": [], "count": 0, "formatted": f"CalDAV list events failed: {e}"}

    async def _create_event(
        self,
        calendar_id: str,
        title: str,
        start_dt: str,
        end_dt: str,
        description: str = "",
    ) -> Dict[str, Any]:
        """Create a calendar event."""
        config = getattr(self, "_config", None) or {}
        url = config.get("calendar_url")
        username = config.get("username")
        password = config.get("password")
        if not url or not username or not password:
            return {
                "event_uid": "",
                "success": False,
                "formatted": "CalDAV plugin: configure calendar_url, username, and password.",
            }
        try:
            import caldav
            start_dt_obj = _parse_date(start_dt)
            end_dt_obj = _parse_date(end_dt)
            if not start_dt_obj or not end_dt_obj:
                return {"event_uid": "", "success": False, "formatted": "Invalid start_dt or end_dt (use ISO 8601)."}

            client = caldav.DAVClient(url=url, username=username, password=password)

            def _create():
                principal = client.principal()
                calendars = principal.get_calendars()
                cal = None
                for c in calendars:
                    if c.url == calendar_id or calendar_id in c.url:
                        cal = c
                        break
                if not cal and calendars:
                    cal = calendars[0]
                if not cal:
                    return None
                return cal.add_event(
                    dtstart=start_dt_obj,
                    dtend=end_dt_obj,
                    summary=title,
                    description=description or "",
                )

            event = await asyncio.to_thread(_create)
            if event is None:
                return {"event_uid": "", "success": False, "formatted": "Calendar not found."}
            ical = getattr(event, "get_icalendar_instance", None) and event.get_icalendar_instance() or getattr(event, "icalendar_component", None)
            uid = ""
            if ical:
                comp = None
                if hasattr(ical, "subcomponents"):
                    for c in ical.subcomponents:
                        if getattr(c, "name", None) == "VEVENT":
                            comp = c
                            break
                if comp and comp.get("UID"):
                    uid = str(comp.get("UID"))
            return {
                "event_uid": uid,
                "success": True,
                "formatted": f"Created event: {title} ({start_dt} - {end_dt})",
            }
        except ImportError:
            return {"event_uid": "", "success": False, "formatted": "CalDAV plugin: install caldav package (pip install caldav)."}
        except Exception as e:
            return {"event_uid": "", "success": False, "formatted": f"CalDAV create event failed: {e}"}
