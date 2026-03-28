"""
CalDAV provider for calendar operations (Google Calendar via CalDAV, Nextcloud, iCloud, etc.).
access_token is JSON: {url, username, password}.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from providers.base_provider import BaseProvider

logger = logging.getLogger(__name__)

_EMAIL_MSG = "CalDAV provider does not support email operations."


def _parse_credentials(access_token: str) -> Dict[str, Any]:
    try:
        data = json.loads(access_token)
        if not isinstance(data, dict):
            raise ValueError("credentials must be a JSON object")
        return data
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("CalDAV invalid credentials JSON: %s", e)
        raise ValueError("Invalid CalDAV credentials") from e


def _parse_date(s: str) -> Optional[datetime]:
    if not s:
        return None
    s = s.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s.replace("Z", "").split("+")[0], fmt.replace("Z", "").rstrip("Z"))
        except ValueError:
            continue
    return None


def _caldav_event_to_dict(ev, calendar_id: str = "") -> Dict[str, Any]:
    """Convert CalDAV event (icalendar VEVENT) to normalized event dict."""
    ical = getattr(ev, "get_icalendar_instance", None) and ev.get_icalendar_instance() or getattr(ev, "icalendar_component", None)
    if not ical:
        return {}
    comp = None
    if hasattr(ical, "subcomponents"):
        for c in ical.subcomponents:
            if getattr(c, "name", None) == "VEVENT":
                comp = c
                break
    if not comp:
        comp = ical.get("VEVENT") or (ical if getattr(ical, "name", None) == "VEVENT" else None)
    if not comp:
        return {}
    uid = str(comp.get("UID", "")) if comp.get("UID") else ""
    title = str(comp.get("SUMMARY", "")) if comp.get("SUMMARY") else "(No subject)"
    dt_start = comp.get("DTSTART")
    dt_end = comp.get("DTEND")
    start_str = str(dt_start.dt) if dt_start and hasattr(dt_start, "dt") else ""
    end_str = str(dt_end.dt) if dt_end and hasattr(dt_end, "dt") else ""
    desc = str(comp.get("DESCRIPTION", "")) if comp.get("DESCRIPTION") else ""
    loc = str(comp.get("LOCATION", "")) if comp.get("LOCATION") else ""
    return {
        "id": uid,
        "subject": title,
        "start_datetime": start_str,
        "end_datetime": end_str,
        "location": loc,
        "body_preview": (desc[:200] + "…") if len(desc) > 200 else desc,
        "body_content": desc,
        "organizer_email": "",
        "organizer_name": "",
        "attendees": [],
        "is_all_day": False,
        "recurrence": "",
        "calendar_id": calendar_id,
        "web_link": "",
    }


class CalDAVProvider(BaseProvider):
    """CalDAV calendar provider. Email methods return error."""

    @property
    def name(self) -> str:
        return "caldav"

    def _creds(self, access_token: str) -> Dict[str, Any]:
        return _parse_credentials(access_token)

    def _client(self, access_token: str):
        creds = self._creds(access_token)
        url = creds.get("url", "").strip()
        username = creds.get("username", "").strip()
        password = creds.get("password", "")
        if not url or not username or not password:
            raise ValueError("url, username, and password required")
        try:
            import caldav
        except ImportError:
            raise RuntimeError("caldav is not installed; add it to connections-service requirements")
        return caldav.DAVClient(url=url, username=username, password=password)

    async def list_calendars(self, access_token: str) -> Dict[str, Any]:
        try:
            client = self._client(access_token)

            def _list():
                principal = client.principal()
                return principal.get_calendars()

            calendars = await asyncio.to_thread(_list)
            out = []
            for cal in calendars:
                name = getattr(cal, "name", None) or (cal.url.split("/")[-1] if cal.url else "Calendar") or "Calendar"
                out.append({
                    "id": cal.url,
                    "name": name,
                    "color": "",
                    "is_default": len(out) == 0,
                    "can_edit": True,
                })
            return {"calendars": out}
        except Exception as e:
            logger.exception("CalDAV list_calendars: %s", e)
            return {"calendars": [], "error": str(e)}

    async def get_calendar_events(
        self,
        access_token: str,
        calendar_id: str = "",
        start_datetime: str = "",
        end_datetime: str = "",
        top: int = 50,
    ) -> Dict[str, Any]:
        try:
            client = self._client(access_token)
            start_dt = _parse_date(start_datetime) or datetime.now()
            end_dt = _parse_date(end_datetime) or datetime.now()

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
            for ev in events[:top]:
                d = _caldav_event_to_dict(ev, calendar_id or (events[0].url if events else ""))
                if d:
                    out.append(d)
            return {"events": out, "total_count": len(out)}
        except Exception as e:
            logger.exception("CalDAV get_calendar_events: %s", e)
            return {"events": [], "total_count": 0, "error": str(e)}

    async def get_event_by_id(self, access_token: str, event_id: str) -> Dict[str, Any]:
        try:
            client = self._client(access_token)

            def _find():
                principal = client.principal()
                for cal in principal.get_calendars():
                    events = cal.search(uid=event_id)
                    if events:
                        return events[0]
                return None

            ev = await asyncio.to_thread(_find)
            if not ev:
                return {"event": None, "error": "Event not found"}
            d = _caldav_event_to_dict(ev)
            return {"event": d}
        except Exception as e:
            logger.exception("CalDAV get_event_by_id: %s", e)
            return {"event": None, "error": str(e)}

    async def create_event(
        self,
        access_token: str,
        subject: str,
        start_datetime: str,
        end_datetime: str,
        location: str = "",
        body: str = "",
        body_is_html: bool = False,
        attendee_emails: Optional[List[str]] = None,
        is_all_day: bool = False,
        calendar_id: str = "",
    ) -> Dict[str, Any]:
        try:
            client = self._client(access_token)
            start_dt = _parse_date(start_datetime)
            end_dt = _parse_date(end_datetime)
            if not start_dt or not end_dt:
                return {"success": False, "event_id": "", "error": "Invalid start_datetime or end_datetime (use ISO 8601)"}

            def _create():
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
                    return None
                return cal.add_event(
                    dtstart=start_dt,
                    dtend=end_dt,
                    summary=subject or "(No subject)",
                    description=body or "",
                )

            event = await asyncio.to_thread(_create)
            if event is None:
                return {"success": False, "event_id": "", "error": "Calendar not found"}
            ical = getattr(event, "get_icalendar_instance", None) and event.get_icalendar_instance() or getattr(event, "icalendar_component", None)
            uid = ""
            if ical and hasattr(ical, "subcomponents"):
                for c in ical.subcomponents:
                    if getattr(c, "name", None) == "VEVENT" and c.get("UID"):
                        uid = str(c.get("UID"))
                        break
            return {"success": True, "event_id": uid}
        except Exception as e:
            logger.exception("CalDAV create_event: %s", e)
            return {"success": False, "event_id": "", "error": str(e)}

    async def update_event(
        self,
        access_token: str,
        event_id: str,
        subject: Optional[str] = None,
        start_datetime: Optional[str] = None,
        end_datetime: Optional[str] = None,
        location: Optional[str] = None,
        body: Optional[str] = None,
        body_is_html: bool = False,
        attendee_emails: Optional[List[str]] = None,
        is_all_day: Optional[bool] = None,
    ) -> Dict[str, Any]:
        try:
            client = self._client(access_token)

            def _update():
                principal = client.principal()
                for cal in principal.get_calendars():
                    events = cal.search(uid=event_id)
                    if events:
                        ev = events[0]
                        ev.load()
                        ical = getattr(ev, "get_icalendar_instance", None) and ev.get_icalendar_instance() or getattr(ev, "icalendar_component", None)
                        if not ical:
                            return False
                        comp = None
                        if hasattr(ical, "subcomponents"):
                            for c in ical.subcomponents:
                                if getattr(c, "name", None) == "VEVENT":
                                    comp = c
                                    break
                        if not comp:
                            return False
                        if subject is not None:
                            comp["SUMMARY"] = subject
                        if start_datetime is not None:
                            comp["DTSTART"] = _parse_date(start_datetime)
                        if end_datetime is not None:
                            comp["DTEND"] = _parse_date(end_datetime)
                        if location is not None:
                            comp["LOCATION"] = location
                        if body is not None:
                            comp["DESCRIPTION"] = body
                        ev.save()
                        return True
                return False

            ok = await asyncio.to_thread(_update)
            return {"success": ok}
        except Exception as e:
            logger.exception("CalDAV update_event: %s", e)
            return {"success": False, "error": str(e)}

    async def delete_event(self, access_token: str, event_id: str) -> Dict[str, Any]:
        try:
            client = self._client(access_token)

            def _delete():
                principal = client.principal()
                for cal in principal.get_calendars():
                    events = cal.search(uid=event_id)
                    if events:
                        events[0].delete()
                        return True
                return False

            ok = await asyncio.to_thread(_delete)
            return {"success": ok}
        except Exception as e:
            logger.exception("CalDAV delete_event: %s", e)
            return {"success": False, "error": str(e)}

    async def get_emails(
        self,
        access_token: str,
        folder_id: str = "inbox",
        top: int = 50,
        skip: int = 0,
        filter_expr: Optional[str] = None,
        unread_only: bool = False,
    ) -> Dict[str, Any]:
        return {"messages": [], "total_count": 0, "error": _EMAIL_MSG}

    async def get_email_by_id(self, access_token: str, message_id: str) -> Dict[str, Any]:
        return {"message": None, "error": _EMAIL_MSG}

    async def get_email_thread(
        self, access_token: str, conversation_id: str
    ) -> Dict[str, Any]:
        return {"messages": [], "error": _EMAIL_MSG}

    async def search_emails(
        self,
        access_token: str,
        query: str,
        top: int = 50,
        from_address: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {"messages": [], "error": _EMAIL_MSG}

    async def send_email(
        self,
        access_token: str,
        to_recipients: List[str],
        subject: str,
        body: str,
        cc_recipients: Optional[List[str]] = None,
        bcc_recipients: Optional[List[str]] = None,
        body_is_html: bool = False,
    ) -> Dict[str, Any]:
        return {"success": False, "error": _EMAIL_MSG}

    async def create_draft(
        self,
        access_token: str,
        to_recipients: List[str],
        subject: str,
        body: str,
        cc_recipients: Optional[List[str]] = None,
        bcc_recipients: Optional[List[str]] = None,
        body_is_html: bool = False,
    ) -> Dict[str, Any]:
        return {"success": False, "error": _EMAIL_MSG}

    async def reply_to_email(
        self,
        access_token: str,
        message_id: str,
        body: str,
        reply_all: bool = False,
        body_is_html: bool = False,
    ) -> Dict[str, Any]:
        return {"success": False, "error": _EMAIL_MSG}

    async def update_email(
        self,
        access_token: str,
        message_id: str,
        is_read: Optional[bool] = None,
        importance: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {"success": False, "error": _EMAIL_MSG}

    async def move_email(
        self,
        access_token: str,
        message_id: str,
        destination_folder_id: str,
    ) -> Dict[str, Any]:
        return {"success": False, "error": _EMAIL_MSG}

    async def delete_email(self, access_token: str, message_id: str) -> Dict[str, Any]:
        return {"success": False, "error": _EMAIL_MSG}

    async def get_folders(self, access_token: str) -> Dict[str, Any]:
        return {"folders": [], "error": _EMAIL_MSG}

    async def sync_folder(
        self,
        access_token: str,
        folder_id: str,
        delta_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {"added": [], "updated": [], "deleted_ids": [], "next_delta_token": "", "error": _EMAIL_MSG}

    async def get_email_statistics(
        self, access_token: str, folder_id: Optional[str] = None
    ) -> Dict[str, Any]:
        return {"total_count": 0, "unread_count": 0, "error": _EMAIL_MSG}
