"""
Microsoft Graph provider for email and calendar operations.
"""

import logging
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx

from config.settings import settings
from providers.base_provider import BaseProvider

logger = logging.getLogger(__name__)

GRAPH_BASE = settings.MICROSOFT_GRAPH_BASE.rstrip("/")
TIMEOUT = settings.GRAPH_REQUEST_TIMEOUT


def _encode_message_id(message_id: str) -> str:
    """URL-encode message ID for use in path (Graph IDs can contain /, +, =)."""
    return quote(message_id, safe="")


def _message_to_dict(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Graph API message to our normalized shape."""
    from_addr = msg.get("from", {}).get("emailAddress", {})
    to_list = msg.get("toRecipients", [])
    cc_list = msg.get("ccRecipients", [])
    body_obj = msg.get("body", {}) or {}
    return {
        "id": msg.get("id", ""),
        "conversation_id": msg.get("conversationId", ""),
        "subject": msg.get("subject", ""),
        "from_address": from_addr.get("address", ""),
        "from_name": from_addr.get("name", ""),
        "to_addresses": [r.get("emailAddress", {}).get("address", "") for r in to_list],
        "cc_addresses": [r.get("emailAddress", {}).get("address", "") for r in cc_list],
        "received_datetime": msg.get("receivedDateTime", ""),
        "is_read": msg.get("isRead", False),
        "has_attachments": msg.get("hasAttachments", False),
        "importance": msg.get("importance", "normal"),
        "body_preview": msg.get("bodyPreview", ""),
        "body_content": body_obj.get("content", ""),
    }


def _folder_to_dict(f: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Graph mailFolder to our shape."""
    return {
        "id": f.get("id", ""),
        "name": f.get("displayName", ""),
        "parent_id": f.get("parentFolderId", ""),
        "unread_count": f.get("unreadItemCount", 0),
        "total_count": f.get("totalItemCount", 0),
    }


def _calendar_to_dict(c: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Graph calendar to our shape."""
    return {
        "id": c.get("id", ""),
        "name": c.get("name", ""),
        "color": c.get("hexColor", ""),
        "is_default": c.get("isDefaultCalendar", False),
        "can_edit": c.get("canEdit", True),
    }


def _event_to_dict(ev: Dict[str, Any], calendar_id: str = "") -> Dict[str, Any]:
    """Convert Graph event to our normalized shape."""
    start_obj = ev.get("start", {}) or {}
    end_obj = ev.get("end", {}) or {}
    body_obj = ev.get("body", {}) or {}
    org = ev.get("organizer", {}).get("emailAddress", {}) or {}
    attendees_raw = ev.get("attendees", [])
    attendees = []
    for a in attendees_raw:
        ea = a.get("emailAddress", {}) or {}
        if ea.get("address"):
            attendees.append({
                "email": ea.get("address", ""),
                "name": ea.get("name", ""),
                "response_status": (a.get("status", {}) or {}).get("response", "none"),
            })
    recurrence = ""
    rec = ev.get("recurrence")
    if rec and isinstance(rec, dict):
        rec_range = rec.get("range", {}) or {}
        rec_type = (rec.get("pattern", {}) or {}).get("type", "")
        if rec_type:
            recurrence = rec_type
        if rec_range.get("type"):
            recurrence = f"{recurrence} ({rec_range.get('type', '')})"
    return {
        "id": ev.get("id", ""),
        "subject": ev.get("subject", "") or "(No subject)",
        "start_datetime": start_obj.get("dateTime", ""),
        "end_datetime": end_obj.get("dateTime", ""),
        "location": (ev.get("location", {}) or {}).get("displayName", "") if isinstance(ev.get("location"), dict) else (ev.get("location") or ""),
        "body_preview": ev.get("bodyPreview", ""),
        "body_content": body_obj.get("content", ""),
        "organizer_email": org.get("address", ""),
        "organizer_name": org.get("name", ""),
        "attendees": attendees,
        "is_all_day": start_obj.get("dateTime") is None,
        "recurrence": recurrence,
        "calendar_id": calendar_id or ev.get("parentFolderId", ""),
        "web_link": ev.get("webLink", ""),
    }


def _contact_to_dict(c: Dict[str, Any], folder_id: str = "") -> Dict[str, Any]:
    """Convert Graph API contact to our normalized shape."""
    email_addresses = []
    for e in c.get("emailAddresses", []) or []:
        addr = (e or {}).get("address", "")
        if addr:
            email_addresses.append({"address": addr, "name": (e or {}).get("name", "")})
    phone_numbers = []
    for ph in (c.get("businessPhones") or []) or []:
        if ph:
            phone_numbers.append({"number": ph, "type": "business"})
    for ph in (c.get("homePhones") or []) or []:
        if ph:
            phone_numbers.append({"number": ph, "type": "home"})
    mobile = (c.get("mobilePhone") or "").strip()
    if mobile:
        phone_numbers.append({"number": mobile, "type": "mobile"})
    return {
        "id": c.get("id", ""),
        "display_name": c.get("displayName", ""),
        "given_name": c.get("givenName", ""),
        "surname": c.get("surname", ""),
        "email_addresses": email_addresses,
        "phone_numbers": phone_numbers,
        "company_name": c.get("companyName", ""),
        "job_title": c.get("jobTitle", ""),
        "birthday": c.get("birthday", ""),
        "notes": c.get("personalNotes", ""),
        "folder_id": folder_id,
    }


class MicrosoftGraphProvider(BaseProvider):
    """Microsoft Graph API implementation for email."""

    @property
    def name(self) -> str:
        return "microsoft"

    def _headers(self, access_token: str) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

    async def _get(
        self, access_token: str, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        url = f"{GRAPH_BASE}{path}"
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(url, headers=self._headers(access_token), params=params)
            resp.raise_for_status()
            return resp.json()

    async def _post(
        self, access_token: str, path: str, json: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        url = f"{GRAPH_BASE}{path}"
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(url, headers=self._headers(access_token), json=json or {})
            if resp.status_code in (200, 201, 202, 204):
                if resp.content:
                    return resp.json()
                return {}
            resp.raise_for_status()
            return resp.json() if resp.content else {}

    async def _patch(
        self, access_token: str, path: str, json: Dict[str, Any]
    ) -> None:
        url = f"{GRAPH_BASE}{path}"
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.patch(url, headers=self._headers(access_token), json=json)
            resp.raise_for_status()

    async def _delete(self, access_token: str, path: str) -> None:
        url = f"{GRAPH_BASE}{path}"
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.delete(url, headers=self._headers(access_token))
            resp.raise_for_status()

    async def get_emails(
        self,
        access_token: str,
        folder_id: str = "inbox",
        top: int = 50,
        skip: int = 0,
        filter_expr: Optional[str] = None,
        unread_only: bool = False,
    ) -> Dict[str, Any]:
        try:
            path = f"/me/mailFolders/{folder_id}/messages"
            params = {"$top": min(top, 1000), "$skip": skip}
            if unread_only:
                params["$filter"] = "isRead eq false"
            elif filter_expr:
                params["$filter"] = filter_expr
            params["$select"] = "id,conversationId,subject,from,toRecipients,ccRecipients,receivedDateTime,isRead,hasAttachments,importance,bodyPreview"
            data = await self._get(access_token, path, params)
            value = data.get("value", [])
            return {
                "messages": [_message_to_dict(m) for m in value],
                "total_count": len(value),
            }
        except httpx.HTTPStatusError as e:
            logger.exception("Graph get_emails failed: %s", e)
            return {"messages": [], "total_count": 0, "error": str(e)}
        except Exception as e:
            logger.exception("Graph get_emails error: %s", e)
            return {"messages": [], "total_count": 0, "error": str(e)}

    async def get_email_by_id(self, access_token: str, message_id: str) -> Dict[str, Any]:
        try:
            encoded_id = _encode_message_id(message_id)
            path = f"/me/messages/{encoded_id}"
            data = await self._get(access_token, path)
            return {"message": _message_to_dict(data)}
        except httpx.HTTPStatusError as e:
            logger.exception("Graph get_email_by_id failed: %s", e)
            return {"message": None, "error": str(e)}
        except Exception as e:
            logger.exception("Graph get_email_by_id error: %s", e)
            return {"message": None, "error": str(e)}

    async def get_email_thread(
        self, access_token: str, conversation_id: str
    ) -> Dict[str, Any]:
        try:
            path = "/me/messages"
            params = {
                "$filter": f"conversationId eq '{conversation_id}'",
                "$orderby": "receivedDateTime asc",
                "$top": 100,
            }
            data = await self._get(access_token, path, params)
            value = data.get("value", [])
            return {"messages": [_message_to_dict(m) for m in value]}
        except Exception as e:
            logger.exception("Graph get_email_thread error: %s", e)
            return {"messages": [], "error": str(e)}

    async def search_emails(
        self,
        access_token: str,
        query: str,
        top: int = 50,
        from_address: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            path = "/me/messages"
            params = {"$top": min(top, 1000), "$search": f'"{query}"'}
            filters = []
            if from_address:
                filters.append(f"from/emailAddress/address eq '{from_address}'")
            if start_date:
                filters.append(f"receivedDateTime ge {start_date}")
            if end_date:
                filters.append(f"receivedDateTime le {end_date}")
            if filters:
                params["$filter"] = " and ".join(filters)
            data = await self._get(access_token, path, params)
            value = data.get("value", [])
            return {"messages": [_message_to_dict(m) for m in value]}
        except Exception as e:
            logger.exception("Graph search_emails error: %s", e)
            return {"messages": [], "error": str(e)}

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
        try:
            to_list = [{"emailAddress": {"address": a}} for a in to_recipients]
            cc_list = [{"emailAddress": {"address": a}} for a in (cc_recipients or [])]
            bcc_list = [{"emailAddress": {"address": a}} for a in (bcc_recipients or [])]
            payload = {
                "message": {
                    "subject": subject,
                    "body": {"contentType": "HTML" if body_is_html else "Text", "content": body},
                    "toRecipients": to_list,
                    "ccRecipients": cc_list,
                    "bccRecipients": bcc_list,
                },
                "saveToSentItems": True,
            }
            await self._post(access_token, "/me/sendMail", json=payload)
            return {"success": True, "message_id": ""}
        except Exception as e:
            logger.exception("Graph send_email error: %s", e)
            return {"success": False, "error": str(e)}

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
        try:
            to_list = [{"emailAddress": {"address": a}} for a in to_recipients]
            cc_list = [{"emailAddress": {"address": a}} for a in (cc_recipients or [])]
            bcc_list = [{"emailAddress": {"address": a}} for a in (bcc_recipients or [])]
            payload = {
                "subject": subject,
                "body": {"contentType": "HTML" if body_is_html else "Text", "content": body},
                "toRecipients": to_list,
                "ccRecipients": cc_list,
                "bccRecipients": bcc_list,
            }
            data = await self._post(access_token, "/me/messages", json=payload)
            message_id = data.get("id", "")
            return {"success": True, "message_id": message_id}
        except Exception as e:
            logger.exception("Graph create_draft error: %s", e)
            return {"success": False, "error": str(e), "message_id": ""}

    async def reply_to_email(
        self,
        access_token: str,
        message_id: str,
        body: str,
        reply_all: bool = False,
        body_is_html: bool = False,
    ) -> Dict[str, Any]:
        try:
            path = f"/me/messages/{_encode_message_id(message_id)}/{'replyAll' if reply_all else 'reply'}"
            await self._post(access_token, path, json={"comment": body})
            return {"success": True, "message_id": message_id}
        except Exception as e:
            logger.exception("Graph reply_to_email error: %s", e)
            return {"success": False, "error": str(e)}

    async def update_email(
        self,
        access_token: str,
        message_id: str,
        is_read: Optional[bool] = None,
        importance: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            payload = {}
            if is_read is not None:
                payload["isRead"] = is_read
            if importance is not None:
                payload["importance"] = importance
            if payload:
                await self._patch(access_token, f"/me/messages/{_encode_message_id(message_id)}", payload)
            return {"success": True}
        except Exception as e:
            logger.exception("Graph update_email error: %s", e)
            return {"success": False, "error": str(e)}

    async def move_email(
        self,
        access_token: str,
        message_id: str,
        destination_folder_id: str,
    ) -> Dict[str, Any]:
        try:
            await self._post(
                access_token,
                f"/me/messages/{_encode_message_id(message_id)}/move",
                json={"destinationId": destination_folder_id},
            )
            return {"success": True}
        except Exception as e:
            logger.exception("Graph move_email error: %s", e)
            return {"success": False, "error": str(e)}

    async def delete_email(self, access_token: str, message_id: str) -> Dict[str, Any]:
        try:
            await self._delete(access_token, f"/me/messages/{_encode_message_id(message_id)}")
            return {"success": True}
        except Exception as e:
            logger.exception("Graph delete_email error: %s", e)
            return {"success": False, "error": str(e)}

    async def get_folders(self, access_token: str) -> Dict[str, Any]:
        try:
            data = await self._get(access_token, "/me/mailFolders")
            value = data.get("value", [])
            return {"folders": [_folder_to_dict(f) for f in value]}
        except Exception as e:
            logger.exception("Graph get_folders error: %s", e)
            return {"folders": [], "error": str(e)}

    async def sync_folder(
        self,
        access_token: str,
        folder_id: str,
        delta_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            path = f"/me/mailFolders/{folder_id}/messages/delta"
            params = {}
            if delta_token:
                params["$deltatoken"] = delta_token
            data = await self._get(access_token, path, params)
            value = data.get("value", [])
            added = []
            updated = []
            deleted_ids = []
            for m in value:
                if "@removed" in m:
                    deleted_ids.append(m.get("id", ""))
                else:
                    msg = _message_to_dict(m)
                    if m.get("id"):
                        updated.append(msg)
                    else:
                        added.append(msg)
            next_link = data.get("@odata.nextLink", "")
            delta_link = data.get("@odata.deltaLink", "")
            next_delta = delta_link.split("$deltatoken=")[-1] if delta_link else ""
            return {
                "added": added,
                "updated": updated,
                "deleted_ids": deleted_ids,
                "next_delta_token": next_delta,
            }
        except Exception as e:
            logger.exception("Graph sync_folder error: %s", e)
            return {"added": [], "updated": [], "deleted_ids": [], "next_delta_token": "", "error": str(e)}

    async def get_email_statistics(
        self, access_token: str, folder_id: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            fid = folder_id or "inbox"
            path = f"/me/mailFolders/{fid}"
            params = {"$select": "totalItemCount,unreadItemCount"}
            data = await self._get(access_token, path, params)
            return {
                "total_count": data.get("totalItemCount", 0),
                "unread_count": data.get("unreadItemCount", 0),
            }
        except Exception as e:
            logger.exception("Graph get_email_statistics error: %s", e)
            return {"total_count": 0, "unread_count": 0, "error": str(e)}

    # -------------------------------------------------------------------------
    # Calendar (Microsoft Graph /me/calendars, /me/events, /me/calendarView)
    # -------------------------------------------------------------------------

    async def list_calendars(self, access_token: str) -> Dict[str, Any]:
        try:
            data = await self._get(access_token, "/me/calendars")
            value = data.get("value", [])
            return {"calendars": [_calendar_to_dict(c) for c in value]}
        except Exception as e:
            logger.exception("Graph list_calendars error: %s", e)
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
            if not start_datetime or not end_datetime:
                return {"events": [], "total_count": 0, "error": "start_datetime and end_datetime are required"}
            base = f"/me/calendars/{calendar_id}" if calendar_id else "/me"
            path = f"{base}/calendarView"
            params = {
                "startDateTime": start_datetime,
                "endDateTime": end_datetime,
                "$top": min(top, 1000),
                "$select": "id,subject,start,end,location,bodyPreview,body,organizer,attendees,isAllDay,recurrence,webLink",
            }
            data = await self._get(access_token, path, params)
            value = data.get("value", [])
            return {
                "events": [_event_to_dict(ev, calendar_id) for ev in value],
                "total_count": len(value),
            }
        except Exception as e:
            logger.exception("Graph get_calendar_events error: %s", e)
            return {"events": [], "total_count": 0, "error": str(e)}

    async def get_event_by_id(self, access_token: str, event_id: str) -> Dict[str, Any]:
        try:
            path = f"/me/events/{event_id}"
            params = {"$select": "id,subject,start,end,location,bodyPreview,body,organizer,attendees,isAllDay,recurrence,webLink,parentFolderId"}
            data = await self._get(access_token, path, params)
            cal_id = data.get("parentFolderId", "")
            return {"event": _event_to_dict(data, cal_id)}
        except Exception as e:
            logger.exception("Graph get_event_by_id error: %s", e)
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
            payload = {
                "subject": subject or "(No subject)",
                "body": {"contentType": "HTML" if body_is_html else "Text", "content": body or ""},
                "isAllDay": is_all_day,
            }
            if is_all_day:
                payload["start"] = {"date": start_datetime[:10]}
                payload["end"] = {"date": end_datetime[:10]}
            else:
                payload["start"] = {"dateTime": start_datetime, "timeZone": "UTC"}
                payload["end"] = {"dateTime": end_datetime, "timeZone": "UTC"}
            if location:
                payload["location"] = {"displayName": location}
            if attendee_emails:
                payload["attendees"] = [
                    {"emailAddress": {"address": a, "name": a}, "type": "required"}
                    for a in attendee_emails if a
                ]
            base = f"/me/calendars/{calendar_id}/events" if calendar_id else "/me/events"
            data = await self._post(access_token, base, json=payload)
            return {"success": True, "event_id": data.get("id", "")}
        except Exception as e:
            logger.exception("Graph create_event error: %s", e)
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
            payload = {}
            if subject is not None:
                payload["subject"] = subject
            if start_datetime is not None:
                payload["start"] = {"dateTime": start_datetime, "timeZone": "UTC"}
            if end_datetime is not None:
                payload["end"] = {"dateTime": end_datetime, "timeZone": "UTC"}
            if location is not None:
                payload["location"] = {"displayName": location}
            if body is not None:
                payload["body"] = {"contentType": "HTML" if body_is_html else "Text", "content": body}
            if attendee_emails is not None:
                payload["attendees"] = [
                    {"emailAddress": {"address": a, "name": a}, "type": "required"}
                    for a in attendee_emails if a
                ]
            if is_all_day is not None:
                payload["isAllDay"] = is_all_day
            if payload:
                await self._patch(access_token, f"/me/events/{event_id}", payload)
            return {"success": True}
        except Exception as e:
            logger.exception("Graph update_event error: %s", e)
            return {"success": False, "error": str(e)}

    async def delete_event(self, access_token: str, event_id: str) -> Dict[str, Any]:
        try:
            await self._delete(access_token, f"/me/events/{event_id}")
            return {"success": True}
        except Exception as e:
            logger.exception("Graph delete_event error: %s", e)
            return {"success": False, "error": str(e)}

    # -------------------------------------------------------------------------
    # Contacts (Microsoft Graph /me/contacts, /me/contactFolders)
    # -------------------------------------------------------------------------

    async def get_contacts(
        self,
        access_token: str,
        folder_id: str = "",
        top: int = 100,
    ) -> Dict[str, Any]:
        try:
            if folder_id:
                path = f"/me/contactFolders/{folder_id}/contacts"
            else:
                path = "/me/contacts"
            params = {"$top": min(top, 1000), "$orderby": "displayName asc"}
            data = await self._get(access_token, path, params)
            value = data.get("value", [])
            return {
                "contacts": [_contact_to_dict(contact, folder_id) for contact in value],
                "total_count": len(value),
            }
        except Exception as e:
            logger.exception("Graph get_contacts error: %s", e)
            return {"contacts": [], "total_count": 0, "error": str(e)}

    async def get_contact_by_id(
        self, access_token: str, contact_id: str
    ) -> Dict[str, Any]:
        try:
            path = f"/me/contacts/{contact_id}"
            data = await self._get(access_token, path)
            folder_id = data.get("parentFolderId", "")
            return {"contact": _contact_to_dict(data, folder_id)}
        except Exception as e:
            logger.exception("Graph get_contact_by_id error: %s", e)
            return {"contact": None, "error": str(e)}

    async def create_contact(
        self,
        access_token: str,
        display_name: str = "",
        given_name: str = "",
        surname: str = "",
        email_addresses: Optional[List[Dict[str, str]]] = None,
        phone_numbers: Optional[List[Dict[str, str]]] = None,
        company_name: str = "",
        job_title: str = "",
        birthday: str = "",
        notes: str = "",
        folder_id: str = "",
    ) -> Dict[str, Any]:
        try:
            payload = {}
            if display_name:
                payload["displayName"] = display_name
            if given_name:
                payload["givenName"] = given_name
            if surname:
                payload["surname"] = surname
            if company_name:
                payload["companyName"] = company_name
            if job_title:
                payload["jobTitle"] = job_title
            if birthday:
                payload["birthday"] = birthday
            if notes:
                payload["personalNotes"] = notes
            if email_addresses:
                payload["emailAddresses"] = [
                    {"address": e.get("address", ""), "name": e.get("name", "")}
                    for e in email_addresses if e.get("address")
                ]
            if phone_numbers:
                business = [p["number"] for p in phone_numbers if (p.get("type") or "").lower() == "business"]
                home = [p["number"] for p in phone_numbers if (p.get("type") or "").lower() == "home"]
                mobile = ""
                for p in phone_numbers:
                    if (p.get("type") or "").lower() == "mobile":
                        mobile = p.get("number", "")
                        break
                if business:
                    payload["businessPhones"] = business
                if home:
                    payload["homePhones"] = home
                if mobile:
                    payload["mobilePhone"] = mobile
            if folder_id:
                path = f"/me/contactFolders/{folder_id}/contacts"
            else:
                path = "/me/contacts"
            data = await self._post(access_token, path, json=payload)
            return {"success": True, "contact_id": data.get("id", "")}
        except Exception as e:
            logger.exception("Graph create_contact error: %s", e)
            return {"success": False, "contact_id": "", "error": str(e)}

    async def update_contact(
        self,
        access_token: str,
        contact_id: str,
        display_name: Optional[str] = None,
        given_name: Optional[str] = None,
        surname: Optional[str] = None,
        email_addresses: Optional[List[Dict[str, str]]] = None,
        phone_numbers: Optional[List[Dict[str, str]]] = None,
        company_name: Optional[str] = None,
        job_title: Optional[str] = None,
        birthday: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            payload = {}
            if display_name is not None:
                payload["displayName"] = display_name
            if given_name is not None:
                payload["givenName"] = given_name
            if surname is not None:
                payload["surname"] = surname
            if company_name is not None:
                payload["companyName"] = company_name
            if job_title is not None:
                payload["jobTitle"] = job_title
            if birthday is not None:
                payload["birthday"] = birthday
            if notes is not None:
                payload["personalNotes"] = notes
            if email_addresses is not None:
                payload["emailAddresses"] = [
                    {"address": e.get("address", ""), "name": e.get("name", "")}
                    for e in email_addresses if e.get("address")
                ]
            if phone_numbers is not None:
                business = [p["number"] for p in phone_numbers if (p.get("type") or "").lower() == "business"]
                home = [p["number"] for p in phone_numbers if (p.get("type") or "").lower() == "home"]
                mobile = ""
                for p in phone_numbers:
                    if (p.get("type") or "").lower() == "mobile":
                        mobile = p.get("number", "")
                        break
                payload["businessPhones"] = business or []
                payload["homePhones"] = home or []
                payload["mobilePhone"] = mobile
            if payload:
                await self._patch(access_token, f"/me/contacts/{contact_id}", payload)
            return {"success": True}
        except Exception as e:
            logger.exception("Graph update_contact error: %s", e)
            return {"success": False, "error": str(e)}

    async def delete_contact(
        self, access_token: str, contact_id: str
    ) -> Dict[str, Any]:
        try:
            await self._delete(access_token, f"/me/contacts/{contact_id}")
            return {"success": True}
        except Exception as e:
            logger.exception("Graph delete_contact error: %s", e)
            return {"success": False, "error": str(e)}
