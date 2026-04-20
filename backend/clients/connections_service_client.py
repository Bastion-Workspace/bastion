"""
Connections Service gRPC Client - Email and external API operations.
"""

import grpc
import json
import logging
import os
from typing import Any, Dict, List, Optional

import sys
from pathlib import Path

from config import settings
try:
    from protos import connections_service_pb2, connections_service_pb2_grpc
except ImportError:
    # Fallback: generated _pb2 modules may be in protos dir or at app root
    _app = Path(__file__).resolve().parent.parent
    _protos_dir = _app / "protos"
    if _protos_dir.is_dir() and str(_protos_dir) not in sys.path:
        sys.path.insert(0, str(_protos_dir))
    import connections_service_pb2
    import connections_service_pb2_grpc
from services.external_connections_service import external_connections_service

logger = logging.getLogger(__name__)


class ConnectionsServiceClient:
    """Client for the Connections Service (email, etc.) via gRPC."""

    def __init__(
        self,
        service_host: Optional[str] = None,
        service_port: Optional[int] = None,
    ):
        self.service_host = service_host or os.getenv(
            "CONNECTIONS_SERVICE_HOST", "connections-service"
        )
        self.service_port = service_port or int(
            os.getenv("CONNECTIONS_SERVICE_PORT", "50057")
        )
        self.service_url = f"{self.service_host}:{self.service_port}"
        self.channel: Optional[grpc.aio.Channel] = None
        self.stub: Optional[connections_service_pb2_grpc.ConnectionsServiceStub] = None
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        options = [
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ]
        self.channel = grpc.aio.insecure_channel(self.service_url, options=options)
        self.stub = connections_service_pb2_grpc.ConnectionsServiceStub(self.channel)
        req = connections_service_pb2.HealthCheckRequest()
        await self.stub.HealthCheck(req, timeout=5.0)
        self._initialized = True
        logger.info("Connected to Connections Service at %s", self.service_url)

    async def close(self) -> None:
        if self.channel:
            await self.channel.close()
        self.channel = None
        self.stub = None
        self._initialized = False

    async def _with_reconnect(self, coro):
        """Run coroutine; on gRPC failure reset channel and retry once."""
        try:
            return await coro()
        except grpc.RpcError as e:
            logger.warning("Connections service RPC failed, reconnecting: %s", e)
            self._initialized = False
            if self.channel:
                try:
                    await self.channel.close()
                except Exception:
                    pass
                self.channel = None
                self.stub = None
            return await coro()

    @staticmethod
    def _m365_service_allowed(
        conn: Optional[Dict[str, Any]], m365_service: Optional[str]
    ) -> bool:
        if not m365_service:
            return True
        if not conn:
            return False
        if (conn.get("provider") or "").lower() != "microsoft":
            return True
        enabled = external_connections_service.get_enabled_services_from_metadata(
            conn.get("provider_metadata")
        )
        return m365_service in enabled

    async def _ensure_token(
        self,
        connection_id: Optional[int],
        user_id: str,
        provider: str = "microsoft",
        rls_context: Optional[Dict[str, str]] = None,
        m365_service: Optional[str] = None,
    ) -> tuple[Optional[str], Optional[int], str]:
        """Resolve connection_id or user's first email connection; return (access_token, connection_id, provider)."""
        if connection_id:
            conn = await external_connections_service.get_connection_by_id(
                connection_id, rls_context=rls_context
            )
            if not conn or conn.get("user_id") != user_id:
                return None, None, provider
            token = await external_connections_service.get_valid_access_token(
                connection_id, rls_context=rls_context
            )
            if token and not self._m365_service_allowed(conn, m365_service):
                return None, connection_id, conn.get("provider", provider)
            return token, connection_id, conn.get("provider", provider)
        connections = await external_connections_service.get_user_connections(
            user_id,
            provider=provider,
            connection_type="email",
            active_only=True,
            rls_context=rls_context,
        )
        if not connections:
            return None, None, provider
        conn = connections[0]
        cid = conn["id"]
        token = await external_connections_service.get_valid_access_token(
            cid, rls_context=rls_context
        )
        if token and not self._m365_service_allowed(conn, m365_service):
            return None, cid, conn.get("provider", provider)
        return token, cid, conn.get("provider", provider)

    async def get_emails(
        self,
        user_id: str,
        connection_id: Optional[int] = None,
        folder_id: str = "inbox",
        top: int = 50,
        skip: int = 0,
        unread_only: bool = False,
        provider: str = "microsoft",
        rls_context: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        token, _, prov = await self._ensure_token(
            connection_id,
            user_id,
            provider,
            rls_context=rls_context,
            m365_service="email",
        )
        if not token:
            return {
                "messages": [],
                "total_count": 0,
                "error": "No valid connection, token, or email service not enabled for this account",
            }

        async def _do():
            await self.initialize()
            req = connections_service_pb2.GetEmailsRequest(
                access_token=token,
                provider=prov,
                folder_id=folder_id,
                top=top,
                skip=skip,
                unread_only=unread_only,
            )
            return await self.stub.GetEmails(req, timeout=30.0)

        resp = await self._with_reconnect(_do)
        messages = [
            {
                "id": m.id,
                "conversation_id": m.conversation_id,
                "subject": m.subject,
                "from_address": m.from_address,
                "from_name": m.from_name,
                "to_addresses": list(m.to_addresses),
                "cc_addresses": list(m.cc_addresses),
                "received_datetime": m.received_datetime,
                "is_read": m.is_read,
                "has_attachments": m.has_attachments,
                "importance": m.importance,
                "body_preview": m.body_preview,
                "body_content": m.body_content,
            }
            for m in resp.messages
        ]
        return {
            "messages": messages,
            "total_count": resp.total_count,
            "error": resp.error if resp.HasField("error") else None,
        }

    async def get_email_by_id(
        self,
        user_id: str,
        message_id: str,
        connection_id: Optional[int] = None,
        provider: str = "microsoft",
    ) -> Dict[str, Any]:
        token, _, prov = await self._ensure_token(
            connection_id, user_id, provider, m365_service="email"
        )
        if not token:
            return {"message": None, "error": "No valid connection or token or email service disabled"}

        async def _do():
            await self.initialize()
            req = connections_service_pb2.GetEmailByIdRequest(
                access_token=token, provider=prov, message_id=message_id
            )
            return await self.stub.GetEmailById(req, timeout=15.0)

        resp = await self._with_reconnect(_do)
        msg = resp.message
        if not msg:
            return {"message": None, "error": resp.error or "Not found"}
        return {
            "message": {
                "id": msg.id,
                "conversation_id": msg.conversation_id,
                "subject": msg.subject,
                "from_address": msg.from_address,
                "from_name": msg.from_name,
                "to_addresses": list(msg.to_addresses),
                "cc_addresses": list(msg.cc_addresses),
                "received_datetime": msg.received_datetime,
                "is_read": msg.is_read,
                "has_attachments": msg.has_attachments,
                "importance": msg.importance,
                "body_preview": msg.body_preview,
                "body_content": msg.body_content,
            },
            "error": resp.error if resp.HasField("error") else None,
        }

    async def search_emails(
        self,
        user_id: str,
        query: str,
        connection_id: Optional[int] = None,
        top: int = 50,
        from_address: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        provider: str = "microsoft",
    ) -> Dict[str, Any]:
        token, _, prov = await self._ensure_token(
            connection_id, user_id, provider, m365_service="email"
        )
        if not token:
            return {"messages": [], "error": "No valid connection or token"}

        async def _do():
            await self.initialize()
            req = connections_service_pb2.SearchEmailsRequest(
                access_token=token,
                provider=prov,
                query=query,
                top=top,
                from_address=from_address or "",
                start_date=start_date or "",
                end_date=end_date or "",
            )
            return await self.stub.SearchEmails(req, timeout=30.0)

        resp = await self._with_reconnect(_do)
        messages = [
            {
                "id": m.id,
                "conversation_id": m.conversation_id,
                "subject": m.subject,
                "from_address": m.from_address,
                "from_name": m.from_name,
                "body_preview": m.body_preview,
                "received_datetime": m.received_datetime,
                "is_read": m.is_read,
            }
            for m in resp.messages
        ]
        return {
            "messages": messages,
            "error": resp.error if resp.HasField("error") else None,
        }

    async def send_email(
        self,
        user_id: str,
        to_recipients: List[str],
        subject: str,
        body: str,
        connection_id: Optional[int] = None,
        cc_recipients: Optional[List[str]] = None,
        body_is_html: bool = False,
        provider: str = "microsoft",
    ) -> Dict[str, Any]:
        token, _, prov = await self._ensure_token(
            connection_id, user_id, provider, m365_service="email"
        )
        if not token:
            return {"success": False, "error": "No valid connection or token"}

        async def _do():
            await self.initialize()
            req = connections_service_pb2.SendEmailRequest(
                access_token=token,
                provider=prov,
                to_recipients=to_recipients,
                subject=subject,
                body=body,
                cc_recipients=cc_recipients or [],
                body_is_html=body_is_html,
            )
            return await self.stub.SendEmail(req, timeout=30.0)

        resp = await self._with_reconnect(_do)
        return {
            "success": resp.success,
            "message_id": resp.message_id,
            "error": resp.error if resp.HasField("error") else None,
        }

    async def create_draft(
        self,
        user_id: str,
        to_recipients: List[str],
        subject: str,
        body: str,
        connection_id: Optional[int] = None,
        cc_recipients: Optional[List[str]] = None,
        body_is_html: bool = False,
        provider: str = "microsoft",
    ) -> Dict[str, Any]:
        token, _, prov = await self._ensure_token(
            connection_id, user_id, provider, m365_service="email"
        )
        if not token:
            return {"success": False, "message_id": "", "error": "No valid connection or token"}

        async def _do():
            await self.initialize()
            req = connections_service_pb2.CreateDraftRequest(
                access_token=token,
                provider=prov,
                to_recipients=to_recipients,
                subject=subject,
                body=body,
                cc_recipients=cc_recipients or [],
                body_is_html=body_is_html,
            )
            return await self.stub.CreateDraft(req, timeout=15.0)

        resp = await self._with_reconnect(_do)
        return {
            "success": resp.success,
            "message_id": resp.message_id,
            "error": resp.error if resp.HasField("error") else None,
        }

    async def reply_to_email(
        self,
        user_id: str,
        message_id: str,
        body: str,
        connection_id: Optional[int] = None,
        reply_all: bool = False,
        body_is_html: bool = False,
        provider: str = "microsoft",
    ) -> Dict[str, Any]:
        token, _, prov = await self._ensure_token(
            connection_id, user_id, provider, m365_service="email"
        )
        if not token:
            return {"success": False, "error": "No valid connection or token"}

        async def _do():
            await self.initialize()
            req = connections_service_pb2.ReplyToEmailRequest(
                access_token=token,
                provider=prov,
                message_id=message_id,
                body=body,
                reply_all=reply_all,
                body_is_html=body_is_html,
            )
            return await self.stub.ReplyToEmail(req, timeout=30.0)

        resp = await self._with_reconnect(_do)
        return {
            "success": resp.success,
            "message_id": resp.message_id,
            "error": resp.error if resp.HasField("error") else None,
        }

    async def get_email_thread(
        self,
        user_id: str,
        conversation_id: str,
        connection_id: Optional[int] = None,
        provider: str = "microsoft",
    ) -> Dict[str, Any]:
        token, _, prov = await self._ensure_token(
            connection_id, user_id, provider, m365_service="email"
        )
        if not token:
            return {"messages": [], "error": "No valid connection or token"}

        async def _do():
            await self.initialize()
            req = connections_service_pb2.GetEmailThreadRequest(
                access_token=token,
                provider=prov,
                conversation_id=conversation_id,
            )
            return await self.stub.GetEmailThread(req, timeout=15.0)

        resp = await self._with_reconnect(_do)
        messages = [
            {
                "id": m.id,
                "subject": m.subject,
                "from_address": m.from_address,
                "from_name": m.from_name,
                "received_datetime": m.received_datetime,
                "body_content": m.body_content,
                "body_preview": m.body_preview,
            }
            for m in resp.messages
        ]
        return {
            "messages": messages,
            "error": resp.error if resp.HasField("error") else None,
        }

    async def update_email(
        self,
        user_id: str,
        message_id: str,
        is_read: Optional[bool] = None,
        importance: Optional[str] = None,
        connection_id: Optional[int] = None,
        provider: str = "microsoft",
    ) -> Dict[str, Any]:
        token, _, prov = await self._ensure_token(
            connection_id, user_id, provider, m365_service="email"
        )
        if not token:
            return {"success": False, "error": "No valid connection or token"}

        async def _do():
            await self.initialize()
            req = connections_service_pb2.UpdateEmailRequest(
                access_token=token,
                provider=prov,
                message_id=message_id,
            )
            if is_read is not None:
                req.is_read = is_read
            if importance is not None:
                req.importance = importance
            return await self.stub.UpdateEmail(req, timeout=10.0)

        resp = await self._with_reconnect(_do)
        return {
            "success": resp.success,
            "error": resp.error if resp.HasField("error") else None,
        }

    async def move_email(
        self,
        user_id: str,
        message_id: str,
        destination_folder_id: str,
        connection_id: Optional[int] = None,
        provider: str = "microsoft",
    ) -> Dict[str, Any]:
        token, _, prov = await self._ensure_token(
            connection_id, user_id, provider, m365_service="email"
        )
        if not token:
            return {"success": False, "error": "No valid connection or token"}

        async def _do():
            await self.initialize()
            req = connections_service_pb2.MoveEmailRequest(
                access_token=token,
                provider=prov,
                message_id=message_id,
                destination_folder_id=destination_folder_id,
            )
            return await self.stub.MoveEmail(req, timeout=10.0)

        resp = await self._with_reconnect(_do)
        return {
            "success": resp.success,
            "error": resp.error if resp.HasField("error") else None,
        }

    async def get_folders(
        self,
        user_id: str,
        connection_id: Optional[int] = None,
        provider: str = "microsoft",
    ) -> Dict[str, Any]:
        token, _, prov = await self._ensure_token(
            connection_id, user_id, provider, m365_service="email"
        )
        if not token:
            return {"folders": [], "error": "No valid connection or token"}

        async def _do():
            await self.initialize()
            req = connections_service_pb2.GetFoldersRequest(access_token=token, provider=prov)
            return await self.stub.GetFolders(req, timeout=15.0)

        resp = await self._with_reconnect(_do)
        folders = [
            {
                "id": f.id,
                "name": f.name,
                "parent_id": f.parent_id,
                "unread_count": f.unread_count,
                "total_count": f.total_count,
            }
            for f in resp.folders
        ]
        return {
            "folders": folders,
            "error": resp.error if resp.HasField("error") else None,
        }

    async def get_email_statistics(
        self,
        user_id: str,
        folder_id: Optional[str] = None,
        connection_id: Optional[int] = None,
        provider: str = "microsoft",
    ) -> Dict[str, Any]:
        token, _, prov = await self._ensure_token(
            connection_id, user_id, provider, m365_service="email"
        )
        if not token:
            return {"total_count": 0, "unread_count": 0, "error": "No valid connection or token"}

        async def _do():
            await self.initialize()
            req = connections_service_pb2.GetEmailStatisticsRequest(
                access_token=token,
                provider=prov,
                folder_id=folder_id or "",
            )
            return await self.stub.GetEmailStatistics(req, timeout=15.0)

        resp = await self._with_reconnect(_do)
        return {
            "total_count": resp.total_count,
            "unread_count": resp.unread_count,
            "error": resp.error if resp.HasField("error") else None,
        }

    async def list_calendars(
        self,
        user_id: str,
        connection_id: Optional[int] = None,
        provider: str = "microsoft",
        rls_context: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        token, _, prov = await self._ensure_token(
            connection_id,
            user_id,
            provider,
            rls_context=rls_context,
            m365_service="calendar",
        )
        if not token:
            return {"calendars": [], "error": "No valid connection or token or calendar service disabled"}

        async def _do():
            await self.initialize()
            req = connections_service_pb2.ListCalendarsRequest(
                access_token=token,
                provider=prov,
            )
            return await self.stub.ListCalendars(req, timeout=15.0)

        resp = await self._with_reconnect(_do)
        calendars = [
            {
                "id": c.id,
                "name": c.name,
                "color": c.color,
                "is_default": c.is_default,
                "can_edit": c.can_edit,
            }
            for c in resp.calendars
        ]
        return {
            "calendars": calendars,
            "error": resp.error if resp.HasField("error") else None,
        }

    async def get_calendar_events(
        self,
        user_id: str,
        start_datetime: str,
        end_datetime: str,
        connection_id: Optional[int] = None,
        calendar_id: str = "",
        top: int = 50,
        provider: str = "microsoft",
        rls_context: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        token, _, prov = await self._ensure_token(
            connection_id,
            user_id,
            provider,
            rls_context=rls_context,
            m365_service="calendar",
        )
        if not token:
            return {"events": [], "total_count": 0, "error": "No valid connection or token or calendar service disabled"}

        async def _do():
            await self.initialize()
            req = connections_service_pb2.GetCalendarEventsRequest(
                access_token=token,
                provider=prov,
                calendar_id=calendar_id,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                top=top,
            )
            return await self.stub.GetCalendarEvents(req, timeout=30.0)

        resp = await self._with_reconnect(_do)
        events = []
        for e in resp.events:
            attendees = [
                {"email": a.email, "name": a.name, "response_status": a.response_status}
                for a in e.attendees
            ]
            events.append({
                "id": e.id,
                "subject": e.subject,
                "start_datetime": e.start_datetime,
                "end_datetime": e.end_datetime,
                "location": e.location,
                "body_preview": e.body_preview,
                "body_content": e.body_content,
                "organizer_email": e.organizer_email,
                "organizer_name": e.organizer_name,
                "attendees": attendees,
                "is_all_day": e.is_all_day,
                "recurrence": e.recurrence,
                "calendar_id": e.calendar_id,
                "web_link": e.web_link,
            })
        return {
            "events": events,
            "total_count": resp.total_count,
            "error": resp.error if resp.HasField("error") else None,
        }

    async def get_event_by_id(
        self,
        user_id: str,
        event_id: str,
        connection_id: Optional[int] = None,
        provider: str = "microsoft",
        rls_context: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        token, _, prov = await self._ensure_token(
            connection_id,
            user_id,
            provider,
            rls_context=rls_context,
            m365_service="calendar",
        )
        if not token:
            return {"event": None, "error": "No valid connection or token or calendar service disabled"}

        async def _do():
            await self.initialize()
            req = connections_service_pb2.GetCalendarEventByIdRequest(
                access_token=token,
                provider=prov,
                event_id=event_id,
            )
            return await self.stub.GetCalendarEventById(req, timeout=15.0)

        resp = await self._with_reconnect(_do)
        ev = resp.event
        if not ev:
            return {"event": None, "error": resp.error or "Not found"}
        attendees = [
            {"email": a.email, "name": a.name, "response_status": a.response_status}
            for a in ev.attendees
        ]
        return {
            "event": {
                "id": ev.id,
                "subject": ev.subject,
                "start_datetime": ev.start_datetime,
                "end_datetime": ev.end_datetime,
                "location": ev.location,
                "body_preview": ev.body_preview,
                "body_content": ev.body_content,
                "organizer_email": ev.organizer_email,
                "organizer_name": ev.organizer_name,
                "attendees": attendees,
                "is_all_day": ev.is_all_day,
                "recurrence": ev.recurrence,
                "calendar_id": ev.calendar_id,
                "web_link": ev.web_link,
            },
            "error": resp.error if resp.HasField("error") else None,
        }

    async def create_event(
        self,
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
        provider: str = "microsoft",
        rls_context: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        token, _, prov = await self._ensure_token(
            connection_id,
            user_id,
            provider,
            rls_context=rls_context,
            m365_service="calendar",
        )
        if not token:
            return {"success": False, "event_id": "", "error": "No valid connection or token or calendar service disabled"}

        async def _do():
            await self.initialize()
            req = connections_service_pb2.CreateCalendarEventRequest(
                access_token=token,
                provider=prov,
                calendar_id=calendar_id,
                subject=subject,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                location=location,
                body=body,
                body_is_html=body_is_html,
                attendee_emails=attendee_emails or [],
                is_all_day=is_all_day,
            )
            return await self.stub.CreateCalendarEvent(req, timeout=15.0)

        resp = await self._with_reconnect(_do)
        return {
            "success": resp.success,
            "event_id": resp.event_id,
            "error": resp.error if resp.HasField("error") else None,
        }

    async def update_event(
        self,
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
        provider: str = "microsoft",
        rls_context: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        token, _, prov = await self._ensure_token(
            connection_id,
            user_id,
            provider,
            rls_context=rls_context,
            m365_service="calendar",
        )
        if not token:
            return {"success": False, "error": "No valid connection or token or calendar service disabled"}

        async def _do():
            await self.initialize()
            req = connections_service_pb2.UpdateCalendarEventRequest(
                access_token=token,
                provider=prov,
                event_id=event_id,
                body_is_html=body_is_html,
            )
            if subject is not None:
                req.subject = subject
            if start_datetime is not None:
                req.start_datetime = start_datetime
            if end_datetime is not None:
                req.end_datetime = end_datetime
            if location is not None:
                req.location = location
            if body is not None:
                req.body = body
            if attendee_emails is not None:
                req.attendee_emails.extend(attendee_emails)
            if is_all_day is not None:
                req.is_all_day = is_all_day
            return await self.stub.UpdateCalendarEvent(req, timeout=15.0)

        resp = await self._with_reconnect(_do)
        return {
            "success": resp.success,
            "error": resp.error if resp.HasField("error") else None,
        }

    async def delete_event(
        self,
        user_id: str,
        event_id: str,
        connection_id: Optional[int] = None,
        provider: str = "microsoft",
        rls_context: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        token, _, prov = await self._ensure_token(
            connection_id,
            user_id,
            provider,
            rls_context=rls_context,
            m365_service="calendar",
        )
        if not token:
            return {"success": False, "error": "No valid connection or token or calendar service disabled"}

        async def _do():
            await self.initialize()
            req = connections_service_pb2.DeleteCalendarEventRequest(
                access_token=token,
                provider=prov,
                event_id=event_id,
            )
            return await self.stub.DeleteCalendarEvent(req, timeout=10.0)

        resp = await self._with_reconnect(_do)
        return {
            "success": resp.success,
            "error": resp.error if resp.HasField("error") else None,
        }

    async def get_contacts(
        self,
        user_id: str,
        connection_id: Optional[int] = None,
        folder_id: str = "",
        top: int = 100,
        provider: str = "microsoft",
        rls_context: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        token, _, prov = await self._ensure_token(
            connection_id,
            user_id,
            provider,
            rls_context=rls_context,
            m365_service="contacts",
        )
        if not token:
            return {"contacts": [], "total_count": 0, "error": "No valid connection or token or contacts service disabled"}

        async def _do():
            await self.initialize()
            req = connections_service_pb2.GetContactsRequest(
                access_token=token,
                provider=prov,
                folder_id=folder_id,
                top=top,
            )
            return await self.stub.GetContacts(req, timeout=15.0)

        resp = await self._with_reconnect(_do)
        contacts = []
        for c in resp.contacts:
            emails = [{"address": e.address, "name": e.name} for e in c.email_addresses]
            phones = [{"number": p.number, "type": p.type} for p in c.phone_numbers]
            contacts.append({
                "id": c.id,
                "display_name": c.display_name,
                "given_name": c.given_name,
                "surname": c.surname,
                "email_addresses": emails,
                "phone_numbers": phones,
                "company_name": c.company_name,
                "job_title": c.job_title,
                "birthday": c.birthday,
                "notes": c.notes,
                "folder_id": c.folder_id,
            })
        return {
            "contacts": contacts,
            "total_count": resp.total_count,
            "error": resp.error if resp.HasField("error") else None,
        }

    async def get_contact_by_id(
        self,
        user_id: str,
        contact_id: str,
        connection_id: Optional[int] = None,
        provider: str = "microsoft",
        rls_context: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        token, _, prov = await self._ensure_token(
            connection_id,
            user_id,
            provider,
            rls_context=rls_context,
            m365_service="contacts",
        )
        if not token:
            return {"contact": None, "error": "No valid connection or token or contacts service disabled"}

        async def _do():
            await self.initialize()
            req = connections_service_pb2.GetContactByIdRequest(
                access_token=token,
                provider=prov,
                contact_id=contact_id,
            )
            return await self.stub.GetContactById(req, timeout=15.0)

        resp = await self._with_reconnect(_do)
        c = resp.contact
        if not c:
            return {"contact": None, "error": resp.error or "Not found"}
        emails = [{"address": e.address, "name": e.name} for e in c.email_addresses]
        phones = [{"number": p.number, "type": p.type} for p in c.phone_numbers]
        return {
            "contact": {
                "id": c.id,
                "display_name": c.display_name,
                "given_name": c.given_name,
                "surname": c.surname,
                "email_addresses": emails,
                "phone_numbers": phones,
                "company_name": c.company_name,
                "job_title": c.job_title,
                "birthday": c.birthday,
                "notes": c.notes,
                "folder_id": c.folder_id,
            },
            "error": resp.error if resp.HasField("error") else None,
        }

    async def create_contact(
        self,
        user_id: str,
        display_name: str = "",
        given_name: str = "",
        surname: str = "",
        connection_id: Optional[int] = None,
        folder_id: str = "",
        email_addresses: Optional[List[Dict[str, str]]] = None,
        phone_numbers: Optional[List[Dict[str, str]]] = None,
        company_name: str = "",
        job_title: str = "",
        birthday: str = "",
        notes: str = "",
        provider: str = "microsoft",
        rls_context: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        token, _, prov = await self._ensure_token(
            connection_id,
            user_id,
            provider,
            rls_context=rls_context,
            m365_service="contacts",
        )
        if not token:
            return {"success": False, "contact_id": "", "error": "No valid connection or token or contacts service disabled"}

        async def _do():
            await self.initialize()
            email_pbs = [
                connections_service_pb2.ContactEmailAddress(address=e.get("address", ""), name=e.get("name", ""))
                for e in (email_addresses or []) if e.get("address")
            ]
            phone_pbs = [
                connections_service_pb2.ContactPhoneNumber(number=p.get("number", ""), type=p.get("type", ""))
                for p in (phone_numbers or []) if p.get("number")
            ]
            req = connections_service_pb2.CreateContactRequest(
                access_token=token,
                provider=prov,
                folder_id=folder_id,
                display_name=display_name,
                given_name=given_name,
                surname=surname,
                email_addresses=email_pbs,
                phone_numbers=phone_pbs,
                company_name=company_name,
                job_title=job_title,
                birthday=birthday,
                notes=notes,
            )
            return await self.stub.CreateContact(req, timeout=15.0)

        resp = await self._with_reconnect(_do)
        return {
            "success": resp.success,
            "contact_id": resp.contact_id,
            "error": resp.error if resp.HasField("error") else None,
        }

    async def update_contact(
        self,
        user_id: str,
        contact_id: str,
        connection_id: Optional[int] = None,
        display_name: Optional[str] = None,
        given_name: Optional[str] = None,
        surname: Optional[str] = None,
        email_addresses: Optional[List[Dict[str, str]]] = None,
        phone_numbers: Optional[List[Dict[str, str]]] = None,
        company_name: Optional[str] = None,
        job_title: Optional[str] = None,
        birthday: Optional[str] = None,
        notes: Optional[str] = None,
        provider: str = "microsoft",
        rls_context: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        token, _, prov = await self._ensure_token(
            connection_id,
            user_id,
            provider,
            rls_context=rls_context,
            m365_service="contacts",
        )
        if not token:
            return {"success": False, "error": "No valid connection or token or contacts service disabled"}

        async def _do():
            await self.initialize()
            req = connections_service_pb2.UpdateContactRequest(
                access_token=token,
                provider=prov,
                contact_id=contact_id,
            )
            if display_name is not None:
                req.display_name = display_name
            if given_name is not None:
                req.given_name = given_name
            if surname is not None:
                req.surname = surname
            if company_name is not None:
                req.company_name = company_name
            if job_title is not None:
                req.job_title = job_title
            if birthday is not None:
                req.birthday = birthday
            if notes is not None:
                req.notes = notes
            if email_addresses is not None:
                req.email_addresses.extend(
                    connections_service_pb2.ContactEmailAddress(address=e.get("address", ""), name=e.get("name", ""))
                    for e in email_addresses
                )
            if phone_numbers is not None:
                req.phone_numbers.extend(
                    connections_service_pb2.ContactPhoneNumber(number=p.get("number", ""), type=p.get("type", ""))
                    for p in phone_numbers
                )
            return await self.stub.UpdateContact(req, timeout=15.0)

        resp = await self._with_reconnect(_do)
        return {
            "success": resp.success,
            "error": resp.error if resp.HasField("error") else None,
        }

    async def delete_contact(
        self,
        user_id: str,
        contact_id: str,
        connection_id: Optional[int] = None,
        provider: str = "microsoft",
        rls_context: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        token, _, prov = await self._ensure_token(
            connection_id,
            user_id,
            provider,
            rls_context=rls_context,
            m365_service="contacts",
        )
        if not token:
            return {"success": False, "error": "No valid connection or token or contacts service disabled"}

        async def _do():
            await self.initialize()
            req = connections_service_pb2.DeleteContactRequest(
                access_token=token,
                provider=prov,
                contact_id=contact_id,
            )
            return await self.stub.DeleteContact(req, timeout=10.0)

        resp = await self._with_reconnect(_do)
        return {
            "success": resp.success,
            "error": resp.error if resp.HasField("error") else None,
        }

    async def register_bot(
        self,
        connection_id: int,
        user_id: str,
        provider: str,
        bot_token: str,
        display_name: str = "",
        config: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Register a messaging bot with the connections-service (start listener)."""

        async def _do():
            await self.initialize()
            req = connections_service_pb2.RegisterBotRequest(
                connection_id=str(connection_id),
                user_id=user_id,
                provider=provider,
                bot_token=bot_token,
                display_name=display_name,
                config=config or {},
            )
            return await self.stub.RegisterBot(req, timeout=30.0)

        resp = await self._with_reconnect(_do)
        return {
            "success": resp.success,
            "bot_username": resp.bot_username,
            "error": resp.error if resp.HasField("error") else None,
        }

    async def unregister_bot(self, connection_id: int) -> Dict[str, Any]:
        """Unregister a messaging bot (stop listener)."""

        async def _do():
            await self.initialize()
            req = connections_service_pb2.UnregisterBotRequest(connection_id=str(connection_id))
            return await self.stub.UnregisterBot(req, timeout=15.0)

        resp = await self._with_reconnect(_do)
        return {
            "success": resp.success,
            "error": resp.error if resp.HasField("error") else None,
        }

    async def get_bot_status(self, connection_id: int) -> Dict[str, Any]:
        """Get status of a messaging bot listener."""

        async def _do():
            await self.initialize()
            req = connections_service_pb2.GetBotStatusRequest(connection_id=str(connection_id))
            return await self.stub.GetBotStatus(req, timeout=5.0)

        resp = await self._with_reconnect(_do)
        return {
            "status": resp.status,
            "bot_username": resp.bot_username,
            "error": resp.error if resp.HasField("error") else None,
        }

    async def send_outbound_message(
        self,
        user_id: str,
        provider: str,
        connection_id: str = "",
        message: str = "",
        format: str = "markdown",
        recipient_chat_id: str = "",
    ) -> Dict[str, Any]:
        """Send a proactive outbound message via a messaging bot (Telegram, Discord)."""

        async def _do():
            await self.initialize()
            req = connections_service_pb2.SendOutboundMessageRequest(
                user_id=user_id,
                provider=provider,
                connection_id=connection_id,
                message=message,
                format=format,
            )
            try:
                setattr(req, "recipient_chat_id", recipient_chat_id or "")
            except (AttributeError, TypeError):
                pass
            return await self.stub.SendOutboundMessage(req, timeout=30.0)

        resp = await self._with_reconnect(_do)
        return {
            "success": resp.success,
            "message_id": resp.message_id,
            "channel": resp.channel,
            "error": resp.error if resp.HasField("error") else None,
        }

    async def execute_connector_endpoint(
        self,
        definition: Dict[str, Any],
        credentials: Dict[str, Any],
        endpoint_id: str,
        params: Optional[Dict[str, Any]] = None,
        max_pages: int = 5,
        oauth_token: Optional[str] = None,
        raw_response: bool = False,
        connector_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a data source connector endpoint via connections-service (third-party API call)."""
        definition_payload = dict(definition) if definition else {}
        if connector_type and "connector_type" not in definition_payload:
            definition_payload["connector_type"] = connector_type

        async def _do():
            await self.initialize()
            req = connections_service_pb2.ExecuteConnectorEndpointRequest(
                definition_json=json.dumps(definition_payload),
                credentials_json=json.dumps(credentials),
                endpoint_id=endpoint_id,
                params_json=json.dumps(params or {}),
                max_pages=max_pages,
                oauth_token=oauth_token or "",
                raw_response=raw_response,
            )
            return await self.stub.ExecuteConnectorEndpoint(req, timeout=60.0)

        resp = await self._with_reconnect(_do)
        records = []
        if resp.records_json:
            try:
                records = json.loads(resp.records_json)
            except json.JSONDecodeError:
                pass
        raw_response_data = None
        if resp.raw_response_json:
            try:
                raw_response_data = json.loads(resp.raw_response_json)
            except json.JSONDecodeError:
                pass
        out = {
            "records": records,
            "count": resp.count,
            "formatted": resp.formatted or "",
            "error": resp.error if resp.HasField("error") else None,
        }
        if raw_response_data is not None:
            out["raw_response"] = raw_response_data
        return out

    async def probe_api_endpoint(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Raw HTTP request for API discovery (no connector definition). Delegates to connections-service."""

        async def _do():
            await self.initialize()
            req = connections_service_pb2.ProbeApiEndpointRequest(
                url=url,
                method=method or "GET",
                headers_json=json.dumps(headers or {}),
                body_json=json.dumps(body) if body is not None else "",
                params_json=json.dumps(params or {}),
            )
            return await self.stub.ProbeApiEndpoint(req, timeout=30.0)

        resp = await self._with_reconnect(_do)
        if not resp.success:
            return {
                "success": False,
                "error": resp.error or "Probe failed",
            }
        return {
            "success": True,
            "status_code": resp.status_code,
            "response_headers": json.loads(resp.response_headers_json) if resp.response_headers_json else {},
            "response_body": resp.response_body or "",
            "content_type": resp.content_type or "",
        }


_connections_client: Optional[ConnectionsServiceClient] = None


async def get_connections_service_client() -> ConnectionsServiceClient:
    global _connections_client
    if _connections_client is None:
        _connections_client = ConnectionsServiceClient()
        await _connections_client.initialize()
    return _connections_client
