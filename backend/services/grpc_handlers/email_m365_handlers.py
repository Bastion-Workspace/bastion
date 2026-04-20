"""gRPC handlers for Email, Calendar, Contacts, and Microsoft 365 operations."""

import json
import logging

import grpc
from protos import tool_service_pb2

logger = logging.getLogger(__name__)


class EmailM365HandlersMixin:
    """Mixin providing Email, Calendar, Contacts, and M365 Graph gRPC handlers.

    Mixed into ToolServiceImplementation; methods rely on lazy imports to
    service/client modules so the mixin itself carries no heavy dependencies.
    """

    # ===== Email operations (via connections-service / email_tools) =====

    async def GetEmails(
        self,
        request: tool_service_pb2.GetEmailsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetEmailsResponse:
        """Get emails for user (inbox or folder)."""
        try:
            from services.langgraph_tools.email_tools import read_recent_emails
            user_id = request.user_id or "system"
            connection_id = getattr(request, "connection_id", 0) or 0
            result = await read_recent_emails(
                user_id=user_id,
                folder=request.folder or "inbox",
                count=request.top or 10,
                unread_only=request.unread_only,
                connection_id=connection_id if connection_id else None,
            )
            return tool_service_pb2.GetEmailsResponse(success=True, result=result)
        except Exception as e:
            logger.error("GetEmails failed: %s", e)
            return tool_service_pb2.GetEmailsResponse(
                success=False, result="", error=str(e)
            )

    async def SearchEmails(
        self,
        request: tool_service_pb2.SearchEmailsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.SearchEmailsResponse:
        """Search emails for user."""
        try:
            from services.langgraph_tools.email_tools import search_emails
            user_id = request.user_id or "system"
            connection_id = getattr(request, "connection_id", 0) or 0
            result = await search_emails(
                user_id=user_id,
                query=request.query,
                top=request.top or 20,
                from_address=request.from_address or None,
                connection_id=connection_id if connection_id else None,
            )
            return tool_service_pb2.SearchEmailsResponse(success=True, result=result)
        except Exception as e:
            logger.error("SearchEmails failed: %s", e)
            return tool_service_pb2.SearchEmailsResponse(
                success=False, result="", error=str(e)
            )

    async def GetEmailThread(
        self,
        request: tool_service_pb2.GetEmailThreadRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetEmailThreadResponse:
        """Get full email thread by conversation_id."""
        try:
            from services.langgraph_tools.email_tools import get_email_thread
            user_id = request.user_id or "system"
            connection_id = getattr(request, "connection_id", 0) or 0
            result = await get_email_thread(
                user_id=user_id,
                conversation_id=request.conversation_id,
                connection_id=connection_id if connection_id else None,
            )
            return tool_service_pb2.GetEmailThreadResponse(success=True, result=result)
        except Exception as e:
            logger.error("GetEmailThread failed: %s", e)
            return tool_service_pb2.GetEmailThreadResponse(
                success=False, result="", error=str(e)
            )

    async def SendEmail(
        self,
        request: tool_service_pb2.SendEmailRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.SendEmailResponse:
        """Send email. from_source: system = Bastion SMTP, user = user's email connection (default)."""
        try:
            from services.langgraph_tools.email_tools import send_email
            user_id = request.user_id or "system"
            from_source = (request.from_source or "user").strip().lower() or "user"
            connection_id = getattr(request, "connection_id", 0) or 0
            result = await send_email(
                user_id=user_id,
                to=list(request.to),
                subject=request.subject,
                body=request.body,
                cc=list(request.cc) if request.cc else None,
                from_source=from_source,
                connection_id=connection_id if connection_id else None,
                body_is_html=getattr(request, "body_is_html", False) or False,
            )
            if result.startswith("Email sent successfully"):
                return tool_service_pb2.SendEmailResponse(success=True, result=result)
            return tool_service_pb2.SendEmailResponse(
                success=False, result=result, error=result
            )
        except Exception as e:
            logger.error("SendEmail failed: %s", e)
            return tool_service_pb2.SendEmailResponse(
                success=False, result="", error=str(e)
            )

    async def ReplyToEmail(
        self,
        request: tool_service_pb2.ReplyToEmailRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ReplyToEmailResponse:
        """Reply to an email."""
        try:
            from services.langgraph_tools.email_tools import reply_to_email
            user_id = request.user_id or "system"
            connection_id = getattr(request, "connection_id", 0) or 0
            result = await reply_to_email(
                user_id=user_id,
                message_id=request.message_id,
                body=request.body,
                reply_all=request.reply_all,
                connection_id=connection_id if connection_id else None,
            )
            return tool_service_pb2.ReplyToEmailResponse(success=True, result=result)
        except Exception as e:
            logger.error("ReplyToEmail failed: %s", e)
            return tool_service_pb2.ReplyToEmailResponse(
                success=False, result="", error=str(e)
            )

    async def GetEmailFolders(
        self,
        request: tool_service_pb2.GetEmailFoldersRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetEmailFoldersResponse:
        """List email folders for user."""
        try:
            from clients.connections_service_client import get_connections_service_client
            user_id = request.user_id or "system"
            connection_id = getattr(request, "connection_id", 0) or 0
            client = await get_connections_service_client()
            data = await client.get_folders(
                user_id=user_id,
                connection_id=connection_id if connection_id else None,
            )
            if data.get("error") and not data.get("folders"):
                return tool_service_pb2.GetEmailFoldersResponse(
                    success=False, result="", error=data.get("error", "No connection")
                )
            folders = data.get("folders", [])
            lines = [
                f"- {f.get('name')} (id={f.get('id')}, unread={f.get('unread_count', 0)})"
                for f in folders
            ]
            result = "\n".join(lines) if lines else "No folders."
            return tool_service_pb2.GetEmailFoldersResponse(success=True, result=result)
        except Exception as e:
            logger.error("GetEmailFolders failed: %s", e)
            return tool_service_pb2.GetEmailFoldersResponse(
                success=False, result="", error=str(e)
            )

    async def GetEmailStatistics(
        self,
        request: tool_service_pb2.GetEmailStatisticsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetEmailStatisticsResponse:
        """Get email statistics (inbox total/unread)."""
        try:
            from services.langgraph_tools.email_tools import get_email_statistics
            user_id = request.user_id or "system"
            connection_id = getattr(request, "connection_id", 0) or 0
            result = await get_email_statistics(
                user_id=user_id,
                connection_id=connection_id if connection_id else None,
            )
            return tool_service_pb2.GetEmailStatisticsResponse(success=True, result=result)
        except Exception as e:
            logger.error("GetEmailStatistics failed: %s", e)
            return tool_service_pb2.GetEmailStatisticsResponse(
                success=False, result="", error=str(e)
            )

    async def MarkEmailRead(
        self,
        request: tool_service_pb2.MarkEmailReadRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.MarkEmailReadResponse:
        """Mark an email as read."""
        try:
            from services.langgraph_tools.email_tools import mark_email_as_read
            user_id = request.user_id or "system"
            result = await mark_email_as_read(
                user_id=user_id,
                message_id=request.message_id,
            )
            return tool_service_pb2.MarkEmailReadResponse(success=True, result=result)
        except Exception as e:
            logger.error("MarkEmailRead failed: %s", e)
            return tool_service_pb2.MarkEmailReadResponse(
                success=False, result="", error=str(e)
            )

    async def GetEmailById(
        self,
        request: tool_service_pb2.GetEmailByIdRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetEmailByIdResponse:
        """Get a single email by message ID (full content)."""
        try:
            from clients.connections_service_client import get_connections_service_client
            user_id = request.user_id or "system"
            message_id = request.message_id or ""
            connection_id = getattr(request, "connection_id", 0) or 0
            if not message_id:
                return tool_service_pb2.GetEmailByIdResponse(
                    success=False, result="", error="message_id is required"
                )
            client = await get_connections_service_client()
            data = await client.get_email_by_id(
                user_id=user_id,
                message_id=message_id,
                connection_id=connection_id if connection_id else None,
            )
            if data.get("error") and not data.get("message"):
                return tool_service_pb2.GetEmailByIdResponse(
                    success=False, result="", error=data.get("error", "Not found")
                )
            msg = data.get("message", {})
            parts = [
                f"Subject: {msg.get('subject', '')}",
                f"From: {msg.get('from_name', '')} <{msg.get('from_address', '')}>",
                f"To: {', '.join(msg.get('to_addresses') or [])}",
                f"Date: {msg.get('received_datetime', '')}",
                f"Read: {msg.get('is_read', False)}",
                f"Has attachments: {msg.get('has_attachments', False)}",
            ]
            body = msg.get("body_content") or msg.get("body_preview") or ""
            if body:
                parts.append(f"\nBody:\n{body}")
            result = "\n".join(parts)
            return tool_service_pb2.GetEmailByIdResponse(success=True, result=result)
        except Exception as e:
            logger.error("GetEmailById failed: %s", e)
            return tool_service_pb2.GetEmailByIdResponse(
                success=False, result="", error=str(e)
            )

    async def MoveEmail(
        self,
        request: tool_service_pb2.MoveEmailRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.MoveEmailResponse:
        """Move an email to a different folder."""
        try:
            from clients.connections_service_client import get_connections_service_client
            user_id = request.user_id or "system"
            message_id = request.message_id or ""
            destination_folder_id = request.destination_folder_id or ""
            connection_id = getattr(request, "connection_id", 0) or 0
            if not message_id or not destination_folder_id:
                return tool_service_pb2.MoveEmailResponse(
                    success=False, result="", error="message_id and destination_folder_id are required"
                )
            client = await get_connections_service_client()
            data = await client.move_email(
                user_id=user_id,
                message_id=message_id,
                destination_folder_id=destination_folder_id,
                connection_id=connection_id if connection_id else None,
            )
            if data.get("success"):
                result = f"Moved email to folder {destination_folder_id}"
                return tool_service_pb2.MoveEmailResponse(success=True, result=result)
            return tool_service_pb2.MoveEmailResponse(
                success=False, result="", error=data.get("error", "Move failed")
            )
        except Exception as e:
            logger.error("MoveEmail failed: %s", e)
            return tool_service_pb2.MoveEmailResponse(
                success=False, result="", error=str(e)
            )

    async def UpdateEmail(
        self,
        request: tool_service_pb2.UpdateEmailRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.UpdateEmailResponse:
        """Update an email (mark read/unread, set importance)."""
        try:
            from clients.connections_service_client import get_connections_service_client
            user_id = request.user_id or "system"
            message_id = request.message_id or ""
            connection_id = getattr(request, "connection_id", 0) or 0
            if not message_id:
                return tool_service_pb2.UpdateEmailResponse(
                    success=False, result="", error="message_id is required"
                )
            is_read = request.is_read if request.HasField("is_read") else None
            importance = request.importance if request.importance else None
            if is_read is None and not importance:
                return tool_service_pb2.UpdateEmailResponse(
                    success=False, result="", error="At least one of is_read or importance is required"
                )
            client = await get_connections_service_client()
            data = await client.update_email(
                user_id=user_id,
                message_id=message_id,
                is_read=is_read,
                importance=importance or None,
                connection_id=connection_id if connection_id else None,
            )
            if data.get("success"):
                result = "Email updated."
                return tool_service_pb2.UpdateEmailResponse(success=True, result=result)
            return tool_service_pb2.UpdateEmailResponse(
                success=False, result="", error=data.get("error", "Update failed")
            )
        except Exception as e:
            logger.error("UpdateEmail failed: %s", e)
            return tool_service_pb2.UpdateEmailResponse(
                success=False, result="", error=str(e)
            )

    async def CreateDraft(
        self,
        request: tool_service_pb2.CreateDraftRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.CreateDraftResponse:
        """Create a draft email (do not send)."""
        try:
            from clients.connections_service_client import get_connections_service_client
            user_id = request.user_id or "system"
            to_list = list(request.to) if request.to else []
            subject = request.subject or ""
            body = request.body or ""
            cc_list = list(request.cc) if request.cc else []
            connection_id = getattr(request, "connection_id", 0) or 0
            if not to_list:
                return tool_service_pb2.CreateDraftResponse(
                    success=False, result="", error="At least one recipient (to) is required"
                )
            client = await get_connections_service_client()
            data = await client.create_draft(
                user_id=user_id,
                to_recipients=to_list,
                subject=subject,
                body=body,
                cc_recipients=cc_list if cc_list else None,
                connection_id=connection_id if connection_id else None,
            )
            if data.get("success"):
                msg_id = data.get("message_id", "")
                result = f"Draft created (ID: {msg_id})" if msg_id else "Draft created."
                return tool_service_pb2.CreateDraftResponse(success=True, result=result)
            return tool_service_pb2.CreateDraftResponse(
                success=False, result="", error=data.get("error", "Create draft failed")
            )
        except Exception as e:
            logger.error("CreateDraft failed: %s", e)
            return tool_service_pb2.CreateDraftResponse(
                success=False, result="", error=str(e)
            )

    # ===== Calendar operations =====

    async def ListCalendars(
        self,
        request: tool_service_pb2.ListCalendarsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ListCalendarsResponse:
        """List user's calendars."""
        try:
            from services.langgraph_tools.calendar_tools import list_calendars
            user_id = request.user_id or "system"
            connection_id = getattr(request, "connection_id", 0) or 0
            result = await list_calendars(
                user_id=user_id,
                connection_id=connection_id if connection_id else None,
            )
            return tool_service_pb2.ListCalendarsResponse(success=True, result=result)
        except Exception as e:
            logger.error("ListCalendars failed: %s", e)
            return tool_service_pb2.ListCalendarsResponse(
                success=False, result="", error=str(e)
            )

    async def GetCalendarEvents(
        self,
        request: tool_service_pb2.GetCalendarEventsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetCalendarEventsResponse:
        """Get calendar events in date range."""
        try:
            from services.langgraph_tools.calendar_tools import get_calendar_events
            user_id = request.user_id or "system"
            connection_id = getattr(request, "connection_id", 0) or 0
            result = await get_calendar_events(
                user_id=user_id,
                start_datetime=request.start_datetime or "",
                end_datetime=request.end_datetime or "",
                calendar_id=request.calendar_id or "",
                top=request.top or 50,
                connection_id=connection_id if connection_id else None,
            )
            return tool_service_pb2.GetCalendarEventsResponse(success=True, result=result)
        except Exception as e:
            logger.error("GetCalendarEvents failed: %s", e)
            return tool_service_pb2.GetCalendarEventsResponse(
                success=False, result="", error=str(e)
            )

    async def GetCalendarEventById(
        self,
        request: tool_service_pb2.GetCalendarEventByIdRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetCalendarEventByIdResponse:
        """Get single calendar event by ID."""
        try:
            from services.langgraph_tools.calendar_tools import get_event_by_id
            user_id = request.user_id or "system"
            connection_id = getattr(request, "connection_id", 0) or 0
            result = await get_event_by_id(
                user_id=user_id,
                event_id=request.event_id or "",
                connection_id=connection_id if connection_id else None,
            )
            return tool_service_pb2.GetCalendarEventByIdResponse(success=True, result=result)
        except Exception as e:
            logger.error("GetCalendarEventById failed: %s", e)
            return tool_service_pb2.GetCalendarEventByIdResponse(
                success=False, result="", error=str(e)
            )

    async def CreateCalendarEvent(
        self,
        request: tool_service_pb2.CreateCalendarEventRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.CreateCalendarEventResponse:
        """Create a calendar event."""
        try:
            from services.langgraph_tools.calendar_tools import create_event
            user_id = request.user_id or "system"
            connection_id = getattr(request, "connection_id", 0) or 0
            result = await create_event(
                user_id=user_id,
                subject=request.subject or "",
                start_datetime=request.start_datetime or "",
                end_datetime=request.end_datetime or "",
                connection_id=connection_id if connection_id else None,
                calendar_id=request.calendar_id or "",
                location=request.location or "",
                body=request.body or "",
                body_is_html=request.body_is_html,
                attendee_emails=list(request.attendee_emails) if request.attendee_emails else None,
                is_all_day=request.is_all_day,
            )
            if "successfully" in result and "Error" not in result:
                return tool_service_pb2.CreateCalendarEventResponse(success=True, result=result)
            return tool_service_pb2.CreateCalendarEventResponse(
                success=False, result=result, error=result
            )
        except Exception as e:
            logger.error("CreateCalendarEvent failed: %s", e)
            return tool_service_pb2.CreateCalendarEventResponse(
                success=False, result="", error=str(e)
            )

    async def UpdateCalendarEvent(
        self,
        request: tool_service_pb2.UpdateCalendarEventRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.UpdateCalendarEventResponse:
        """Update a calendar event."""
        try:
            from services.langgraph_tools.calendar_tools import update_event
            user_id = request.user_id or "system"
            connection_id = getattr(request, "connection_id", 0) or 0
            result = await update_event(
                user_id=user_id,
                event_id=request.event_id or "",
                connection_id=connection_id if connection_id else None,
                subject=request.subject if request.subject else None,
                start_datetime=request.start_datetime if request.start_datetime else None,
                end_datetime=request.end_datetime if request.end_datetime else None,
                location=request.location if request.location else None,
                body=request.body if request.body else None,
                body_is_html=request.body_is_html,
                attendee_emails=list(request.attendee_emails) if request.attendee_emails else None,
                is_all_day=request.is_all_day,
            )
            if result == "Event updated successfully.":
                return tool_service_pb2.UpdateCalendarEventResponse(success=True, result=result)
            return tool_service_pb2.UpdateCalendarEventResponse(
                success=False, result=result, error=result
            )
        except Exception as e:
            logger.error("UpdateCalendarEvent failed: %s", e)
            return tool_service_pb2.UpdateCalendarEventResponse(
                success=False, result="", error=str(e)
            )

    async def DeleteCalendarEvent(
        self,
        request: tool_service_pb2.DeleteCalendarEventRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.DeleteCalendarEventResponse:
        """Delete a calendar event."""
        try:
            from services.langgraph_tools.calendar_tools import delete_event
            user_id = request.user_id or "system"
            connection_id = getattr(request, "connection_id", 0) or 0
            result = await delete_event(
                user_id=user_id,
                event_id=request.event_id or "",
                connection_id=connection_id if connection_id else None,
            )
            if result == "Event deleted successfully.":
                return tool_service_pb2.DeleteCalendarEventResponse(success=True, result=result)
            return tool_service_pb2.DeleteCalendarEventResponse(
                success=False, result=result, error=result
            )
        except Exception as e:
            logger.error("DeleteCalendarEvent failed: %s", e)
            return tool_service_pb2.DeleteCalendarEventResponse(
                success=False, result="", error=str(e)
            )

    # ===== Contacts operations =====

    async def GetContacts(
        self,
        request: tool_service_pb2.GetContactsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetContactsResponse:
        """Get contacts. sources: all (O365+org), microsoft, org, caldav."""
        try:
            from services.langgraph_tools.contact_tools import (
                get_contacts,
                get_contacts_unified,
                _get_org_contacts_for_tool,
                _format_contacts,
            )
            import json

            user_id = request.user_id or "system"
            connection_id = getattr(request, "connection_id", 0) or 0
            sources = (getattr(request, "sources", None) or "all").strip().lower() or "all"
            folder_id = request.folder_id or ""
            top = request.top or 100

            if sources == "microsoft":
                result = await get_contacts(
                    user_id=user_id,
                    connection_id=connection_id if connection_id else None,
                    folder_id=folder_id,
                    top=top,
                )
            elif sources == "org":
                org_list = await _get_org_contacts_for_tool(user_id, limit=top)
                formatted = _format_contacts(org_list, max_items=top, include_source=True)
                result = json.dumps({"contacts": org_list, "formatted": formatted})
            else:
                result = await get_contacts_unified(
                    user_id=user_id,
                    connection_id=connection_id if connection_id else None,
                    folder_id=folder_id,
                    top=top,
                )
            return tool_service_pb2.GetContactsResponse(success=True, result=result)
        except Exception as e:
            logger.error("GetContacts failed: %s", e)
            return tool_service_pb2.GetContactsResponse(
                success=False, result="", error=str(e)
            )

    async def GetContactById(
        self,
        request: tool_service_pb2.GetContactByIdRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetContactByIdResponse:
        """Get single O365 contact by ID."""
        try:
            from services.langgraph_tools.contact_tools import get_contact_by_id
            user_id = request.user_id or "system"
            connection_id = getattr(request, "connection_id", 0) or 0
            result = await get_contact_by_id(
                user_id=user_id,
                contact_id=request.contact_id or "",
                connection_id=connection_id if connection_id else None,
            )
            return tool_service_pb2.GetContactByIdResponse(success=True, result=result)
        except Exception as e:
            logger.error("GetContactById failed: %s", e)
            return tool_service_pb2.GetContactByIdResponse(
                success=False, result="", error=str(e)
            )

    async def CreateContact(
        self,
        request: tool_service_pb2.CreateContactRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.CreateContactResponse:
        """Create an O365 contact."""
        try:
            import json
            from services.langgraph_tools.contact_tools import create_contact
            user_id = request.user_id or "system"
            connection_id = getattr(request, "connection_id", 0) or 0
            email_addresses = None
            if request.email_addresses_json:
                try:
                    email_addresses = json.loads(request.email_addresses_json)
                except json.JSONDecodeError:
                    pass
            phone_numbers = None
            if request.phone_numbers_json:
                try:
                    phone_numbers = json.loads(request.phone_numbers_json)
                except json.JSONDecodeError:
                    pass
            result = await create_contact(
                user_id=user_id,
                display_name=request.display_name or "",
                given_name=request.given_name or "",
                surname=request.surname or "",
                connection_id=connection_id if connection_id else None,
                folder_id=request.folder_id or "",
                email_addresses=email_addresses,
                phone_numbers=phone_numbers,
                company_name=request.company_name or "",
                job_title=request.job_title or "",
                birthday=request.birthday or "",
                notes=request.notes or "",
            )
            if "successfully" in result and "Error" not in result:
                return tool_service_pb2.CreateContactResponse(success=True, result=result)
            return tool_service_pb2.CreateContactResponse(
                success=False, result=result, error=result
            )
        except Exception as e:
            logger.error("CreateContact failed: %s", e)
            return tool_service_pb2.CreateContactResponse(
                success=False, result="", error=str(e)
            )

    async def UpdateContact(
        self,
        request: tool_service_pb2.UpdateContactRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.UpdateContactResponse:
        """Update an O365 contact."""
        try:
            import json
            from services.langgraph_tools.contact_tools import update_contact
            user_id = request.user_id or "system"
            connection_id = getattr(request, "connection_id", 0) or 0
            email_addresses = None
            if request.email_addresses_json:
                try:
                    email_addresses = json.loads(request.email_addresses_json)
                except json.JSONDecodeError:
                    pass
            phone_numbers = None
            if request.phone_numbers_json:
                try:
                    phone_numbers = json.loads(request.phone_numbers_json)
                except json.JSONDecodeError:
                    pass
            result = await update_contact(
                user_id=user_id,
                contact_id=request.contact_id or "",
                connection_id=connection_id if connection_id else None,
                display_name=request.display_name if request.display_name else None,
                given_name=request.given_name if request.given_name else None,
                surname=request.surname if request.surname else None,
                email_addresses=email_addresses,
                phone_numbers=phone_numbers,
                company_name=request.company_name if request.company_name else None,
                job_title=request.job_title if request.job_title else None,
                birthday=request.birthday if request.birthday else None,
                notes=request.notes if request.notes else None,
            )
            if result == "Contact updated successfully.":
                return tool_service_pb2.UpdateContactResponse(success=True, result=result)
            return tool_service_pb2.UpdateContactResponse(
                success=False, result=result, error=result
            )
        except Exception as e:
            logger.error("UpdateContact failed: %s", e)
            return tool_service_pb2.UpdateContactResponse(
                success=False, result="", error=str(e)
            )

    async def DeleteContact(
        self,
        request: tool_service_pb2.DeleteContactRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.DeleteContactResponse:
        """Delete an O365 contact."""
        try:
            from services.langgraph_tools.contact_tools import delete_contact
            user_id = request.user_id or "system"
            connection_id = getattr(request, "connection_id", 0) or 0
            result = await delete_contact(
                user_id=user_id,
                contact_id=request.contact_id or "",
                connection_id=connection_id if connection_id else None,
            )
            if result == "Contact deleted successfully.":
                return tool_service_pb2.DeleteContactResponse(success=True, result=result)
            return tool_service_pb2.DeleteContactResponse(
                success=False, result=result, error=result
            )
        except Exception as e:
            logger.error("DeleteContact failed: %s", e)
            return tool_service_pb2.DeleteContactResponse(
                success=False, result="", error=str(e)
            )

    async def SearchContacts(
        self,
        request: tool_service_pb2.SearchContactsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.SearchContactsResponse:
        """Search contacts by query (substring match on name, email, company)."""
        try:
            from services.langgraph_tools.contact_tools import search_contacts as search_contacts_impl

            user_id = request.user_id or "system"
            connection_id = getattr(request, "connection_id", 0) or 0
            query = (request.query or "").strip()
            sources = (request.sources or "all").strip().lower() or "all"
            top = request.top or 20
            result = await search_contacts_impl(
                user_id=user_id,
                query=query,
                sources=sources,
                top=top,
                connection_id=connection_id if connection_id else None,
            )
            return tool_service_pb2.SearchContactsResponse(success=True, result=result)
        except Exception as e:
            logger.error("SearchContacts failed: %s", e)
            return tool_service_pb2.SearchContactsResponse(
                success=False, result="", error=str(e)
            )

    # ===== M365 Graph & Account operations =====

    async def M365GraphInvoke(
        self,
        request: tool_service_pb2.M365GraphInvokeRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.M365GraphInvokeResponse:
        """Microsoft 365 Graph workloads (To Do, OneDrive, OneNote, Planner) via connections-service."""
        try:
            from clients.connections_service_client import get_connections_service_client
            from services.m365_graph_tool_dispatch import dispatch_m365_graph, parse_m365_params

            user_id = request.user_id or "system"
            cid = int(getattr(request, "connection_id", 0) or 0)
            operation = (request.operation or "").strip()
            params = parse_m365_params(getattr(request, "params_json", "") or "")

            client = await get_connections_service_client()
            ok, data, err = await dispatch_m365_graph(
                client,
                user_id,
                cid if cid else None,
                operation,
                params,
            )
            if not ok:
                return tool_service_pb2.M365GraphInvokeResponse(
                    success=False,
                    result_json="{}",
                    error=err or "M365 operation failed",
                )
            return tool_service_pb2.M365GraphInvokeResponse(
                success=True,
                result_json=json.dumps(data if data is not None else {}),
                error="",
            )
        except Exception as e:
            logger.error("M365GraphInvoke failed: %s", e)
            return tool_service_pb2.M365GraphInvokeResponse(
                success=False, result_json="{}", error=str(e)
            )

    async def ListUserAccounts(
        self,
        request: tool_service_pb2.ListUserAccountsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ListUserAccountsResponse:
        """List active external_connections (email, calendar, contacts, etc.) for the user."""
        try:
            from services.database_manager.database_helpers import fetch_all
            from services.external_connections_service import external_connections_service

            user_id = request.user_id or "system"
            service_type = (request.service_type or "all").strip().lower() or "all"

            if service_type == "all":
                rows = await fetch_all(
                    """
                    SELECT id, provider, connection_type, account_identifier, display_name,
                           provider_metadata
                    FROM external_connections
                    WHERE user_id = $1 AND is_active = true
                    ORDER BY connection_type, provider
                    """,
                    user_id,
                )
            else:
                rows = await fetch_all(
                    """
                    SELECT id, provider, connection_type, account_identifier, display_name,
                           provider_metadata
                    FROM external_connections
                    WHERE user_id = $1 AND is_active = true AND connection_type = $2
                    ORDER BY connection_type, provider
                    """,
                    user_id,
                    service_type,
                )

            accounts = []
            for r in rows:
                prov = (r.get("provider") or "").strip().lower()
                entry = {
                    "connection_id": int(r["id"]),
                    "provider": r["provider"] or "",
                    "type": r["connection_type"] or "",
                    "label": (r.get("display_name") or r.get("account_identifier") or "").strip()
                    or (r.get("account_identifier") or ""),
                    "address": r.get("account_identifier") or "",
                }
                if prov == "microsoft":
                    entry["enabled_services"] = (
                        external_connections_service.get_enabled_services_from_metadata(
                            r.get("provider_metadata")
                        )
                    )
                accounts.append(entry)
            return tool_service_pb2.ListUserAccountsResponse(
                success=True, result=json.dumps(accounts)
            )
        except Exception as e:
            logger.error("ListUserAccounts failed: %s", e)
            return tool_service_pb2.ListUserAccountsResponse(
                success=False, result="[]", error=str(e)
            )
