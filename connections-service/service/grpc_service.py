"""
gRPC Service Implementation - Connections Service (email, messaging, data source connectors).
"""

import json
import logging
import sys
from typing import Any, Dict, List

sys.path.insert(0, "/app")

from connections_service_pb2 import (
    CalendarEvent,
    CalendarEventAttendee,
    CalendarInfo,
    Contact,
    ContactEmailAddress,
    ContactPhoneNumber,
    CreateCalendarEventRequest,
    CreateCalendarEventResponse,
    CreateContactRequest,
    CreateContactResponse,
    CreateDriveFolderRequest,
    CreateDriveFolderResponse,
    CreateDraftRequest,
    CreateDraftResponse,
    CreateOneNotePageRequest,
    CreateOneNotePageResponse,
    CreatePlannerTaskRequest,
    CreatePlannerTaskResponse,
    CreateTodoTaskRequest,
    CreateTodoTaskResponse,
    DeleteCalendarEventRequest,
    DeleteCalendarEventResponse,
    DeleteContactRequest,
    DeleteContactResponse,
    DeleteDriveItemRequest,
    DeleteDriveItemResponse,
    DeleteEmailRequest,
    DeleteEmailResponse,
    DeletePlannerTaskRequest,
    DeletePlannerTaskResponse,
    DeleteTodoTaskRequest,
    DeleteTodoTaskResponse,
    DriveItem,
    EmailMessage,
    EmailFolder,
    ExecuteConnectorEndpointRequest,
    ExecuteConnectorEndpointResponse,
    ProbeApiEndpointRequest,
    ProbeApiEndpointResponse,
    GetBotStatusRequest,
    GetBotStatusResponse,
    GetCalendarEventByIdRequest,
    GetCalendarEventByIdResponse,
    GetCalendarEventsRequest,
    GetCalendarEventsResponse,
    GetDriveItemRequest,
    GetDriveItemResponse,
    GetFileContentRequest,
    GetFileContentResponse,
    GetContactByIdRequest,
    GetContactByIdResponse,
    GetContactsRequest,
    GetContactsResponse,
    GetEmailByIdRequest,
    GetEmailByIdResponse,
    GetEmailStatisticsRequest,
    GetEmailStatisticsResponse,
    GetEmailsRequest,
    GetEmailsResponse,
    GetEmailThreadRequest,
    GetEmailThreadResponse,
    GetFoldersRequest,
    GetFoldersResponse,
    GetOneNotePageContentRequest,
    GetOneNotePageContentResponse,
    GetPlannerTasksRequest,
    GetPlannerTasksResponse,
    GetTodoTasksRequest,
    GetTodoTasksResponse,
    HealthCheckRequest,
    HealthCheckResponse,
    ListCalendarsRequest,
    ListCalendarsResponse,
    ListDriveItemsRequest,
    ListDriveItemsResponse,
    ListOneNoteNotebooksRequest,
    ListOneNoteNotebooksResponse,
    ListOneNotePagesRequest,
    ListOneNotePagesResponse,
    ListOneNoteSectionsRequest,
    ListOneNoteSectionsResponse,
    ListPlannerPlansRequest,
    ListPlannerPlansResponse,
    ListTodoListsRequest,
    ListTodoListsResponse,
    MoveDriveItemRequest,
    MoveDriveItemResponse,
    MoveEmailRequest,
    MoveEmailResponse,
    RegisterBotRequest,
    RegisterBotResponse,
    ReplyToEmailRequest,
    ReplyToEmailResponse,
    SearchEmailsRequest,
    SearchEmailsResponse,
    SendEmailRequest,
    SendEmailResponse,
    SendOutboundMessageRequest,
    SendOutboundMessageResponse,
    SyncFolderRequest,
    SyncFolderResponse,
    UnregisterBotRequest,
    UnregisterBotResponse,
    OneNoteNotebook,
    OneNotePage,
    OneNoteSection,
    PlannerPlan,
    PlannerTask,
    SearchDriveRequest,
    SearchDriveResponse,
    TodoList,
    TodoTask,
    UpdateCalendarEventRequest,
    UpdateCalendarEventResponse,
    UpdateContactRequest,
    UpdateContactResponse,
    UpdateEmailRequest,
    UpdateEmailResponse,
    UpdatePlannerTaskRequest,
    UpdatePlannerTaskResponse,
    UpdateTodoTaskRequest,
    UpdateTodoTaskResponse,
    UploadFileRequest,
    UploadFileResponse,
)
from connections_service_pb2_grpc import ConnectionsServiceServicer
from config.settings import settings
from service.connector_executor import (
    execute_connector as connector_execute_connector,
    probe_api_endpoint as connector_probe_api_endpoint,
)
from service.provider_router import get_provider

logger = logging.getLogger(__name__)


def _body_looks_like_html(body: str) -> bool:
    """Treat body as HTML if it looks like a document (DOCTYPE or <html)."""
    if not body or not body.strip():
        return False
    stripped = body.strip().lower()
    return stripped.startswith("<!doctype") or stripped.startswith("<html")

def _dict_to_email_message(d: Dict[str, Any]) -> EmailMessage:
    return EmailMessage(
        id=d.get("id", ""),
        conversation_id=d.get("conversation_id", ""),
        subject=d.get("subject", ""),
        from_address=d.get("from_address", ""),
        from_name=d.get("from_name", ""),
        to_addresses=d.get("to_addresses") or [],
        cc_addresses=d.get("cc_addresses") or [],
        received_datetime=d.get("received_datetime", ""),
        is_read=d.get("is_read", False),
        has_attachments=d.get("has_attachments", False),
        importance=d.get("importance", "normal"),
        body_preview=d.get("body_preview", ""),
        body_content=d.get("body_content", ""),
    )


def _dict_to_folder(d: Dict[str, Any]) -> EmailFolder:
    return EmailFolder(
        id=d.get("id", ""),
        name=d.get("name", ""),
        parent_id=d.get("parent_id", ""),
        unread_count=d.get("unread_count", 0),
        total_count=d.get("total_count", 0),
    )


def _dict_to_calendar_info(d: Dict[str, Any]) -> CalendarInfo:
    return CalendarInfo(
        id=d.get("id", ""),
        name=d.get("name", ""),
        color=d.get("color", ""),
        is_default=d.get("is_default", False),
        can_edit=d.get("can_edit", True),
    )


def _dict_to_calendar_event(d: Dict[str, Any]) -> CalendarEvent:
    attendees = []
    for a in d.get("attendees") or []:
        attendees.append(
            CalendarEventAttendee(
                email=a.get("email", ""),
                name=a.get("name", ""),
                response_status=a.get("response_status", "none"),
            )
        )
    return CalendarEvent(
        id=d.get("id", ""),
        subject=d.get("subject", ""),
        start_datetime=d.get("start_datetime", ""),
        end_datetime=d.get("end_datetime", ""),
        location=d.get("location", ""),
        body_preview=d.get("body_preview", ""),
        body_content=d.get("body_content", ""),
        organizer_email=d.get("organizer_email", ""),
        organizer_name=d.get("organizer_name", ""),
        attendees=attendees,
        is_all_day=d.get("is_all_day", False),
        recurrence=d.get("recurrence", ""),
        calendar_id=d.get("calendar_id", ""),
        web_link=d.get("web_link", ""),
    )


def _dict_to_contact(d: Dict[str, Any]) -> Contact:
    emails = [
        ContactEmailAddress(address=e.get("address", ""), name=e.get("name", ""))
        for e in d.get("email_addresses") or []
    ]
    phones = [
        ContactPhoneNumber(number=p.get("number", ""), type=p.get("type", ""))
        for p in d.get("phone_numbers") or []
    ]
    return Contact(
        id=d.get("id", ""),
        display_name=d.get("display_name", ""),
        given_name=d.get("given_name", ""),
        surname=d.get("surname", ""),
        email_addresses=emails,
        phone_numbers=phones,
        company_name=d.get("company_name", ""),
        job_title=d.get("job_title", ""),
        birthday=d.get("birthday", ""),
        notes=d.get("notes", ""),
        folder_id=d.get("folder_id", ""),
    )


def _dict_to_todo_list(d: Dict[str, Any]) -> TodoList:
    return TodoList(
        id=d.get("id", ""),
        display_name=d.get("display_name", ""),
        is_owner=d.get("is_owner", False),
        is_shared=d.get("is_shared", False),
        well_known_list_name=d.get("well_known_list_name", ""),
    )


def _dict_to_todo_task(d: Dict[str, Any]) -> TodoTask:
    return TodoTask(
        id=d.get("id", ""),
        list_id=d.get("list_id", ""),
        title=d.get("title", ""),
        status=d.get("status", ""),
        body=d.get("body", ""),
        due_datetime=d.get("due_datetime", ""),
        importance=d.get("importance", "normal"),
    )


def _dict_to_drive_item(d: Dict[str, Any]) -> DriveItem:
    return DriveItem(
        id=d.get("id", ""),
        name=d.get("name", ""),
        web_url=d.get("web_url", ""),
        is_folder=d.get("is_folder", False),
        mime_type=d.get("mime_type", ""),
        size=int(d.get("size") or 0),
        parent_id=d.get("parent_id", ""),
        last_modified=d.get("last_modified", ""),
    )


def _dict_to_onenote_notebook(d: Dict[str, Any]) -> OneNoteNotebook:
    return OneNoteNotebook(
        id=d.get("id", ""),
        display_name=d.get("display_name", ""),
        web_url=d.get("web_url", ""),
    )


def _dict_to_onenote_section(d: Dict[str, Any]) -> OneNoteSection:
    return OneNoteSection(
        id=d.get("id", ""),
        display_name=d.get("display_name", ""),
        notebook_id=d.get("notebook_id", ""),
        web_url=d.get("web_url", ""),
    )


def _dict_to_onenote_page(d: Dict[str, Any]) -> OneNotePage:
    return OneNotePage(
        id=d.get("id", ""),
        title=d.get("title", ""),
        section_id=d.get("section_id", ""),
        web_url=d.get("web_url", ""),
        created_time=d.get("created_time", ""),
    )


def _dict_to_planner_plan(d: Dict[str, Any]) -> PlannerPlan:
    return PlannerPlan(
        id=d.get("id", ""),
        title=d.get("title", ""),
        owner=d.get("owner", ""),
    )


def _dict_to_planner_task(d: Dict[str, Any]) -> PlannerTask:
    return PlannerTask(
        id=d.get("id", ""),
        plan_id=d.get("plan_id", ""),
        title=d.get("title", ""),
        percent_complete=int(d.get("percent_complete") or 0),
        due_datetime=d.get("due_datetime", ""),
    )


class ConnectionsServiceImplementation(ConnectionsServiceServicer):
    """Connections service gRPC implementation with provider routing."""

    def __init__(self, listener_manager=None):
        self._initialized = False
        self._listener_manager = listener_manager

    async def initialize(self):
        self._initialized = True
        logger.info("Connections Service initialized")

    async def HealthCheck(self, request: HealthCheckRequest, context) -> HealthCheckResponse:
        return HealthCheckResponse(
            status="healthy",
            service_name=settings.SERVICE_NAME,
            version="1.0.0",
        )

    async def GetEmails(self, request: GetEmailsRequest, context) -> GetEmailsResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return GetEmailsResponse(messages=[], total_count=0, error="Unknown provider")
        result = await provider.get_emails(
            request.access_token,
            folder_id=request.folder_id or "inbox",
            top=request.top or 50,
            skip=request.skip or 0,
            filter_expr=request.filter or None,
            unread_only=request.unread_only,
        )
        messages = [_dict_to_email_message(m) for m in result.get("messages", [])]
        err = result.get("error")
        return GetEmailsResponse(
            messages=messages,
            total_count=result.get("total_count", 0),
            error=err if err else None,
        )

    async def GetEmailById(self, request: GetEmailByIdRequest, context) -> GetEmailByIdResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return GetEmailByIdResponse(error="Unknown provider")
        result = await provider.get_email_by_id(request.access_token, request.message_id)
        msg = result.get("message")
        return GetEmailByIdResponse(
            message=_dict_to_email_message(msg) if msg else None,
            error=result.get("error") or None,
        )

    async def GetEmailThread(
        self, request: GetEmailThreadRequest, context
    ) -> GetEmailThreadResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return GetEmailThreadResponse(messages=[], error="Unknown provider")
        result = await provider.get_email_thread(
            request.access_token, request.conversation_id
        )
        messages = [_dict_to_email_message(m) for m in result.get("messages", [])]
        return GetEmailThreadResponse(
            messages=messages,
            error=result.get("error") or None,
        )

    async def SearchEmails(
        self, request: SearchEmailsRequest, context
    ) -> SearchEmailsResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return SearchEmailsResponse(messages=[], error="Unknown provider")
        result = await provider.search_emails(
            request.access_token,
            query=request.query or "",
            top=request.top or 50,
            from_address=request.from_address or None,
            start_date=request.start_date or None,
            end_date=request.end_date or None,
        )
        messages = [_dict_to_email_message(m) for m in result.get("messages", [])]
        return SearchEmailsResponse(
            messages=messages,
            error=result.get("error") or None,
        )

    async def SendEmail(self, request: SendEmailRequest, context) -> SendEmailResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return SendEmailResponse(success=False, error="Unknown provider")
        body = request.body or ""
        body_is_html = request.body_is_html or _body_looks_like_html(body)
        result = await provider.send_email(
            request.access_token,
            to_recipients=list(request.to_recipients),
            subject=request.subject or "",
            body=body,
            cc_recipients=list(request.cc_recipients) if request.cc_recipients else None,
            bcc_recipients=list(request.bcc_recipients) if request.bcc_recipients else None,
            body_is_html=body_is_html,
        )
        return SendEmailResponse(
            success=result.get("success", False),
            message_id=result.get("message_id", ""),
            error=result.get("error") or None,
        )

    async def CreateDraft(self, request: CreateDraftRequest, context) -> CreateDraftResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return CreateDraftResponse(success=False, error="Unknown provider")
        body = request.body or ""
        body_is_html = request.body_is_html or _body_looks_like_html(body)
        result = await provider.create_draft(
            request.access_token,
            to_recipients=list(request.to_recipients),
            subject=request.subject or "",
            body=body,
            cc_recipients=list(request.cc_recipients) if request.cc_recipients else None,
            bcc_recipients=list(request.bcc_recipients) if request.bcc_recipients else None,
            body_is_html=body_is_html,
        )
        return CreateDraftResponse(
            success=result.get("success", False),
            message_id=result.get("message_id", ""),
            error=result.get("error") or None,
        )

    async def ReplyToEmail(
        self, request: ReplyToEmailRequest, context
    ) -> ReplyToEmailResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return ReplyToEmailResponse(success=False, error="Unknown provider")
        body = request.body or ""
        body_is_html = request.body_is_html or _body_looks_like_html(body)
        result = await provider.reply_to_email(
            request.access_token,
            request.message_id,
            body,
            reply_all=request.reply_all,
            body_is_html=body_is_html,
        )
        return ReplyToEmailResponse(
            success=result.get("success", False),
            message_id=result.get("message_id", ""),
            error=result.get("error") or None,
        )

    async def UpdateEmail(
        self, request: UpdateEmailRequest, context
    ) -> UpdateEmailResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return UpdateEmailResponse(success=False, error="Unknown provider")
        result = await provider.update_email(
            request.access_token,
            request.message_id,
            is_read=request.is_read if request.HasField("is_read") else None,
            importance=request.importance if request.importance else None,
        )
        return UpdateEmailResponse(
            success=result.get("success", False),
            error=result.get("error") or None,
        )

    async def MoveEmail(self, request: MoveEmailRequest, context) -> MoveEmailResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return MoveEmailResponse(success=False, error="Unknown provider")
        result = await provider.move_email(
            request.access_token,
            request.message_id,
            request.destination_folder_id,
        )
        return MoveEmailResponse(
            success=result.get("success", False),
            error=result.get("error") or None,
        )

    async def DeleteEmail(
        self, request: DeleteEmailRequest, context
    ) -> DeleteEmailResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return DeleteEmailResponse(success=False, error="Unknown provider")
        result = await provider.delete_email(request.access_token, request.message_id)
        return DeleteEmailResponse(
            success=result.get("success", False),
            error=result.get("error") or None,
        )

    async def GetFolders(self, request: GetFoldersRequest, context) -> GetFoldersResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return GetFoldersResponse(folders=[], error="Unknown provider")
        result = await provider.get_folders(request.access_token)
        folders = [_dict_to_folder(f) for f in result.get("folders", [])]
        return GetFoldersResponse(
            folders=folders,
            error=result.get("error") or None,
        )

    async def SyncFolder(
        self, request: SyncFolderRequest, context
    ) -> SyncFolderResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return SyncFolderResponse(error="Unknown provider")
        result = await provider.sync_folder(
            request.access_token,
            request.folder_id or "inbox",
            delta_token=request.delta_token or None,
        )
        added = [_dict_to_email_message(m) for m in result.get("added", [])]
        updated = [_dict_to_email_message(m) for m in result.get("updated", [])]
        return SyncFolderResponse(
            added=added,
            updated=updated,
            deleted_ids=result.get("deleted_ids", []),
            next_delta_token=result.get("next_delta_token", ""),
            error=result.get("error") or None,
        )

    async def GetEmailStatistics(
        self, request: GetEmailStatisticsRequest, context
    ) -> GetEmailStatisticsResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return GetEmailStatisticsResponse(
                total_count=0, unread_count=0, error="Unknown provider"
            )
        result = await provider.get_email_statistics(
            request.access_token,
            folder_id=request.folder_id or None,
        )
        return GetEmailStatisticsResponse(
            total_count=result.get("total_count", 0),
            unread_count=result.get("unread_count", 0),
            error=result.get("error") or None,
        )

    async def ListCalendars(
        self, request: ListCalendarsRequest, context
    ) -> ListCalendarsResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return ListCalendarsResponse(calendars=[], error="Unknown provider")
        result = await provider.list_calendars(request.access_token)
        calendars = [_dict_to_calendar_info(c) for c in result.get("calendars", [])]
        return ListCalendarsResponse(
            calendars=calendars,
            error=result.get("error") or None,
        )

    async def GetCalendarEvents(
        self, request: GetCalendarEventsRequest, context
    ) -> GetCalendarEventsResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return GetCalendarEventsResponse(events=[], total_count=0, error="Unknown provider")
        result = await provider.get_calendar_events(
            request.access_token,
            calendar_id=request.calendar_id or "",
            start_datetime=request.start_datetime or "",
            end_datetime=request.end_datetime or "",
            top=request.top or 50,
        )
        events = [_dict_to_calendar_event(e) for e in result.get("events", [])]
        return GetCalendarEventsResponse(
            events=events,
            total_count=result.get("total_count", 0),
            error=result.get("error") or None,
        )

    async def GetCalendarEventById(
        self, request: GetCalendarEventByIdRequest, context
    ) -> GetCalendarEventByIdResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return GetCalendarEventByIdResponse(error="Unknown provider")
        result = await provider.get_event_by_id(request.access_token, request.event_id)
        ev = result.get("event")
        return GetCalendarEventByIdResponse(
            event=_dict_to_calendar_event(ev) if ev else None,
            error=result.get("error") or None,
        )

    async def CreateCalendarEvent(
        self, request: CreateCalendarEventRequest, context
    ) -> CreateCalendarEventResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return CreateCalendarEventResponse(success=False, error="Unknown provider")
        result = await provider.create_event(
            request.access_token,
            subject=request.subject or "",
            start_datetime=request.start_datetime or "",
            end_datetime=request.end_datetime or "",
            location=request.location or "",
            body=request.body or "",
            body_is_html=request.body_is_html,
            attendee_emails=list(request.attendee_emails) if request.attendee_emails else None,
            is_all_day=request.is_all_day,
            calendar_id=request.calendar_id or "",
        )
        return CreateCalendarEventResponse(
            success=result.get("success", False),
            event_id=result.get("event_id", ""),
            error=result.get("error") or None,
        )

    async def UpdateCalendarEvent(
        self, request: UpdateCalendarEventRequest, context
    ) -> UpdateCalendarEventResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return UpdateCalendarEventResponse(success=False, error="Unknown provider")
        subject = request.subject if request.subject else None
        start_dt = request.start_datetime if request.start_datetime else None
        end_dt = request.end_datetime if request.end_datetime else None
        location = request.location if request.location else None
        body = request.body if request.body else None
        attendee_emails = list(request.attendee_emails) if request.attendee_emails else None
        is_all_day = request.is_all_day if request.HasField("is_all_day") else None
        result = await provider.update_event(
            request.access_token,
            request.event_id,
            subject=subject,
            start_datetime=start_dt,
            end_datetime=end_dt,
            location=location,
            body=body,
            body_is_html=request.body_is_html,
            attendee_emails=attendee_emails,
            is_all_day=is_all_day,
        )
        return UpdateCalendarEventResponse(
            success=result.get("success", False),
            error=result.get("error") or None,
        )

    async def DeleteCalendarEvent(
        self, request: DeleteCalendarEventRequest, context
    ) -> DeleteCalendarEventResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return DeleteCalendarEventResponse(success=False, error="Unknown provider")
        result = await provider.delete_event(request.access_token, request.event_id)
        return DeleteCalendarEventResponse(
            success=result.get("success", False),
            error=result.get("error") or None,
        )

    async def GetContacts(
        self, request: GetContactsRequest, context
    ) -> GetContactsResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return GetContactsResponse(contacts=[], total_count=0, error="Unknown provider")
        result = await provider.get_contacts(
            request.access_token,
            folder_id=request.folder_id or "",
            top=request.top or 100,
        )
        contacts = [_dict_to_contact(c) for c in result.get("contacts", [])]
        return GetContactsResponse(
            contacts=contacts,
            total_count=result.get("total_count", 0),
            error=result.get("error") or None,
        )

    async def GetContactById(
        self, request: GetContactByIdRequest, context
    ) -> GetContactByIdResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return GetContactByIdResponse(error="Unknown provider")
        result = await provider.get_contact_by_id(
            request.access_token, request.contact_id
        )
        contact = result.get("contact")
        return GetContactByIdResponse(
            contact=_dict_to_contact(contact) if contact else None,
            error=result.get("error") or None,
        )

    async def CreateContact(
        self, request: CreateContactRequest, context
    ) -> CreateContactResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return CreateContactResponse(success=False, error="Unknown provider")
        email_addresses = [
            {"address": e.address, "name": e.name}
            for e in (request.email_addresses or [])
        ]
        phone_numbers = [
            {"number": p.number, "type": p.type}
            for p in (request.phone_numbers or [])
        ]
        result = await provider.create_contact(
            request.access_token,
            display_name=request.display_name or "",
            given_name=request.given_name or "",
            surname=request.surname or "",
            email_addresses=email_addresses if email_addresses else None,
            phone_numbers=phone_numbers if phone_numbers else None,
            company_name=request.company_name or "",
            job_title=request.job_title or "",
            birthday=request.birthday or "",
            notes=request.notes or "",
            folder_id=request.folder_id or "",
        )
        return CreateContactResponse(
            success=result.get("success", False),
            contact_id=result.get("contact_id", ""),
            error=result.get("error") or None,
        )

    async def UpdateContact(
        self, request: UpdateContactRequest, context
    ) -> UpdateContactResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return UpdateContactResponse(success=False, error="Unknown provider")
        display_name = request.display_name if request.display_name else None
        given_name = request.given_name if request.given_name else None
        surname = request.surname if request.surname else None
        company_name = request.company_name if request.company_name else None
        job_title = request.job_title if request.job_title else None
        birthday = request.birthday if request.birthday else None
        notes = request.notes if request.notes else None
        email_addresses = [
            {"address": e.address, "name": e.name}
            for e in (request.email_addresses or [])
        ] or None
        phone_numbers = [
            {"number": p.number, "type": p.type}
            for p in (request.phone_numbers or [])
        ] or None
        result = await provider.update_contact(
            request.access_token,
            request.contact_id,
            display_name=display_name,
            given_name=given_name,
            surname=surname,
            email_addresses=email_addresses,
            phone_numbers=phone_numbers,
            company_name=company_name,
            job_title=job_title,
            birthday=birthday,
            notes=notes,
        )
        return UpdateContactResponse(
            success=result.get("success", False),
            error=result.get("error") or None,
        )

    async def DeleteContact(
        self, request: DeleteContactRequest, context
    ) -> DeleteContactResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return DeleteContactResponse(success=False, error="Unknown provider")
        result = await provider.delete_contact(
            request.access_token, request.contact_id
        )
        return DeleteContactResponse(
            success=result.get("success", False),
            error=result.get("error") or None,
        )

    async def ListTodoLists(
        self, request: ListTodoListsRequest, context
    ) -> ListTodoListsResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return ListTodoListsResponse(lists=[], error="Unknown provider")
        result = await provider.list_todo_lists(request.access_token)
        lists = [_dict_to_todo_list(x) for x in result.get("lists", [])]
        return ListTodoListsResponse(lists=lists, error=result.get("error") or None)

    async def GetTodoTasks(
        self, request: GetTodoTasksRequest, context
    ) -> GetTodoTasksResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return GetTodoTasksResponse(tasks=[], error="Unknown provider")
        result = await provider.get_todo_tasks(
            request.access_token, request.list_id, top=request.top or 50
        )
        tasks = [_dict_to_todo_task(x) for x in result.get("tasks", [])]
        return GetTodoTasksResponse(tasks=tasks, error=result.get("error") or None)

    async def CreateTodoTask(
        self, request: CreateTodoTaskRequest, context
    ) -> CreateTodoTaskResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return CreateTodoTaskResponse(success=False, error="Unknown provider")
        result = await provider.create_todo_task(
            request.access_token,
            request.list_id,
            title=request.title or "",
            body=request.body or "",
            due_datetime=request.due_datetime or "",
            importance=request.importance or "normal",
        )
        return CreateTodoTaskResponse(
            success=result.get("success", False),
            task_id=result.get("task_id", ""),
            error=result.get("error") or None,
        )

    async def UpdateTodoTask(
        self, request: UpdateTodoTaskRequest, context
    ) -> UpdateTodoTaskResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return UpdateTodoTaskResponse(success=False, error="Unknown provider")
        title = request.title if request.HasField("title") else None
        body = request.body if request.HasField("body") else None
        status = request.status if request.HasField("status") else None
        due = request.due_datetime if request.HasField("due_datetime") else None
        imp = request.importance if request.HasField("importance") else None
        result = await provider.update_todo_task(
            request.access_token,
            request.list_id,
            request.task_id,
            title=title,
            body=body,
            status=status,
            due_datetime=due,
            importance=imp,
        )
        return UpdateTodoTaskResponse(
            success=result.get("success", False),
            error=result.get("error") or None,
        )

    async def DeleteTodoTask(
        self, request: DeleteTodoTaskRequest, context
    ) -> DeleteTodoTaskResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return DeleteTodoTaskResponse(success=False, error="Unknown provider")
        result = await provider.delete_todo_task(
            request.access_token, request.list_id, request.task_id
        )
        return DeleteTodoTaskResponse(
            success=result.get("success", False),
            error=result.get("error") or None,
        )

    async def ListDriveItems(
        self, request: ListDriveItemsRequest, context
    ) -> ListDriveItemsResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return ListDriveItemsResponse(items=[], error="Unknown provider")
        result = await provider.list_drive_items(
            request.access_token,
            parent_item_id=request.parent_item_id or "",
            top=request.top or 50,
        )
        items = [_dict_to_drive_item(x) for x in result.get("items", [])]
        return ListDriveItemsResponse(items=items, error=result.get("error") or None)

    async def GetDriveItem(
        self, request: GetDriveItemRequest, context
    ) -> GetDriveItemResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return GetDriveItemResponse(error="Unknown provider")
        result = await provider.get_drive_item(request.access_token, request.item_id)
        it = result.get("item")
        return GetDriveItemResponse(
            item=_dict_to_drive_item(it) if it else None,
            error=result.get("error") or None,
        )

    async def SearchDrive(
        self, request: SearchDriveRequest, context
    ) -> SearchDriveResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return SearchDriveResponse(items=[], error="Unknown provider")
        result = await provider.search_drive(
            request.access_token, request.query or "", top=request.top or 25
        )
        items = [_dict_to_drive_item(x) for x in result.get("items", [])]
        return SearchDriveResponse(items=items, error=result.get("error") or None)

    async def GetFileContent(
        self, request: GetFileContentRequest, context
    ) -> GetFileContentResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return GetFileContentResponse(error="Unknown provider")
        result = await provider.get_file_content(request.access_token, request.item_id)
        return GetFileContentResponse(
            content_base64=result.get("content_base64", ""),
            mime_type=result.get("mime_type", ""),
            error=result.get("error") or None,
        )

    async def UploadFile(
        self, request: UploadFileRequest, context
    ) -> UploadFileResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return UploadFileResponse(success=False, error="Unknown provider")
        result = await provider.upload_file(
            request.access_token,
            request.parent_item_id or "",
            request.name or "upload.bin",
            request.content_base64 or "",
            mime_type=request.mime_type or "application/octet-stream",
        )
        return UploadFileResponse(
            success=result.get("success", False),
            item_id=result.get("item_id", ""),
            error=result.get("error") or None,
        )

    async def CreateDriveFolder(
        self, request: CreateDriveFolderRequest, context
    ) -> CreateDriveFolderResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return CreateDriveFolderResponse(success=False, error="Unknown provider")
        result = await provider.create_drive_folder(
            request.access_token,
            request.parent_item_id or "",
            request.name or "New folder",
        )
        return CreateDriveFolderResponse(
            success=result.get("success", False),
            item_id=result.get("item_id", ""),
            error=result.get("error") or None,
        )

    async def MoveDriveItem(
        self, request: MoveDriveItemRequest, context
    ) -> MoveDriveItemResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return MoveDriveItemResponse(success=False, error="Unknown provider")
        result = await provider.move_drive_item(
            request.access_token, request.item_id, request.new_parent_item_id
        )
        return MoveDriveItemResponse(
            success=result.get("success", False),
            error=result.get("error") or None,
        )

    async def DeleteDriveItem(
        self, request: DeleteDriveItemRequest, context
    ) -> DeleteDriveItemResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return DeleteDriveItemResponse(success=False, error="Unknown provider")
        result = await provider.delete_drive_item(request.access_token, request.item_id)
        return DeleteDriveItemResponse(
            success=result.get("success", False),
            error=result.get("error") or None,
        )

    async def ListOneNoteNotebooks(
        self, request: ListOneNoteNotebooksRequest, context
    ) -> ListOneNoteNotebooksResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return ListOneNoteNotebooksResponse(notebooks=[], error="Unknown provider")
        result = await provider.list_onenote_notebooks(request.access_token)
        nbs = [_dict_to_onenote_notebook(x) for x in result.get("notebooks", [])]
        return ListOneNoteNotebooksResponse(
            notebooks=nbs, error=result.get("error") or None
        )

    async def ListOneNoteSections(
        self, request: ListOneNoteSectionsRequest, context
    ) -> ListOneNoteSectionsResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return ListOneNoteSectionsResponse(sections=[], error="Unknown provider")
        result = await provider.list_onenote_sections(
            request.access_token, request.notebook_id
        )
        secs = [_dict_to_onenote_section(x) for x in result.get("sections", [])]
        return ListOneNoteSectionsResponse(
            sections=secs, error=result.get("error") or None
        )

    async def ListOneNotePages(
        self, request: ListOneNotePagesRequest, context
    ) -> ListOneNotePagesResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return ListOneNotePagesResponse(pages=[], error="Unknown provider")
        result = await provider.list_onenote_pages(
            request.access_token, request.section_id, top=request.top or 50
        )
        pages = [_dict_to_onenote_page(x) for x in result.get("pages", [])]
        return ListOneNotePagesResponse(pages=pages, error=result.get("error") or None)

    async def GetOneNotePageContent(
        self, request: GetOneNotePageContentRequest, context
    ) -> GetOneNotePageContentResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return GetOneNotePageContentResponse(error="Unknown provider")
        result = await provider.get_onenote_page_content(
            request.access_token, request.page_id
        )
        return GetOneNotePageContentResponse(
            html_content=result.get("html_content", ""),
            error=result.get("error") or None,
        )

    async def CreateOneNotePage(
        self, request: CreateOneNotePageRequest, context
    ) -> CreateOneNotePageResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return CreateOneNotePageResponse(success=False, error="Unknown provider")
        result = await provider.create_onenote_page(
            request.access_token,
            request.section_id,
            request.html or "",
            title=request.title or "",
        )
        return CreateOneNotePageResponse(
            success=result.get("success", False),
            page_id=result.get("page_id", ""),
            error=result.get("error") or None,
        )

    async def ListPlannerPlans(
        self, request: ListPlannerPlansRequest, context
    ) -> ListPlannerPlansResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return ListPlannerPlansResponse(plans=[], error="Unknown provider")
        result = await provider.list_planner_plans(request.access_token)
        plans = [_dict_to_planner_plan(x) for x in result.get("plans", [])]
        return ListPlannerPlansResponse(plans=plans, error=result.get("error") or None)

    async def GetPlannerTasks(
        self, request: GetPlannerTasksRequest, context
    ) -> GetPlannerTasksResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return GetPlannerTasksResponse(tasks=[], error="Unknown provider")
        result = await provider.get_planner_tasks(
            request.access_token, request.plan_id
        )
        tasks = [_dict_to_planner_task(x) for x in result.get("tasks", [])]
        return GetPlannerTasksResponse(tasks=tasks, error=result.get("error") or None)

    async def CreatePlannerTask(
        self, request: CreatePlannerTaskRequest, context
    ) -> CreatePlannerTaskResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return CreatePlannerTaskResponse(success=False, error="Unknown provider")
        result = await provider.create_planner_task(
            request.access_token,
            request.plan_id,
            title=request.title or "",
            bucket_id=request.bucket_id or "",
        )
        return CreatePlannerTaskResponse(
            success=result.get("success", False),
            task_id=result.get("task_id", ""),
            error=result.get("error") or None,
        )

    async def UpdatePlannerTask(
        self, request: UpdatePlannerTaskRequest, context
    ) -> UpdatePlannerTaskResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return UpdatePlannerTaskResponse(success=False, error="Unknown provider")
        title = request.title if request.HasField("title") else None
        pct = request.percent_complete if request.HasField("percent_complete") else None
        due = request.due_datetime if request.HasField("due_datetime") else None
        result = await provider.update_planner_task(
            request.access_token,
            request.task_id,
            title=title,
            percent_complete=pct,
            due_datetime=due,
        )
        return UpdatePlannerTaskResponse(
            success=result.get("success", False),
            error=result.get("error") or None,
        )

    async def DeletePlannerTask(
        self, request: DeletePlannerTaskRequest, context
    ) -> DeletePlannerTaskResponse:
        provider = get_provider(request.provider or "microsoft")
        if not provider:
            return DeletePlannerTaskResponse(success=False, error="Unknown provider")
        result = await provider.delete_planner_task(
            request.access_token, request.task_id, etag=request.etag or ""
        )
        return DeletePlannerTaskResponse(
            success=result.get("success", False),
            error=result.get("error") or None,
        )

    async def RegisterBot(self, request: RegisterBotRequest, context) -> RegisterBotResponse:
        if not self._listener_manager:
            return RegisterBotResponse(success=False, error="Listener manager not configured")
        config = dict(request.config) if request.config else {}
        result = await self._listener_manager.register_bot(
            connection_id=str(request.connection_id),
            user_id=request.user_id or "",
            provider=request.provider or "",
            bot_token=request.bot_token or "",
            display_name=request.display_name or "",
            config=config,
        )
        return RegisterBotResponse(
            success=result.get("success", False),
            bot_username=result.get("bot_username", ""),
            error=result.get("error") or None,
        )

    async def UnregisterBot(self, request: UnregisterBotRequest, context) -> UnregisterBotResponse:
        if not self._listener_manager:
            return UnregisterBotResponse(success=False, error="Listener manager not configured")
        result = await self._listener_manager.unregister_bot(str(request.connection_id))
        return UnregisterBotResponse(
            success=result.get("success", False),
            error=result.get("error") or None,
        )

    async def GetBotStatus(self, request: GetBotStatusRequest, context) -> GetBotStatusResponse:
        if not self._listener_manager:
            return GetBotStatusResponse(status="stopped", error="Listener manager not configured")
        result = self._listener_manager.get_status(str(request.connection_id))
        return GetBotStatusResponse(
            status=result.get("status", "stopped"),
            bot_username=result.get("bot_username", ""),
            error=result.get("error") or None,
        )

    async def SendOutboundMessage(
        self, request: SendOutboundMessageRequest, context
    ) -> SendOutboundMessageResponse:
        if not self._listener_manager:
            return SendOutboundMessageResponse(
                success=False, error="Listener manager not configured"
            )
        result = await self._listener_manager.send_outbound(
            user_id=request.user_id or "",
            provider=request.provider or "",
            connection_id=request.connection_id or "",
            message=request.message or "",
            format=request.format or "markdown",
            recipient_chat_id=getattr(request, "recipient_chat_id", None) or "",
        )
        return SendOutboundMessageResponse(
            success=result.get("success", False),
            message_id=result.get("message_id", ""),
            channel=result.get("channel", ""),
            error=result.get("error") or None,
        )

    async def ExecuteConnectorEndpoint(
        self, request: ExecuteConnectorEndpointRequest, context
    ) -> ExecuteConnectorEndpointResponse:
        """Execute a data source connector endpoint; definition and credentials from backend via gRPC."""
        try:
            definition = {}
            if request.definition_json:
                definition = json.loads(request.definition_json)
            credentials = {}
            if request.credentials_json:
                credentials = json.loads(request.credentials_json)
            params = {}
            if request.params_json:
                params = json.loads(request.params_json)
            oauth_token = request.oauth_token or None
            if oauth_token == "":
                oauth_token = None
            result = await connector_execute_connector(
                definition=definition,
                credentials=credentials,
                endpoint_id=request.endpoint_id or "",
                params=params,
                max_pages=request.max_pages if request.max_pages > 0 else 5,
                oauth_token=oauth_token,
                raw_response=request.raw_response,
            )
            records_json = json.dumps(result.get("records", []))
            raw_response_json = ""
            if result.get("raw_response") is not None:
                raw_response_json = json.dumps(result["raw_response"])
            return ExecuteConnectorEndpointResponse(
                success=not result.get("error"),
                records_json=records_json,
                count=result.get("count", 0),
                formatted=result.get("formatted", ""),
                raw_response_json=raw_response_json,
                error=result.get("error") or None,
            )
        except json.JSONDecodeError as e:
            logger.exception("ExecuteConnectorEndpoint JSON decode failed: %s", e)
            return ExecuteConnectorEndpointResponse(
                success=False,
                records_json="[]",
                count=0,
                formatted="",
                raw_response_json="",
                error=str(e),
            )
        except Exception as e:
            logger.exception("ExecuteConnectorEndpoint failed: %s", e)
            return ExecuteConnectorEndpointResponse(
                success=False,
                records_json="[]",
                count=0,
                formatted="",
                raw_response_json="",
                error=str(e),
            )

    async def ProbeApiEndpoint(
        self, request: ProbeApiEndpointRequest, context
    ) -> ProbeApiEndpointResponse:
        """Raw HTTP request for API discovery (no connector definition)."""
        try:
            headers = {}
            if request.headers_json:
                try:
                    headers = json.loads(request.headers_json)
                except json.JSONDecodeError:
                    return ProbeApiEndpointResponse(
                        success=False, error="Invalid headers_json"
                    )
            body = None
            if request.body_json:
                try:
                    body = json.loads(request.body_json)
                except json.JSONDecodeError:
                    return ProbeApiEndpointResponse(
                        success=False, error="Invalid body_json"
                    )
            params = None
            if request.params_json:
                try:
                    params = json.loads(request.params_json)
                except json.JSONDecodeError:
                    return ProbeApiEndpointResponse(
                        success=False, error="Invalid params_json"
                    )
            result = await connector_probe_api_endpoint(
                url=request.url or "",
                method=request.method or "GET",
                headers=headers,
                body=body,
                params=params,
            )
            if not result.get("success"):
                return ProbeApiEndpointResponse(
                    success=False,
                    error=result.get("error", "Probe failed"),
                )
            return ProbeApiEndpointResponse(
                success=True,
                status_code=result.get("status_code", 0),
                response_headers_json=json.dumps(result.get("response_headers", {})),
                response_body=result.get("response_body", ""),
                content_type=result.get("content_type", ""),
            )
        except Exception as e:
            logger.exception("ProbeApiEndpoint failed: %s", e)
            return ProbeApiEndpointResponse(success=False, error=str(e))
