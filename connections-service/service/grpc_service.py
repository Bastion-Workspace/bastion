"""
gRPC Service Implementation - Connections Service (email, calendar, etc.)
"""

import logging
import sys
from typing import Any, Dict, List

sys.path.insert(0, "/app")

from connections_service_pb2 import (
    DeleteEmailRequest,
    DeleteEmailResponse,
    EmailMessage,
    EmailFolder,
    GetBotStatusRequest,
    GetBotStatusResponse,
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
    HealthCheckRequest,
    HealthCheckResponse,
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
    SyncFolderRequest,
    SyncFolderResponse,
    UnregisterBotRequest,
    UnregisterBotResponse,
    UpdateEmailRequest,
    UpdateEmailResponse,
)
from connections_service_pb2_grpc import ConnectionsServiceServicer
from config.settings import settings
from service.provider_router import get_provider

logger = logging.getLogger(__name__)


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
        result = await provider.send_email(
            request.access_token,
            to_recipients=list(request.to_recipients),
            subject=request.subject or "",
            body=request.body or "",
            cc_recipients=list(request.cc_recipients) if request.cc_recipients else None,
            bcc_recipients=list(request.bcc_recipients) if request.bcc_recipients else None,
            body_is_html=request.body_is_html,
        )
        return SendEmailResponse(
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
        result = await provider.reply_to_email(
            request.access_token,
            request.message_id,
            request.body or "",
            reply_all=request.reply_all,
            body_is_html=request.body_is_html,
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
