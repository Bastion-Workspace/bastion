"""
Connections Service gRPC Client - Email and external API operations.
"""

import grpc
import logging
import os
from typing import Any, Dict, List, Optional

from config import settings
from protos import connections_service_pb2, connections_service_pb2_grpc
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
            self._initialized = False

    async def _ensure_token(
        self, connection_id: Optional[int], user_id: str, provider: str = "microsoft"
    ) -> tuple[Optional[str], Optional[int], str]:
        """Resolve connection_id or user's first email connection; return (access_token, connection_id, provider)."""
        if connection_id:
            conn = await external_connections_service.get_connection_by_id(connection_id)
            if not conn or conn.get("user_id") != user_id:
                return None, None, provider
            token = await external_connections_service.get_valid_access_token(connection_id)
            return token, connection_id, conn.get("provider", provider)
        connections = await external_connections_service.get_user_connections(
            user_id, provider=provider, connection_type="email", active_only=True
        )
        if not connections:
            return None, None, provider
        conn = connections[0]
        cid = conn["id"]
        token = await external_connections_service.get_valid_access_token(cid)
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
    ) -> Dict[str, Any]:
        token, _, prov = await self._ensure_token(connection_id, user_id, provider)
        if not token:
            return {"messages": [], "total_count": 0, "error": "No valid connection or token"}
        await self.initialize()
        req = connections_service_pb2.GetEmailsRequest(
            access_token=token,
            provider=prov,
            folder_id=folder_id,
            top=top,
            skip=skip,
            unread_only=unread_only,
        )
        resp = await self.stub.GetEmails(req, timeout=30.0)
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
        token, _, prov = await self._ensure_token(connection_id, user_id, provider)
        if not token:
            return {"message": None, "error": "No valid connection or token"}
        await self.initialize()
        req = connections_service_pb2.GetEmailByIdRequest(
            access_token=token, provider=prov, message_id=message_id
        )
        resp = await self.stub.GetEmailById(req, timeout=15.0)
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
        token, _, prov = await self._ensure_token(connection_id, user_id, provider)
        if not token:
            return {"messages": [], "error": "No valid connection or token"}
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
        resp = await self.stub.SearchEmails(req, timeout=30.0)
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
        token, _, prov = await self._ensure_token(connection_id, user_id, provider)
        if not token:
            return {"success": False, "error": "No valid connection or token"}
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
        resp = await self.stub.SendEmail(req, timeout=30.0)
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
        token, _, prov = await self._ensure_token(connection_id, user_id, provider)
        if not token:
            return {"success": False, "error": "No valid connection or token"}
        await self.initialize()
        req = connections_service_pb2.ReplyToEmailRequest(
            access_token=token,
            provider=prov,
            message_id=message_id,
            body=body,
            reply_all=reply_all,
            body_is_html=body_is_html,
        )
        resp = await self.stub.ReplyToEmail(req, timeout=30.0)
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
        token, _, prov = await self._ensure_token(connection_id, user_id, provider)
        if not token:
            return {"messages": [], "error": "No valid connection or token"}
        await self.initialize()
        req = connections_service_pb2.GetEmailThreadRequest(
            access_token=token,
            provider=prov,
            conversation_id=conversation_id,
        )
        resp = await self.stub.GetEmailThread(req, timeout=15.0)
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
        token, _, prov = await self._ensure_token(connection_id, user_id, provider)
        if not token:
            return {"success": False, "error": "No valid connection or token"}
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
        resp = await self.stub.UpdateEmail(req, timeout=10.0)
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
        token, _, prov = await self._ensure_token(connection_id, user_id, provider)
        if not token:
            return {"folders": [], "error": "No valid connection or token"}
        await self.initialize()
        req = connections_service_pb2.GetFoldersRequest(access_token=token, provider=prov)
        resp = await self.stub.GetFolders(req, timeout=15.0)
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
        token, _, prov = await self._ensure_token(connection_id, user_id, provider)
        if not token:
            return {"total_count": 0, "unread_count": 0, "error": "No valid connection or token"}
        await self.initialize()
        req = connections_service_pb2.GetEmailStatisticsRequest(
            access_token=token,
            provider=prov,
            folder_id=folder_id or "",
        )
        resp = await self.stub.GetEmailStatistics(req, timeout=15.0)
        return {
            "total_count": resp.total_count,
            "unread_count": resp.unread_count,
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
        await self.initialize()
        req = connections_service_pb2.RegisterBotRequest(
            connection_id=str(connection_id),
            user_id=user_id,
            provider=provider,
            bot_token=bot_token,
            display_name=display_name,
            config=config or {},
        )
        resp = await self.stub.RegisterBot(req, timeout=30.0)
        return {
            "success": resp.success,
            "bot_username": resp.bot_username,
            "error": resp.error if resp.HasField("error") else None,
        }

    async def unregister_bot(self, connection_id: int) -> Dict[str, Any]:
        """Unregister a messaging bot (stop listener)."""
        await self.initialize()
        req = connections_service_pb2.UnregisterBotRequest(connection_id=str(connection_id))
        resp = await self.stub.UnregisterBot(req, timeout=15.0)
        return {
            "success": resp.success,
            "error": resp.error if resp.HasField("error") else None,
        }

    async def get_bot_status(self, connection_id: int) -> Dict[str, Any]:
        """Get status of a messaging bot listener."""
        await self.initialize()
        req = connections_service_pb2.GetBotStatusRequest(connection_id=str(connection_id))
        resp = await self.stub.GetBotStatus(req, timeout=5.0)
        return {
            "status": resp.status,
            "bot_username": resp.bot_username,
            "error": resp.error if resp.HasField("error") else None,
        }


_connections_client: Optional[ConnectionsServiceClient] = None


async def get_connections_service_client() -> ConnectionsServiceClient:
    global _connections_client
    if _connections_client is None:
        _connections_client = ConnectionsServiceClient()
        await _connections_client.initialize()
    return _connections_client
