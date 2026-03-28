"""
Backend HTTP client for calling the backend internal API (external-chat).
"""

import logging
from typing import Any, Dict, List, Optional

import httpx

from config.settings import settings
from providers.base_messaging_provider import InboundMessage

logger = logging.getLogger(__name__)


class BackendClient:
    """HTTP client for backend internal API."""

    def __init__(self) -> None:
        self._base_url = getattr(settings, "BACKEND_URL", "").rstrip("/")
        self._service_key = getattr(settings, "INTERNAL_SERVICE_KEY", "")
        self._timeout = getattr(settings, "EXTERNAL_CHAT_TIMEOUT", 300.0)

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "X-Internal-Service-Key": self._service_key,
        }

    async def send_external_message(
        self,
        user_id: str,
        conversation_id: str,
        query: str,
        platform: str,
        platform_chat_id: str,
        sender_name: str,
        images: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        POST /api/internal/external-chat.
        Returns: {"response": str, "images": list, "conversation_id": str} or {"error": str}.
        """
        if not self._base_url or not self._service_key:
            logger.warning("BACKEND_URL or INTERNAL_SERVICE_KEY not set; cannot call backend")
            return {"error": "Connections service not configured for external chat"}

        url = f"{self._base_url}/api/internal/external-chat"
        payload: Dict[str, Any] = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "query": query,
            "platform": platform,
            "platform_chat_id": platform_chat_id,
            "sender_name": sender_name,
        }
        if images:
            payload["images"] = images

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json=payload, headers=self._headers())
                resp.raise_for_status()
                data = resp.json()
                return data
        except httpx.HTTPStatusError as e:
            logger.exception("Backend external-chat HTTP error: %s", e)
            return {"error": str(e.response.text) if e.response.text else str(e)}
        except Exception as e:
            logger.exception("Backend external-chat error: %s", e)
            return {"error": str(e)}

    async def start_new_conversation(
        self,
        user_id: str,
        conversation_id: str,
        platform: str,
        platform_chat_id: str,
    ) -> Dict[str, Any]:
        """
        POST /api/internal/external-chat with start_new_conversation=True.
        Returns: {"response": str, "conversation_id": str} or {"error": str}.
        """
        if not self._base_url or not self._service_key:
            return {"error": "Connections service not configured for external chat"}
        url = f"{self._base_url}/api/internal/external-chat"
        payload: Dict[str, Any] = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "query": "",
            "platform": platform,
            "platform_chat_id": platform_chat_id,
            "sender_name": "",
            "start_new_conversation": True,
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, json=payload, headers=self._headers())
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            logger.warning("Backend start_new_conversation HTTP error: %s", e)
            return {"error": str(e.response.text) if e.response.text else str(e)}
        except Exception as e:
            logger.exception("Backend start_new_conversation error: %s", e)
            return {"error": str(e)}

    async def list_models(
        self, user_id: str, conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        POST /api/internal/external-chat-models.
        Returns: {"models": [{"index": int, "id": str, "name": str}, ...], "current_model_id": str or null} or {"error": str}.
        When conversation_id is provided, backend returns current_model_id for that conversation.
        """
        if not self._base_url or not self._service_key:
            return {"error": "Connections service not configured for external chat"}
        url = f"{self._base_url}/api/internal/external-chat-models"
        payload: Dict[str, Any] = {"user_id": user_id}
        if conversation_id:
            payload["conversation_id"] = conversation_id
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(url, json=payload, headers=self._headers())
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            logger.warning("Backend list_models HTTP error: %s", e)
            return {"error": str(e.response.text) if e.response.text else str(e)}
        except Exception as e:
            logger.exception("Backend list_models error: %s", e)
            return {"error": str(e)}

    async def set_model(
        self,
        user_id: str,
        conversation_id: str,
        model_index: int,
    ) -> Dict[str, Any]:
        """
        POST /api/internal/external-chat-set-model.
        Returns: {"response": str, "model_name": str, "model_id": str} or {"error": str}.
        """
        if not self._base_url or not self._service_key:
            return {"error": "Connections service not configured for external chat"}
        url = f"{self._base_url}/api/internal/external-chat-set-model"
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    url,
                    json={
                        "user_id": user_id,
                        "conversation_id": conversation_id,
                        "model_index": model_index,
                    },
                    headers=self._headers(),
                )
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            logger.warning("Backend set_model HTTP error: %s", e)
            return {"error": str(e.response.text) if e.response.text else str(e)}
        except Exception as e:
            logger.exception("Backend set_model error: %s", e)
            return {"error": str(e)}

    async def update_outbound_chat_id(
        self,
        connection_id: str,
        chat_id: str,
        sender_name: str = "",
    ) -> Dict[str, Any]:
        """
        POST /api/internal/connection-outbound-chat-id to persist the chat_id
        for future outbound messaging. Stored in external_connections.provider_metadata.
        """
        if not self._base_url or not self._service_key:
            return {"error": "Connections service not configured"}
        url = f"{self._base_url}/api/internal/connection-outbound-chat-id"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    url,
                    json={
                        "connection_id": connection_id,
                        "outbound_chat_id": chat_id,
                        "sender_name": sender_name,
                    },
                    headers=self._headers(),
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.warning("Failed to persist outbound_chat_id for connection %s: %s", connection_id, e)
            return {"error": str(e)}

    async def get_outbound_chat_id(self, connection_id: str) -> Dict[str, Any]:
        """
        GET /api/internal/connection-outbound-chat-id?connection_id=X.
        Returns: {"outbound_chat_id": str} or {"error": str}.
        """
        if not self._base_url or not self._service_key:
            return {"error": "Connections service not configured"}
        url = f"{self._base_url}/api/internal/connection-outbound-chat-id"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    url,
                    params={"connection_id": connection_id},
                    headers=self._headers(),
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.warning("Failed to get outbound_chat_id for connection %s: %s", connection_id, e)
            return {"error": str(e)}

    async def add_known_chat(
        self,
        connection_id: str,
        chat_id: str,
        chat_title: str = "",
        chat_username: str = "",
        chat_type: str = "",
    ) -> Dict[str, Any]:
        """
        POST /api/internal/connection-known-chat to add/update a chat in known_chats
        for the connection (used for recipient dropdown). Stored in provider_metadata.
        """
        if not self._base_url or not self._service_key:
            return {"error": "Connections service not configured"}
        url = f"{self._base_url}/api/internal/connection-known-chat"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    url,
                    json={
                        "connection_id": connection_id,
                        "chat_id": chat_id,
                        "chat_title": chat_title,
                        "chat_username": chat_username,
                        "chat_type": chat_type,
                    },
                    headers=self._headers(),
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.warning("Failed to add known_chat for connection %s: %s", connection_id, e)
            return {"error": str(e)}

    async def list_user_conversations(
        self, user_id: str, limit: int = 10
    ) -> Dict[str, Any]:
        """
        GET /api/internal/user-conversations.
        Returns: {"conversations": [{"conversation_id", "title", "updated_at", "message_count"}, ...]} or {"error": str}.
        """
        if not self._base_url or not self._service_key:
            return {"error": "Connections service not configured", "conversations": []}
        url = f"{self._base_url}/api/internal/user-conversations"
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    url,
                    params={"user_id": user_id, "limit": limit},
                    headers=self._headers(),
                )
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            logger.warning("Backend list_user_conversations HTTP error: %s", e)
            return {"error": str(e.response.text) if e.response.text else str(e), "conversations": []}
        except Exception as e:
            logger.exception("Backend list_user_conversations error: %s", e)
            return {"error": str(e), "conversations": []}

    async def validate_conversation(
        self, user_id: str, conversation_id: str
    ) -> Dict[str, Any]:
        """
        GET /api/internal/validate-conversation.
        Returns: {"valid": bool, "title": str} or {"error": str}.
        """
        if not self._base_url or not self._service_key:
            return {"error": "Connections service not configured", "valid": False, "title": ""}
        url = f"{self._base_url}/api/internal/validate-conversation"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    url,
                    params={"user_id": user_id, "conversation_id": conversation_id},
                    headers=self._headers(),
                )
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            logger.warning("Backend validate_conversation HTTP error: %s", e)
            return {"error": str(e.response.text) if e.response.text else str(e), "valid": False, "title": ""}
        except Exception as e:
            logger.exception("Backend validate_conversation error: %s", e)
            return {"error": str(e), "valid": False, "title": ""}

    async def persist_active_conversation(
        self, connection_id: str, chat_id: str, conversation_id: str
    ) -> Dict[str, Any]:
        """
        POST /api/internal/connection-active-conversation to persist the active
        conversation for this (connection_id, chat_id) in provider_metadata.
        """
        if not self._base_url or not self._service_key:
            return {"error": "Connections service not configured"}
        url = f"{self._base_url}/api/internal/connection-active-conversation"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    url,
                    json={
                        "connection_id": connection_id,
                        "chat_id": chat_id,
                        "conversation_id": conversation_id,
                    },
                    headers=self._headers(),
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.warning(
                "Failed to persist active_conversation for connection %s chat %s: %s",
                connection_id, chat_id, e,
            )
            return {"error": str(e)}

    async def get_active_conversation(
        self, connection_id: str, chat_id: str
    ) -> Dict[str, Any]:
        """
        GET /api/internal/connection-active-conversation.
        Returns: {"conversation_id": str} or {"error": str}.
        """
        if not self._base_url or not self._service_key:
            return {"error": "Connections service not configured", "conversation_id": ""}
        url = f"{self._base_url}/api/internal/connection-active-conversation"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    url,
                    params={"connection_id": connection_id, "chat_id": chat_id},
                    headers=self._headers(),
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.warning(
                "Failed to get active_conversation for connection %s chat %s: %s",
                connection_id, chat_id, e,
            )
            return {"error": str(e), "conversation_id": ""}

    async def get_active_chat_bots(self) -> Dict[str, Any]:
        """
        GET /api/internal/active-chat-bots.
        Returns: {"bots": [{connection_id, user_id, provider, bot_token, display_name, config}, ...]} or {"error": str}.
        """
        if not self._base_url or not self._service_key:
            return {"error": "Connections service not configured", "bots": []}
        url = f"{self._base_url}/api/internal/active-chat-bots"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url, headers=self._headers())
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.warning("Failed to get active chat bots: %s", e)
            return {"error": str(e), "bots": []}
