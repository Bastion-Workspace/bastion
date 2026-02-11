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

    async def list_models(self, user_id: str) -> Dict[str, Any]:
        """
        POST /api/internal/external-chat-models.
        Returns: {"models": [{"index": int, "id": str, "name": str}, ...]} or {"error": str}.
        """
        if not self._base_url or not self._service_key:
            return {"error": "Connections service not configured for external chat"}
        url = f"{self._base_url}/api/internal/external-chat-models"
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    url, json={"user_id": user_id}, headers=self._headers()
                )
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
