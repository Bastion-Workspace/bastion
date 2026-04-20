"""
Publish document_status_update payloads to Redis so the backend can relay to WebSockets.
"""

import json
import logging
import os
from typing import Any, Dict, Optional

import redis.asyncio as redis

logger = logging.getLogger(__name__)


def _channel() -> str:
    return os.getenv("DOCUMENT_STATUS_REDIS_CHANNEL", "bastion:document_status")


class RedisDocumentStatusBridge:
    """Drop-in replacement for websocket_manager.send_document_status_update."""

    def __init__(self, redis_url: str) -> None:
        self._redis_url = redis_url
        self._client: Optional[redis.Redis] = None

    async def _ensure(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.from_url(self._redis_url, decode_responses=True)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def send_document_status_update(
        self,
        document_id: str,
        status: str,
        folder_id: str = None,
        user_id: str = None,
        filename: str = None,
        proposal_data: Dict[str, Any] = None,
        updated_at: str = None,
        content_source: str = "embedding",
        **extras: Any,
    ) -> None:
        try:
            r = await self._ensure()
            payload: Dict[str, Any] = {
                "document_id": document_id,
                "status": status,
                "folder_id": folder_id,
                "user_id": user_id,
                "filename": filename,
                "content_source": content_source or "embedding",
            }
            if updated_at is not None:
                payload["updated_at"] = updated_at
            for k, v in extras.items():
                if v is not None:
                    payload[k] = v
            await r.publish(_channel(), json.dumps({k: v for k, v in payload.items() if v is not None}))
        except Exception as e:
            logger.error("Failed to publish document status to Redis: %s", e)
