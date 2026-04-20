"""
Publish folder/file UI events to Redis so the backend can relay them to WebSocket clients.

DS code calls get_websocket_manager().send_to_session(...) / .broadcast(...) like the real backend.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_channel = os.getenv("FOLDER_EVENTS_REDIS_CHANNEL", "bastion:folder_events")
_redis_client: Any = None


async def _get_redis_client():
    global _redis_client
    if _redis_client is None:
        import redis.asyncio as redis_mod

        url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        _redis_client = redis_mod.from_url(url, decode_responses=True)
    return _redis_client


class _RedisFolderEventBridge:
    """Minimal async surface used by ds_services/folder_service and WebSocketNotifier."""

    async def send_to_session(self, message: Any, user_id: str) -> None:
        try:
            r = await _get_redis_client()
            envelope = json.dumps(
                {"message": message, "target_user_id": user_id, "broadcast": False}
            )
            await r.publish(_channel, envelope)
        except Exception as e:
            logger.warning("Redis folder event publish failed: %s", e)

    async def broadcast(self, message: Any) -> None:
        try:
            r = await _get_redis_client()
            envelope = json.dumps({"message": message, "broadcast": True})
            await r.publish(_channel, envelope)
        except Exception as e:
            logger.warning("Redis folder event broadcast failed: %s", e)


_bridge: Optional[_RedisFolderEventBridge] = None


def get_websocket_manager():
    global _bridge
    if _bridge is None:
        _bridge = _RedisFolderEventBridge()
    return _bridge
