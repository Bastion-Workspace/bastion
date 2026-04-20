"""
Subscribe to document-service Redis pub/sub channels and forward to WebSockets.
"""

import asyncio
import json
import logging
from typing import Optional

import redis.asyncio as redis

from config import settings

logger = logging.getLogger(__name__)


async def _subscriber_loop() -> None:
    status_channel = getattr(
        settings, "DOCUMENT_STATUS_REDIS_CHANNEL", "bastion:document_status"
    )
    folder_channel = getattr(
        settings, "FOLDER_EVENTS_REDIS_CHANNEL", "bastion:folder_events"
    )
    client = redis.from_url(settings.REDIS_URL, decode_responses=True)
    pubsub = client.pubsub()
    await pubsub.subscribe(status_channel, folder_channel)
    logger.info(
        "Subscribed to Redis channels %s, %s for WebSocket relay",
        status_channel,
        folder_channel,
    )
    try:
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message is None:
                continue
            if message.get("type") != "message":
                continue
            raw = message.get("data")
            if not raw:
                continue
            ch = message.get("channel")
            if isinstance(ch, bytes):
                ch = ch.decode()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON on Redis relay channel: %s", raw[:200])
                continue
            from utils.websocket_manager import get_websocket_manager

            mgr = get_websocket_manager()
            if not mgr:
                continue

            if ch == folder_channel:
                inner = payload.get("message")
                if inner is None:
                    continue
                if payload.get("broadcast"):
                    await mgr.broadcast(inner)
                else:
                    uid = payload.get("target_user_id")
                    if uid:
                        await mgr.send_to_session(inner, uid)
                    else:
                        await mgr.broadcast(inner)
                continue

            # document_status channel (flat payload)
            await mgr.send_document_status_update(
                document_id=payload.get("document_id", ""),
                status=payload.get("status", ""),
                folder_id=payload.get("folder_id"),
                user_id=payload.get("user_id"),
                filename=payload.get("filename"),
                updated_at=payload.get("updated_at"),
                content_source=payload.get("content_source"),
            )
    except asyncio.CancelledError:
        logger.info("Document status Redis subscriber cancelled")
        raise
    except Exception as e:
        logger.error("Document status Redis subscriber error: %s", e)
    finally:
        try:
            await pubsub.unsubscribe(status_channel, folder_channel)
        except Exception:
            pass
        await client.aclose()


class DocumentStatusRedisSubscriber:
    """Runs the subscriber until cancelled."""

    def __init__(self) -> None:
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(_subscriber_loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None


document_status_redis_subscriber = DocumentStatusRedisSubscriber()
