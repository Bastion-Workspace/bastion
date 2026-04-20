"""
Slack bot provider using Slack Bolt (Socket Mode).
"""

import asyncio
import base64
import logging
from typing import Any, Dict, List, Optional

import httpx

from providers.base_messaging_provider import (
    BaseMessagingProvider,
    InboundMessage,
    MessageCallback,
    OutboundImage,
)
from providers.formatting import format_text_for_platform

logger = logging.getLogger(__name__)

# Lazy import to avoid requiring the library when not used
_bolt = None


def _get_bolt():
    global _bolt
    if _bolt is None:
        from slack_bolt.async_app import AsyncApp
        from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

        _bolt = (AsyncApp, AsyncSocketModeHandler)
    return _bolt


class SlackProvider(BaseMessagingProvider):
    """Slack Bot implementation using Bolt and Socket Mode."""

    def __init__(self) -> None:
        self._app: Optional[Any] = None
        self._handler: Optional[Any] = None
        self._handler_task: Optional[Any] = None
        self._connection_id: Optional[str] = None
        self._message_callback: Optional[MessageCallback] = None
        self._bot_token: Optional[str] = None
        self._running = False
        self._user_name_cache: Dict[str, str] = {}

    @property
    def name(self) -> str:
        return "slack"

    async def start(
        self,
        bot_token: str,
        config: Dict[str, Any],
        message_callback: MessageCallback,
        connection_id: str,
    ) -> None:
        AsyncApp, AsyncSocketModeHandler = _get_bolt()
        app_token = (config or {}).get("app_token", "").strip()
        if not app_token:
            raise ValueError("Slack Socket Mode requires app_token (xapp-...) in config")

        self._bot_token = bot_token
        self._connection_id = connection_id
        self._message_callback = message_callback

        app = AsyncApp(token=bot_token)

        @app.event("message")
        async def handle_message(event: Dict[str, Any], say: Any, client: Any) -> None:
            await self._handle_message(event, say, client)

        self._app = app
        self._handler = AsyncSocketModeHandler(app, app_token)
        self._running = True
        self._handler_task = asyncio.create_task(self._handler.connect_async())
        logger.info("Slack bot started for connection_id=%s", connection_id)

    async def _handle_message(
        self,
        event: Dict[str, Any],
        say: Any,
        client: Any,
    ) -> None:
        if not self._message_callback:
            return
        if event.get("bot_id"):
            return
        subtype = event.get("subtype", "")
        if subtype in ("bot_message", "message_changed", "message_deleted", "channel_join", "channel_leave"):
            return
        channel_id = event.get("channel", "")
        user_id = event.get("user", "")
        text = (event.get("text") or "").strip()
        ts = event.get("ts", "")

        sender_name = self._user_name_cache.get(user_id)
        if sender_name is None and user_id:
            try:
                resp = await client.users_info(user=user_id)
                if resp.get("ok") and resp.get("user"):
                    u = resp["user"]
                    sender_name = u.get("real_name") or u.get("name") or user_id
                else:
                    sender_name = user_id
            except Exception as e:
                logger.debug("Slack users_info failed for %s: %s", user_id, e)
                sender_name = user_id
            self._user_name_cache[user_id] = sender_name or user_id
        sender_name = sender_name or user_id

        images: List[Dict[str, Any]] = []
        audio_part: Optional[Dict[str, Any]] = None
        for f in event.get("files") or []:
            mime = (f.get("mimetype") or "").split(";")[0].strip()
            url = f.get("url_private_download")
            if not url:
                continue
            if mime.startswith("image/"):
                try:
                    async with httpx.AsyncClient(timeout=30.0) as client_http:
                        resp = await client_http.get(
                            url,
                            headers={"Authorization": f"Bearer {self._bot_token}"},
                        )
                        resp.raise_for_status()
                        data_b64 = base64.b64encode(resp.content).decode("utf-8")
                    images.append({"data": data_b64, "mime": mime or "image/png"})
                except Exception as e:
                    logger.warning("Failed to download Slack file: %s", e)
            elif mime.startswith("audio/") and audio_part is None:
                try:
                    async with httpx.AsyncClient(timeout=60.0) as client_http:
                        resp = await client_http.get(
                            url,
                            headers={"Authorization": f"Bearer {self._bot_token}"},
                        )
                        resp.raise_for_status()
                        raw = resp.content
                    fn = (f.get("name") or f.get("title") or "").strip() or "audio.bin"
                    audio_part = {
                        "data": base64.b64encode(raw).decode("utf-8"),
                        "mime": mime or "application/octet-stream",
                        "filename": fn,
                    }
                except Exception as e:
                    logger.warning("Failed to download Slack audio file: %s", e)

        if not text and not images and not audio_part:
            return
        if not text and images and not audio_part:
            text = "[Image]"

        chat_title = event.get("channel") or channel_id
        inbound = InboundMessage(
            sender_id=user_id,
            sender_name=sender_name,
            chat_id=channel_id,
            text=text,
            images=images,
            platform="slack",
            timestamp=ts,
            connection_id=self._connection_id,
            chat_title=chat_title,
            chat_username=None,
            chat_type=None,
            audio=audio_part,
        )
        try:
            await self._message_callback(inbound)
        except Exception as e:
            logger.exception("Slack message_callback failed: %s", e)
            try:
                await say(text=f"Sorry, an error occurred: {str(e)[:200]}")
            except Exception:
                pass

    async def stop(self) -> None:
        self._running = False
        if self._handler_task:
            self._handler_task.cancel()
            try:
                await self._handler_task
            except Exception:
                pass
            self._handler_task = None
        if self._handler:
            try:
                await self._handler.close()
            except Exception as e:
                logger.warning("Slack handler close error: %s", e)
            self._handler = None
        self._app = None
        self._user_name_cache.clear()
        logger.info("Slack bot stopped for connection_id=%s", self._connection_id)

    async def send_message(
        self,
        chat_id: str,
        text: str,
        images: Optional[List[OutboundImage]] = None,
    ) -> Dict[str, Any]:
        if not self._app or not self._running:
            return {"success": False, "error": "Bot not running"}
        try:
            from slack_sdk.web.async_client import AsyncWebClient

            client = AsyncWebClient(token=self._bot_token)
            mrkdwn = format_text_for_platform(text or "", "slack")
            message_ts = ""
            if mrkdwn:
                chunk_size = 4000
                for i in range(0, len(mrkdwn), chunk_size):
                    chunk = mrkdwn[i : i + chunk_size]
                    resp = await client.chat_postMessage(channel=chat_id, text=chunk)
                    if resp.get("ok") and resp.get("ts"):
                        message_ts = resp["ts"]
            if images:
                for img in images:
                    if not img.data or not isinstance(img.data, bytes):
                        logger.warning("Slack send_message: skipping image with no/invalid data")
                        continue
                    caption = (img.caption or "")[:500]
                    cap_mrkdwn = format_text_for_platform(caption, "slack") if caption else None
                    await client.files_upload_v2(
                        channel=chat_id,
                        file=img.data,
                        filename="image.png",
                        initial_comment=cap_mrkdwn,
                    )
            return {"success": True, "message_id": message_ts}
        except Exception as e:
            err_msg = str(e)
            logger.exception("Slack send_message failed: %s", e)
            return {"success": False, "error": err_msg}

    async def send_typing_indicator(self, chat_id: str) -> None:
        """Slack has no persistent typing indicator API for bots; no-op."""
        pass

    async def get_bot_info(
        self, bot_token: str, config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            from slack_sdk.web.async_client import AsyncWebClient

            client = AsyncWebClient(token=bot_token)
            resp = await client.auth_test()
            if resp.get("ok"):
                return {"username": resp.get("user", ""), "id": resp.get("user_id", "")}
            return {"username": "", "error": resp.get("error", "auth_test failed")}
        except Exception as e:
            logger.warning("Slack get_bot_info failed: %s", e)
            return {"username": "", "error": str(e)}
