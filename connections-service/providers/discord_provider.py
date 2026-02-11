"""
Discord bot provider using discord.py.
"""

import asyncio
import base64
import io
import logging
from typing import Any, Dict, List, Optional

import httpx

from providers.base_messaging_provider import (
    BaseMessagingProvider,
    InboundMessage,
    MessageCallback,
    OutboundImage,
)

logger = logging.getLogger(__name__)

_discord = None


def _get_discord():
    global _discord
    if _discord is None:
        import discord
        from discord import Intents

        _discord = (discord, Intents)
    return _discord


class DiscordProvider(BaseMessagingProvider):
    """Discord Bot API implementation using discord.py."""

    def __init__(self) -> None:
        self._client: Optional[Any] = None
        self._connection_id: Optional[str] = None
        self._message_callback: Optional[MessageCallback] = None
        self._bot_token: Optional[str] = None
        self._running = False
        self._task: Optional[Any] = None

    @property
    def name(self) -> str:
        return "discord"

    async def start(
        self,
        bot_token: str,
        config: Dict[str, Any],
        message_callback: MessageCallback,
        connection_id: str,
    ) -> None:
        discord_module, Intents = _get_discord()
        self._bot_token = bot_token
        self._connection_id = connection_id
        self._message_callback = message_callback

        intents = Intents.default()
        intents.message_content = True
        intents.messages = True
        intents.guilds = True

        class BotClient(discord_module.Client):
            def __init__(outer_self, provider: "DiscordProvider") -> None:
                super().__init__(intents=intents)
                outer_self._provider = provider

            async def on_ready(outer_self) -> None:
                logger.info("Discord bot logged in as %s for connection_id=%s", outer_self.user, connection_id)

            async def on_message(outer_self, message: Any) -> None:
                if message.author.bot:
                    return
                if not outer_self._provider._message_callback:
                    return
                channel_id = str(message.channel.id)
                sender_id = str(message.author.id)
                sender_name = message.author.display_name or message.author.name or "Unknown"
                text = (message.content or "").strip()
                images: List[Dict[str, Any]] = []

                for attachment in message.attachments:
                    if attachment.content_type and attachment.content_type.startswith("image/"):
                        try:
                            async with httpx.AsyncClient(timeout=30.0) as client:
                                resp = await client.get(attachment.url)
                                resp.raise_for_status()
                                data_b64 = base64.b64encode(resp.content).decode("utf-8")
                            images.append({"data": data_b64, "mime": attachment.content_type or "image/png"})
                        except Exception as e:
                            logger.warning("Failed to download Discord attachment: %s", e)

                if not text and not images:
                    return
                if not text:
                    text = "[Image]"

                inbound = InboundMessage(
                    sender_id=sender_id,
                    sender_name=sender_name,
                    chat_id=channel_id,
                    text=text,
                    images=images,
                    platform="discord",
                    timestamp=message.created_at.isoformat() if message.created_at else None,
                    connection_id=outer_self._provider._connection_id,
                )
                try:
                    await outer_self._provider._message_callback(inbound)
                except Exception as e:
                    logger.exception("Discord message_callback failed: %s", e)
                    try:
                        await message.channel.send(content=f"Sorry, an error occurred: {str(e)[:200]}")
                    except Exception:
                        pass

        client = BotClient(self)
        self._client = client
        self._running = True
        self._task = asyncio.create_task(client.start(bot_token))
        logger.info("Discord bot start task created for connection_id=%s", connection_id)

    async def stop(self) -> None:
        self._running = False
        if self._client:
            try:
                await self._client.close()
            except Exception as e:
                logger.warning("Discord close error: %s", e)
            self._client = None
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning("Discord task cancel: %s", e)
        self._task = None
        logger.info("Discord bot stopped for connection_id=%s", self._connection_id)

    async def send_message(
        self,
        chat_id: str,
        text: str,
        images: Optional[List[OutboundImage]] = None,
    ) -> Dict[str, Any]:
        if not self._client or not self._running:
            return {"success": False, "error": "Bot not running"}
        try:
            channel = self._client.get_channel(int(chat_id))
            if not channel:
                return {"success": False, "error": f"Channel {chat_id} not found"}
            discord_module, _ = _get_discord()

            has_any_caption = bool(
                images and any(getattr(img, "caption", None) for img in images)
            )

            if has_any_caption:
                if text:
                    while text:
                        chunk = text[:2000]
                        text = text[2000:]
                        await channel.send(content=chunk)
                for i, img in enumerate(images):
                    fp = io.BytesIO(img.data)
                    ext = "png" if "png" in img.mime else "jpg"
                    content = (getattr(img, "caption", None) or "")[:2000]
                    await channel.send(content=content or None, files=[discord_module.File(fp, filename=f"image_{i}.{ext}")])
                return {"success": True}

            files: List[Any] = []
            if images:
                for i, img in enumerate(images):
                    fp = io.BytesIO(img.data)
                    ext = "png" if "png" in img.mime else "jpg"
                    files.append(discord_module.File(fp, filename=f"image_{i}.{ext}"))
            if text:
                while text:
                    chunk = text[:2000]
                    text = text[2000:]
                    if files:
                        await channel.send(content=chunk, files=files)
                        files = []
                    else:
                        await channel.send(content=chunk)
            elif files:
                await channel.send(files=files)
            return {"success": True}
        except Exception as e:
            logger.exception("Discord send_message failed: %s", e)
            return {"success": False, "error": str(e)}

    async def send_typing_indicator(self, chat_id: str) -> None:
        """Show typing indicator in the channel for ~4s (discord.py keeps it while inside context)."""
        if not self._client or not self._running:
            return
        try:
            channel = self._client.get_channel(int(chat_id))
            if not channel:
                return
            async with channel.typing():
                await asyncio.sleep(4)
        except Exception as e:
            logger.debug("Discord send_typing_indicator failed: %s", e)

    async def get_bot_info(self, bot_token: str) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://discord.com/api/v10/users/@me",
                    headers={"Authorization": f"Bot {bot_token}"},
                )
                resp.raise_for_status()
                data = resp.json()
            username = data.get("username", "") or ""
            return {"username": username, "id": str(data.get("id", ""))}
        except Exception as e:
            logger.warning("Discord get_bot_info failed: %s", e)
            return {"username": "", "error": str(e)}
