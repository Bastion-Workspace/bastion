"""
Telegram bot provider using python-telegram-bot.
"""

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
from providers.formatting import markdown_to_telegram_html, split_telegram_html_for_sends

logger = logging.getLogger(__name__)

# Lazy import to avoid requiring the library when not used
_telegram = None


def _get_telegram():
    global _telegram
    if _telegram is None:
        from telegram import Update
        from telegram.ext import Application, ContextTypes, MessageHandler, filters

        _telegram = (Update, Application, ContextTypes, MessageHandler, filters)
    return _telegram


class TelegramProvider(BaseMessagingProvider):
    """Telegram Bot API implementation using python-telegram-bot."""

    def __init__(self) -> None:
        self._application: Optional[Any] = None
        self._connection_id: Optional[str] = None
        self._message_callback: Optional[MessageCallback] = None
        self._bot_token: Optional[str] = None
        self._running = False

    @property
    def name(self) -> str:
        return "telegram"

    async def start(
        self,
        bot_token: str,
        config: Dict[str, Any],
        message_callback: MessageCallback,
        connection_id: str,
    ) -> None:
        Update, Application, ContextTypes, MessageHandler, filters = _get_telegram()
        from telegram.request import HTTPXRequest

        self._bot_token = bot_token
        self._connection_id = connection_id
        self._message_callback = message_callback

        # Longer timeouts so send_message does not time out when agent response is slow or network is slow
        request = HTTPXRequest(
            read_timeout=60.0,
            write_timeout=60.0,
            connect_timeout=15.0,
            pool_timeout=5.0,
            media_write_timeout=90.0,
        )
        application = (
            Application.builder()
            .token(bot_token)
            .request(request)
            .post_init(self._post_init)
            .build()
        )
        inbound_media = (
            filters.TEXT
            | filters.PHOTO
            | filters.Document.IMAGE
            | filters.VOICE
            | filters.AUDIO
            | filters.Document.Category("audio/")
        )
        application.add_handler(MessageHandler(inbound_media, self._handle_message))
        self._application = application
        self._running = True
        await application.initialize()
        await application.start()
        await application.updater.start_polling(drop_pending_updates=True)
        logger.info("Telegram bot started for connection_id=%s", connection_id)

    async def _post_init(self, application: Any) -> None:
        pass

    async def _download_telegram_file(self, context: Any, file_id: str) -> bytes:
        file = await context.bot.get_file(file_id)
        url = f"https://api.telegram.org/file/bot{self._bot_token}/{file.file_path}"
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content

    async def _handle_message(self, update: Any, context: Any) -> None:
        if not update.message or not self._message_callback:
            return
        message = update.message
        chat_id = str(message.chat.id)
        sender = message.from_user
        sender_id = str(sender.id) if sender else ""
        sender_name = (sender.first_name or "") + (" " + (sender.last_name or "")).strip() if sender else "Unknown"
        text = (message.text or "").strip()
        caption = (message.caption or "").strip()
        images: List[Dict[str, Any]] = []
        audio_part: Optional[Dict[str, Any]] = None

        if message.photo:
            photo = message.photo[-1]
            try:
                file = await context.bot.get_file(photo.file_id)
                url = f"https://api.telegram.org/file/bot{self._bot_token}/{file.file_path}"
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    data_b64 = base64.b64encode(resp.content).decode("utf-8")
                images.append({"data": data_b64, "mime": "image/jpeg"})
            except Exception as e:
                logger.warning("Failed to download Telegram photo: %s", e)
        if message.document and message.document.mime_type and message.document.mime_type.startswith("image/"):
            try:
                file = await context.bot.get_file(message.document.file_id)
                url = f"https://api.telegram.org/file/bot{self._bot_token}/{file.file_path}"
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    data_b64 = base64.b64encode(resp.content).decode("utf-8")
                images.append({"data": data_b64, "mime": message.document.mime_type or "image/png"})
            except Exception as e:
                logger.warning("Failed to download Telegram document image: %s", e)

        try:
            if message.voice:
                raw = await self._download_telegram_file(context, message.voice.file_id)
                mime = (message.voice.mime_type or "audio/ogg").strip() or "audio/ogg"
                ext = "ogg"
                if "/" in mime:
                    ext = mime.split("/")[-1].split(";")[0].strip() or "ogg"
                audio_part = {
                    "data": base64.b64encode(raw).decode("utf-8"),
                    "mime": mime,
                    "filename": f"voice.{ext}",
                }
                if caption:
                    audio_part["prompt"] = caption
                    text = caption if not text else f"{text}\n{caption}"
            elif message.audio:
                raw = await self._download_telegram_file(context, message.audio.file_id)
                af = message.audio
                mime = (af.mime_type or "audio/mpeg").strip() or "audio/mpeg"
                fn = (af.file_name or "").strip() or "audio.bin"
                audio_part = {
                    "data": base64.b64encode(raw).decode("utf-8"),
                    "mime": mime,
                    "filename": fn,
                }
                if caption:
                    audio_part["prompt"] = caption
                    text = caption if not text else f"{text}\n{caption}"
            elif message.document and message.document.mime_type and message.document.mime_type.startswith(
                "audio/"
            ):
                raw = await self._download_telegram_file(context, message.document.file_id)
                df = message.document
                mime = (df.mime_type or "application/octet-stream").strip()
                fn = (df.file_name or "").strip() or "audio.bin"
                audio_part = {
                    "data": base64.b64encode(raw).decode("utf-8"),
                    "mime": mime,
                    "filename": fn,
                }
                if caption:
                    audio_part["prompt"] = caption
                    text = caption if not text else f"{text}\n{caption}"
        except Exception as e:
            logger.warning("Failed to download Telegram audio/voice: %s", e)

        if not text and not images and not audio_part:
            return
        if not text and images and not audio_part:
            text = "[Image]"

        chat = message.chat
        chat_title = getattr(chat, "title", None) or (getattr(chat, "first_name", "") or "").strip()
        if chat_title and getattr(chat, "last_name", None):
            chat_title = (chat_title + " " + (chat.last_name or "")).strip()
        chat_username = getattr(chat, "username", None)
        chat_type = getattr(chat, "type", None)

        inbound = InboundMessage(
            sender_id=sender_id,
            sender_name=sender_name,
            chat_id=chat_id,
            text=text,
            images=images,
            platform="telegram",
            timestamp=message.date.isoformat() if message.date else None,
            connection_id=self._connection_id,
            chat_title=chat_title or None,
            chat_username=("@" + chat_username) if chat_username else None,
            chat_type=str(chat_type) if chat_type else None,
            audio=audio_part,
        )
        try:
            await self._message_callback(inbound)
        except Exception as e:
            logger.exception("Telegram message_callback failed: %s", e)
            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"Sorry, an error occurred: {str(e)[:200]}",
                )
            except Exception:
                pass

    async def stop(self) -> None:
        if not self._application:
            return
        self._running = False
        try:
            await self._application.updater.stop()
            await self._application.stop()
            await self._application.shutdown()
        except Exception as e:
            logger.warning("Telegram stop error: %s", e)
        self._application = None
        logger.info("Telegram bot stopped for connection_id=%s", self._connection_id)

    async def send_message(
        self,
        chat_id: str,
        text: str,
        images: Optional[List[OutboundImage]] = None,
    ) -> Dict[str, Any]:
        if not self._application or not self._running:
            return {"success": False, "error": "Bot not running"}
        bot = self._application.bot
        try:
            html = markdown_to_telegram_html(text) if text else ""
            for chunk, parse_mode in split_telegram_html_for_sends(html, max_len=4096):
                if not chunk:
                    continue
                send_kw: Dict[str, Any] = {"chat_id": chat_id, "text": chunk}
                if parse_mode:
                    send_kw["parse_mode"] = parse_mode
                await bot.send_message(**send_kw)
            if images:
                for i, img in enumerate(images):
                    if not img.data or not isinstance(img.data, bytes):
                        logger.warning("Telegram send_message: skipping image with no/invalid data")
                        continue
                    caption = img.caption
                    cap_html = None
                    cap_parse: Optional[str] = None
                    if caption and str(caption).strip():
                        cap_conv = markdown_to_telegram_html(str(caption).strip())
                        cap_parts = split_telegram_html_for_sends(cap_conv, max_len=1024)
                        if cap_parts:
                            cap_html, cap_pm = cap_parts[0]
                            cap_parse = cap_pm or None
                    photo_kw: Dict[str, Any] = {
                        "chat_id": chat_id,
                        "photo": img.data,
                        "caption": cap_html,
                    }
                    if cap_html and cap_parse:
                        photo_kw["parse_mode"] = cap_parse
                    await bot.send_photo(**photo_kw)
            return {"success": True}
        except Exception as e:
            err_msg = str(e)
            if "Chat not found" in err_msg or "chat not found" in err_msg.lower():
                err_msg = (
                    "Chat not found. Use numeric chat_id for DMs (recipient must have messaged the bot first), "
                    "or @channelname for public channels (bot must be admin). Cannot send to a private user by @username."
                )
            logger.exception("Telegram send_message failed: %s", e)
            return {"success": False, "error": err_msg}

    async def send_typing_indicator(self, chat_id: str) -> None:
        """Show 'typing...' in the chat. Telegram shows it for ~5 seconds."""
        if not self._application or not self._running:
            return
        try:
            bot = self._application.bot
            await bot.send_chat_action(chat_id=chat_id, action="typing")
        except Exception as e:
            logger.debug("Telegram send_typing_indicator failed: %s", e)

    async def get_bot_info(
        self, bot_token: str, config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            Update, Application, ContextTypes, MessageHandler, filters = _get_telegram()
            application = Application.builder().token(bot_token).build()
            await application.initialize()
            me = await application.bot.get_me()
            await application.shutdown()
            username = me.username or ""
            return {"username": username, "id": me.id}
        except Exception as e:
            logger.warning("Telegram get_bot_info failed: %s", e)
            return {"username": "", "error": str(e)}
