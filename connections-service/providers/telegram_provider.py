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
from providers.formatting import markdown_to_telegram_html

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
        self._bot_token = bot_token
        self._connection_id = connection_id
        self._message_callback = message_callback

        application = (
            Application.builder()
            .token(bot_token)
            .post_init(self._post_init)
            .build()
        )
        application.add_handler(
            MessageHandler(
                filters.TEXT | filters.PHOTO | filters.Document.IMAGE,
                self._handle_message,
            )
        )
        self._application = application
        self._running = True
        await application.initialize()
        await application.start()
        await application.updater.start_polling(drop_pending_updates=True)
        logger.info("Telegram bot started for connection_id=%s", connection_id)

    async def _post_init(self, application: Any) -> None:
        pass

    async def _handle_message(self, update: Any, context: Any) -> None:
        if not update.message or not self._message_callback:
            return
        message = update.message
        chat_id = str(message.chat.id)
        sender = message.from_user
        sender_id = str(sender.id) if sender else ""
        sender_name = (sender.first_name or "") + (" " + (sender.last_name or "")).strip() if sender else "Unknown"
        text = (message.text or "").strip()
        images: List[Dict[str, Any]] = []

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

        if not text and not images:
            return
        if not text:
            text = "[Image]"

        inbound = InboundMessage(
            sender_id=sender_id,
            sender_name=sender_name,
            chat_id=chat_id,
            text=text,
            images=images,
            platform="telegram",
            timestamp=message.date.isoformat() if message.date else None,
            connection_id=self._connection_id,
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
            html_text = markdown_to_telegram_html(text) if text else ""
            # Send agent response text first, then images with captions (so order matches UI)
            if html_text:
                while html_text:
                    chunk = html_text[:4096]
                    html_text = html_text[4096:]
                    await bot.send_message(
                        chat_id=chat_id,
                        text=chunk,
                        parse_mode="HTML",
                    )
            if images:
                for i, img in enumerate(images):
                    if not img.data or not isinstance(img.data, bytes):
                        logger.warning("Telegram send_message: skipping image with no/invalid data")
                        continue
                    caption = img.caption
                    cap_html = markdown_to_telegram_html(caption)[:1024] if caption else None
                    await bot.send_photo(
                        chat_id=chat_id,
                        photo=img.data,
                        caption=cap_html,
                        parse_mode="HTML",
                    )
            return {"success": True}
        except Exception as e:
            logger.exception("Telegram send_message failed: %s", e)
            return {"success": False, "error": str(e)}

    async def send_typing_indicator(self, chat_id: str) -> None:
        """Show 'typing...' in the chat. Telegram shows it for ~5 seconds."""
        if not self._application or not self._running:
            return
        try:
            bot = self._application.bot
            await bot.send_chat_action(chat_id=chat_id, action="typing")
        except Exception as e:
            logger.debug("Telegram send_typing_indicator failed: %s", e)

    async def get_bot_info(self, bot_token: str) -> Dict[str, Any]:
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
