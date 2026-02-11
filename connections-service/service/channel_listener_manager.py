"""
Channel listener manager: lifecycle of messaging bot listeners and routing inbound messages.
Handles /newchat (start new conversation) and /model (list or set model) commands.
"""

import asyncio
import logging
import uuid
from typing import Any, Dict, Optional, Tuple

from providers.base_messaging_provider import BaseMessagingProvider, InboundMessage
from service.backend_client import BackendClient
from service.provider_router import get_messaging_provider

logger = logging.getLogger(__name__)


class ChannelListenerManager:
    """Manages active bot listeners and routes inbound messages to the backend."""

    def __init__(self) -> None:
        self._backend_client = BackendClient()
        self._bots: Dict[str, Dict[str, Any]] = {}  # connection_id -> {user_id, provider, instance, status}
        self._current_conversation: Dict[Tuple[str, str], str] = {}  # (connection_id, chat_id) -> conversation_id

    async def register_bot(
        self,
        connection_id: str,
        user_id: str,
        provider: str,
        bot_token: str,
        display_name: str = "",
        config: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Start a bot listener. Returns {success, bot_username, error}."""
        if connection_id in self._bots:
            await self.unregister_bot(connection_id)

        prov_class = get_messaging_provider(provider)
        if not prov_class:
            return {"success": False, "bot_username": "", "error": f"Unknown provider: {provider}"}

        instance = prov_class()
        bot_info = await instance.get_bot_info(bot_token)
        if bot_info.get("error"):
            return {"success": False, "bot_username": "", "error": bot_info["error"]}

        try:
            await instance.start(
                bot_token=bot_token,
                config=dict(config or {}),
                message_callback=self._on_message,
                connection_id=connection_id,
            )
        except Exception as e:
            logger.exception("Failed to start bot connection_id=%s: %s", connection_id, e)
            return {"success": False, "bot_username": bot_info.get("username", ""), "error": str(e)}

        self._bots[connection_id] = {
            "user_id": user_id,
            "provider": provider,
            "instance": instance,
            "status": "running",
            "bot_username": bot_info.get("username", ""),
        }
        return {"success": True, "bot_username": bot_info.get("username", "")}

    async def unregister_bot(self, connection_id: str) -> Dict[str, Any]:
        """Stop and remove a bot listener."""
        if connection_id not in self._bots:
            return {"success": True}
        entry = self._bots.pop(connection_id)
        instance = entry.get("instance")
        if instance:
            try:
                await instance.stop()
            except Exception as e:
                logger.warning("Error stopping bot connection_id=%s: %s", connection_id, e)
                return {"success": False, "error": str(e)}
        return {"success": True}

    def get_status(self, connection_id: str) -> Dict[str, Any]:
        """Return status for a connection: status (running/stopped/error), bot_username, error."""
        if connection_id not in self._bots:
            return {"status": "stopped", "bot_username": "", "error": None}
        entry = self._bots[connection_id]
        return {
            "status": entry.get("status", "running"),
            "bot_username": entry.get("bot_username", ""),
            "error": None,
        }

    def _get_current_conversation_id(self, connection_id: str, platform: str, chat_id: str) -> str:
        """Return the current conversation id for this chat, or the default deterministic one."""
        key = (connection_id, chat_id)
        default = f"{platform}:{connection_id}:{chat_id}"
        return self._current_conversation.get(key, default)

    def _set_current_conversation_id(self, connection_id: str, chat_id: str, conversation_id: str) -> None:
        """Set the current conversation id for this chat (e.g. after /newchat)."""
        self._current_conversation[(connection_id, chat_id)] = conversation_id

    async def _on_message(self, msg: InboundMessage) -> None:
        """Callback from providers: handle /newchat, /model, /model N, or call backend and send reply."""
        connection_id = msg.connection_id or ""
        if connection_id not in self._bots:
            logger.warning("Received message for unknown connection_id=%s", connection_id)
            return
        entry = self._bots[connection_id]
        user_id = entry["user_id"]
        instance: BaseMessagingProvider = entry["instance"]
        chat_key = (connection_id, msg.chat_id)
        conversation_id = self._get_current_conversation_id(connection_id, msg.platform, msg.chat_id)
        text = (msg.text or "").strip()
        text_lower = text.lower()

        if text_lower == "/newchat":
            new_id = f"{msg.platform}:{connection_id}:{msg.chat_id}:{uuid.uuid4()}"
            self._set_current_conversation_id(connection_id, msg.chat_id, new_id)
            result = await self._backend_client.start_new_conversation(
                user_id=user_id,
                conversation_id=new_id,
                platform=msg.platform,
                platform_chat_id=msg.chat_id,
            )
            reply = result.get("response", "Started a new chat. How can I help?") if not result.get("error") else f"Sorry: {result.get('error', 'Unknown error')[:200]}"
            await instance.send_message(msg.chat_id, reply, None)
            return

        if text_lower == "/model":
            result = await self._backend_client.list_models(user_id)
            if result.get("error"):
                reply = f"Sorry: {result.get('error', 'Unknown error')[:200]}"
            else:
                models = result.get("models", [])
                if not models:
                    reply = "No enabled models configured. Ask an admin to enable models in Settings."
                else:
                    lines = [f"{m.get('index', i+1)}) {m.get('name', m.get('id', '?'))}" for i, m in enumerate(models)]
                    reply = "Reply with /model <number> to select a model.\n\n" + "\n".join(lines)
            await instance.send_message(msg.chat_id, reply, None)
            return

        if text_lower.startswith("/model "):
            rest = text_lower[7:].strip()
            if rest.isdigit():
                model_index = int(rest)
                result = await self._backend_client.set_model(user_id, conversation_id, model_index)
                if result.get("error"):
                    reply = f"Sorry: {result.get('error', 'Unknown error')[:200]}"
                else:
                    reply = result.get("response", "Model set.")
                await instance.send_message(msg.chat_id, reply, None)
                return

        # Show typing indicator while agent processes (refresh every 4s; Telegram shows ~5s)
        async def _typing_loop() -> None:
            while True:
                await instance.send_typing_indicator(msg.chat_id)
                await asyncio.sleep(4)

        typing_task = asyncio.create_task(_typing_loop())
        try:
            result = await self._backend_client.send_external_message(
                user_id=user_id,
                conversation_id=conversation_id,
                query=msg.text,
                platform=msg.platform,
                platform_chat_id=msg.chat_id,
                sender_name=msg.sender_name,
                images=msg.images if msg.images else None,
            )
        finally:
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass

        if result.get("error"):
            logger.warning("Backend external-chat error for connection_id=%s: %s", connection_id, result["error"])
            reply_text = f"Sorry, an error occurred: {result['error'][:200]}"
            await instance.send_message(msg.chat_id, reply_text, None)
            return

        response_text = result.get("response", "")
        response_images = result.get("images", [])
        outbound_images = None
        if response_images:
            import base64
            from providers.base_messaging_provider import OutboundImage
            outbound_images = []
            for img in response_images:
                data = img.get("data")
                if data is None:
                    # Backend may send url-only (e.g. /api/documents/xxx/file); we can't send that as Telegram photo
                    continue
                if isinstance(data, str):
                    data = base64.b64decode(data)
                if not data:
                    continue
                outbound_images.append(OutboundImage(data=data, mime=img.get("mime", "image/png"), caption=img.get("caption")))
            if not outbound_images:
                outbound_images = None
        await instance.send_message(msg.chat_id, response_text, outbound_images)
