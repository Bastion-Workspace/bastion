"""
Channel listener manager: lifecycle of messaging bot listeners and routing inbound messages.
Handles /newchat, /chats, /loadchat, /chat, and /model (list or set model) commands.
"""

import asyncio
import base64
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from providers.base_messaging_provider import BaseMessagingProvider, InboundMessage
from service.backend_client import BackendClient
from service.provider_router import get_messaging_provider

logger = logging.getLogger(__name__)


def _format_relative_time(updated_at: Optional[str]) -> str:
    """Format updated_at ISO string as '2h ago', 'yesterday', '3d ago'."""
    if not updated_at:
        return "unknown"
    try:
        if isinstance(updated_at, str) and "T" in updated_at:
            dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        else:
            return str(updated_at)[:10]
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta = now - dt
        if delta < timedelta(minutes=1):
            return "just now"
        if delta < timedelta(hours=1):
            m = int(delta.total_seconds() / 60)
            return f"{m}m ago"
        if delta < timedelta(hours=24):
            h = int(delta.total_seconds() / 3600)
            return f"{h}h ago"
        if delta < timedelta(days=2):
            return "yesterday"
        if delta < timedelta(days=7):
            d = delta.days
            return f"{d}d ago"
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return "unknown"


class ChannelListenerManager:
    """Manages active bot listeners and routes inbound messages to the backend."""

    def __init__(self) -> None:
        self._backend_client = BackendClient()
        self._bots: Dict[str, Dict[str, Any]] = {}  # connection_id -> {user_id, provider, instance, status}
        self._current_conversation: Dict[Tuple[str, str], str] = {}  # (connection_id, chat_id) -> conversation_id
        self._outbound_chat_ids: Dict[str, str] = {}  # connection_id -> chat_id (for proactive messaging)
        self._chat_listing_cache: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}  # (connection_id, chat_id) -> list from /chats

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
        bot_info = await instance.get_bot_info(bot_token, config)
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

    async def send_outbound(
        self,
        user_id: str,
        provider: str,
        connection_id: str = "",
        message: str = "",
        format: str = "markdown",
        recipient_chat_id: str = "",
    ) -> Dict[str, Any]:
        """Send a proactive outbound message to a user via a messaging bot.

        If connection_id is empty, finds the first running bot for the given
        provider that belongs to the user. If recipient_chat_id is set, sends to
        that chat/channel; otherwise resolves the outbound_chat_id from the
        in-memory cache or by loading from the backend.
        """
        if not message:
            return {"success": False, "message_id": "", "channel": provider, "error": "Empty message"}

        # Resolve bot entry
        entry: Optional[Dict[str, Any]] = None
        resolved_connection_id = connection_id

        if connection_id and connection_id in self._bots:
            entry = self._bots[connection_id]
        elif connection_id:
            return {
                "success": False, "message_id": "", "channel": provider,
                "error": f"Bot not running for connection_id={connection_id}",
            }
        else:
            # Find first running bot for this provider and user
            for cid, bot_entry in self._bots.items():
                if bot_entry.get("provider") == provider and bot_entry.get("user_id") == user_id:
                    entry = bot_entry
                    resolved_connection_id = cid
                    break
            if entry is None:
                return {
                    "success": False, "message_id": "", "channel": provider,
                    "error": f"No running {provider} bot found for user {user_id}",
                }

        # Resolve destination chat_id: explicit recipient override, or stored outbound_chat_id
        chat_id = (recipient_chat_id or "").strip()
        if not chat_id:
            chat_id = self._outbound_chat_ids.get(resolved_connection_id)
        if not chat_id:
            result = await self._backend_client.get_outbound_chat_id(resolved_connection_id)
            chat_id = result.get("outbound_chat_id", "")
            if chat_id:
                self._outbound_chat_ids[resolved_connection_id] = chat_id

        if not chat_id:
            return {
                "success": False, "message_id": "", "channel": provider,
                "error": (
                    f"No outbound chat_id for connection {resolved_connection_id}. "
                    "The user must send at least one message to the bot first."
                ),
            }

        instance: BaseMessagingProvider = entry["instance"]
        try:
            result = await instance.send_message(chat_id, message[:4000], None)
            success = result.get("success", False)
            return {
                "success": success,
                "message_id": result.get("message_id", ""),
                "channel": provider,
                "error": result.get("error") if not success else None,
            }
        except Exception as e:
            logger.exception("send_outbound failed for connection %s: %s", resolved_connection_id, e)
            return {"success": False, "message_id": "", "channel": provider, "error": str(e)}

    async def _get_current_conversation_id(self, connection_id: str, platform: str, chat_id: str) -> str:
        """Return the current conversation id for this chat, or the default deterministic one.
        On cache miss, restores from provider_metadata (backend) so selection survives restarts.
        """
        key = (connection_id, chat_id)
        default = f"{platform}:{connection_id}:{chat_id}"
        if key in self._current_conversation:
            return self._current_conversation[key]
        result = await self._backend_client.get_active_conversation(connection_id, chat_id)
        stored = (result or {}).get("conversation_id", "").strip()
        if stored:
            self._current_conversation[key] = stored
            return stored
        return default

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
        conversation_id = await self._get_current_conversation_id(connection_id, msg.platform, msg.chat_id)
        text = (msg.text or "").strip()

        if msg.audio:
            aud = msg.audio or {}
            raw = aud.get("data") or ""
            try:
                if isinstance(raw, str):
                    audio_bytes = base64.b64decode(raw)
                elif isinstance(raw, (bytes, bytearray)):
                    audio_bytes = bytes(raw)
                else:
                    raise ValueError("invalid audio data type")
            except Exception as e:
                logger.warning("Invalid inbound audio payload: %s", e)
                await instance.send_message(msg.chat_id, "Sorry, invalid audio data.", None)
                return
            filename = (aud.get("filename") or "voice.bin").strip() or "voice.bin"
            mime = aud.get("mime") or "application/octet-stream"
            whisper_prompt = aud.get("prompt")
            wp = str(whisper_prompt).strip() if whisper_prompt is not None else ""
            tr = await self._backend_client.transcribe_audio(
                user_id=user_id,
                audio_bytes=audio_bytes,
                filename=filename,
                content_type=mime,
                prompt=wp or None,
            )
            if tr.get("error"):
                await instance.send_message(
                    msg.chat_id,
                    f"Sorry, I couldn't transcribe that: {tr['error'][:200]}",
                    None,
                )
                return
            transcript = (tr.get("text") or "").strip()
            if not transcript:
                await instance.send_message(
                    msg.chat_id,
                    "Sorry, transcription was empty.",
                    None,
                )
                return
            text = f"{text}\n\n{transcript}".strip() if text else transcript

        text_lower = text.lower()

        # Persist outbound chat_id for proactive messaging (fire-and-forget)
        if msg.chat_id and connection_id not in self._outbound_chat_ids:
            self._outbound_chat_ids[connection_id] = msg.chat_id
            asyncio.create_task(
                self._backend_client.update_outbound_chat_id(
                    connection_id=connection_id,
                    chat_id=msg.chat_id,
                    sender_name=msg.sender_name or "",
                )
            )
        # Record this chat in known_chats for recipient dropdown (fire-and-forget)
        if msg.chat_id and connection_id:
            asyncio.create_task(
                self._backend_client.add_known_chat(
                    connection_id=connection_id,
                    chat_id=msg.chat_id,
                    chat_title=getattr(msg, "chat_title", None) or "",
                    chat_username=getattr(msg, "chat_username", None) or "",
                    chat_type=getattr(msg, "chat_type", None) or "",
                )
            )

        if text_lower in ("/chatid", "/chat_id"):
            title_part = f"\nTitle: {msg.chat_title}" if getattr(msg, "chat_title", None) else ""
            type_part = f"\nType: {msg.chat_type}" if getattr(msg, "chat_type", None) else ""
            username_part = f"\nUsername: {msg.chat_username}" if getattr(msg, "chat_username", None) else ""
            reply = f"This chat's ID: `{msg.chat_id}`{title_part}{type_part}{username_part}"
            await instance.send_message(msg.chat_id, reply, None)
            return

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
            result = await self._backend_client.list_models(user_id, conversation_id=conversation_id or "")
            if result.get("error"):
                reply = f"Sorry: {result.get('error', 'Unknown error')[:200]}"
            else:
                models = result.get("models", [])
                if not models:
                    reply = "No enabled models configured. Ask an admin to enable models in Settings."
                else:
                    current_id = result.get("current_model_id")
                    line_parts = []
                    for i, m in enumerate(models):
                        idx = m.get("index", i + 1)
                        name = m.get("name", m.get("id", "?"))
                        if current_id and m.get("id") == current_id:
                            line_parts.append(f"{idx}) **{name}** *")
                        else:
                            line_parts.append(f"{idx}) {name}")
                    reply = "Reply with /model <number> to select a model.\n\n" + "\n".join(line_parts)
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

        if text_lower == "/chats":
            result = await self._backend_client.list_user_conversations(user_id, limit=10)
            if result.get("error"):
                reply = f"Sorry: {result.get('error', 'Unknown error')[:200]}"
            else:
                convos = result.get("conversations") or []
                self._chat_listing_cache[chat_key] = convos
                if not convos:
                    reply = "No conversations yet. Send a message to start one, or use /newchat."
                else:
                    lines = ["Your conversations:"]
                    for i, c in enumerate(convos, 1):
                        title = (c.get("title") or "Untitled")[:50]
                        when = _format_relative_time(c.get("updated_at"))
                        count = c.get("message_count", 0)
                        lines.append(f"{i}) {title} ({when}, {count} msgs)")
                    lines.append("")
                    lines.append("Use /loadchat N to switch, /newchat to start fresh.")
                    current_title = None
                    for c in convos:
                        if c.get("conversation_id") == conversation_id:
                            current_title = c.get("title") or "Untitled"
                            break
                    if current_title:
                        lines.append(f"Currently in: {current_title[:50]}")
                    reply = "\n".join(lines)
            await instance.send_message(msg.chat_id, reply, None)
            return

        if text_lower.startswith("/loadchat "):
            rest = text_lower[len("/loadchat "):].strip()
            if not rest.isdigit():
                reply = "Use /loadchat N (e.g. /loadchat 2). Run /chats to see the list."
                await instance.send_message(msg.chat_id, reply, None)
                return
            idx = int(rest)
            cached = self._chat_listing_cache.get(chat_key)
            if not cached or idx < 1 or idx > len(cached):
                reply = "Run /chats first to see your conversations, then use /loadchat N."
                await instance.send_message(msg.chat_id, reply, None)
                return
            chosen = cached[idx - 1]
            conv_id = chosen.get("conversation_id")
            if not conv_id:
                reply = "That conversation is no longer available. Run /chats again."
                await instance.send_message(msg.chat_id, reply, None)
                return
            val = await self._backend_client.validate_conversation(user_id, conv_id)
            if not val.get("valid"):
                reply = "That conversation is no longer available. Run /chats again."
                await instance.send_message(msg.chat_id, reply, None)
                return
            self._set_current_conversation_id(connection_id, msg.chat_id, conv_id)
            await self._backend_client.persist_active_conversation(
                connection_id=connection_id, chat_id=msg.chat_id, conversation_id=conv_id
            )
            title = (val.get("title") or "Untitled Conversation").strip()[:50]
            reply = f"Switched to: {title}"
            await instance.send_message(msg.chat_id, reply, None)
            return

        if text_lower in ("/chat", "/current"):
            val = await self._backend_client.validate_conversation(user_id, conversation_id)
            title = (val.get("title") or "Untitled Conversation").strip() if val.get("valid") else "(new chat)"
            when = "unknown"
            count_str = ""
            cached = self._chat_listing_cache.get(chat_key)
            if cached:
                for c in cached:
                    if c.get("conversation_id") == conversation_id:
                        when = _format_relative_time(c.get("updated_at"))
                        count_str = f"{c.get('message_count', 0)} msgs"
                        break
            if not count_str:
                count_str = "—"
            reply = f"Current chat: {title[:50]}\nMessages: {count_str} | Last active: {when}\nUse /chats to see all, /newchat to start fresh."
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
                query=text,
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
