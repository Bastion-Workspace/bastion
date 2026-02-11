"""
Base provider interface for messaging platforms (Telegram, Discord).
All messaging providers must implement this interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class InboundMessage:
    """Normalized inbound message from a messaging platform."""

    sender_id: str
    sender_name: str
    chat_id: str
    text: str
    images: List[Dict[str, Any]]  # [{"data": base64_or_bytes, "mime": "image/jpeg"}, ...]
    platform: str  # "telegram" or "discord"
    timestamp: Optional[str] = None  # ISO format
    connection_id: Optional[str] = None  # external_connections.id for routing


@dataclass
class OutboundImage:
    """Image to send back to the platform."""

    data: bytes
    mime: str = "image/png"
    caption: Optional[str] = None


MessageCallback = Callable[[InboundMessage], Any]


class BaseMessagingProvider(ABC):
    """Abstract base class for messaging platform providers (Telegram, Discord)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g. 'telegram', 'discord')."""
        pass

    @abstractmethod
    async def start(
        self,
        bot_token: str,
        config: Dict[str, Any],
        message_callback: MessageCallback,
        connection_id: str,
    ) -> None:
        """Start the bot listener. Invoke message_callback for each inbound message."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the bot listener."""
        pass

    @abstractmethod
    async def send_message(
        self,
        chat_id: str,
        text: str,
        images: Optional[List[OutboundImage]] = None,
    ) -> Dict[str, Any]:
        """Send a message (and optional images) to the chat. Returns success/error dict."""
        pass

    async def send_typing_indicator(self, chat_id: str) -> None:
        """Show a typing indicator in the chat (e.g. 'typing...'). No-op if not supported."""
        pass

    @abstractmethod
    async def get_bot_info(self, bot_token: str) -> Dict[str, Any]:
        """Validate token and return bot identity. Keys: username (str), error (optional)."""
        pass
