"""
Provider router - returns the appropriate provider implementation by name.
"""

import logging
from typing import Any, Dict, Optional, Type

from providers.base_provider import BaseProvider
from providers.base_messaging_provider import BaseMessagingProvider
from providers.microsoft_provider import MicrosoftGraphProvider
from providers.telegram_provider import TelegramProvider
from providers.discord_provider import DiscordProvider

logger = logging.getLogger(__name__)

_providers: Optional[Dict[str, BaseProvider]] = None
_messaging_providers: Optional[Dict[str, Type[BaseMessagingProvider]]] = None


def get_messaging_providers() -> Dict[str, Type[BaseMessagingProvider]]:
    """Return registry of messaging provider name -> class."""
    global _messaging_providers
    if _messaging_providers is None:
        _messaging_providers = {
            "telegram": TelegramProvider,
            "discord": DiscordProvider,
        }
    return _messaging_providers


def get_messaging_provider(provider_name: str) -> Optional[Type[BaseMessagingProvider]]:
    """Return messaging provider class for the given name."""
    providers = get_messaging_providers()
    name = (provider_name or "").strip().lower()
    return providers.get(name)


def get_providers() -> Dict[str, BaseProvider]:
    """Return registry of provider name -> implementation."""
    global _providers
    if _providers is None:
        _providers = {
            "microsoft": MicrosoftGraphProvider(),
            "": MicrosoftGraphProvider(),  # default
        }
    return _providers


def get_provider(provider_name: str) -> Optional[BaseProvider]:
    """Return provider for the given name, or default (microsoft) if empty."""
    providers = get_providers()
    name = (provider_name or "").strip().lower() or "microsoft"
    return providers.get(name) or providers.get("microsoft")
