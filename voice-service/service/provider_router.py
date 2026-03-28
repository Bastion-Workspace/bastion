"""
Provider router - returns STT and TTS provider instances by name (singleton).
"""

import logging
from typing import Optional

from config.settings import settings
from providers.stt.base_stt_provider import BaseSTTProvider
from providers.stt.whisper_provider import WhisperProvider
from providers.tts.base_tts_provider import BaseTTSProvider
from providers.tts.elevenlabs_provider import ElevenLabsProvider
from providers.tts.openai_provider import OpenAIProvider
from providers.tts.piper_provider import PiperProvider

logger = logging.getLogger(__name__)

_stt_instances: dict = {}
_tts_instances: dict = {}

_STT_REGISTRY = {"whisper": WhisperProvider}
_TTS_REGISTRY = {
    "elevenlabs": ElevenLabsProvider,
    "openai": OpenAIProvider,
    "piper": PiperProvider,
}


def get_stt_provider(provider_name: str = "") -> BaseSTTProvider:
    """Return STT provider for the given name, or default from settings."""
    name = (provider_name or settings.VOICE_STT_PROVIDER or "whisper").strip().lower()
    if name not in _stt_instances:
        cls = _STT_REGISTRY.get(name) or WhisperProvider
        _stt_instances[name] = cls()
    return _stt_instances[name]


def get_tts_provider(provider_name: str = "") -> BaseTTSProvider:
    """Return TTS provider for the given name, or default from settings."""
    name = (provider_name or settings.VOICE_TTS_PROVIDER or "elevenlabs").strip().lower()
    if name not in _tts_instances:
        cls = _TTS_REGISTRY.get(name)
        if cls is None:
            cls = ElevenLabsProvider
            if not settings.ELEVENLABS_API_KEY and settings.OPENAI_API_KEY:
                cls = OpenAIProvider
            elif not settings.ELEVENLABS_API_KEY and not settings.OPENAI_API_KEY:
                cls = PiperProvider
        _tts_instances[name] = cls()
    return _tts_instances[name]
