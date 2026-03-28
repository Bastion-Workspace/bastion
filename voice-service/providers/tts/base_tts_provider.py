"""
Base Text-to-Speech provider interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, List


@dataclass
class SynthesisResult:
    """Result of a TTS synthesis call."""

    audio_data: bytes
    audio_format: str
    duration_seconds: float = 0.0


@dataclass
class VoiceInfo:
    """Metadata for an available TTS voice."""

    voice_id: str
    name: str
    provider: str
    language: str = "en"
    gender: str = "neutral"


class BaseTTSProvider(ABC):
    """Abstract base class for TTS providers."""

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice_id: str,
        output_format: str,
        api_key: str = "",
        base_url: str = "",
        model_id: str = "",
    ) -> SynthesisResult:
        """Synthesize text to speech. Optional api_key/base_url override env defaults."""
        pass

    async def synthesize_stream(
        self,
        text: str,
        voice_id: str,
        output_format: str,
        api_key: str = "",
        base_url: str = "",
        model_id: str = "",
    ) -> AsyncIterator[bytes]:
        """
        Yield audio chunks as they arrive from the provider.
        Default implementation buffers via synthesize() and yields one chunk.
        """
        result = await self.synthesize(
            text, voice_id, output_format, api_key, base_url, model_id
        )
        if result.audio_data:
            yield result.audio_data

    @abstractmethod
    async def list_voices(
        self, api_key: str = "", base_url: str = ""
    ) -> List[VoiceInfo]:
        """Return available voices for this provider."""
        pass

    @abstractmethod
    async def is_available(self, api_key: str = "", base_url: str = "") -> bool:
        """Check if provider is configured and ready (per-request key when provided)."""
        pass

    @abstractmethod
    def provider_name(self) -> str:
        """Provider identifier."""
        pass
