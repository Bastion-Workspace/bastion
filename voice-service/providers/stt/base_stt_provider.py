"""
Base Speech-to-Text provider interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TranscriptionResult:
    """Result of a transcription (STT) call."""

    text: str
    detected_language: str = "en"
    confidence: float = 0.0
    duration_seconds: float = 0.0


class BaseSTTProvider(ABC):
    """Abstract base class for STT providers."""

    @abstractmethod
    async def transcribe(
        self,
        audio_data: bytes,
        audio_format: str,
        language: str,
        api_key: str = "",
        base_url: str = "",
    ) -> TranscriptionResult:
        """Transcribe audio to text. Optional api_key/base_url for cloud STT (unused by local whisper)."""
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if provider is configured and ready."""
        pass

    @abstractmethod
    def provider_name(self) -> str:
        """Provider identifier (e.g. 'whisper')."""
        pass
