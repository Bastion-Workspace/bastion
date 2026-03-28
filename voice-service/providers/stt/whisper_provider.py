"""
Whisper STT provider using faster-whisper (lazy model loading).
"""

import asyncio
import logging
import tempfile
from pathlib import Path

from config.settings import settings
from providers.stt.base_stt_provider import BaseSTTProvider, TranscriptionResult

logger = logging.getLogger(__name__)


class WhisperProvider(BaseSTTProvider):
    """faster-whisper based speech-to-text provider."""

    def __init__(self) -> None:
        self._model = None
        self._model_lock = asyncio.Lock()

    def provider_name(self) -> str:
        return "whisper"

    async def _get_model(self):
        """Load model on first use (lazy init)."""
        async with self._model_lock:
            if self._model is not None:
                return self._model
            device = settings.get_whisper_device()
            compute_type = (
                "float16" if device == "cuda" else settings.WHISPER_COMPUTE_TYPE
            )
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: __import__("faster_whisper").WhisperModel(
                    settings.WHISPER_MODEL_SIZE,
                    device=device,
                    compute_type=compute_type,
                ),
            )
            logger.info(
                "Whisper model loaded: size=%s device=%s",
                settings.WHISPER_MODEL_SIZE,
                device,
            )
            return self._model

    async def is_available(self) -> bool:
        try:
            await self._get_model()
            return True
        except Exception as e:
            logger.warning("Whisper not available: %s", e)
            return False

    async def transcribe(
        self,
        audio_data: bytes,
        audio_format: str,
        language: str,
        api_key: str = "",
        base_url: str = "",
    ) -> TranscriptionResult:
        from service.audio_utils import convert_to_wav

        wav_bytes = convert_to_wav(audio_data, audio_format or "wav")
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as tmp:
            tmp.write(wav_bytes)
            path = tmp.name
        try:
            model = await self._get_model()
            lang_param = None if (language or "").strip().lower() == "auto" else (language or "en")
            loop = asyncio.get_event_loop()
            segments, info = await loop.run_in_executor(
                None,
                lambda: model.transcribe(path, language=lang_param),
            )
            text_parts = [s.text for s in segments]
            text = " ".join(text_parts).strip() if text_parts else ""
            detected = getattr(info, "language", "en") or "en"
            return TranscriptionResult(
                text=text,
                detected_language=detected,
                confidence=getattr(info, "language_probability", 0.0) or 0.0,
                duration_seconds=0.0,
            )
        finally:
            Path(path).unlink(missing_ok=True)
