"""
Voice Service Configuration
"""

import os
from typing import Optional


class Settings:
    """Voice service settings from environment variables."""

    # Service Configuration
    SERVICE_NAME: str = os.getenv("SERVICE_NAME", "voice-service")
    GRPC_PORT: int = int(os.getenv("GRPC_PORT", "50059"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    SERVICE_VERSION: str = os.getenv("SERVICE_VERSION", "1.0.0")  # voice-service

    # STT (Speech-to-Text) - default provider
    VOICE_STT_PROVIDER: str = os.getenv("VOICE_STT_PROVIDER", "whisper")

    # TTS (Text-to-Speech) - default provider
    VOICE_TTS_PROVIDER: str = os.getenv("VOICE_TTS_PROVIDER", "elevenlabs")

    # Whisper (faster-whisper)
    WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "small")
    WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "auto")
    WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

    # ElevenLabs
    ELEVENLABS_API_KEY: Optional[str] = os.getenv("ELEVENLABS_API_KEY")

    # Hedra TTS (and other generations)
    HEDRA_API_KEY: Optional[str] = os.getenv("HEDRA_API_KEY")

    # OpenAI TTS
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_TTS_MODEL: str = os.getenv("OPENAI_TTS_MODEL", "tts-1")
    OPENAI_TTS_VOICE: str = os.getenv("OPENAI_TTS_VOICE", "alloy")

    # OpenRouter TTS (OpenAI-compatible /audio/speech)
    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    OPENROUTER_TTS_MODEL: str = os.getenv("OPENROUTER_TTS_MODEL", "")
    OPENROUTER_TTS_VOICE: str = os.getenv("OPENROUTER_TTS_VOICE", "alloy")

    # Piper (local)
    PIPER_MODEL_PATH: str = os.getenv("PIPER_MODEL_PATH", "/app/models/piper")
    PIPER_VOICE: str = os.getenv("PIPER_VOICE", "en_US-arctic-medium")

    # Safety / limits
    MAX_AUDIO_DURATION_SECONDS: int = int(
        os.getenv("MAX_AUDIO_DURATION_SECONDS", "300")
    )

    VALID_STT_PROVIDERS = frozenset({"whisper"})
    VALID_TTS_PROVIDERS = frozenset({"elevenlabs", "hedra", "openai", "openrouter", "piper"})

    @classmethod
    def validate(cls) -> None:
        """Validate provider names and API key presence when required."""
        if cls.VOICE_STT_PROVIDER not in cls.VALID_STT_PROVIDERS:
            raise ValueError(
                f"VOICE_STT_PROVIDER must be one of {cls.VALID_STT_PROVIDERS}, got: {cls.VOICE_STT_PROVIDER}"
            )
        if cls.VOICE_TTS_PROVIDER not in cls.VALID_TTS_PROVIDERS:
            raise ValueError(
                f"VOICE_TTS_PROVIDER must be one of {cls.VALID_TTS_PROVIDERS}, got: {cls.VOICE_TTS_PROVIDER}"
            )
        if cls.VOICE_TTS_PROVIDER == "elevenlabs" and not cls.ELEVENLABS_API_KEY:
            raise ValueError(
                "ELEVENLABS_API_KEY is required when VOICE_TTS_PROVIDER=elevenlabs"
            )
        if cls.VOICE_TTS_PROVIDER == "hedra" and not cls.HEDRA_API_KEY:
            raise ValueError(
                "HEDRA_API_KEY is required when VOICE_TTS_PROVIDER=hedra"
            )
        if cls.VOICE_TTS_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is required when VOICE_TTS_PROVIDER=openai"
            )
        if cls.VOICE_TTS_PROVIDER == "openrouter" and not cls.OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY is required when VOICE_TTS_PROVIDER=openrouter"
            )
        if cls.WHISPER_MODEL_SIZE not in (
            "tiny",
            "base",
            "small",
            "medium",
            "large-v2",
            "large-v3",
        ):
            raise ValueError(
                f"WHISPER_MODEL_SIZE must be tiny/base/small/medium/large-v2/large-v3, got: {cls.WHISPER_MODEL_SIZE}"
            )
        if cls.WHISPER_DEVICE not in ("cpu", "cuda", "auto"):
            raise ValueError(
                f"WHISPER_DEVICE must be cpu/cuda/auto, got: {cls.WHISPER_DEVICE}"
            )

    @classmethod
    def get_whisper_device(cls) -> str:
        """Resolve Whisper device (auto -> cpu or cuda)."""
        if cls.WHISPER_DEVICE != "auto":
            return cls.WHISPER_DEVICE
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"


# Global settings instance
settings = Settings()
