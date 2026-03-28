"""Text-to-Speech providers."""

from providers.tts.base_tts_provider import (
    BaseTTSProvider,
    SynthesisResult,
    VoiceInfo,
)

__all__ = ["BaseTTSProvider", "SynthesisResult", "VoiceInfo"]
