"""
Voice Service Client - gRPC client for the voice-service (STT/TTS).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import grpc
from protos import voice_service_pb2, voice_service_pb2_grpc

logger = logging.getLogger(__name__)

_GRPC_MESSAGE_LIMIT = 100 * 1024 * 1024

_voice_client: Optional["VoiceServiceClient"] = None


class VoiceServiceClient:
    """gRPC client for the voice-service (Transcribe, Synthesize)."""

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
        self.host = host or os.getenv("VOICE_SERVICE_HOST", "voice-service")
        self.port = port or int(os.getenv("VOICE_SERVICE_PORT", "50059"))
        self.address = f"{self.host}:{self.port}"
        self._channel: Optional[grpc.aio.Channel] = None
        self._stub: Optional[voice_service_pb2_grpc.VoiceServiceStub] = None
        logger.info("Voice Service Client configured for %s", self.address)

    async def connect(self) -> None:
        if self._channel is None:
            options = [
                ("grpc.max_send_message_length", _GRPC_MESSAGE_LIMIT),
                ("grpc.max_receive_message_length", _GRPC_MESSAGE_LIMIT),
            ]
            self._channel = grpc.aio.insecure_channel(self.address, options=options)
            self._stub = voice_service_pb2_grpc.VoiceServiceStub(self._channel)

    async def close(self) -> None:
        if self._channel:
            await self._channel.close()
            self._channel = None
            self._stub = None

    async def _ensure_connected(self) -> None:
        if self._stub is None:
            await self.connect()

    async def transcribe(
        self,
        audio_data: bytes,
        audio_format: str = "wav",
        language: str = "auto",
        provider: str = "",
    ) -> Dict[str, Any]:
        """Transcribe audio to text. Returns dict with text, detected_language, confidence, duration_seconds, error."""
        await self._ensure_connected()
        req = voice_service_pb2.TranscribeRequest(
            audio_data=audio_data,
            audio_format=audio_format,
            language=language,
            provider=provider,
        )
        try:
            resp = await self._stub.Transcribe(req, timeout=60.0)
            out = {
                "text": resp.text,
                "detected_language": resp.detected_language,
                "confidence": resp.confidence,
                "duration_seconds": resp.duration_seconds,
            }
            if resp.HasField("error") and resp.error:
                out["error"] = resp.error
            return out
        except grpc.RpcError as e:
            return {
                "text": "",
                "detected_language": "en",
                "confidence": 0.0,
                "duration_seconds": 0.0,
                "error": e.details() or str(e),
            }


def get_voice_service_client() -> VoiceServiceClient:
    """Return the singleton Voice service client."""
    global _voice_client
    if _voice_client is None:
        _voice_client = VoiceServiceClient()
    return _voice_client
