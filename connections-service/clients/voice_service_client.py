"""
Voice Service gRPC Client (STT/TTS).
"""

import logging
from typing import Any, Dict, Optional

import grpc

logger = logging.getLogger(__name__)

# Proto imports (generated at Docker build when voice_service.proto is added)
try:
    from protos import voice_service_pb2, voice_service_pb2_grpc
except ImportError:
    voice_service_pb2 = None
    voice_service_pb2_grpc = None


_voice_client: Optional["VoiceServiceClient"] = None


class VoiceServiceClient:
    """gRPC client for the voice-service (Transcribe, Synthesize)."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> None:
        from config.settings import settings
        self.host = host or settings.VOICE_SERVICE_HOST
        self.port = port or settings.VOICE_SERVICE_PORT
        self._url = f"{self.host}:{self.port}"
        self._channel: Optional[grpc.aio.Channel] = None
        self._stub = None
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        if voice_service_pb2_grpc is None:
            raise RuntimeError("Voice service proto not available")
        options = [
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ]
        self._channel = grpc.aio.insecure_channel(self._url, options=options)
        self._stub = voice_service_pb2_grpc.VoiceServiceStub(self._channel)
        try:
            req = voice_service_pb2.HealthCheckRequest()
            resp = await self._stub.HealthCheck(req, timeout=5.0)
            if resp.status == "healthy":
                self._initialized = True
                logger.info("Voice service client connected at %s", self._url)
            else:
                logger.warning("Voice service health: %s", resp.status)
        except Exception as e:
            logger.error("Voice service connection failed: %s", e)
            await self._channel.close()
            raise

    async def close(self) -> None:
        if self._channel:
            await self._channel.close()
        self._initialized = False

    async def transcribe(
        self,
        audio_data: bytes,
        audio_format: str = "ogg",
        language: str = "auto",
        provider: str = "",
    ) -> Dict[str, Any]:
        """Transcribe audio to text. Returns dict with text, detected_language, confidence, duration_seconds, error."""
        if not self._initialized:
            await self.initialize()
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

    async def synthesize(
        self,
        text: str,
        voice_id: str = "",
        provider: str = "",
        output_format: str = "ogg",
        model_id: str = "",
    ) -> Dict[str, Any]:
        """Synthesize text to speech. Returns dict with audio_data, audio_format, duration_seconds, error."""
        if not self._initialized:
            await self.initialize()
        req = voice_service_pb2.SynthesizeRequest(
            text=text,
            voice_id=voice_id,
            provider=provider,
            output_format=output_format,
            model_id=model_id or "",
        )
        try:
            resp = await self._stub.Synthesize(req, timeout=60.0)
            out = {
                "audio_data": bytes(resp.audio_data),
                "audio_format": resp.audio_format,
                "duration_seconds": resp.duration_seconds,
            }
            if resp.HasField("error") and resp.error:
                out["error"] = resp.error
            return out
        except grpc.RpcError as e:
            return {
                "audio_data": b"",
                "audio_format": output_format,
                "duration_seconds": 0.0,
                "error": e.details() or str(e),
            }


async def get_voice_service_client() -> VoiceServiceClient:
    """Return singleton VoiceServiceClient (lazy init)."""
    global _voice_client
    if _voice_client is None:
        _voice_client = VoiceServiceClient()
        await _voice_client.initialize()
    return _voice_client
