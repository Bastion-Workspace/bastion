"""
Voice Service gRPC client for backend API routes.
"""

import logging
from typing import Any, AsyncIterator, Dict, List, Optional

import grpc

from config import settings

logger = logging.getLogger(__name__)

try:
    from protos import voice_service_pb2, voice_service_pb2_grpc
except ImportError:
    voice_service_pb2 = None
    voice_service_pb2_grpc = None


class VoiceServiceClient:
    """Client wrapper for voice-service gRPC calls."""

    def __init__(
        self,
        service_host: Optional[str] = None,
        service_port: Optional[int] = None,
    ) -> None:
        self.service_host = service_host or settings.VOICE_SERVICE_HOST
        self.service_port = service_port or settings.VOICE_SERVICE_PORT
        self.service_url = f"{self.service_host}:{self.service_port}"
        self.channel: Optional[grpc.aio.Channel] = None
        self.stub = None
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        if voice_service_pb2_grpc is None:
            raise RuntimeError("Voice service protobuf modules are unavailable")
        options = [
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ]
        self.channel = grpc.aio.insecure_channel(self.service_url, options=options)
        self.stub = voice_service_pb2_grpc.VoiceServiceStub(self.channel)
        req = voice_service_pb2.HealthCheckRequest()
        await self.stub.HealthCheck(req, timeout=5.0)
        self._initialized = True
        logger.info("Connected to Voice Service at %s", self.service_url)

    async def close(self) -> None:
        if self.channel:
            await self.channel.close()
        self.channel = None
        self.stub = None
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.initialize()

    async def _with_reconnect(self, coro):
        try:
            return await coro()
        except grpc.RpcError as e:
            logger.warning("Voice service RPC failed, reconnecting: %s", e)
            self._initialized = False
            if self.channel:
                try:
                    await self.channel.close()
                except Exception:
                    pass
            self.channel = None
            self.stub = None
            return await coro()

    async def health_check(self) -> Dict[str, Any]:
        async def _do():
            await self._ensure_initialized()
            req = voice_service_pb2.HealthCheckRequest()
            return await self.stub.HealthCheck(req, timeout=5.0)

        resp = await self._with_reconnect(_do)
        return {
            "status": resp.status,
            "service_version": resp.service_version,
            "device": resp.device,
            "details": dict(resp.details or {}),
        }

    async def synthesize(
        self,
        text: str,
        voice_id: str = "",
        provider: str = "",
        output_format: str = "mp3",
        api_key: str = "",
        base_url: str = "",
        model_id: str = "",
    ) -> Dict[str, Any]:
        async def _do():
            await self._ensure_initialized()
            req = voice_service_pb2.SynthesizeRequest(
                text=text,
                voice_id=voice_id or "",
                provider=provider or "",
                output_format=output_format or "mp3",
                api_key=api_key or "",
                base_url=base_url or "",
                model_id=model_id or "",
            )
            return await self.stub.Synthesize(req, timeout=120.0)

        resp = await self._with_reconnect(_do)
        out = {
            "audio_data": bytes(resp.audio_data),
            "audio_format": resp.audio_format or output_format or "mp3",
            "duration_seconds": resp.duration_seconds,
        }
        if resp.HasField("error") and resp.error:
            out["error"] = resp.error
        return out

    async def synthesize_stream(
        self,
        text: str,
        voice_id: str = "",
        provider: str = "",
        output_format: str = "ogg",
        api_key: str = "",
        base_url: str = "",
        model_id: str = "",
    ) -> AsyncIterator[bytes]:
        """Yield raw audio chunks from voice-service streaming RPC."""
        await self._ensure_initialized()
        req = voice_service_pb2.SynthesizeRequest(
            text=text,
            voice_id=voice_id or "",
            provider=provider or "",
            output_format=output_format or "ogg",
            api_key=api_key or "",
            base_url=base_url or "",
            model_id=model_id or "",
        )
        try:
            call = self.stub.SynthesizeStream(req, timeout=120.0)
            async for resp in call:
                err = getattr(resp, "error", "") or ""
                if err:
                    raise RuntimeError(err)
                if resp.audio_data:
                    yield bytes(resp.audio_data)
        except grpc.RpcError as e:
            raise RuntimeError(e.details() or str(e)) from e

    async def list_voices(
        self,
        provider: str = "",
        api_key: str = "",
        base_url: str = "",
    ) -> Dict[str, Any]:
        async def _do():
            await self._ensure_initialized()
            req = voice_service_pb2.ListVoicesRequest(
                provider=provider or "",
                api_key=api_key or "",
                base_url=base_url or "",
            )
            return await self.stub.ListVoices(req, timeout=30.0)

        resp = await self._with_reconnect(_do)
        voices: List[Dict[str, Any]] = [
            {
                "voice_id": v.voice_id,
                "name": v.name,
                "provider": v.provider,
                "language": v.language,
                "gender": v.gender,
            }
            for v in resp.voices
        ]
        out: Dict[str, Any] = {"voices": voices}
        if resp.HasField("error") and resp.error:
            out["error"] = resp.error
        return out


_voice_client: Optional[VoiceServiceClient] = None


async def get_voice_service_client() -> VoiceServiceClient:
    global _voice_client
    if _voice_client is None:
        _voice_client = VoiceServiceClient()
        await _voice_client.initialize()
    return _voice_client
