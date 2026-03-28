"""
gRPC Service Implementation - Voice Service
"""

import grpc
import logging
import sys

sys.path.insert(0, "/app")

from config.settings import settings
from service.provider_router import get_stt_provider, get_tts_provider

logger = logging.getLogger(__name__)

# Proto imports (generated at Docker build)
from protos import voice_service_pb2, voice_service_pb2_grpc


class VoiceServiceImplementation(voice_service_pb2_grpc.VoiceServiceServicer):
    """Voice Service gRPC implementation."""

    def __init__(self) -> None:
        self._initialized = False

    async def initialize(self) -> None:
        """Pre-load default STT/TTS providers (warm up Whisper)."""
        try:
            stt = get_stt_provider()
            await stt.is_available()
            tts = get_tts_provider()
            await tts.is_available()
            self._initialized = True
            logger.info("Voice service initialized")
        except Exception as e:
            logger.warning("Voice service init warning: %s", e)
            self._initialized = True

    async def cleanup(self) -> None:
        """Release resources."""
        pass

    async def HealthCheck(self, request, context):
        """Health check endpoint."""
        try:
            device = "cpu"
            try:
                from config.settings import settings as s
                device = s.get_whisper_device()
            except Exception:
                pass
            return voice_service_pb2.HealthCheckResponse(
                status="healthy",
                service_version=settings.SERVICE_VERSION,
                device=device,
                details={
                    "stt_provider": settings.VOICE_STT_PROVIDER,
                    "tts_provider": settings.VOICE_TTS_PROVIDER,
                },
            )
        except Exception as e:
            logger.error("Health check failed: %s", e)
            return voice_service_pb2.HealthCheckResponse(
                status="unhealthy",
                service_version=settings.SERVICE_VERSION,
                device="unknown",
                details={"error": str(e)},
            )

    async def Transcribe(self, request, context):
        """Transcribe audio to text."""
        try:
            provider = get_stt_provider(request.provider or "")
            result = await provider.transcribe(
                request.audio_data,
                request.audio_format or "wav",
                request.language or "auto",
                api_key=request.api_key or "",
                base_url=request.base_url or "",
            )
            return voice_service_pb2.TranscribeResponse(
                text=result.text,
                detected_language=result.detected_language,
                confidence=result.confidence,
                duration_seconds=result.duration_seconds,
            )
        except Exception as e:
            logger.error("Transcribe failed: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return voice_service_pb2.TranscribeResponse(error=str(e))

    async def Synthesize(self, request, context):
        """Synthesize text to speech."""
        try:
            provider = get_tts_provider(request.provider or "")
            result = await provider.synthesize(
                request.text,
                request.voice_id or "",
                request.output_format or "mp3",
                api_key=request.api_key or "",
                base_url=request.base_url or "",
                model_id=request.model_id or "",
            )
            return voice_service_pb2.SynthesizeResponse(
                audio_data=result.audio_data,
                audio_format=result.audio_format,
                duration_seconds=result.duration_seconds,
            )
        except Exception as e:
            logger.error("Synthesize failed: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return voice_service_pb2.SynthesizeResponse(error=str(e))

    async def SynthesizeStream(self, request, context):
        """Stream synthesized audio chunks (lower time-to-first-byte)."""
        text = (request.text or "").strip()
        if not text:
            yield voice_service_pb2.SynthesizeChunk(
                error="Text is required",
                is_final=True,
            )
            return
        try:
            provider = get_tts_provider(request.provider or "")
            out_fmt = (request.output_format or "ogg").strip().lower()
            # ElevenLabs byte-stream API emits MP3 (unless caller requested WAV buffer path).
            if provider.provider_name() == "elevenlabs" and out_fmt != "wav":
                label_fmt = "mp3"
            else:
                label_fmt = out_fmt
            sent_audio = False
            async for chunk in provider.synthesize_stream(
                text,
                request.voice_id or "",
                out_fmt,
                api_key=request.api_key or "",
                base_url=request.base_url or "",
                model_id=request.model_id or "",
            ):
                if not chunk:
                    continue
                msg = voice_service_pb2.SynthesizeChunk(
                    audio_data=chunk,
                    is_final=False,
                )
                if not sent_audio:
                    msg.audio_format = label_fmt
                    sent_audio = True
                yield msg
            if not sent_audio:
                yield voice_service_pb2.SynthesizeChunk(
                    error="Voice service returned no audio",
                    is_final=True,
                )
                return
            yield voice_service_pb2.SynthesizeChunk(
                audio_format=label_fmt,
                is_final=True,
            )
        except Exception as e:
            logger.error("SynthesizeStream failed: %s", e)
            yield voice_service_pb2.SynthesizeChunk(error=str(e), is_final=True)

    async def ListVoices(self, request, context):
        """List available TTS voices (optionally filtered by provider)."""
        try:
            provider_name = (request.provider or "").strip().lower()
            if provider_name:
                provider = get_tts_provider(provider_name)
                req_key = request.api_key or ""
                req_base = request.base_url or ""
                if not await provider.is_available(req_key, req_base):
                    return voice_service_pb2.ListVoicesResponse(voices=[])
                voices_list = await provider.list_voices(
                    api_key=req_key, base_url=req_base
                )
            else:
                voices_list = []
                for name in ("elevenlabs", "openai", "piper"):
                    try:
                        p = get_tts_provider(name)
                        if await p.is_available():
                            voices_list.extend(await p.list_voices())
                    except Exception:
                        continue
            return voice_service_pb2.ListVoicesResponse(
                voices=[
                    voice_service_pb2.VoiceInfo(
                        voice_id=v.voice_id,
                        name=v.name,
                        provider=v.provider,
                        language=v.language,
                        gender=v.gender,
                    )
                    for v in voices_list
                ]
            )
        except Exception as e:
            logger.error("ListVoices failed: %s", e)
            return voice_service_pb2.ListVoicesResponse(error=str(e))
