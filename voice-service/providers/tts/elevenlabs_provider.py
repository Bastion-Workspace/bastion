"""
ElevenLabs TTS provider (cloud).
"""

import asyncio
import logging
from typing import List

from config.settings import settings
from providers.tts.base_tts_provider import (
    BaseTTSProvider,
    SynthesisResult,
    VoiceInfo,
)

logger = logging.getLogger(__name__)


class ElevenLabsProvider(BaseTTSProvider):
    """ElevenLabs text-to-speech provider."""

    def __init__(self) -> None:
        self._clients: dict = {}

    def provider_name(self) -> str:
        return "elevenlabs"

    def _effective_key(self, api_key: str) -> str:
        return (api_key or "").strip() or (settings.ELEVENLABS_API_KEY or "")

    def _get_client(self, api_key: str = ""):
        from elevenlabs.client import ElevenLabs

        key = self._effective_key(api_key)
        if key not in self._clients:
            self._clients[key] = ElevenLabs(api_key=key)
        return self._clients[key]

    async def is_available(self, api_key: str = "", base_url: str = "") -> bool:
        del base_url  # ElevenLabs uses fixed API host
        if not self._effective_key(api_key):
            return False
        try:
            client = self._get_client(api_key)
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: client.voices.get_all()
            )
            return True
        except Exception as e:
            logger.warning("ElevenLabs not available: %s", e)
            return False

    def _elevenlabs_output_format(self, output_format: str) -> tuple:
        """Return (canonical_format, elevenlabs_api_format).

        ElevenLabs no longer accepts ogg_44100_128; use opus_* or mp3_* from their
        documented output_format list (see API validation errors).
        """
        fmt = (output_format or "mp3").strip().lower()
        if fmt in ("ogg", "opus"):
            return "ogg", "opus_48000_128"
        if fmt == "wav":
            return "wav", "pcm_44100"
        return fmt or "mp3", "mp3_44100_128"

    async def synthesize(
        self,
        text: str,
        voice_id: str,
        output_format: str,
        api_key: str = "",
        base_url: str = "",
        model_id: str = "",
    ) -> SynthesisResult:
        del base_url
        client = self._get_client(api_key)
        fmt, out_fmt = self._elevenlabs_output_format(output_format)
        vid = voice_id or "21m00Tcm4TlvDq8ikWAM"
        mid = (model_id or "").strip()

        def _generate():
            kwargs = {
                "voice_id": vid,
                "text": text,
                "output_format": out_fmt,
            }
            if mid:
                kwargs["model_id"] = mid
            resp = client.text_to_speech.convert(**kwargs)
            if hasattr(resp, "__iter__") and not isinstance(resp, (bytes, str)):
                return b"".join(resp)
            if hasattr(resp, "read"):
                return resp.read()
            return resp if isinstance(resp, bytes) else b""

        audio_bytes = await asyncio.get_event_loop().run_in_executor(None, _generate)
        if fmt == "ogg":
            return SynthesisResult(
                audio_data=audio_bytes,
                audio_format="ogg",
                duration_seconds=0.0,
            )
        if fmt == "wav" and out_fmt == "pcm_44100":
            from service.audio_utils import convert_audio

            audio_bytes = convert_audio(audio_bytes, "s16le", "wav")
        return SynthesisResult(
            audio_data=audio_bytes,
            audio_format=fmt or "mp3",
            duration_seconds=0.0,
        )

    async def synthesize_stream(
        self,
        text: str,
        voice_id: str,
        output_format: str,
        api_key: str = "",
        base_url: str = "",
        model_id: str = "",
    ):
        del base_url
        fmt, out_fmt = self._elevenlabs_output_format(output_format)
        if fmt == "wav" and out_fmt == "pcm_44100":
            result = await self.synthesize(
                text,
                voice_id,
                output_format,
                api_key,
                base_url="",
                model_id=model_id,
            )
            if result.audio_data:
                yield result.audio_data
            return

        # Streaming API emits MP3 chunks only (no OGG stream); use mp3 for low TTFB.
        stream_out = "mp3_44100_128"
        client = self._get_client(api_key)
        vid = voice_id or "21m00Tcm4TlvDq8ikWAM"
        mid = (model_id or "").strip()
        loop = asyncio.get_running_loop()

        def _stream_kwargs():
            k = {"voice_id": vid, "text": text, "output_format": stream_out}
            if mid:
                k["model_id"] = mid
            return k

        def _make_stream_iterator():
            stream_fn = getattr(client.text_to_speech, "convert_as_stream", None)
            skw = _stream_kwargs()
            if stream_fn is None:
                resp = client.text_to_speech.convert(**skw)
                if hasattr(resp, "__iter__") and not isinstance(resp, (bytes, str)):
                    return iter(resp)
                if hasattr(resp, "read"):
                    data = resp.read()
                    return iter([data]) if data else iter([])
                if isinstance(resp, bytes):
                    return iter([resp]) if resp else iter([])
                return iter([])
            return iter(stream_fn(**skw))

        iterator = await loop.run_in_executor(None, _make_stream_iterator)
        _done = object()

        def _advance():
            try:
                return next(iterator)
            except StopIteration:
                return _done

        while True:
            chunk = await loop.run_in_executor(None, _advance)
            if chunk is _done:
                break
            if chunk:
                if isinstance(chunk, bytes):
                    yield chunk
                else:
                    yield bytes(chunk)

    async def list_voices(
        self, api_key: str = "", base_url: str = ""
    ) -> List[VoiceInfo]:
        del base_url
        client = self._get_client(api_key)
        resp = await asyncio.get_event_loop().run_in_executor(
            None, lambda: client.voices.get_all()
        )
        voices = []
        for v in getattr(resp, "voices", []) or []:
            voices.append(
                VoiceInfo(
                    voice_id=getattr(v, "voice_id", "") or getattr(v, "id", ""),
                    name=getattr(v, "name", "Unknown"),
                    provider="elevenlabs",
                    language="en",
                    gender=getattr(v, "labels", {}).get("gender", "neutral")
                    if hasattr(v, "labels")
                    else "neutral",
                )
            )
        return voices
