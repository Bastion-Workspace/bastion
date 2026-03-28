"""
OpenAI TTS provider (cloud).
"""

import asyncio
import threading
from typing import List, Optional

from config.settings import settings
from providers.tts.base_tts_provider import (
    BaseTTSProvider,
    SynthesisResult,
    VoiceInfo,
)

OPENAI_VOICES = (
    "alloy",
    "echo",
    "fable",
    "onyx",
    "nova",
    "shimmer",
    "ash",
    "ballad",
    "coral",
    "sage",
    "verse",
    "marin",
    "cedar",
)


class OpenAIProvider(BaseTTSProvider):
    """OpenAI text-to-speech provider."""

    def __init__(self) -> None:
        self._clients: dict = {}

    def provider_name(self) -> str:
        return "openai"

    def _effective_key(self, api_key: str) -> str:
        return (api_key or "").strip() or (settings.OPENAI_API_KEY or "")

    def _effective_base_url(self, base_url: str) -> Optional[str]:
        u = (base_url or "").strip()
        return u if u else None

    def _get_client(self, api_key: str = "", base_url: str = ""):
        from openai import OpenAI

        key = self._effective_key(api_key)
        bu = self._effective_base_url(base_url)
        cache_key = (key, bu or "")
        if cache_key not in self._clients:
            if bu:
                self._clients[cache_key] = OpenAI(api_key=key, base_url=bu)
            else:
                self._clients[cache_key] = OpenAI(api_key=key)
        return self._clients[cache_key]

    async def is_available(self, api_key: str = "", base_url: str = "") -> bool:
        if not self._effective_key(api_key):
            return False
        return True

    async def synthesize(
        self,
        text: str,
        voice_id: str,
        output_format: str,
        api_key: str = "",
        base_url: str = "",
        model_id: str = "",
    ) -> SynthesisResult:
        del model_id
        client = self._get_client(api_key, base_url)
        voice = (voice_id or settings.OPENAI_TTS_VOICE or "alloy").strip().lower()
        if voice not in OPENAI_VOICES:
            voice = "alloy"
        fmt = (output_format or "mp3").strip().lower()
        if fmt not in ("mp3", "opus", "aac", "flac", "wav", "pcm"):
            fmt = "mp3"
        resp = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.audio.speech.create(
                model=settings.OPENAI_TTS_MODEL or "tts-1",
                voice=voice,
                input=text,
                response_format=fmt,
            ),
        )
        audio_bytes = resp.content if hasattr(resp, "content") else resp.read()
        return SynthesisResult(
            audio_data=audio_bytes,
            audio_format=fmt,
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
        del model_id
        client = self._get_client(api_key, base_url)
        voice = (voice_id or settings.OPENAI_TTS_VOICE or "alloy").strip().lower()
        if voice not in OPENAI_VOICES:
            voice = "alloy"
        fmt = (output_format or "mp3").strip().lower()
        if fmt == "ogg":
            resp_fmt = "opus"
        elif fmt not in ("mp3", "opus", "aac", "flac", "wav", "pcm"):
            resp_fmt = "mp3"
        else:
            resp_fmt = fmt

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=32)

        def worker():
            try:
                with client.audio.speech.with_streaming_response.create(
                    model=settings.OPENAI_TTS_MODEL or "tts-1",
                    voice=voice,
                    input=text,
                    response_format=resp_fmt,
                ) as response:
                    for chunk in response.iter_bytes(chunk_size=4096):
                        if not chunk:
                            continue
                        fut = asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
                        fut.result()
                fut = asyncio.run_coroutine_threadsafe(queue.put(None), loop)
                fut.result()
            except Exception as e:
                fut = asyncio.run_coroutine_threadsafe(queue.put(e), loop)
                fut.result()

        threading.Thread(target=worker, daemon=True).start()

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    async def list_voices(
        self, api_key: str = "", base_url: str = ""
    ) -> List[VoiceInfo]:
        return [
            VoiceInfo(
                voice_id=v,
                name=v.capitalize(),
                provider="openai",
                language="en",
                gender="neutral",
            )
            for v in OPENAI_VOICES
        ]
