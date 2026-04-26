"""
OpenRouter TTS (OpenAI-compatible /v1/audio/speech).
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

# Suggested voices; model-specific lists live on OpenRouter / model pages.
OPENROUTER_SUGGESTED_VOICES = (
    "alloy",
    "echo",
    "fable",
    "onyx",
    "nova",
    "shimmer",
)


class OpenRouterProvider(BaseTTSProvider):
    """OpenRouter text-to-speech via OpenAI-compatible Audio API."""

    def __init__(self) -> None:
        self._clients: dict = {}

    def provider_name(self) -> str:
        return "openrouter"

    def _default_v1_root(self) -> str:
        return (settings.OPENROUTER_BASE_URL or "https://openrouter.ai/api/v1").strip().rstrip(
            "/"
        )

    def _effective_root(self, base_url: str) -> str:
        u = (base_url or "").strip().rstrip("/")
        if not u:
            u = self._default_v1_root()
        if not u.endswith("/v1"):
            u = f"{u}/v1"
        return u

    def _effective_key(self, api_key: str) -> str:
        return (api_key or "").strip() or (settings.OPENROUTER_API_KEY or "")

    def _get_client(self, api_key: str = "", base_url: str = ""):
        from openai import OpenAI

        key = self._effective_key(api_key)
        root = self._effective_root(base_url)
        cache_key = (key, root)
        if cache_key not in self._clients:
            self._clients[cache_key] = OpenAI(api_key=key, base_url=root)
        return self._clients[cache_key]

    def _effective_model(self, model_id: str) -> str:
        m = (model_id or "").strip() or (settings.OPENROUTER_TTS_MODEL or "").strip()
        return m

    def _map_response_format(self, output_format: str) -> str:
        fmt = (output_format or "mp3").strip().lower()
        if fmt in ("pcm", "wav"):
            return "pcm"
        return "mp3"

    async def is_available(self, api_key: str = "", base_url: str = "") -> bool:
        return bool(self._effective_key(api_key))

    async def synthesize(
        self,
        text: str,
        voice_id: str,
        output_format: str,
        api_key: str = "",
        base_url: str = "",
        model_id: str = "",
    ) -> SynthesisResult:
        model = self._effective_model(model_id)
        if not model:
            raise ValueError(
                "OpenRouter TTS requires a model_id or OPENROUTER_TTS_MODEL in voice-service env."
            )
        client = self._get_client(api_key, base_url)
        voice = (voice_id or settings.OPENROUTER_TTS_VOICE or "alloy").strip()
        resp_fmt = self._map_response_format(output_format)
        resp = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                response_format=resp_fmt,
            ),
        )
        audio_bytes = resp.content if hasattr(resp, "content") else resp.read()
        return SynthesisResult(
            audio_data=audio_bytes,
            audio_format=resp_fmt,
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
        model = self._effective_model(model_id)
        if not model:
            raise ValueError(
                "OpenRouter TTS requires a model_id or OPENROUTER_TTS_MODEL in voice-service env."
            )
        client = self._get_client(api_key, base_url)
        voice = (voice_id or settings.OPENROUTER_TTS_VOICE or "alloy").strip()
        out_fmt = (output_format or "mp3").strip().lower()
        resp_fmt = self._map_response_format("mp3" if out_fmt == "ogg" else out_fmt)

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=32)

        def worker():
            try:
                with client.audio.speech.with_streaming_response.create(
                    model=model,
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
                provider="openrouter",
                language="en",
                gender="neutral",
            )
            for v in OPENROUTER_SUGGESTED_VOICES
        ]
