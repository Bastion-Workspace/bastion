"""
Piper TTS provider (local).
"""

import asyncio
import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import List

from config.settings import settings
from providers.tts.base_tts_provider import (
    BaseTTSProvider,
    SynthesisResult,
    VoiceInfo,
)

logger = logging.getLogger(__name__)


class PiperProvider(BaseTTSProvider):
    """
    Local Piper text-to-speech provider.

    Streaming path pipes PCM from Piper into ffmpeg (libopus + OGG) for
    incremental chunks suitable for MediaSource playback.
    """

    def __init__(self) -> None:
        self._voice = None
        self._voice_lock = asyncio.Lock()

    def provider_name(self) -> str:
        return "piper"

    def _find_model_path(self, voice_id: str) -> str:
        base = Path(settings.PIPER_MODEL_PATH)
        if not base.exists():
            return ""
        preferred = voice_id or settings.PIPER_VOICE or ""
        for p in base.rglob("*.onnx"):
            name = p.stem
            if preferred and preferred in name:
                return str(p)
        for p in base.rglob("*.onnx"):
            return str(p)
        return ""

    async def _get_voice(self, voice_id: str = ""):
        async with self._voice_lock:
            if self._voice is not None:
                return self._voice
            path = self._find_model_path(voice_id)
            if not path:
                raise FileNotFoundError(
                    f"No Piper .onnx model found under {settings.PIPER_MODEL_PATH}"
                )
            config_path = path + ".json"
            if not os.path.isfile(config_path):
                config_path = ""
            loop = asyncio.get_event_loop()
            try:
                from piper import PiperVoice
                self._voice = await loop.run_in_executor(
                    None,
                    lambda: PiperVoice.load(path, config_path or None),
                )
            except ImportError:
                from piper.voice import PiperVoice
                self._voice = await loop.run_in_executor(
                    None,
                    lambda: PiperVoice.load(path, config_path or None),
                )
            return self._voice

    async def is_available(self, api_key: str = "", base_url: str = "") -> bool:
        del api_key, base_url
        try:
            path = self._find_model_path("")
            return bool(path)
        except Exception:
            return False

    async def synthesize(
        self,
        text: str,
        voice_id: str,
        output_format: str,
        api_key: str = "",
        base_url: str = "",
        model_id: str = "",
    ) -> SynthesisResult:
        del api_key, base_url, model_id
        voice = await self._get_voice(voice_id)
        loop = asyncio.get_event_loop()

        def _chunk_to_pcm(chunk) -> bytes:
            if isinstance(chunk, (bytes, bytearray)):
                return bytes(chunk)
            # piper-tts 1.2+ yields AudioChunk dataclasses (not raw bytes)
            if hasattr(chunk, "audio_int16_bytes"):
                return chunk.audio_int16_bytes
            raise TypeError(
                f"Piper yielded unexpected chunk type: {type(chunk).__name__}"
            )

        def _collect_pcm() -> bytes:
            parts: List[bytes] = []
            if hasattr(voice, "synthesize_stream_raw"):
                iterator = voice.synthesize_stream_raw(text)
            else:
                iterator = voice.synthesize(text)
            for chunk in iterator:
                parts.append(_chunk_to_pcm(chunk))
            return b"".join(parts)

        raw = await loop.run_in_executor(None, _collect_pcm)
        fmt = (output_format or "wav").strip().lower()
        try:
            pcm_rate = int(voice.config.sample_rate)
        except (TypeError, ValueError, AttributeError):
            pcm_rate = 22050

        # Piper yields raw s16le PCM (no RIFF header); do not pass source_format "wav".
        if raw:
            from service.audio_utils import convert_audio

            if fmt == "wav":
                raw = convert_audio(
                    raw, "pcm", "wav", source_pcm_frame_rate=pcm_rate
                )
            elif fmt in ("mp3", "ogg", "opus", "flac", "webm"):
                target = "ogg" if fmt == "opus" else fmt
                raw = convert_audio(
                    raw, "pcm", target, source_pcm_frame_rate=pcm_rate
                )

        return SynthesisResult(
            audio_data=raw,
            audio_format=fmt or "wav",
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
        del api_key, base_url, model_id
        fmt = (output_format or "ogg").strip().lower()
        if fmt == "opus":
            fmt = "ogg"
        if fmt != "ogg":
            async for chunk in super().synthesize_stream(
                text,
                voice_id,
                output_format,
                api_key="",
                base_url="",
                model_id="",
            ):
                if chunk:
                    yield chunk
            return

        voice = await self._get_voice(voice_id)
        try:
            pcm_rate = int(voice.config.sample_rate)
        except (TypeError, ValueError, AttributeError):
            pcm_rate = 22050

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=64)

        def _chunk_to_pcm(chunk) -> bytes:
            if isinstance(chunk, (bytes, bytearray)):
                return bytes(chunk)
            if hasattr(chunk, "audio_int16_bytes"):
                return chunk.audio_int16_bytes
            raise TypeError(
                f"Piper yielded unexpected chunk type: {type(chunk).__name__}"
            )

        def worker():
            proc = None
            try:
                proc = subprocess.Popen(
                    [
                        "ffmpeg",
                        "-loglevel",
                        "error",
                        "-f",
                        "s16le",
                        "-ar",
                        str(pcm_rate),
                        "-ac",
                        "1",
                        "-i",
                        "pipe:0",
                        "-c:a",
                        "libopus",
                        "-f",
                        "ogg",
                        "pipe:1",
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )

                def feed_piper():
                    try:
                        if hasattr(voice, "synthesize_stream_raw"):
                            iterator = voice.synthesize_stream_raw(text)
                        else:
                            iterator = voice.synthesize(text)
                        for chunk in iterator:
                            pcm = _chunk_to_pcm(chunk)
                            if pcm and proc.stdin:
                                proc.stdin.write(pcm)
                    except Exception as err:
                        logger.warning("Piper stream feed failed: %s", err)
                    finally:
                        try:
                            if proc.stdin:
                                proc.stdin.close()
                        except Exception:
                            pass

                threading.Thread(target=feed_piper, daemon=True).start()

                assert proc.stdout is not None
                while True:
                    data = proc.stdout.read(8192)
                    if not data:
                        break
                    fut = asyncio.run_coroutine_threadsafe(queue.put(data), loop)
                    fut.result(timeout=120.0)
                proc.wait(timeout=60)
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result(
                    timeout=5.0
                )
            except Exception as e:
                if proc and proc.poll() is None:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                asyncio.run_coroutine_threadsafe(queue.put(e), loop).result(
                    timeout=5.0
                )

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
        del api_key, base_url
        base = Path(settings.PIPER_MODEL_PATH)
        voices = []
        if base.exists():
            for p in base.rglob("*.onnx"):
                name = p.stem
                voices.append(
                    VoiceInfo(
                        voice_id=name,
                        name=name,
                        provider="piper",
                        language="en",
                        gender="neutral",
                    )
                )
        return voices
