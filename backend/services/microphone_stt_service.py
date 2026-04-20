"""
Shared STT for web microphone and internal (connections-service) transcribe API.
Uses per-user STT from user_voice_provider_service, else admin OPENAI_API_KEY.
"""

import io
import logging
from typing import Any, Dict, Optional

import httpx

from config import settings
from services.user_voice_provider_service import user_voice_provider_service

logger = logging.getLogger(__name__)

# Align with Telegram bot voice limits (~20MB practical); avoid huge uploads.
MAX_TRANSCRIBE_AUDIO_BYTES = 25 * 1024 * 1024


async def transcribe_audio_bytes_for_user(
    user_id: str,
    audio_bytes: bytes,
    filename: str,
    content_type: Optional[str] = None,
    prompt: Optional[str] = None,
) -> str:
    """
    Transcribe audio bytes for a user. Same provider resolution as /api/audio/transcribe.

    Raises:
        ValueError: empty payload, oversize file, or missing filename when required by caller.
        RuntimeError: no STT configured, provider failure, or empty transcript (message is user-safe detail where appropriate).
    """
    fn = (filename or "").strip() or "audio.bin"
    if not audio_bytes:
        raise ValueError("Empty audio file")
    if len(audio_bytes) > MAX_TRANSCRIBE_AUDIO_BYTES:
        raise ValueError(
            f"Audio exceeds maximum size ({MAX_TRANSCRIBE_AUDIO_BYTES // (1024 * 1024)}MB)"
        )

    ctx = await user_voice_provider_service.get_voice_context(user_id, "stt")
    transcript_text: Optional[str] = None

    try:
        if ctx and ctx.get("provider_type") == "deepgram":
            ctype = content_type or "application/octet-stream"
            async with httpx.AsyncClient(timeout=120.0) as http_client:
                r = await http_client.post(
                    "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true",
                    headers={
                        "Authorization": f"Token {ctx['api_key']}",
                        "Content-Type": ctype,
                    },
                    content=audio_bytes,
                )
                r.raise_for_status()
                data = r.json()
                channels = (data.get("results") or {}).get("channels") or []
                if channels:
                    alts = channels[0].get("alternatives") or []
                    if alts:
                        transcript_text = alts[0].get("transcript")

        elif ctx and ctx.get("provider_type") in ("openai", "whisper_api"):
            from openai import AsyncOpenAI

            bu = (ctx.get("base_url") or "").strip() or None
            client = AsyncOpenAI(api_key=ctx["api_key"], base_url=bu)
            model_name = "whisper-1"
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = fn
            create_kwargs: Dict[str, Any] = {
                "model": model_name,
                "file": audio_file,
            }
            if prompt:
                create_kwargs["prompt"] = prompt
            result = await client.audio.transcriptions.create(**create_kwargs)
            transcript_text = getattr(result, "text", None) or getattr(
                result, "data", {}
            ).get("text")

        elif settings.OPENAI_API_KEY:
            from openai import AsyncOpenAI

            logger.info("Using OpenAI Whisper for transcription (admin key)")
            client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            model_name = "whisper-1"
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = fn
            create_kwargs: Dict[str, Any] = {
                "model": model_name,
                "file": audio_file,
            }
            if prompt:
                create_kwargs["prompt"] = prompt
            result = await client.audio.transcriptions.create(**create_kwargs)
            transcript_text = getattr(result, "text", None) or getattr(
                result, "data", {}
            ).get("text")

        elif settings.OPENROUTER_API_KEY:
            raise RuntimeError(
                "OpenRouter does not support Whisper audio transcription in this setup. "
                "Configure OPENAI_API_KEY or add your own STT provider in Settings."
            )
        else:
            raise RuntimeError(
                "No API key configured for transcription. "
                "Configure OPENAI_API_KEY or add your own STT provider in Settings."
            )

    except RuntimeError:
        raise
    except Exception as e:
        logger.warning("Primary transcription path failed: %s", e)
        transcript_text = None

    if not transcript_text:
        raise RuntimeError("Transcription service unavailable")

    out = (
        transcript_text.strip()
        if isinstance(transcript_text, str)
        else str(transcript_text).strip()
    )
    if not out:
        raise RuntimeError("Transcription service unavailable")
    return out
