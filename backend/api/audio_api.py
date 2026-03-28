"""
Audio Transcription API

Provides a simple endpoint to accept microphone recordings from the frontend
and transcribe them using the configured provider. The request is a multipart
form with a single file field named "file".
"""

import io
import logging
from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends

from utils.auth_middleware import get_current_user, AuthenticatedUserResponse
from config import settings
from services.user_voice_provider_service import user_voice_provider_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Audio"])


@router.post("/api/audio/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> Dict[str, Any]:
    """Transcribe an uploaded audio file and return plain text.

    Accepts common webm/ogg/mp3/wav containers from MediaRecorder.
    Uses per-user STT credentials when configured, else admin OpenAI key.
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")

        transcript_text = None
        ctx = await user_voice_provider_service.get_voice_context(
            current_user.user_id, "stt"
        )

        try:
            if ctx and ctx.get("provider_type") == "deepgram":
                ctype = file.content_type or "application/octet-stream"
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
                audio_file.name = file.filename
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
                audio_file.name = file.filename
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

        except Exception as e:
            logger.warning("Primary transcription path failed: %s", e)
            transcript_text = None

        if not transcript_text:
            raise HTTPException(
                status_code=502, detail="Transcription service unavailable"
            )

        return {
            "success": True,
            "text": transcript_text.strip()
            if isinstance(transcript_text, str)
            else str(transcript_text),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Transcription failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
