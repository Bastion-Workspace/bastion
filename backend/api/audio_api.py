"""
Audio Transcription API

Provides a simple endpoint to accept microphone recordings from the frontend
and transcribe them using the configured provider. The request is a multipart
form with a single file field named "file".
"""

import io
import logging
from typing import Dict, Any

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends

from utils.auth_middleware import get_current_user, AuthenticatedUserResponse
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Audio"])


@router.post("/api/audio/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    """Transcribe an uploaded audio file and return plain text.

    Accepts common webm/ogg/mp3/wav containers from MediaRecorder.
    Uses OpenAI-compatible transcription via OpenRouter when available.
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        # Read file bytes into memory buffer
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")

        # Prefer native OpenAI Whisper when OPENAI_API_KEY is configured
        transcript_text = None

        try:
            from openai import AsyncOpenAI

            client: AsyncOpenAI
            model_name = "whisper-1"

            if settings.OPENAI_API_KEY:
                # Use OpenAI directly for Whisper
                logger.info("🎙️ Using OpenAI Whisper for transcription")
                client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            elif settings.OPENROUTER_API_KEY:
                # OpenRouter generally does not support audio.transcriptions → warn and fail fast
                raise RuntimeError("OpenRouter does not support Whisper audio transcription in this setup. Configure OPENAI_API_KEY.")
            else:
                raise RuntimeError("No API key configured for transcription (OPENAI_API_KEY required)")

            # Build a file-like object for upload
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = file.filename

            # Use OpenAI v1 audio transcriptions API
            result = await client.audio.transcriptions.create(
                model=model_name,
                file=(audio_file),
            )

            # Extract text depending on provider response shape
            # OpenAI returns .text; some providers wrap under data
            transcript_text = getattr(result, "text", None) or getattr(result, "data", {}).get("text")

        except Exception as e:
            logger.warning(f"⚠️ Primary transcription path failed: {e}")
            transcript_text = None

        if not transcript_text:
            # Minimal fallback: return an error rather than silently failing
            raise HTTPException(status_code=502, detail="Transcription service unavailable")

        return {
            "success": True,
            "text": transcript_text.strip() if isinstance(transcript_text, str) else str(transcript_text),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


