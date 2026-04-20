"""
Audio Transcription API

Provides a simple endpoint to accept microphone recordings from the frontend
and transcribe them using the configured provider. The request is a multipart
form with a single file field named "file".
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends

from utils.auth_middleware import get_current_user, AuthenticatedUserResponse
from services.microphone_stt_service import transcribe_audio_bytes_for_user

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
        try:
            text = await transcribe_audio_bytes_for_user(
                current_user.user_id,
                audio_bytes,
                file.filename,
                content_type=file.content_type,
                prompt=prompt,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except RuntimeError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e

        return {"success": True, "text": text}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Transcription failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e
