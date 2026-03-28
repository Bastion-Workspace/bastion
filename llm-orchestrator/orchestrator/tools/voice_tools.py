"""
Voice tools - Speech-to-text (transcription) via voice-service.
"""
from __future__ import annotations

import base64
import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.cli_service_client import get_cli_service_client
from orchestrator.utils.action_io_registry import register_action
from orchestrator.voice_service_client import get_voice_service_client

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {"mp4", "mkv", "webm", "mov", "avi", "m4v"}
AUDIO_EXTENSIONS = {"wav", "mp3", "ogg", "opus", "webm", "flac", "m4a", "aac"}


def _content_to_bytes(content: str) -> bytes:
    if not content:
        return b""
    try:
        return base64.b64decode(content)
    except Exception:
        return content.encode("utf-8")


def _ext_from_filename(filename: str) -> str:
    if not filename or "." not in filename:
        return "wav"
    return filename.rsplit(".", 1)[-1].lower()


class TranscribeAudioInputs(BaseModel):
    document_id: str = Field(description="Audio or video document ID to transcribe")


class TranscribeAudioParams(BaseModel):
    language: str = Field(default="auto", description="Language code or 'auto' for detection")


class TranscribeAudioOutputs(BaseModel):
    success: bool = Field(description="Whether the operation succeeded")
    transcript: str = Field(description="Full transcript text")
    detected_language: Optional[str] = Field(default=None)
    confidence: Optional[float] = Field(default=None)
    duration_seconds: Optional[float] = Field(default=None)
    error: Optional[str] = Field(default=None)
    formatted: str = Field(description="Human-readable summary")


async def transcribe_audio_tool(
    document_id: str,
    language: str = "auto",
    user_id: str = "system",
) -> Dict[str, Any]:
    """Transcribe an audio or video document to text using the voice-service (Whisper)."""
    try:
        backend = await get_backend_tool_client()
        content = await backend.get_document_content(document_id, user_id=user_id)
        if content is None:
            return {
                "success": False,
                "transcript": "",
                "error": "Document not found",
                "formatted": "Document not found.",
            }
        raw = _content_to_bytes(content)
        doc = await backend.get_document(document_id, user_id=user_id)
        filename = doc.get("filename", "input") if doc else "input"
        ext = _ext_from_filename(filename)

        if ext in VIDEO_EXTENSIONS:
            cli = get_cli_service_client()
            result = await cli.extract_audio(
                input_data=raw,
                input_filename=filename,
                output_format="wav",
            )
            if not result.get("success"):
                return {
                    "success": False,
                    "transcript": "",
                    "error": result.get("error", "Failed to extract audio from video"),
                    "formatted": result.get("formatted", "Extract audio failed."),
                }
            audio_bytes = result.get("output_data", b"")
            audio_format = "wav"
        elif ext in AUDIO_EXTENSIONS:
            audio_bytes = raw
            audio_format = ext
        else:
            audio_bytes = raw
            audio_format = "wav"

        if not audio_bytes:
            return {
                "success": False,
                "transcript": "",
                "error": "No audio data to transcribe",
                "formatted": "No audio data to transcribe.",
            }

        voice_client = get_voice_service_client()
        transcribe_result = await voice_client.transcribe(
            audio_data=audio_bytes,
            audio_format=audio_format,
            language=language,
        )

        if transcribe_result.get("error"):
            return {
                "success": False,
                "transcript": transcribe_result.get("text", ""),
                "detected_language": transcribe_result.get("detected_language"),
                "confidence": transcribe_result.get("confidence"),
                "duration_seconds": transcribe_result.get("duration_seconds"),
                "error": transcribe_result.get("error"),
                "formatted": transcribe_result.get("error", "Transcription failed."),
            }

        text = transcribe_result.get("text", "").strip()
        return {
            "success": True,
            "transcript": text,
            "detected_language": transcribe_result.get("detected_language"),
            "confidence": transcribe_result.get("confidence"),
            "duration_seconds": transcribe_result.get("duration_seconds"),
            "formatted": f"Transcript ({transcribe_result.get('detected_language', 'unknown')}): {text[:500]}{'...' if len(text) > 500 else ''}",
        }
    except Exception as e:
        logger.exception("transcribe_audio_tool failed")
        return {
            "success": False,
            "transcript": "",
            "error": str(e),
            "formatted": str(e),
        }


register_action(
    name="transcribe_audio",
    category="media",
    description="Transcribe an audio or video document to text (speech-to-text via voice-service).",
    inputs_model=TranscribeAudioInputs,
    params_model=TranscribeAudioParams,
    outputs_model=TranscribeAudioOutputs,
    tool_function=transcribe_audio_tool,
)
