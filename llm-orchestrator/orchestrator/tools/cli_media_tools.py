"""
CLI media tools - FFmpeg, ExifTool, yt-dlp via CLI worker.
"""
from __future__ import annotations

import base64
import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.cli_service_client import get_cli_service_client
from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)


def _content_to_bytes(content: str) -> bytes:
    """Convert document content string to bytes; support base64 for binary docs."""
    if not content:
        return b""
    try:
        return base64.b64decode(content)
    except Exception:
        return content.encode("utf-8")


async def _store_result(
    output_data: bytes,
    output_filename: str,
    user_id: str,
    is_text: bool,
) -> Dict[str, Any]:
    """Store CLI output as a new document."""
    backend = await get_backend_tool_client()
    if is_text:
        stored = await backend.create_user_file(
            filename=output_filename,
            content=output_data.decode("utf-8", errors="replace"),
            user_id=user_id,
        )
    else:
        stored = await backend.create_user_file(
            filename=output_filename,
            content="",
            user_id=user_id,
            content_bytes=output_data,
        )
    return stored


# ----- Transcode media -----
class TranscodeMediaInputs(BaseModel):
    document_id: str = Field(description="Source document ID (video/audio)")
    output_format: str = Field(description="Output format: mp3, mp4, wav, ogg, webm, flac, aac, mkv, mov")


class TranscodeMediaParams(BaseModel):
    audio_bitrate_kbps: Optional[int] = Field(default=None, ge=8, le=320)
    video_bitrate_kbps: Optional[int] = Field(default=None, ge=100, le=50000)


class TranscodeMediaOutputs(BaseModel):
    success: bool = Field(description="Whether the operation succeeded")
    document_id: Optional[str] = Field(default=None)
    output_filename: Optional[str] = Field(default=None)
    error: Optional[str] = Field(default=None)
    formatted: str = Field(description="Human-readable summary")


async def transcode_media_tool(
    document_id: str,
    output_format: str = "mp4",
    audio_bitrate_kbps: Optional[int] = None,
    video_bitrate_kbps: Optional[int] = None,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Transcode a media document to another format."""
    try:
        backend = await get_backend_tool_client()
        content = await backend.get_document_content(document_id, user_id=user_id)
        if content is None:
            return {"success": False, "error": "Document not found", "formatted": "Document not found."}
        raw = _content_to_bytes(content)
        doc = await backend.get_document(document_id, user_id=user_id)
        filename = doc.get("filename", "input") if doc else "input"
        cli = get_cli_service_client()
        result = await cli.transcode_media(
            input_data=raw,
            input_filename=filename,
            output_format=output_format,
            audio_bitrate_kbps=audio_bitrate_kbps,
            video_bitrate_kbps=video_bitrate_kbps,
        )
        if not result.get("success"):
            return {"success": False, "error": result.get("error"), "formatted": result.get("formatted", result.get("error", "Transcode failed."))}
        out_data = result.get("output_data", b"")
        out_name = result.get("output_filename", f"out.{output_format}")
        stored = await _store_result(out_data, out_name, user_id, is_text=False)
        if not stored.get("success"):
            return {"success": False, "error": stored.get("error"), "formatted": stored.get("error", "Storage failed.")}
        return {
            "success": True,
            "document_id": stored.get("document_id"),
            "output_filename": out_name,
            "formatted": f"Transcoded to {output_format}. New document: {stored.get('filename', out_name)} (ID: {stored.get('document_id', '')}).",
        }
    except Exception as e:
        logger.exception("transcode_media_tool failed")
        return {"success": False, "error": str(e), "formatted": str(e)}


register_action(
    name="transcode_media",
    category="media",
    description="Transcode a media document to another format (e.g. video to mp3).",
    inputs_model=TranscodeMediaInputs,
    params_model=TranscodeMediaParams,
    outputs_model=TranscodeMediaOutputs,
    tool_function=transcode_media_tool,
)


# ----- Extract audio -----
class ExtractAudioInputs(BaseModel):
    document_id: str = Field(description="Source video document ID")
    output_format: str = Field(description="Audio format: mp3, wav, ogg, flac, aac")


class ExtractAudioParams(BaseModel):
    bitrate_kbps: Optional[int] = Field(default=128, ge=8, le=320)


class ExtractAudioOutputs(BaseModel):
    success: bool = Field(description="Whether the operation succeeded")
    document_id: Optional[str] = Field(default=None)
    output_filename: Optional[str] = Field(default=None)
    error: Optional[str] = Field(default=None)
    formatted: str = Field(description="Human-readable summary")


async def extract_audio_tool(
    document_id: str,
    output_format: str = "mp3",
    bitrate_kbps: Optional[int] = 128,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Extract audio track from a video document."""
    try:
        backend = await get_backend_tool_client()
        content = await backend.get_document_content(document_id, user_id=user_id)
        if content is None:
            return {"success": False, "error": "Document not found", "formatted": "Document not found."}
        raw = _content_to_bytes(content)
        doc = await backend.get_document(document_id, user_id=user_id)
        filename = doc.get("filename", "input") if doc else "input"
        cli = get_cli_service_client()
        result = await cli.extract_audio(raw, filename, output_format, bitrate_kbps=bitrate_kbps)
        if not result.get("success"):
            return {"success": False, "error": result.get("error"), "formatted": result.get("formatted", result.get("error", "Extract audio failed."))}
        out_data = result.get("output_data", b"")
        out_name = result.get("output_filename", f"audio.{output_format}")
        stored = await _store_result(out_data, out_name, user_id, is_text=False)
        if not stored.get("success"):
            return {"success": False, "error": stored.get("error"), "formatted": stored.get("error", "Storage failed.")}
        return {
            "success": True,
            "document_id": stored.get("document_id"),
            "output_filename": out_name,
            "formatted": f"Extracted audio as {output_format}. New document: {stored.get('filename', out_name)}.",
        }
    except Exception as e:
        logger.exception("extract_audio_tool failed")
        return {"success": False, "error": str(e), "formatted": str(e)}


register_action(
    name="extract_audio",
    category="media",
    description="Extract the audio track from a video document.",
    inputs_model=ExtractAudioInputs,
    params_model=ExtractAudioParams,
    outputs_model=ExtractAudioOutputs,
    tool_function=extract_audio_tool,
)


# ----- Trim media -----
class TrimMediaInputs(BaseModel):
    document_id: str = Field(description="Source media document ID")
    start_time: str = Field(description="Start timestamp (HH:MM:SS or seconds)")
    end_time: str = Field(description="End timestamp (HH:MM:SS or seconds)")


class TrimMediaParams(BaseModel):
    output_format: str = Field(
        default="mp4",
        description="Output format: mp3, mp4, wav, ogg, webm, flac, aac, mkv, mov",
    )


class TrimMediaOutputs(BaseModel):
    success: bool = Field(description="Whether the operation succeeded")
    document_id: Optional[str] = Field(default=None)
    output_filename: Optional[str] = Field(default=None)
    error: Optional[str] = Field(default=None)
    formatted: str = Field(description="Human-readable summary")


async def trim_media_tool(
    document_id: str,
    start_time: str,
    end_time: str,
    output_format: str = "mp4",
    user_id: str = "system",
) -> Dict[str, Any]:
    """Trim a media document to a time range (start_time to end_time)."""
    try:
        backend = await get_backend_tool_client()
        content = await backend.get_document_content(document_id, user_id=user_id)
        if content is None:
            return {"success": False, "error": "Document not found", "formatted": "Document not found."}
        raw = _content_to_bytes(content)
        doc = await backend.get_document(document_id, user_id=user_id)
        filename = doc.get("filename", "input") if doc else "input"
        cli = get_cli_service_client()
        result = await cli.trim_media(
            input_data=raw,
            input_filename=filename,
            output_format=output_format,
            start_time=start_time,
            end_time=end_time,
        )
        if not result.get("success"):
            return {
                "success": False,
                "error": result.get("error"),
                "formatted": result.get("formatted", result.get("error", "Trim failed.")),
            }
        out_data = result.get("output_data", b"")
        out_name = result.get("output_filename", f"trimmed.{output_format}")
        stored = await _store_result(out_data, out_name, user_id, is_text=False)
        if not stored.get("success"):
            return {"success": False, "error": stored.get("error"), "formatted": stored.get("error", "Storage failed.")}
        return {
            "success": True,
            "document_id": stored.get("document_id"),
            "output_filename": out_name,
            "formatted": f"Trimmed to {start_time}-{end_time}. New document: {stored.get('filename', out_name)} (ID: {stored.get('document_id', '')}).",
        }
    except Exception as e:
        logger.exception("trim_media_tool failed")
        return {"success": False, "error": str(e), "formatted": str(e)}


register_action(
    name="trim_media",
    category="media",
    description="Trim a media document to a time range (start_time to end_time).",
    inputs_model=TrimMediaInputs,
    params_model=TrimMediaParams,
    outputs_model=TrimMediaOutputs,
    tool_function=trim_media_tool,
)


# ----- Get media info -----
class GetMediaInfoInputs(BaseModel):
    document_id: str = Field(description="Media document ID")


class GetMediaInfoOutputs(BaseModel):
    success: bool = Field(description="Whether the operation succeeded")
    duration_seconds: Optional[float] = Field(default=None)
    width: Optional[int] = Field(default=None)
    height: Optional[int] = Field(default=None)
    video_codec: Optional[str] = Field(default=None)
    audio_codec: Optional[str] = Field(default=None)
    metadata_json: Optional[str] = Field(default=None)
    error: Optional[str] = Field(default=None)
    formatted: str = Field(description="Human-readable summary")


async def get_media_info_tool(document_id: str, user_id: str = "system") -> Dict[str, Any]:
    """Get metadata (duration, codecs, resolution) for a media document."""
    try:
        backend = await get_backend_tool_client()
        content = await backend.get_document_content(document_id, user_id=user_id)
        if content is None:
            return {"success": False, "error": "Document not found", "formatted": "Document not found."}
        raw = _content_to_bytes(content)
        doc = await backend.get_document(document_id, user_id=user_id)
        filename = doc.get("filename", "input") if doc else "input"
        cli = get_cli_service_client()
        result = await cli.get_media_info(raw, filename)
        return {
            "success": result.get("success", False),
            "duration_seconds": result.get("duration_seconds"),
            "width": result.get("width"),
            "height": result.get("height"),
            "video_codec": result.get("video_codec"),
            "audio_codec": result.get("audio_codec"),
            "metadata_json": result.get("metadata_json"),
            "error": result.get("error"),
            "formatted": result.get("formatted", result.get("error", "Get media info failed.")),
        }
    except Exception as e:
        logger.exception("get_media_info_tool failed")
        return {"success": False, "error": str(e), "formatted": str(e)}


register_action(
    name="get_media_info",
    category="media",
    description="Get metadata (duration, codecs, resolution) for a media document.",
    inputs_model=GetMediaInfoInputs,
    outputs_model=GetMediaInfoOutputs,
    tool_function=get_media_info_tool,
)


# ----- Download media -----
class DownloadMediaInputs(BaseModel):
    url: str = Field(description="URL to download (e.g. YouTube, media link)")


class DownloadMediaParams(BaseModel):
    format_preference: str = Field(default="best", description="best, bestaudio, or bestvideo")
    max_filesize_mb: Optional[int] = Field(default=None, ge=1, le=2000)


class DownloadMediaOutputs(BaseModel):
    success: bool = Field(description="Whether the download succeeded")
    document_id: Optional[str] = Field(default=None)
    output_filename: Optional[str] = Field(default=None)
    error: Optional[str] = Field(default=None)
    formatted: str = Field(description="Human-readable summary")


async def download_media_tool(
    url: str,
    format_preference: str = "best",
    max_filesize_mb: Optional[int] = None,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Download media from a URL (e.g. YouTube) and store as a document."""
    try:
        cli = get_cli_service_client()
        result = await cli.download_media(url, format_preference=format_preference, max_filesize_mb=max_filesize_mb)
        if not result.get("success"):
            return {"success": False, "error": result.get("error"), "formatted": result.get("formatted", result.get("error", "Download failed."))}
        out_data = result.get("output_data", b"")
        out_name = result.get("output_filename", "downloaded")
        stored = await _store_result(out_data, out_name, user_id, is_text=False)
        if not stored.get("success"):
            return {"success": False, "error": stored.get("error"), "formatted": stored.get("error", "Storage failed.")}
        return {
            "success": True,
            "document_id": stored.get("document_id"),
            "output_filename": out_name,
            "formatted": f"Downloaded: {out_name}. New document ID: {stored.get('document_id', '')}.",
        }
    except Exception as e:
        logger.exception("download_media_tool failed")
        return {"success": False, "error": str(e), "formatted": str(e)}


register_action(
    name="download_media",
    category="media",
    description="Download media from a URL (e.g. YouTube) and save as a document.",
    inputs_model=DownloadMediaInputs,
    params_model=DownloadMediaParams,
    outputs_model=DownloadMediaOutputs,
    tool_function=download_media_tool,
)


# ----- Burn subtitles -----
class BurnSubtitlesInputs(BaseModel):
    document_id: str = Field(description="Source video document ID")
    subtitle_text: str = Field(description="Subtitle content in SRT format")


class BurnSubtitlesParams(BaseModel):
    output_format: str = Field(default="mp4", description="Output format: mp4, mkv, webm, mov")
    font_size: Optional[int] = Field(default=24, ge=8, le=72)
    font_color: Optional[str] = Field(default=None, description="Hex color e.g. FFFFFF")


class BurnSubtitlesOutputs(BaseModel):
    success: bool = Field(description="Whether the operation succeeded")
    document_id: Optional[str] = Field(default=None)
    output_filename: Optional[str] = Field(default=None)
    error: Optional[str] = Field(default=None)
    formatted: str = Field(description="Human-readable summary")


async def burn_subtitles_tool(
    document_id: str,
    subtitle_text: str,
    output_format: str = "mp4",
    font_size: Optional[int] = 24,
    font_color: Optional[str] = None,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Burn subtitle text (SRT format) into a video document."""
    try:
        backend = await get_backend_tool_client()
        content = await backend.get_document_content(document_id, user_id=user_id)
        if content is None:
            return {"success": False, "error": "Document not found", "formatted": "Document not found."}
        raw = _content_to_bytes(content)
        doc = await backend.get_document(document_id, user_id=user_id)
        filename = doc.get("filename", "input") if doc else "input"
        subtitle_data = subtitle_text.encode("utf-8")
        cli = get_cli_service_client()
        result = await cli.burn_subtitles(
            input_data=raw,
            input_filename=filename,
            subtitle_data=subtitle_data,
            subtitle_filename="subs.srt",
            output_format=output_format,
            font_size=font_size,
            font_color=font_color,
        )
        if not result.get("success"):
            return {
                "success": False,
                "error": result.get("error"),
                "formatted": result.get("formatted", result.get("error", "Burn subtitles failed.")),
            }
        out_data = result.get("output_data", b"")
        out_name = result.get("output_filename", f"burned.{output_format}")
        stored = await _store_result(out_data, out_name, user_id, is_text=False)
        if not stored.get("success"):
            return {"success": False, "error": stored.get("error"), "formatted": stored.get("error", "Storage failed.")}
        return {
            "success": True,
            "document_id": stored.get("document_id"),
            "output_filename": out_name,
            "formatted": f"Burned subtitles into video. New document: {stored.get('filename', out_name)} (ID: {stored.get('document_id', '')}).",
        }
    except Exception as e:
        logger.exception("burn_subtitles_tool failed")
        return {"success": False, "error": str(e), "formatted": str(e)}


register_action(
    name="burn_subtitles",
    category="media",
    description="Burn subtitle text (SRT format) into a video document.",
    inputs_model=BurnSubtitlesInputs,
    params_model=BurnSubtitlesParams,
    outputs_model=BurnSubtitlesOutputs,
    tool_function=burn_subtitles_tool,
)


# ----- Read media metadata (ExifTool) -----
class ReadMediaMetadataInputs(BaseModel):
    document_id: str = Field(description="Image or media document ID")


class ReadMediaMetadataOutputs(BaseModel):
    success: bool = Field(description="Whether the operation succeeded")
    metadata_json: Optional[str] = Field(default=None)
    error: Optional[str] = Field(default=None)
    formatted: str = Field(description="Human-readable summary")


async def read_media_metadata_tool(document_id: str, user_id: str = "system") -> Dict[str, Any]:
    """Read EXIF/IPTC/XMP metadata from an image or media document."""
    try:
        backend = await get_backend_tool_client()
        content = await backend.get_document_content(document_id, user_id=user_id)
        if content is None:
            return {"success": False, "error": "Document not found", "formatted": "Document not found."}
        raw = _content_to_bytes(content)
        doc = await backend.get_document(document_id, user_id=user_id)
        filename = doc.get("filename", "input") if doc else "input"
        cli = get_cli_service_client()
        result = await cli.read_media_metadata(raw, filename)
        return {
            "success": result.get("success", False),
            "metadata_json": result.get("metadata_json"),
            "error": result.get("error"),
            "formatted": result.get("formatted", result.get("error", "Read metadata failed.")),
        }
    except Exception as e:
        logger.exception("read_media_metadata_tool failed")
        return {"success": False, "error": str(e), "formatted": str(e)}


register_action(
    name="read_media_metadata",
    category="media",
    description="Read EXIF/IPTC/XMP metadata from an image or media document.",
    inputs_model=ReadMediaMetadataInputs,
    outputs_model=ReadMediaMetadataOutputs,
    tool_function=read_media_metadata_tool,
)
