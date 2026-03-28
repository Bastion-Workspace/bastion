"""
Audio format conversion using pydub and ffmpeg.
Normalizes input to WAV (16 kHz mono for Whisper) and supports cross-format conversion.
"""

import logging
from io import BytesIO
from typing import Optional

from pydub import AudioSegment

logger = logging.getLogger(__name__)

_FORMAT_MAP = {
    "ogg": "ogg",
    "opus": "ogg",
    "webm": "webm",
    "mp3": "mp3",
    "wav": "wav",
    "flac": "flac",
    "pcm": "s16le",
}

WAV_SAMPLE_RATE = 16000
WAV_CHANNELS = 1


def _pydub_format(source_format: str) -> str:
    fmt = (source_format or "wav").strip().lower()
    return _FORMAT_MAP.get(fmt, "wav")


def _load_segment(
    audio_data: bytes,
    source_format: str,
    *,
    pcm_frame_rate: Optional[int] = None,
) -> AudioSegment:
    fmt = _pydub_format(source_format)
    if fmt == "s16le":
        rate = pcm_frame_rate if pcm_frame_rate is not None else WAV_SAMPLE_RATE
        return AudioSegment(
            data=audio_data,
            sample_width=2,
            frame_rate=rate,
            channels=1,
        )
    return AudioSegment.from_file(BytesIO(audio_data), format=fmt)


def convert_to_wav(
    audio_data: bytes,
    source_format: str,
    sample_rate: int = WAV_SAMPLE_RATE,
    channels: int = WAV_CHANNELS,
    source_pcm_frame_rate: Optional[int] = None,
) -> bytes:
    """Normalize any input to WAV (16 kHz mono by default, suitable for Whisper)."""
    if not audio_data:
        raise ValueError("audio_data is empty")
    segment = _load_segment(
        audio_data, source_format, pcm_frame_rate=source_pcm_frame_rate
    )
    segment = segment.set_frame_rate(sample_rate).set_channels(channels)
    buffer = BytesIO()
    segment.export(buffer, format="wav")
    return buffer.getvalue()


def convert_audio(
    audio_data: bytes,
    source_format: str,
    target_format: str,
    sample_rate: Optional[int] = None,
    channels: Optional[int] = None,
    source_pcm_frame_rate: Optional[int] = None,
) -> bytes:
    """Convert audio from source_format to target_format."""
    if not audio_data:
        raise ValueError("audio_data is empty")
    segment = _load_segment(
        audio_data, source_format, pcm_frame_rate=source_pcm_frame_rate
    )
    if sample_rate is not None:
        segment = segment.set_frame_rate(sample_rate)
    if channels is not None:
        segment = segment.set_channels(channels)
    target = (target_format or "wav").strip().lower()
    if target == "pcm":
        target = "s16le"
    if target not in ("wav", "mp3", "ogg", "flac", "webm", "s16le"):
        target = "wav"
    buffer = BytesIO()
    segment.export(buffer, format=target)
    return buffer.getvalue()