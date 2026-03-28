"""
FFmpeg command builder - transcode, extract audio, thumbnail, trim, probe.
Allowlisted formats and codecs only.
"""
from __future__ import annotations

from cli_worker.service.command_builders.base_builder import BaseCommandBuilder

OUTPUT_FORMATS = {"mp3", "mp4", "wav", "ogg", "webm", "flac", "aac", "mkv", "mov"}
VIDEO_OUTPUT_FORMATS = {"mp4", "webm", "mkv", "mov"}
AUDIO_FORMATS = {"mp3", "wav", "ogg", "flac", "aac"}
THUMBNAIL_FORMATS = {"jpg", "jpeg", "png"}
VIDEO_CODECS = {"libx264", "libx265", "libvpx", "libvpx-vp9"}
AUDIO_CODECS = {"aac", "libmp3lame", "libvorbis", "flac"}


class FFmpegCommandBuilder(BaseCommandBuilder):
    @staticmethod
    def build_transcode(
        input_path: str,
        output_path: str,
        output_format: str,
        audio_bitrate_kbps: int | None = None,
        video_bitrate_kbps: int | None = None,
        max_width: int | None = None,
        max_height: int | None = None,
    ) -> list[str]:
        out_fmt =         BaseCommandBuilder.validate_enum(output_format, OUTPUT_FORMATS, "output_format")
        cmd = ["ffmpeg", "-y", "-i", input_path]
        if audio_bitrate_kbps is not None:
            BaseCommandBuilder.validate_range(audio_bitrate_kbps, 8, 320, "audio_bitrate_kbps")
            cmd.extend(["-b:a", f"{audio_bitrate_kbps}k"])
        if video_bitrate_kbps is not None:
            BaseCommandBuilder.validate_range(video_bitrate_kbps, 100, 50000, "video_bitrate_kbps")
            cmd.extend(["-b:v", f"{video_bitrate_kbps}k"])
        if max_width is not None or max_height is not None:
            scale = "scale="
            if max_width is not None and max_height is not None:
                BaseCommandBuilder.validate_range(max_width, 1, 8192, "max_width")
                BaseCommandBuilder.validate_range(max_height, 1, 8192, "max_height")
                scale += f"{max_width}:{max_height}"
            elif max_width is not None:
                BaseCommandBuilder.validate_range(max_width, 1, 8192, "max_width")
                scale += f"{max_width}:-2"
            else:
                BaseCommandBuilder.validate_range(max_height, 1, 8192, "max_height")
                scale += f"-2:{max_height}"
            cmd.extend(["-vf", scale])
        cmd.append(output_path)
        return cmd

    @staticmethod
    def build_extract_audio(
        input_path: str,
        output_path: str,
        output_format: str,
        bitrate_kbps: int | None = None,
    ) -> list[str]:
        out_fmt = BaseCommandBuilder.validate_enum(output_format, AUDIO_FORMATS, "output_format")
        cmd = ["ffmpeg", "-y", "-i", input_path, "-vn"]
        if bitrate_kbps is not None:
            BaseCommandBuilder.validate_range(bitrate_kbps, 8, 320, "bitrate_kbps")
            cmd.extend(["-b:a", f"{bitrate_kbps}k"])
        cmd.append(output_path)
        return cmd

    @staticmethod
    def build_thumbnail(
        input_path: str,
        output_path: str,
        output_format: str,
        width: int,
        height: int,
        timestamp: str | None = None,
        quality: int | None = None,
    ) -> list[str]:
        BaseCommandBuilder.validate_enum(output_format, THUMBNAIL_FORMATS, "output_format")
        BaseCommandBuilder.validate_range(width, 1, 4096, "width")
        BaseCommandBuilder.validate_range(height, 1, 4096, "height")
        cmd = ["ffmpeg", "-y", "-i", input_path]
        if timestamp:
            cmd.extend(["-ss", timestamp])
        cmd.extend(["-vframes", "1", "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease"])
        if quality is not None and output_format.lower() in ("jpg", "jpeg"):
            BaseCommandBuilder.validate_range(quality, 1, 100, "quality")
            cmd.extend(["-q:v", str(round(31 * (100 - quality) / 99))])
        cmd.append(output_path)
        return cmd

    @staticmethod
    def build_probe(input_path: str) -> list[str]:
        return ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", input_path]

    @staticmethod
    def build_trim(
        input_path: str,
        output_path: str,
        output_format: str,
        start_time: str,
        end_time: str,
    ) -> list[str]:
        BaseCommandBuilder.validate_enum(output_format, OUTPUT_FORMATS, "output_format")
        return [
            "ffmpeg", "-y", "-i", input_path,
            "-ss", start_time, "-to", end_time,
            "-c", "copy", output_path,
        ]

    @staticmethod
    def build_burn_subtitles(
        input_path: str,
        subtitle_path: str,
        output_path: str,
        output_format: str,
        font_size: int | None = None,
        font_color: str | None = None,
    ) -> list[str]:
        """Burn subtitle file into video. Output format must be video (mp4, mkv, webm, mov)."""
        BaseCommandBuilder.validate_enum(output_format, VIDEO_OUTPUT_FORMATS, "output_format")
        style_parts = []
        if font_size is not None:
            BaseCommandBuilder.validate_range(font_size, 8, 72, "font_size")
            style_parts.append(f"FontSize={font_size}")
        if font_color is not None and font_color.strip():
            color = font_color.strip().lstrip("#").upper()
            if len(color) == 6 and all(c in "0123456789ABCDEF" for c in color):
                style_parts.append(f"PrimaryColour=&H00{color[4:6]}{color[2:4]}{color[0:2]}&")
        force_style = ":force_style='" + ",".join(style_parts) + "'" if style_parts else ""
        sub_escaped = subtitle_path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        filter_str = f"subtitles={sub_escaped}{force_style}"
        return ["ffmpeg", "-y", "-i", input_path, "-vf", filter_str, output_path]
