"""
yt-dlp command builder - download media from URL.
"""
from __future__ import annotations

from cli_worker.service.command_builders.base_builder import BaseCommandBuilder

FORMAT_PREFERENCE = {"best", "bestaudio", "bestvideo"}


class YtDlpCommandBuilder(BaseCommandBuilder):
    @staticmethod
    def build_download(
        url: str,
        output_path: str,
        format_preference: str = "best",
        max_filesize_mb: int | None = None,
    ) -> list[str]:
        BaseCommandBuilder.validate_enum(format_preference, FORMAT_PREFERENCE, "format_preference")
        cmd = [
            "yt-dlp",
            "--no-playlist",
            "--no-warnings",
            "-o", output_path,
        ]
        if format_preference == "bestaudio":
            cmd.extend(["-x", "-f", "bestaudio/best"])
        elif format_preference == "bestvideo":
            cmd.extend(["-f", "bestvideo/best"])
        else:
            cmd.extend(["-f", "best"])
        if max_filesize_mb is not None:
            BaseCommandBuilder.validate_positive(max_filesize_mb, "max_filesize_mb")
            cmd.extend(["--max-filesize", f"{max_filesize_mb}M"])
        cmd.append(url)
        return cmd
