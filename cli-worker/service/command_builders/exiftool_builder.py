"""
ExifTool command builder - read/write media metadata.
Write: allowlisted tag names only.
"""
from __future__ import annotations

from cli_worker.service.command_builders.base_builder import BaseCommandBuilder

WRITABLE_TAGS = {
    "Title", "Artist", "Copyright", "Description", "Comment",
    "Author", "Creator", "Keywords", "Subject", "Rating",
    "ImageDescription", "XPTitle", "XPComment", "XPKeywords",
}


class ExifToolCommandBuilder(BaseCommandBuilder):
    @staticmethod
    def build_read(input_path: str) -> list[str]:
        return ["exiftool", "-j", "-G", input_path]

    @staticmethod
    def build_write(input_path: str, output_path: str, metadata: dict[str, str]) -> list[str]:
        cmd = ["exiftool", "-o", output_path]
        for k, v in metadata.items():
            key = k.strip()
            if key not in WRITABLE_TAGS:
                raise ValueError(f"metadata key not allowed: {key}. Allowed: {sorted(WRITABLE_TAGS)}")
            cmd.extend([f"-{key}={v}"])
        cmd.append(input_path)
        return cmd
