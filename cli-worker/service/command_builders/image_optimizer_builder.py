"""
Image optimizer command builder - optipng (PNG), jpegoptim (JPEG).
Dispatches by input format; only PNG and JPEG are supported.
"""
from __future__ import annotations

import os

from cli_worker.service.command_builders.base_builder import BaseCommandBuilder

PNG_EXTENSIONS = {".png"}
JPEG_EXTENSIONS = {".jpg", ".jpeg"}


class ImageOptimizerCommandBuilder(BaseCommandBuilder):
    @staticmethod
    def build_optipng(input_path: str, strip_metadata: bool = False) -> list[str]:
        cmd = ["optipng", "-o2", "-quiet"]
        if strip_metadata:
            cmd.append("-strip")
        cmd.append(input_path)
        return cmd

    @staticmethod
    def build_jpegoptim(
        input_path: str,
        quality: int | None = None,
        strip_metadata: bool = False,
    ) -> list[str]:
        cmd = ["jpegoptim", "--quiet"]
        if strip_metadata:
            cmd.append("--strip-all")
        if quality is not None:
            BaseCommandBuilder.validate_range(quality, 1, 100, "quality")
            cmd.extend(["--max", str(quality)])
        cmd.append(input_path)
        return cmd

    @staticmethod
    def format_from_filename(filename: str) -> str:
        ext = (os.path.splitext(filename or "")[1] or "").lower()
        if ext in PNG_EXTENSIONS:
            return "png"
        if ext in JPEG_EXTENSIONS:
            return "jpeg"
        raise ValueError(f"OptimizeImage supports only PNG and JPEG, got {filename!r}")
