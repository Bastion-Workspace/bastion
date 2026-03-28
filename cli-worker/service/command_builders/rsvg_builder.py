"""
SVG rendering command builder - rsvg-convert (librsvg).
"""
from __future__ import annotations

from cli_worker.service.command_builders.base_builder import BaseCommandBuilder

OUTPUT_FORMATS = {"png", "pdf"}


class RsvgCommandBuilder(BaseCommandBuilder):
    @staticmethod
    def build_rsvg_convert(
        input_path: str,
        output_path: str,
        output_format: str,
        width: int | None = None,
        height: int | None = None,
        dpi: int | None = None,
    ) -> list[str]:
        BaseCommandBuilder.validate_enum(output_format, OUTPUT_FORMATS, "output_format")
        cmd = ["rsvg-convert", "-o", output_path]
        if width is not None:
            BaseCommandBuilder.validate_positive(width, "width")
            cmd.extend(["--width", str(width)])
        if height is not None:
            BaseCommandBuilder.validate_positive(height, "height")
            cmd.extend(["--height", str(height)])
        if dpi is not None:
            BaseCommandBuilder.validate_range(dpi, 72, 600, "dpi")
            cmd.extend(["--dpi-x", str(dpi), "--dpi-y", str(dpi)])
        cmd.append(input_path)
        return cmd
