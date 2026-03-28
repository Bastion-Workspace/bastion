"""
ImageMagick convert command builder - resize, crop, format, quality.
"""
from __future__ import annotations

from cli_worker.service.command_builders.base_builder import BaseCommandBuilder

OUTPUT_FORMATS = {"png", "jpg", "jpeg", "webp", "gif", "tiff", "bmp", "svg"}


class ImageMagickCommandBuilder(BaseCommandBuilder):
    @staticmethod
    def build_convert(
        input_path: str,
        output_path: str,
        output_format: str,
        width: int | None = None,
        height: int | None = None,
        quality: int | None = None,
        crop_x: int | None = None,
        crop_y: int | None = None,
        crop_width: int | None = None,
        crop_height: int | None = None,
    ) -> list[str]:
        BaseCommandBuilder.validate_enum(output_format, OUTPUT_FORMATS, "output_format")
        cmd = ["convert", input_path]
        if crop_width is not None and crop_height is not None:
            x = crop_x or 0
            y = crop_y or 0
            BaseCommandBuilder.validate_range(crop_width, 1, 8192, "crop_width")
            BaseCommandBuilder.validate_range(crop_height, 1, 8192, "crop_height")
            cmd.extend(["-crop", f"{crop_width}x{crop_height}+{x}+{y}", "+repage"])
        if width is not None or height is not None:
            dim = ""
            if width is not None and height is not None:
                BaseCommandBuilder.validate_range(width, 1, 8192, "width")
                BaseCommandBuilder.validate_range(height, 1, 8192, "height")
                dim = f"{width}x{height}"
            elif width is not None:
                BaseCommandBuilder.validate_range(width, 1, 8192, "width")
                dim = f"{width}x"
            else:
                BaseCommandBuilder.validate_range(height, 1, 8192, "height")
                dim = f"x{height}"
            cmd.extend(["-resize", dim])
        if quality is not None:
            BaseCommandBuilder.validate_range(quality, 1, 100, "quality")
            cmd.extend(["-quality", str(quality)])
        cmd.append(output_path)
        return cmd
