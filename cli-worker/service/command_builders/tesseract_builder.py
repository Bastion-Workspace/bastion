"""
Tesseract OCR command builder.
"""
from __future__ import annotations

from cli_worker.service.command_builders.base_builder import BaseCommandBuilder

OUTPUT_FORMATS = {"text", "hocr", "tsv", "pdf"}
LANGUAGES = {"eng", "fra", "deu"}


class TesseractCommandBuilder(BaseCommandBuilder):
    @staticmethod
    def build_ocr(
        input_path: str,
        output_base_path: str,
        output_format: str,
        languages: list[str] | None = None,
    ) -> list[str]:
        BaseCommandBuilder.validate_enum(output_format, OUTPUT_FORMATS, "output_format")
        lang_list = languages or ["eng"]
        for l in lang_list:
            BaseCommandBuilder.validate_enum(l.strip().lower(), LANGUAGES, "languages")
        cmd = ["tesseract", input_path, output_base_path]
        if lang_list:
            cmd.extend(["-l", "+".join(lang_list)])
        cmd.append(output_format)
        return cmd
