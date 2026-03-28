"""
Poppler utils command builder - pdftotext, pdfseparate, pdfunite.
"""
from __future__ import annotations

from cli_worker.service.command_builders.base_builder import BaseCommandBuilder


class PopplerCommandBuilder(BaseCommandBuilder):
    @staticmethod
    def build_pdftotext(
        input_path: str,
        output_path: str,
        first_page: int | None = None,
        last_page: int | None = None,
    ) -> list[str]:
        cmd = ["pdftotext", input_path, output_path]
        if first_page is not None:
            BaseCommandBuilder.validate_positive(first_page, "first_page")
            cmd.extend(["-f", str(first_page)])
        if last_page is not None:
            BaseCommandBuilder.validate_positive(last_page, "last_page")
            cmd.extend(["-l", str(last_page)])
        return cmd

    @staticmethod
    def build_pdfseparate(input_path: str, output_pattern: str) -> list[str]:
        return ["pdfseparate", input_path, output_pattern]

    @staticmethod
    def build_pdfunite(input_paths: list[str], output_path: str) -> list[str]:
        if len(input_paths) < 2:
            raise ValueError("pdfunite requires at least 2 input PDFs")
        return ["pdfunite"] + input_paths + [output_path]

    @staticmethod
    def build_pdftoppm(
        input_path: str,
        output_prefix: str,
        output_format: str,
        first_page: int | None = None,
        last_page: int | None = None,
        dpi: int = 150,
    ) -> list[str]:
        BaseCommandBuilder.validate_enum(output_format, {"png", "jpeg"}, "output_format")
        BaseCommandBuilder.validate_range(dpi, 72, 600, "dpi")
        cmd = ["pdftoppm", f"-{output_format}", "-r", str(dpi)]
        if first_page is not None:
            BaseCommandBuilder.validate_positive(first_page, "first_page")
            cmd.extend(["-f", str(first_page)])
        if last_page is not None:
            BaseCommandBuilder.validate_positive(last_page, "last_page")
            cmd.extend(["-l", str(last_page)])
        cmd.extend([input_path, output_prefix])
        return cmd
