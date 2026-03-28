"""
Ghostscript command builder - PDF compression and PDF/A conversion.
"""
from __future__ import annotations

from cli_worker.service.command_builders.base_builder import BaseCommandBuilder

COMPRESS_QUALITY = {"screen", "ebook", "printer", "prepress"}


class GhostscriptCommandBuilder(BaseCommandBuilder):
    @staticmethod
    def build_compress(
        input_path: str,
        output_path: str,
        quality: str,
    ) -> list[str]:
        BaseCommandBuilder.validate_enum(quality, COMPRESS_QUALITY, "quality")
        # gs -sDEVICE=pdfwrite -dPDFSETTINGS=/{quality} -o out.pdf in.pdf
        return [
            "gs",
            "-dNOPAUSE",
            "-dBATCH",
            "-dSAFER",
            "-sDEVICE=pdfwrite",
            f"-dPDFSETTINGS=/{quality}",
            "-dCompatibilityLevel=1.4",
            f"-sOutputFile={output_path}",
            input_path,
        ]

    @staticmethod
    def build_pdfa(input_path: str, output_path: str) -> list[str]:
        # PDF/A-2b: -dPDFA=2 -dPDFACompatibilityPolicy=1
        return [
            "gs",
            "-dNOPAUSE",
            "-dBATCH",
            "-dSAFER",
            "-sDEVICE=pdfwrite",
            "-dPDFA=2",
            "-dPDFACompatibilityPolicy=1",
            f"-sOutputFile={output_path}",
            input_path,
        ]
