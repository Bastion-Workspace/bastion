"""
Pandoc command builder - document format conversion.
"""
from __future__ import annotations

from cli_worker.service.command_builders.base_builder import BaseCommandBuilder

INPUT_FORMATS = {"markdown", "md", "org", "html", "latex", "rst", "docx", "epub"}
OUTPUT_FORMATS = {"pdf", "docx", "html", "epub", "latex", "rst", "plain", "markdown", "md"}


class PandocCommandBuilder(BaseCommandBuilder):
    @staticmethod
    def build_convert(
        input_path: str,
        output_path: str,
        input_format: str,
        output_format: str,
    ) -> list[str]:
        inf = BaseCommandBuilder.validate_enum(
            input_format.lower().replace("markdown", "md"), {f for f in INPUT_FORMATS},
            "input_format",
        )
        if inf == "md":
            inf = "markdown"
        outf = BaseCommandBuilder.validate_enum(
            output_format.lower().replace("markdown", "md"), {f for f in OUTPUT_FORMATS},
            "output_format",
        )
        if outf == "md":
            outf = "markdown"
        cmd = [
            "pandoc", input_path,
            "-f", inf,
            "-t", outf,
            "-o", output_path,
        ]
        if outf == "pdf":
            cmd.extend(["--pdf-engine=pdflatex"])
        return cmd
