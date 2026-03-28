"""
Graphviz dot command builder - render DOT to image/PDF.
"""
from __future__ import annotations

from cli_worker.service.command_builders.base_builder import BaseCommandBuilder

OUTPUT_FORMATS = {"png", "svg", "pdf"}
ENGINES = {"dot", "neato", "fdp", "sfdp", "circo", "twopi"}


class GraphvizCommandBuilder(BaseCommandBuilder):
    @staticmethod
    def build_render(
        input_path: str,
        output_path: str,
        output_format: str,
        engine: str = "dot",
    ) -> list[str]:
        BaseCommandBuilder.validate_enum(output_format, OUTPUT_FORMATS, "output_format")
        BaseCommandBuilder.validate_enum(engine, ENGINES, "engine")
        return [engine, "-T" + output_format, "-o", output_path, input_path]
