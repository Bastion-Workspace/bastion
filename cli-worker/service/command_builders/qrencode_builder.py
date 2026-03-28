"""
QR code command builder - qrencode.
"""
from __future__ import annotations

from cli_worker.service.command_builders.base_builder import BaseCommandBuilder

OUTPUT_FORMATS = {"png", "svg"}
ERROR_CORRECTION = {"l", "m", "q", "h"}


class QrencodeCommandBuilder(BaseCommandBuilder):
    @staticmethod
    def build_qrencode(
        content: str,
        output_path: str,
        output_format: str,
        size: int = 256,
        error_correction: str = "M",
    ) -> list[str]:
        BaseCommandBuilder.validate_enum(output_format, OUTPUT_FORMATS, "output_format")
        BaseCommandBuilder.validate_range(size, 32, 2048, "size")
        ec = (error_correction or "M").strip().upper()
        if ec not in {"L", "M", "Q", "H"}:
            raise ValueError(f"error_correction must be L/M/Q/H, got {error_correction!r}")
        cmd = ["qrencode", "-o", output_path]
        if output_format == "png":
            cmd.extend(["-s", str(max(1, size // 32))])
        cmd.extend(["-l", ec])
        cmd.extend(["-t", output_format])
        cmd.append(content)
        return cmd
