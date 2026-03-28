"""
CLI utility tools - QR code generation and SVG rendering via CLI worker.
"""
from __future__ import annotations

import base64
import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.cli_service_client import get_cli_service_client
from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)


def _content_to_bytes(content: str) -> bytes:
    if not content:
        return b""
    try:
        return base64.b64decode(content)
    except Exception:
        return content.encode("utf-8")


async def _store_result(
    output_data: bytes,
    output_filename: str,
    user_id: str,
    is_text: bool,
) -> Dict[str, Any]:
    backend = await get_backend_tool_client()
    if is_text:
        stored = await backend.create_user_file(
            filename=output_filename,
            content=output_data.decode("utf-8", errors="replace"),
            user_id=user_id,
        )
    else:
        stored = await backend.create_user_file(
            filename=output_filename,
            content="",
            user_id=user_id,
            content_bytes=output_data,
        )
    return stored


# ----- Generate QR code (qrencode) -----
class GenerateQrCodeInputs(BaseModel):
    content: str = Field(description="Text or URL to encode in the QR code")
    output_format: str = Field(default="png", description="Output format: png, svg")


class GenerateQrCodeParams(BaseModel):
    size: Optional[int] = Field(default=256, ge=32, le=2048)
    error_correction: Optional[str] = Field(default="M", description="Error correction: L, M, Q, H")


class GenerateQrCodeOutputs(BaseModel):
    success: bool = Field(description="Whether the operation succeeded")
    document_id: Optional[str] = Field(default=None)
    output_filename: Optional[str] = Field(default=None)
    error: Optional[str] = Field(default=None)
    formatted: str = Field(description="Human-readable summary")


async def generate_qr_code_tool(
    content: str,
    output_format: str = "png",
    size: int = 256,
    error_correction: str = "M",
    user_id: str = "system",
) -> Dict[str, Any]:
    """Generate a QR code image from text or URL."""
    try:
        cli = get_cli_service_client()
        result = await cli.generate_qr_code(
            content=content,
            output_format=output_format,
            size=size,
            error_correction=error_correction,
        )
        if not result.get("success"):
            return {"success": False, "error": result.get("error"), "formatted": result.get("formatted", result.get("error", "Generate QR code failed."))}
        out_data = result.get("output_data", b"")
        out_name = result.get("output_filename", f"qrcode.{output_format}")
        stored = await _store_result(out_data, out_name, user_id, is_text=False)
        if not stored.get("success"):
            return {"success": False, "error": stored.get("error"), "formatted": stored.get("error", "Storage failed.")}
        return {
            "success": True,
            "document_id": stored.get("document_id"),
            "output_filename": out_name,
            "formatted": f"Generated QR code. New document: {stored.get('filename', out_name)}.",
        }
    except Exception as e:
        logger.exception("generate_qr_code_tool failed")
        return {"success": False, "error": str(e), "formatted": str(e)}


register_action(
    name="generate_qr_code",
    category="utility",
    description="Generate a QR code image from text or URL (PNG or SVG).",
    inputs_model=GenerateQrCodeInputs,
    params_model=GenerateQrCodeParams,
    outputs_model=GenerateQrCodeOutputs,
    tool_function=generate_qr_code_tool,
)


# ----- Render SVG (librsvg) -----
class RenderSvgInputs(BaseModel):
    document_id: str = Field(description="SVG document ID (or content as document)")
    output_format: str = Field(default="png", description="Output format: png, pdf")


class RenderSvgParams(BaseModel):
    width: Optional[int] = Field(default=None, ge=1)
    height: Optional[int] = Field(default=None, ge=1)
    dpi: Optional[int] = Field(default=None, ge=72, le=600)


class RenderSvgOutputs(BaseModel):
    success: bool = Field(description="Whether the operation succeeded")
    document_id: Optional[str] = Field(default=None)
    output_filename: Optional[str] = Field(default=None)
    error: Optional[str] = Field(default=None)
    formatted: str = Field(description="Human-readable summary")


async def render_svg_tool(
    document_id: str,
    output_format: str = "png",
    width: Optional[int] = None,
    height: Optional[int] = None,
    dpi: Optional[int] = None,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Render an SVG document to PNG or PDF."""
    try:
        backend = await get_backend_tool_client()
        content = await backend.get_document_content(document_id, user_id=user_id)
        if content is None:
            return {"success": False, "error": "Document not found", "formatted": "Document not found."}
        raw = _content_to_bytes(content)
        doc = await backend.get_document(document_id, user_id=user_id)
        filename = doc.get("filename", "input.svg") if doc else "input.svg"
        cli = get_cli_service_client()
        result = await cli.render_svg(
            raw,
            output_format=output_format,
            width=width,
            height=height,
            dpi=dpi,
        )
        if not result.get("success"):
            return {"success": False, "error": result.get("error"), "formatted": result.get("formatted", result.get("error", "Render SVG failed."))}
        out_data = result.get("output_data", b"")
        out_name = result.get("output_filename", f"out.{output_format}")
        stored = await _store_result(out_data, out_name, user_id, is_text=False)
        if not stored.get("success"):
            return {"success": False, "error": stored.get("error"), "formatted": stored.get("error", "Storage failed.")}
        return {
            "success": True,
            "document_id": stored.get("document_id"),
            "output_filename": out_name,
            "formatted": f"Rendered SVG to {output_format}. New document: {stored.get('filename', out_name)}.",
        }
    except Exception as e:
        logger.exception("render_svg_tool failed")
        return {"success": False, "error": str(e), "formatted": str(e)}


register_action(
    name="render_svg",
    category="image",
    description="Render an SVG document to PNG or PDF.",
    inputs_model=RenderSvgInputs,
    params_model=RenderSvgParams,
    outputs_model=RenderSvgOutputs,
    tool_function=render_svg_tool,
)
