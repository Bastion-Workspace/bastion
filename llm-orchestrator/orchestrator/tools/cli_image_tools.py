"""
CLI image tools - ImageMagick and Graphviz via CLI worker.
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


# ----- Convert image (ImageMagick) -----
class ConvertImageInputs(BaseModel):
    document_id: str = Field(description="Image document ID")
    output_format: str = Field(description="Output format: png, jpg, jpeg, webp, gif, tiff, bmp, svg")


class ConvertImageParams(BaseModel):
    width: Optional[int] = Field(default=None, ge=1, le=8192)
    height: Optional[int] = Field(default=None, ge=1, le=8192)
    quality: Optional[int] = Field(default=None, ge=1, le=100)


class ConvertImageOutputs(BaseModel):
    success: bool = Field(description="Whether the operation succeeded")
    document_id: Optional[str] = Field(default=None)
    output_filename: Optional[str] = Field(default=None)
    error: Optional[str] = Field(default=None)
    formatted: str = Field(description="Human-readable summary")


async def convert_image_tool(
    document_id: str,
    output_format: str = "png",
    width: Optional[int] = None,
    height: Optional[int] = None,
    quality: Optional[int] = None,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Convert an image to another format, optionally resize or set quality."""
    try:
        backend = await get_backend_tool_client()
        content = await backend.get_document_content(document_id, user_id=user_id)
        if content is None:
            return {"success": False, "error": "Document not found", "formatted": "Document not found."}
        raw = _content_to_bytes(content)
        doc = await backend.get_document(document_id, user_id=user_id)
        filename = doc.get("filename", "input") if doc else "input"
        cli = get_cli_service_client()
        result = await cli.convert_image(
            input_data=raw,
            input_filename=filename,
            output_format=output_format,
            width=width,
            height=height,
            quality=quality,
        )
        if not result.get("success"):
            return {"success": False, "error": result.get("error"), "formatted": result.get("formatted", result.get("error", "Convert image failed."))}
        out_data = result.get("output_data", b"")
        out_name = result.get("output_filename", f"out.{output_format}")
        stored = await _store_result(out_data, out_name, user_id, is_text=False)
        if not stored.get("success"):
            return {"success": False, "error": stored.get("error"), "formatted": stored.get("error", "Storage failed.")}
        return {
            "success": True,
            "document_id": stored.get("document_id"),
            "output_filename": out_name,
            "formatted": f"Converted to {output_format}. New document: {stored.get('filename', out_name)}.",
        }
    except Exception as e:
        logger.exception("convert_image_tool failed")
        return {"success": False, "error": str(e), "formatted": str(e)}


register_action(
    name="convert_image",
    category="image",
    description="Convert an image to another format (e.g. PNG to JPEG), optionally resize or set quality.",
    inputs_model=ConvertImageInputs,
    params_model=ConvertImageParams,
    outputs_model=ConvertImageOutputs,
    tool_function=convert_image_tool,
)


# ----- Optimize image (optipng, jpegoptim) -----
class OptimizeImageInputs(BaseModel):
    document_id: str = Field(description="Image document ID (PNG or JPEG)")


class OptimizeImageParams(BaseModel):
    quality: Optional[int] = Field(default=None, ge=1, le=100, description="JPEG quality 1-100 (JPEG only)")
    strip_metadata: bool = Field(default=False, description="Remove EXIF and other metadata")


class OptimizeImageOutputs(BaseModel):
    success: bool = Field(description="Whether the operation succeeded")
    document_id: Optional[str] = Field(default=None)
    output_filename: Optional[str] = Field(default=None)
    error: Optional[str] = Field(default=None)
    formatted: str = Field(description="Human-readable summary")


async def optimize_image_tool(
    document_id: str,
    quality: Optional[int] = None,
    strip_metadata: bool = False,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Optimize a PNG or JPEG image (lossless for PNG, optional quality for JPEG)."""
    try:
        backend = await get_backend_tool_client()
        content = await backend.get_document_content(document_id, user_id=user_id)
        if content is None:
            return {"success": False, "error": "Document not found", "formatted": "Document not found."}
        raw = _content_to_bytes(content)
        doc = await backend.get_document(document_id, user_id=user_id)
        filename = doc.get("filename", "input") if doc else "input"
        cli = get_cli_service_client()
        result = await cli.optimize_image(
            input_data=raw,
            input_filename=filename,
            quality=quality,
            strip_metadata=strip_metadata,
        )
        if not result.get("success"):
            return {"success": False, "error": result.get("error"), "formatted": result.get("formatted", result.get("error", "Optimize image failed."))}
        out_data = result.get("output_data", b"")
        out_name = result.get("output_filename", filename)
        stored = await _store_result(out_data, out_name, user_id, is_text=False)
        if not stored.get("success"):
            return {"success": False, "error": stored.get("error"), "formatted": stored.get("error", "Storage failed.")}
        return {
            "success": True,
            "document_id": stored.get("document_id"),
            "output_filename": out_name,
            "formatted": f"Optimized image. New document: {stored.get('filename', out_name)}.",
        }
    except Exception as e:
        logger.exception("optimize_image_tool failed")
        return {"success": False, "error": str(e), "formatted": str(e)}


register_action(
    name="optimize_image",
    category="image",
    description="Optimize a PNG or JPEG image (smaller file size; JPEG quality optional).",
    inputs_model=OptimizeImageInputs,
    params_model=OptimizeImageParams,
    outputs_model=OptimizeImageOutputs,
    tool_function=optimize_image_tool,
)


# ----- Render diagram (Graphviz) -----
class RenderDiagramInputs(BaseModel):
    dot_content: str = Field(description="DOT source code for the diagram")
    output_format: str = Field(description="Output format: png, svg, pdf")


class RenderDiagramParams(BaseModel):
    engine: str = Field(default="dot", description="Graphviz engine: dot, neato, fdp, sfdp, circo, twopi")


class RenderDiagramOutputs(BaseModel):
    success: bool = Field(description="Whether the operation succeeded")
    document_id: Optional[str] = Field(default=None)
    output_filename: Optional[str] = Field(default=None)
    error: Optional[str] = Field(default=None)
    formatted: str = Field(description="Human-readable summary")


async def render_diagram_tool(
    dot_content: str,
    output_format: str = "png",
    engine: str = "dot",
    user_id: str = "system",
) -> Dict[str, Any]:
    """Render a diagram from DOT source (Graphviz) to an image or PDF."""
    try:
        raw = dot_content.encode("utf-8")
        cli = get_cli_service_client()
        result = await cli.render_diagram(raw, output_format=output_format, engine=engine)
        if not result.get("success"):
            return {"success": False, "error": result.get("error"), "formatted": result.get("formatted", result.get("error", "Render diagram failed."))}
        out_data = result.get("output_data", b"")
        out_name = result.get("output_filename", f"diagram.{output_format}")
        stored = await _store_result(out_data, out_name, user_id, is_text=False)
        if not stored.get("success"):
            return {"success": False, "error": stored.get("error"), "formatted": stored.get("error", "Storage failed.")}
        return {
            "success": True,
            "document_id": stored.get("document_id"),
            "output_filename": out_name,
            "formatted": f"Rendered diagram as {output_format}. New document: {stored.get('filename', out_name)}.",
        }
    except Exception as e:
        logger.exception("render_diagram_tool failed")
        return {"success": False, "error": str(e), "formatted": str(e)}


register_action(
    name="render_diagram",
    category="image",
    description="Render a diagram from DOT source (Graphviz) to PNG, SVG, or PDF.",
    inputs_model=RenderDiagramInputs,
    params_model=RenderDiagramParams,
    outputs_model=RenderDiagramOutputs,
    tool_function=render_diagram_tool,
)
