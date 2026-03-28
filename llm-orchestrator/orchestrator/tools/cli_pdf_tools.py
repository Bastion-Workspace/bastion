"""
CLI PDF tools - Render PDF pages, compress PDF, convert to PDF/A via CLI worker.
"""
from __future__ import annotations

import base64
import logging
from typing import Any, Dict, List, Optional

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


# ----- Render PDF pages (pdftoppm) -----
class RenderPdfPagesInputs(BaseModel):
    document_id: str = Field(description="PDF document ID")
    output_format: str = Field(default="png", description="Output image format: png, jpeg")


class RenderPdfPagesParams(BaseModel):
    first_page: Optional[int] = Field(default=None, ge=1)
    last_page: Optional[int] = Field(default=None, ge=1)
    dpi: Optional[int] = Field(default=150, ge=72, le=600)


class RenderPdfPagesOutputs(BaseModel):
    success: bool = Field(description="Whether the operation succeeded")
    document_ids: List[str] = Field(default_factory=list)
    output_filenames: List[str] = Field(default_factory=list)
    error: Optional[str] = Field(default=None)
    formatted: str = Field(description="Human-readable summary")


async def render_pdf_pages_tool(
    document_id: str,
    output_format: str = "png",
    first_page: Optional[int] = None,
    last_page: Optional[int] = None,
    dpi: int = 150,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Render PDF pages to images (e.g. for vision analysis)."""
    try:
        backend = await get_backend_tool_client()
        content = await backend.get_document_content(document_id, user_id=user_id)
        if content is None:
            return {"success": False, "error": "Document not found", "document_ids": [], "output_filenames": [], "formatted": "Document not found."}
        raw = _content_to_bytes(content)
        doc = await backend.get_document(document_id, user_id=user_id)
        filename = doc.get("filename", "input.pdf") if doc else "input.pdf"
        cli = get_cli_service_client()
        result = await cli.render_pdf_pages(
            raw, filename,
            output_format=output_format,
            first_page=first_page,
            last_page=last_page,
            dpi=dpi,
        )
        if not result.get("success"):
            return {"success": False, "error": result.get("error"), "document_ids": [], "output_filenames": [], "formatted": result.get("formatted", result.get("error", "Render PDF pages failed."))}
        doc_ids = []
        names = result.get("output_filenames", [])
        for i, data in enumerate(result.get("output_data", [])):
            out_name = names[i] if i < len(names) else f"page_{i+1}.{output_format}"
            stored = await _store_result(bytes(data), out_name, user_id, is_text=False)
            if stored.get("success"):
                doc_ids.append(stored.get("document_id", ""))
        return {
            "success": True,
            "document_ids": doc_ids,
            "output_filenames": names,
            "formatted": f"Rendered {len(doc_ids)} page(s) to {output_format}. New document IDs: {', '.join(doc_ids[:5])}{'...' if len(doc_ids) > 5 else ''}.",
        }
    except Exception as e:
        logger.exception("render_pdf_pages_tool failed")
        return {"success": False, "error": str(e), "document_ids": [], "output_filenames": [], "formatted": str(e)}


register_action(
    name="render_pdf_pages",
    category="document",
    description="Render PDF pages to images (PNG or JPEG) for vision analysis or embedding.",
    inputs_model=RenderPdfPagesInputs,
    params_model=RenderPdfPagesParams,
    outputs_model=RenderPdfPagesOutputs,
    tool_function=render_pdf_pages_tool,
)


# ----- Compress PDF (Ghostscript) -----
class CompressPdfInputs(BaseModel):
    document_id: str = Field(description="PDF document ID")
    quality: str = Field(default="ebook", description="Compression level: screen, ebook, printer, prepress")


class CompressPdfOutputs(BaseModel):
    success: bool = Field(description="Whether the operation succeeded")
    document_id: Optional[str] = Field(default=None)
    output_filename: Optional[str] = Field(default=None)
    error: Optional[str] = Field(default=None)
    formatted: str = Field(description="Human-readable summary")


async def compress_pdf_tool(
    document_id: str,
    quality: str = "ebook",
    user_id: str = "system",
) -> Dict[str, Any]:
    """Compress a PDF to reduce file size."""
    try:
        backend = await get_backend_tool_client()
        content = await backend.get_document_content(document_id, user_id=user_id)
        if content is None:
            return {"success": False, "error": "Document not found", "formatted": "Document not found."}
        raw = _content_to_bytes(content)
        doc = await backend.get_document(document_id, user_id=user_id)
        filename = doc.get("filename", "input.pdf") if doc else "input.pdf"
        cli = get_cli_service_client()
        result = await cli.compress_pdf(raw, filename, quality=quality)
        if not result.get("success"):
            return {"success": False, "error": result.get("error"), "formatted": result.get("formatted", result.get("error", "Compress PDF failed."))}
        out_data = result.get("output_data", b"")
        out_name = result.get("output_filename", "compressed.pdf")
        stored = await _store_result(out_data, out_name, user_id, is_text=False)
        if not stored.get("success"):
            return {"success": False, "error": stored.get("error"), "formatted": stored.get("error", "Storage failed.")}
        return {
            "success": True,
            "document_id": stored.get("document_id"),
            "output_filename": out_name,
            "formatted": f"Compressed PDF ({quality}). New document: {stored.get('filename', out_name)}.",
        }
    except Exception as e:
        logger.exception("compress_pdf_tool failed")
        return {"success": False, "error": str(e), "formatted": str(e)}


register_action(
    name="compress_pdf",
    category="document",
    description="Compress a PDF to reduce file size (screen, ebook, printer, prepress).",
    inputs_model=CompressPdfInputs,
    outputs_model=CompressPdfOutputs,
    tool_function=compress_pdf_tool,
)


# ----- Convert to PDF/A (Ghostscript) -----
class ConvertPdfAInputs(BaseModel):
    document_id: str = Field(description="PDF document ID")


class ConvertPdfAOutputs(BaseModel):
    success: bool = Field(description="Whether the operation succeeded")
    document_id: Optional[str] = Field(default=None)
    output_filename: Optional[str] = Field(default=None)
    error: Optional[str] = Field(default=None)
    formatted: str = Field(description="Human-readable summary")


async def convert_pdfa_tool(document_id: str, user_id: str = "system") -> Dict[str, Any]:
    """Convert a PDF to PDF/A-2b (archival format)."""
    try:
        backend = await get_backend_tool_client()
        content = await backend.get_document_content(document_id, user_id=user_id)
        if content is None:
            return {"success": False, "error": "Document not found", "formatted": "Document not found."}
        raw = _content_to_bytes(content)
        doc = await backend.get_document(document_id, user_id=user_id)
        filename = doc.get("filename", "input.pdf") if doc else "input.pdf"
        cli = get_cli_service_client()
        result = await cli.convert_pdfa(raw, filename)
        if not result.get("success"):
            return {"success": False, "error": result.get("error"), "formatted": result.get("formatted", result.get("error", "Convert PDF/A failed."))}
        out_data = result.get("output_data", b"")
        out_name = result.get("output_filename", "output.pdf")
        stored = await _store_result(out_data, out_name, user_id, is_text=False)
        if not stored.get("success"):
            return {"success": False, "error": stored.get("error"), "formatted": stored.get("error", "Storage failed.")}
        return {
            "success": True,
            "document_id": stored.get("document_id"),
            "output_filename": out_name,
            "formatted": f"Converted to PDF/A-2b. New document: {stored.get('filename', out_name)}.",
        }
    except Exception as e:
        logger.exception("convert_pdfa_tool failed")
        return {"success": False, "error": str(e), "formatted": str(e)}


register_action(
    name="convert_pdfa",
    category="document",
    description="Convert a PDF to PDF/A-2b archival format.",
    inputs_model=ConvertPdfAInputs,
    outputs_model=ConvertPdfAOutputs,
    tool_function=convert_pdfa_tool,
)
