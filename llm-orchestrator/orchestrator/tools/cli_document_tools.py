"""
CLI document tools - Pandoc, Poppler, Tesseract via CLI worker.
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


# ----- Convert document (Pandoc) -----
TEXT_OUTPUT_FORMATS = {"markdown", "md", "html", "latex", "rst", "plain"}
BINARY_OUTPUT_FORMATS = {"pdf", "docx", "epub"}


class ConvertDocumentInputs(BaseModel):
    document_id: str = Field(description="Source document ID")
    output_format: str = Field(description="Output format: pdf, docx, html, epub, latex, rst, plain, markdown")


class ConvertDocumentParams(BaseModel):
    input_format: Optional[str] = Field(default=None, description="Input format (auto-detected from filename if omitted)")


class ConvertDocumentOutputs(BaseModel):
    success: bool = Field(description="Whether the operation succeeded")
    document_id: Optional[str] = Field(default=None)
    output_filename: Optional[str] = Field(default=None)
    error: Optional[str] = Field(default=None)
    formatted: str = Field(description="Human-readable summary")


async def convert_document_tool(
    document_id: str,
    output_format: str = "pdf",
    input_format: Optional[str] = None,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Convert a document to another format (e.g. markdown to PDF, org to docx)."""
    try:
        backend = await get_backend_tool_client()
        content = await backend.get_document_content(document_id, user_id=user_id)
        if content is None:
            return {"success": False, "error": "Document not found", "formatted": "Document not found."}
        raw = _content_to_bytes(content)
        doc = await backend.get_document(document_id, user_id=user_id)
        filename = doc.get("filename", "input") if doc else "input"
        inf = input_format or ("markdown" if filename.lower().endswith((".md", ".markdown")) else "html" if filename.lower().endswith((".html", ".htm")) else "markdown")
        cli = get_cli_service_client()
        result = await cli.convert_document(
            input_data=raw,
            input_filename=filename,
            input_format=inf,
            output_format=output_format,
        )
        if not result.get("success"):
            return {"success": False, "error": result.get("error"), "formatted": result.get("formatted", result.get("error", "Convert failed."))}
        out_data = result.get("output_data", b"")
        out_name = result.get("output_filename", f"out.{output_format}")
        is_text = output_format.lower() in TEXT_OUTPUT_FORMATS
        stored = await _store_result(out_data, out_name, user_id, is_text=is_text)
        if not stored.get("success"):
            return {"success": False, "error": stored.get("error"), "formatted": stored.get("error", "Storage failed.")}
        return {
            "success": True,
            "document_id": stored.get("document_id"),
            "output_filename": out_name,
            "formatted": f"Converted to {output_format}. New document: {stored.get('filename', out_name)}.",
        }
    except Exception as e:
        logger.exception("convert_document_tool failed")
        return {"success": False, "error": str(e), "formatted": str(e)}


register_action(
    name="convert_document",
    category="document",
    description="Convert a document to another format (e.g. markdown to PDF, org to docx).",
    inputs_model=ConvertDocumentInputs,
    params_model=ConvertDocumentParams,
    outputs_model=ConvertDocumentOutputs,
    tool_function=convert_document_tool,
)


# ----- OCR image -----
class OcrImageInputs(BaseModel):
    document_id: str = Field(description="Image document ID")
    output_format: str = Field(default="text", description="Output format: text, hocr, tsv, pdf")


class OcrImageParams(BaseModel):
    languages: Optional[List[str]] = Field(default=None, description="OCR languages: eng, fra, deu")


class OcrImageOutputs(BaseModel):
    success: bool = Field(description="Whether the operation succeeded")
    document_id: Optional[str] = Field(default=None)
    output_filename: Optional[str] = Field(default=None)
    error: Optional[str] = Field(default=None)
    formatted: str = Field(description="Human-readable summary")


async def ocr_image_tool(
    document_id: str,
    output_format: str = "text",
    languages: Optional[List[str]] = None,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Extract text from an image document using OCR."""
    try:
        backend = await get_backend_tool_client()
        content = await backend.get_document_content(document_id, user_id=user_id)
        if content is None:
            return {"success": False, "error": "Document not found", "formatted": "Document not found."}
        raw = _content_to_bytes(content)
        doc = await backend.get_document(document_id, user_id=user_id)
        filename = doc.get("filename", "input") if doc else "input"
        cli = get_cli_service_client()
        result = await cli.ocr_image(raw, filename, output_format=output_format, languages=languages)
        if not result.get("success"):
            return {"success": False, "error": result.get("error"), "formatted": result.get("formatted", result.get("error", "OCR failed."))}
        out_data = result.get("output_data", b"")
        out_name = result.get("output_filename", "out.txt")
        is_text = output_format == "text"
        stored = await _store_result(out_data, out_name, user_id, is_text=is_text)
        if not stored.get("success"):
            return {"success": False, "error": stored.get("error"), "formatted": stored.get("error", "Storage failed.")}
        return {
            "success": True,
            "document_id": stored.get("document_id"),
            "output_filename": out_name,
            "formatted": f"OCR completed. New document: {stored.get('filename', out_name)}.",
        }
    except Exception as e:
        logger.exception("ocr_image_tool failed")
        return {"success": False, "error": str(e), "formatted": str(e)}


register_action(
    name="ocr_image",
    category="document",
    description="Extract text from an image document using OCR.",
    inputs_model=OcrImageInputs,
    params_model=OcrImageParams,
    outputs_model=OcrImageOutputs,
    tool_function=ocr_image_tool,
)


# ----- Extract PDF text -----
class ExtractPdfTextInputs(BaseModel):
    document_id: str = Field(description="PDF document ID")


class ExtractPdfTextParams(BaseModel):
    first_page: Optional[int] = Field(default=None, ge=1)
    last_page: Optional[int] = Field(default=None, ge=1)


class ExtractPdfTextOutputs(BaseModel):
    success: bool = Field(description="Whether the operation succeeded")
    document_id: Optional[str] = Field(default=None)
    output_filename: Optional[str] = Field(default=None)
    error: Optional[str] = Field(default=None)
    formatted: str = Field(description="Human-readable summary")


async def extract_pdf_text_tool(
    document_id: str,
    first_page: Optional[int] = None,
    last_page: Optional[int] = None,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Extract text from a PDF document."""
    try:
        backend = await get_backend_tool_client()
        content = await backend.get_document_content(document_id, user_id=user_id)
        if content is None:
            return {"success": False, "error": "Document not found", "formatted": "Document not found."}
        raw = _content_to_bytes(content)
        doc = await backend.get_document(document_id, user_id=user_id)
        filename = doc.get("filename", "input.pdf") if doc else "input.pdf"
        cli = get_cli_service_client()
        result = await cli.extract_pdf_text(raw, filename, first_page=first_page, last_page=last_page)
        if not result.get("success"):
            return {"success": False, "error": result.get("error"), "formatted": result.get("formatted", result.get("error", "Extract PDF text failed."))}
        out_data = result.get("output_data", b"")
        out_name = result.get("output_filename", "out.txt")
        stored = await _store_result(out_data, out_name, user_id, is_text=True)
        if not stored.get("success"):
            return {"success": False, "error": stored.get("error"), "formatted": stored.get("error", "Storage failed.")}
        return {
            "success": True,
            "document_id": stored.get("document_id"),
            "output_filename": out_name,
            "formatted": f"Extracted PDF text. New document: {stored.get('filename', out_name)}.",
        }
    except Exception as e:
        logger.exception("extract_pdf_text_tool failed")
        return {"success": False, "error": str(e), "formatted": str(e)}


register_action(
    name="extract_pdf_text",
    category="document",
    description="Extract text from a PDF document.",
    inputs_model=ExtractPdfTextInputs,
    params_model=ExtractPdfTextParams,
    outputs_model=ExtractPdfTextOutputs,
    tool_function=extract_pdf_text_tool,
)


# ----- Split PDF -----
class SplitPdfInputs(BaseModel):
    document_id: str = Field(description="PDF document ID to split into one file per page")


class SplitPdfOutputs(BaseModel):
    success: bool = Field(description="Whether the operation succeeded")
    document_ids: List[str] = Field(default_factory=list)
    output_filenames: List[str] = Field(default_factory=list)
    error: Optional[str] = Field(default=None)
    formatted: str = Field(description="Human-readable summary")


async def split_pdf_tool(document_id: str, user_id: str = "system") -> Dict[str, Any]:
    """Split a PDF into one document per page."""
    try:
        backend = await get_backend_tool_client()
        content = await backend.get_document_content(document_id, user_id=user_id)
        if content is None:
            return {"success": False, "error": "Document not found", "document_ids": [], "output_filenames": [], "formatted": "Document not found."}
        raw = _content_to_bytes(content)
        doc = await backend.get_document(document_id, user_id=user_id)
        filename = doc.get("filename", "input.pdf") if doc else "input.pdf"
        cli = get_cli_service_client()
        result = await cli.split_pdf(raw, filename)
        if not result.get("success"):
            return {"success": False, "error": result.get("error"), "document_ids": [], "output_filenames": [], "formatted": result.get("formatted", result.get("error", "Split PDF failed."))}
        doc_ids = []
        names = result.get("output_filenames", [])
        for i, data in enumerate(result.get("output_data", [])):
            out_name = names[i] if i < len(names) else f"page_{i+1}.pdf"
            stored = await _store_result(bytes(data), out_name, user_id, is_text=False)
            if stored.get("success"):
                doc_ids.append(stored.get("document_id", ""))
        return {
            "success": True,
            "document_ids": doc_ids,
            "output_filenames": names,
            "formatted": f"Split into {len(doc_ids)} page(s). New document IDs: {', '.join(doc_ids[:5])}{'...' if len(doc_ids) > 5 else ''}.",
        }
    except Exception as e:
        logger.exception("split_pdf_tool failed")
        return {"success": False, "error": str(e), "document_ids": [], "output_filenames": [], "formatted": str(e)}


register_action(
    name="split_pdf",
    category="document",
    description="Split a PDF into one document per page.",
    inputs_model=SplitPdfInputs,
    outputs_model=SplitPdfOutputs,
    tool_function=split_pdf_tool,
)


# ----- Merge PDFs -----
class MergePdfsInputs(BaseModel):
    document_ids: str = Field(description="Comma-separated list of PDF document IDs to merge in order")


class MergePdfsOutputs(BaseModel):
    success: bool = Field(description="Whether the operation succeeded")
    document_id: Optional[str] = Field(default=None)
    output_filename: Optional[str] = Field(default=None)
    error: Optional[str] = Field(default=None)
    formatted: str = Field(description="Human-readable summary")


async def merge_pdfs_tool(document_ids: str, user_id: str = "system") -> Dict[str, Any]:
    """Merge multiple PDF documents into one."""
    try:
        ids = [x.strip() for x in document_ids.split(",") if x.strip()]
        if len(ids) < 2:
            return {"success": False, "error": "At least 2 document IDs required", "formatted": "At least 2 document IDs required."}
        backend = await get_backend_tool_client()
        inputs = []
        for did in ids:
            content = await backend.get_document_content(did, user_id=user_id)
            if content is None:
                return {"success": False, "error": f"Document not found: {did}", "formatted": f"Document not found: {did}."}
            inputs.append(_content_to_bytes(content))
        cli = get_cli_service_client()
        result = await cli.merge_pdfs(inputs)
        if not result.get("success"):
            return {"success": False, "error": result.get("error"), "formatted": result.get("formatted", result.get("error", "Merge failed."))}
        out_data = result.get("output_data", b"")
        out_name = result.get("output_filename", "merged.pdf")
        stored = await _store_result(out_data, out_name, user_id, is_text=False)
        if not stored.get("success"):
            return {"success": False, "error": stored.get("error"), "formatted": stored.get("error", "Storage failed.")}
        return {
            "success": True,
            "document_id": stored.get("document_id"),
            "output_filename": out_name,
            "formatted": f"Merged {len(ids)} PDF(s). New document: {stored.get('filename', out_name)}.",
        }
    except Exception as e:
        logger.exception("merge_pdfs_tool failed")
        return {"success": False, "error": str(e), "formatted": str(e)}


register_action(
    name="merge_pdfs",
    category="document",
    description="Merge multiple PDF documents into one (order by document_ids).",
    inputs_model=MergePdfsInputs,
    outputs_model=MergePdfsOutputs,
    tool_function=merge_pdfs_tool,
)
