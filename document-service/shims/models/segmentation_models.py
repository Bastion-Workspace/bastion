"""Minimal types for optional enhanced-PDF paths (unused in typical document-service flow)."""

from typing import Any, Optional
from pydantic import BaseModel


class PDFExtractionRequest(BaseModel):
    document_id: str
    extract_images: bool = False
    image_dpi: int = 300
    image_format: str = "PNG"


class PDFExtractionResult(BaseModel):
    pages_extracted: int = 0
    raw_text: str = ""
    metadata: Optional[Any] = None
