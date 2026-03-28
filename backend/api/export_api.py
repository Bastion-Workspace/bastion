import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from api.auth_api import get_current_user, AuthenticatedUserResponse
from models.api_models import (
    EpubExportRequest,
    PdfExportRequest,
    PdfHeadingOutlineRequest,
    PdfHeadingOutlineResponse,
)
from services.epub_export_service import EpubExportService
from services.pdf_export_service import PdfExportService


logger = logging.getLogger(__name__)

router = APIRouter(tags=["Export"])


@router.post("/api/export/epub")
async def export_epub(
    request: EpubExportRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    try:
        service = EpubExportService()
        epub_bytes = await service.export_markdown_to_epub(
            request.content,
            options={
                "include_toc": request.include_toc,
                "include_cover": request.include_cover,
                "split_on_headings": request.split_on_headings,
                "split_on_heading_levels": request.split_on_heading_levels,
                "metadata": {
                    **(request.metadata or {}),
                    "document_id": request.document_id,
                    "folder_id": request.folder_id,
                },
                "heading_alignments": request.heading_alignments,
                "indent_paragraphs": request.indent_paragraphs,
                "no_indent_first_paragraph": request.no_indent_first_paragraph,
            },
            user_id=current_user.user_id,
        )

        title = request.metadata.get("title") if request.metadata else None
        filename = f"{title or 'export'}.epub"

        return Response(
            content=epub_bytes,
            media_type="application/epub+zip",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    except Exception as e:
        logger.error(f"❌ EPUB export failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to export EPUB")


@router.post("/api/export/pdf")
async def export_pdf(
    request: PdfExportRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    try:
        service = PdfExportService()
        pdf_bytes = await service.export_to_pdf(
            request,
            user_id=current_user.user_id,
        )

        # Determine filename from request
        if request.title:
            filename = f"{request.title}.pdf"
        elif request.kind.value == "markdown_document" and request.metadata and request.metadata.get("title"):
            filename = f"{request.metadata.get('title')}.pdf"
        elif request.kind.value == "conversation" and request.conversation_title:
            filename = f"{request.conversation_title}.pdf"
        elif request.kind.value == "chat_message":
            filename = "chat-message.pdf"
        else:
            filename = "export.pdf"

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    except Exception as e:
        logger.error("PDF export failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to export PDF")


@router.post("/api/export/pdf/heading-outline", response_model=PdfHeadingOutlineResponse)
async def pdf_export_heading_outline(
    request: PdfHeadingOutlineRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    try:
        service = PdfExportService()
        headings = service.get_heading_outline(
            request.content,
            source_format=request.source_format or "markdown",
        )
        return PdfHeadingOutlineResponse(headings=headings)
    except Exception as e:
        logger.error("PDF heading outline failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to build heading outline")





























