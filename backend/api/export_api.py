import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from api.auth_api import get_current_user, AuthenticatedUserResponse
from models.api_models import EpubExportRequest
from services.epub_export_service import EpubExportService


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/export", tags=["Export"])


@router.post("/epub")
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
                "metadata": request.metadata,
                "heading_alignments": request.heading_alignments,
            },
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





























