"""Daily markdown notes and wikilink stub creation for Zettelkasten."""

import logging
import re
from datetime import date, datetime
from typing import Any, Dict, Optional, Tuple

from models.zettelkasten_models import DailyNoteFormat, ZettelkastenSettings
from services.file_manager.models.file_placement_models import (
    FilePlacementRequest,
    SourceType,
)
from models.api_models import DocumentType, DocumentCategory

logger = logging.getLogger(__name__)


def _daily_filename(d: date, fmt: DailyNoteFormat) -> str:
    if fmt == DailyNoteFormat.ISO:
        return f"{d.isoformat()}.md"
    if fmt == DailyNoteFormat.ISO_DAY:
        return f"{d.strftime('%Y-%m-%d')}-{d.strftime('%A')}.md"
    if fmt == DailyNoteFormat.COMPACT:
        return f"{d.strftime('%Y%m%d')}.md"
    return f"{d.isoformat()}.md"


def _safe_wikilink_filename(title: str) -> str:
    base = re.sub(r"[^a-zA-Z0-9_\-\s]", "", (title or "").strip()).replace(" ", "-")[:120]
    if not base:
        base = "note"
    if not base.lower().endswith(".md"):
        base = f"{base}.md"
    return base


async def _resolve_folder_id(user_id: str, rel_path: Optional[str]) -> Optional[str]:
    if not rel_path or not str(rel_path).strip():
        return None
    from services.service_container import get_service_container

    container = await get_service_container()
    fs = container.folder_service
    parts = [p.strip() for p in str(rel_path).replace("\\", "/").split("/") if p.strip()]
    parent_id = None
    for i, part in enumerate(parts):
        if parent_id is None and i == 0:
            folder = await fs.get_or_create_root_folder(
                folder_name=part,
                user_id=user_id,
                collection_type="user",
                current_user_role="user",
                admin_user_id=user_id,
            )
        else:
            folder = await fs.get_or_create_subfolder(
                parent_folder_id=parent_id,
                folder_name=part,
                user_id=user_id,
                collection_type="user",
                current_user_role="user",
                admin_user_id=user_id,
            )
        parent_id = folder.folder_id
    return parent_id


async def _place_markdown(
    user_id: str,
    *,
    title: str,
    filename: str,
    content: str,
    target_folder_id: Optional[str],
) -> Tuple[bool, Optional[str], Optional[str]]:
    from services.file_manager import get_file_manager

    fm = await get_file_manager()
    if not fm._initialized:
        await fm.initialize()
    req = FilePlacementRequest(
        content=content,
        title=title,
        filename=filename,
        source_type=SourceType.MANUAL,
        doc_type=DocumentType.MD,
        category=DocumentCategory.OTHER,
        user_id=user_id,
        collection_type="user",
        target_folder_id=target_folder_id,
        process_immediately=True,
    )
    try:
        resp = await fm.place_file(req)
        return True, resp.document_id, resp.filename
    except Exception as e:
        logger.warning("place_file failed: %s", e)
        return False, None, str(e)


class ZettelkastenDailyNoteService:
    async def get_or_create_daily_note(
        self, user_id: str, settings: ZettelkastenSettings, day: Optional[date] = None
    ) -> Dict[str, Any]:
        d = day or date.today()
        folder_id = await _resolve_folder_id(user_id, settings.daily_note_location)
        fname = _daily_filename(d, settings.daily_note_format)

        from repositories.document_repository import DocumentRepository

        repo = DocumentRepository()
        await repo.initialize()
        existing = await repo.find_by_filename_and_context(
            fname, user_id, "user", folder_id, case_insensitive=True
        )
        if existing:
            return {
                "success": True,
                "document_id": existing.document_id,
                "filename": existing.filename,
                "created": False,
            }

        body = settings.daily_note_template or ""
        if not body.strip():
            body = f"# {d.isoformat()}\n\n"
        ok, doc_id, err = await _place_markdown(
            user_id,
            title=fname.replace(".md", ""),
            filename=fname,
            content=body,
            target_folder_id=folder_id,
        )
        if not ok:
            return {"success": False, "error": err or "place_file failed"}
        return {"success": True, "document_id": doc_id, "filename": fname, "created": True}

    async def create_wikilink_note(
        self, user_id: str, settings: ZettelkastenSettings, title: str
    ) -> Dict[str, Any]:
        raw = (title or "").strip()
        if not raw:
            return {"success": False, "error": "empty title"}
        fname = _safe_wikilink_filename(raw)
        stem_for_title = fname[:-3] if fname.lower().endswith(".md") else fname

        from repositories.document_repository import DocumentRepository

        repo = DocumentRepository()
        await repo.initialize()

        folder_id = await _resolve_folder_id(user_id, settings.daily_note_location)
        existing = await repo.find_by_filename_and_context(
            fname, user_id, "user", folder_id, case_insensitive=True
        )
        if existing:
            return {
                "success": True,
                "document_id": existing.document_id,
                "filename": existing.filename,
                "created": False,
            }

        prefix = ""
        if settings.note_id_prefix:
            prefix = datetime.utcnow().strftime("%Y%m%d%H%M")
            if not fname.lower().startswith(prefix.lower()):
                fname = f"{prefix}-{fname}"

        content = f"# {stem_for_title}\n\n"
        ok, doc_id, err = await _place_markdown(
            user_id,
            title=stem_for_title,
            filename=fname,
            content=content,
            target_folder_id=folder_id,
        )
        if not ok:
            return {"success": False, "error": err or "place_file failed"}
        return {"success": True, "document_id": doc_id, "filename": fname, "created": True}


_daily_svc: Optional[ZettelkastenDailyNoteService] = None


async def get_zettelkasten_daily_note_service() -> ZettelkastenDailyNoteService:
    global _daily_svc
    if _daily_svc is None:
        _daily_svc = ZettelkastenDailyNoteService()
    return _daily_svc
