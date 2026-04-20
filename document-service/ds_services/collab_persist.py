"""
Write collaborative document plain text to disk and queue re-indexing (Yjs room flush).
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Any, Optional

from ds_config import settings
from ds_models.api_models import ProcessingStatus
from ds_db.database_manager.database_helpers import fetch_one
from ds_services.document_version_service import snapshot_before_write

logger = logging.getLogger(__name__)

_ADMIN_RLS = {"user_id": "", "user_role": "admin"}


async def _resolve_file_path(doc_info: Any, folder_service: Any) -> Optional[Path]:
    filename = getattr(doc_info, "filename", None) or ""
    if not filename:
        return None
    user_id = getattr(doc_info, "user_id", None)
    folder_id = getattr(doc_info, "folder_id", None)
    collection_type = getattr(doc_info, "collection_type", "user")
    team_id = getattr(doc_info, "team_id", None)
    if team_id is not None and not isinstance(team_id, str):
        team_id = str(team_id)
    try:
        file_path_str = await folder_service.get_document_file_path(
            filename=filename,
            folder_id=folder_id,
            user_id=user_id,
            collection_type=collection_type,
            team_id=team_id,
        )
        fp = Path(file_path_str)
        if fp.exists():
            return fp
    except Exception as e:
        logger.warning("collab_persist: folder path resolution failed: %s", e)
    upload_dir = Path(settings.UPLOAD_DIR)
    doc_id = getattr(doc_info, "document_id", None)
    legacy_paths = [
        upload_dir / f"{doc_id}_{filename}",
        upload_dir / filename,
    ]
    if filename.lower().endswith(".md"):
        legacy_paths.extend(
            glob.glob(str(upload_dir / "web_sources" / "rss_articles" / "*" / filename))
        )
        legacy_paths.extend(
            glob.glob(str(upload_dir / "web_sources" / "scraped_content" / "*" / filename))
        )
    for candidate in legacy_paths:
        if isinstance(candidate, str):
            candidate = Path(candidate)
        if candidate.exists():
            return candidate
    return None


async def _broadcast_library_saved(document_id: str, document_service: Any) -> None:
    """Notify all clients of document_metadata.updated_at after canonical file persist (collaborators, file tree)."""
    try:
        row = await fetch_one(
            """
            SELECT updated_at FROM document_metadata WHERE document_id = $1
            """,
            document_id,
            rls_context=_ADMIN_RLS,
        )
        if not row or not row.get("updated_at"):
            return
        ts = row["updated_at"]
        updated_at_iso = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        meta = await document_service.document_repository.get_document_metadata(document_id)
        folder_id = meta.get("folder_id") if meta else None
        filename = meta.get("filename") if meta else None
        from utils.websocket_manager import get_websocket_manager

        mgr = get_websocket_manager()
        if mgr:
            await mgr.send_document_status_update(
                document_id=document_id,
                status="library_saved",
                folder_id=folder_id,
                user_id=None,
                filename=filename,
                updated_at=updated_at_iso,
            )
    except Exception as e:
        logger.warning("collab_persist: library_saved broadcast failed: %s", e)


async def flush_collaborative_document(document_id: str, new_content: str) -> None:
    """
    Persist Y.Text content to the canonical file and queue embedding reprocess.
    Uses document owner for version snapshot and exempt checks.
    """
    from shims.services.service_container import get_service_container

    container = await get_service_container()
    document_service = container.document_service
    folder_service = container.folder_service
    if not document_service or not folder_service:
        raise RuntimeError("Document services not initialized")

    doc_info = await document_service.get_document(document_id)
    if not doc_info:
        raise ValueError(f"Document not found: {document_id}")

    owner_id = getattr(doc_info, "user_id", None) or "system"
    filename = getattr(doc_info, "filename", "") or ""

    editable_exts = (".txt", ".md", ".org")
    if not str(filename).lower().endswith(editable_exts):
        raise ValueError("Collaborative flush only supports .txt, .md, .org")

    file_path = await _resolve_file_path(doc_info, folder_service)
    if file_path is None:
        raise FileNotFoundError(f"Could not resolve file path for document {document_id}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            if f.read() == new_content:
                logger.info("collab_persist: no-op flush for %s", document_id)
                return
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning("collab_persist: could not compare existing file: %s", e)

    try:
        await snapshot_before_write(document_id, owner_id, "collab_flush", None, None)
    except Exception as verr:
        logger.warning("collab_persist: version snapshot failed (non-fatal): %s", verr)

    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    logger.info("collab_persist: wrote %s (%s chars)", file_path, len(new_content))

    try:
        await document_service.document_repository.update_file_size(
            document_id, len(new_content.encode("utf-8"))
        )
    except Exception as e:
        logger.warning("collab_persist: update_file_size failed: %s", e)

    if str(filename).lower().endswith((".org", ".md", ".txt")):
        try:
            from ds_services.link_extraction_service import get_link_extraction_service

            link_service = await get_link_extraction_service()
            rls_context = {"user_id": owner_id, "user_role": "user"}
            await link_service.extract_and_store_links(document_id, new_content, rls_context)
        except Exception as link_err:
            logger.warning("collab_persist: link extraction failed: %s", link_err)

    is_exempt = await document_service.document_repository.is_document_exempt(document_id, owner_id)
    if is_exempt:
        await document_service.document_repository.update_status(document_id, ProcessingStatus.COMPLETED)
        await _broadcast_library_saved(document_id, document_service)
        return

    await document_service.document_repository.update_status(document_id, ProcessingStatus.EMBEDDING)
    await _broadcast_library_saved(document_id, document_service)
    await document_service._emit_document_status_update(
        document_id, ProcessingStatus.EMBEDDING.value, owner_id
    )
    try:
        await document_service.embedding_manager.delete_document_chunks(document_id)
    except Exception as e:
        logger.warning("collab_persist: delete chunks failed: %s", e)
    if document_service.kg_service:
        try:
            await document_service.kg_service.delete_document_entities(document_id)
        except Exception as e:
            logger.warning("collab_persist: delete KG entities failed: %s", e)

    import asyncio

    from ds_services.collab_reprocess_helper import schedule_reprocess_after_save

    asyncio.create_task(schedule_reprocess_after_save(document_id, owner_id))


async def load_collab_state_row(document_id: str) -> Optional[bytes]:
    row = await fetch_one(
        """
        SELECT ydoc_state FROM document_collab_state
        WHERE document_id = $1
        """,
        document_id,
        rls_context=_ADMIN_RLS,
    )
    if not row or not row.get("ydoc_state"):
        return None
    return bytes(row["ydoc_state"])


async def save_collab_state_row(document_id: str, ydoc_state: bytes) -> None:
    from ds_db.database_manager.database_helpers import execute

    await execute(
        """
        INSERT INTO document_collab_state (document_id, ydoc_state, updated_at)
        VALUES ($1, $2, NOW())
        ON CONFLICT (document_id) DO UPDATE
        SET ydoc_state = EXCLUDED.ydoc_state,
            updated_at = NOW()
        """,
        document_id,
        ydoc_state,
        rls_context=_ADMIN_RLS,
    )


async def read_document_plaintext_for_collab(document_id: str) -> str:
    """Load UTF-8 text from disk via the same path rules as content API."""
    from shims.services.service_container import get_service_container

    container = await get_service_container()
    document_service = container.document_service
    folder_service = container.folder_service
    doc_info = await document_service.get_document(document_id)
    if not doc_info:
        return ""
    fp = await _resolve_file_path(doc_info, folder_service)
    if fp is None or not fp.exists():
        return ""
    try:
        return fp.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("collab_persist: read failed for %s: %s", document_id, e)
        return ""
