"""In-process reprocess after collab save (replaces Celery in document-service)."""

import logging
from pathlib import Path

from ds_config import settings
from shims.services.service_container import get_service_container

logger = logging.getLogger(__name__)


async def schedule_reprocess_after_save(document_id: str, user_id: str) -> None:
    """Resolve path and run full reprocess on the in-process document service."""
    try:
        container = await get_service_container()
        document_service = container.document_service
        if not document_service:
            logger.warning("collab_reprocess: no document_service in container")
            return

        doc_info = await document_service.get_document(document_id)
        if not doc_info:
            logger.warning("collab_reprocess: document not found %s", document_id)
            return

        path_user_id = getattr(doc_info, "user_id", None) or user_id
        folder_service = container.folder_service
        collection_type = getattr(doc_info, "collection_type", None) or "user"
        team_id = getattr(doc_info, "team_id", None)
        if team_id is not None and not isinstance(team_id, str):
            team_id = str(team_id)

        file_path = None
        try:
            folder_path = await folder_service.get_document_file_path(
                filename=doc_info.filename,
                folder_id=getattr(doc_info, "folder_id", None),
                user_id=path_user_id,
                collection_type=collection_type,
                team_id=team_id,
            )
            if folder_path and Path(folder_path).exists():
                file_path = Path(folder_path)
            else:
                filename_with_id = f"{document_id}_{doc_info.filename}"
                folder_path = await folder_service.get_document_file_path(
                    filename=filename_with_id,
                    folder_id=getattr(doc_info, "folder_id", None),
                    user_id=path_user_id,
                    collection_type=collection_type,
                    team_id=team_id,
                )
                if folder_path and Path(folder_path).exists():
                    file_path = Path(folder_path)
        except Exception as e:
            logger.warning("collab_reprocess: folder path failed: %s", e)

        if not file_path or not file_path.exists():
            upload_dir = Path(settings.UPLOAD_DIR)
            for potential_file in upload_dir.glob(f"{document_id}_*"):
                file_path = potential_file
                break

        if not file_path or not file_path.exists():
            logger.warning("collab_reprocess: file not on disk for %s", document_id)
            return

        doc_type = document_service._detect_document_type(doc_info.filename)
        await document_service._process_document_async(
            document_id, file_path, doc_type, path_user_id
        )
    except Exception as e:
        logger.exception("collab_reprocess failed: %s", e)
