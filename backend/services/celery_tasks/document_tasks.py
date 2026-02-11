"""
Document Celery Tasks
Background reprocessing after document content save so the save response returns immediately.
"""

import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

from services.celery_app import celery_app
from config import settings

logger = logging.getLogger(__name__)


async def _async_reprocess_document_after_save(doc_id: str, user_id: str) -> Dict[str, Any]:
    """
    Resolve document file path and run full reprocess (re-embed + entity extraction).
    Called from Celery after content has been written to disk by update_document_content.
    """
    from services.service_container import get_service_container

    container = await get_service_container()
    document_service = container.document_service
    folder_service = container.folder_service

    doc_info = await document_service.get_document(doc_id)
    if not doc_info:
        logger.warning(f"Document not found for reprocess: {doc_id}")
        return {"success": False, "error": "Document not found", "document_id": doc_id}

    file_path = None
    try:
        folder_path = await folder_service.get_document_file_path(
            filename=doc_info.filename,
            folder_id=getattr(doc_info, "folder_id", None),
            user_id=user_id,
            collection_type="user",
        )
        if folder_path and Path(folder_path).exists():
            file_path = Path(folder_path)
        else:
            filename_with_id = f"{doc_id}_{doc_info.filename}"
            folder_path = await folder_service.get_document_file_path(
                filename=filename_with_id,
                folder_id=getattr(doc_info, "folder_id", None),
                user_id=user_id,
                collection_type="user",
            )
            if folder_path and Path(folder_path).exists():
                file_path = Path(folder_path)
    except Exception as e:
        logger.warning(f"Failed to resolve path with folder service: {e}")

    if not file_path or not file_path.exists():
        upload_dir = Path(settings.UPLOAD_DIR)
        for potential_file in upload_dir.glob(f"{doc_id}_*"):
            file_path = potential_file
            break

    if not file_path or not file_path.exists():
        logger.warning(f"Original file not found for reprocess: {doc_id}")
        return {"success": False, "error": "File not found on disk", "document_id": doc_id}

    doc_type = document_service._detect_document_type(doc_info.filename)
    await document_service._process_document_async(doc_id, file_path, doc_type, user_id)
    return {"success": True, "document_id": doc_id}


@celery_app.task(bind=True, name="services.celery_tasks.document_tasks.reprocess_document_after_save")
def reprocess_document_after_save_task(self, doc_id: str, user_id: str) -> Dict[str, Any]:
    """
    Celery task: run vector re-embedding and entity extraction after document save.
    Save API returns immediately after writing to disk; this task runs in the background.
    """
    try:
        logger.info(f"Document reprocess task started: {doc_id}")
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(_async_reprocess_document_after_save(doc_id, user_id))
        finally:
            loop.close()
        logger.info(f"Document reprocess task completed: {doc_id}")
        return result
    except Exception as e:
        logger.error(f"Document reprocess task failed: {doc_id} - {e}")
        return {
            "success": False,
            "error": str(e),
            "document_id": doc_id,
            "message": "Background re-indexing failed",
        }
