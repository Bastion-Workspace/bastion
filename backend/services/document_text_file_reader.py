"""
Resolve on-disk path for a user document and read UTF-8 text (for TTS export, etc.).
"""

import logging
from pathlib import Path
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)

_TEXT_SUFFIXES = (".md", ".txt", ".org", ".markdown")


def is_text_document_filename(filename: Optional[str]) -> bool:
    if not filename:
        return False
    lower = filename.lower()
    return any(lower.endswith(s) for s in _TEXT_SUFFIXES)


async def read_user_document_text(doc_id: str, user_id: str) -> Optional[str]:
    """
    Load plain text from the document file on disk.
    Returns None if document missing, not a text file, or file not found.
    """
    from services.service_container import get_service_container

    container = await get_service_container()
    document_service = container.document_service
    folder_service = container.folder_service

    doc_info = await document_service.get_document(doc_id)
    if not doc_info or not doc_info.filename:
        logger.warning("Document not found or no filename: %s", doc_id)
        return None

    if not is_text_document_filename(doc_info.filename):
        logger.warning("Not a text document for TTS export: %s", doc_info.filename)
        return None

    collection_type = getattr(doc_info, "collection_type", "user")
    folder_id = getattr(doc_info, "folder_id", None)
    file_path: Optional[Path] = None

    try:
        folder_path = await folder_service.get_document_file_path(
            filename=doc_info.filename,
            folder_id=folder_id,
            user_id=user_id,
            collection_type=collection_type,
        )
        if folder_path and Path(folder_path).exists():
            file_path = Path(folder_path)
        else:
            alt = f"{doc_id}_{doc_info.filename}"
            folder_path = await folder_service.get_document_file_path(
                filename=alt,
                folder_id=folder_id,
                user_id=user_id,
                collection_type=collection_type,
            )
            if folder_path and Path(folder_path).exists():
                file_path = Path(folder_path)
    except Exception as e:
        logger.warning("Folder service path resolution failed: %s", e)

    if not file_path or not file_path.exists():
        upload_dir = Path(settings.UPLOAD_DIR)
        for potential_file in upload_dir.glob(f"{doc_id}_*"):
            if potential_file.is_file():
                file_path = potential_file
                break

    if not file_path or not file_path.exists():
        logger.warning("Document file not on disk: %s", doc_id)
        return None

    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("File is not valid UTF-8 text: %s", file_path)
        return None
    except Exception as e:
        logger.error("Failed to read document file %s: %s", file_path, e)
        return None
