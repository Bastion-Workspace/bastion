"""
Resolve document text via document-service (for TTS export, etc.).
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_TEXT_SUFFIXES = (".md", ".txt", ".org", ".markdown", ".json")


def is_text_document_filename(filename: Optional[str]) -> bool:
    if not filename:
        return False
    lower = filename.lower()
    return any(lower.endswith(s) for s in _TEXT_SUFFIXES)


async def read_user_document_text(doc_id: str, user_id: str) -> Optional[str]:
    """
    Load plain text for a document from document-service.
    Returns None if document missing, not a text file, or read fails.
    """
    from services.service_container import get_service_container
    from clients.document_service_client import get_document_service_client

    container = await get_service_container()
    document_service = container.document_service

    doc_info = await document_service.get_document(doc_id)
    if not doc_info or not doc_info.filename:
        logger.warning("Document not found or no filename: %s", doc_id)
        return None

    if not is_text_document_filename(doc_info.filename):
        logger.warning("Not a text document for TTS export: %s", doc_info.filename)
        return None

    dsc = get_document_service_client()
    try:
        await dsc.initialize(required=True)
        resp = await dsc.get_document_content_grpc(doc_id, user_id)
        text = (resp.content or "") if resp else ""
        return text if text else None
    except Exception as e:
        logger.error("Failed to read document %s via document-service: %s", doc_id, e)
        return None
