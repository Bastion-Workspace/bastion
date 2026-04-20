"""
Write collaborative document plain text and queue re-indexing (Yjs room flush).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from services.database_manager.database_helpers import fetch_one

logger = logging.getLogger(__name__)

_ADMIN_RLS = {"user_id": "", "user_role": "admin"}


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
    Persist Y.Text content to the canonical file and queue embedding reprocess (document-service).
    """
    from clients.document_service_client import get_document_service_client

    dsc = get_document_service_client()
    await dsc.initialize(required=True)
    await dsc.flush_collaborative_document(document_id, new_content)


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
    from services.database_manager.database_helpers import execute

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
    """Load UTF-8 text via document-service GetDocumentContent."""
    import grpc

    from clients.document_service_client import get_document_service_client
    from services.service_container import get_service_container

    container = await get_service_container()
    document_service = container.document_service
    doc_info = await document_service.get_document(document_id)
    if not doc_info:
        return ""
    uid = getattr(doc_info, "user_id", None) or ""
    dsc = get_document_service_client()
    try:
        await dsc.initialize(required=True)
        resp = await dsc.get_document_content_grpc(document_id, uid)
        return resp.content or ""
    except grpc.RpcError as e:
        logger.warning("collab_persist: DS read failed for %s: %s", document_id, e)
        return ""
    except Exception as e:
        logger.warning("collab_persist: read failed for %s: %s", document_id, e)
        return ""
