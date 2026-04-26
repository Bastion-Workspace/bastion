"""Background embedding for code workspace chunks left in embedding_pending state."""

import logging
from typing import Any, Dict

from services.celery_app import celery_app
from services.celery_tasks.async_runner import run_async

logger = logging.getLogger(__name__)


async def _embed_pending_async(workspace_id: str, user_id: str) -> Dict[str, Any]:
    from services.database_manager.database_helpers import fetch_all
    from services import code_workspace_index_service as cidx

    rls = {"user_id": user_id, "user_role": "user"}
    rows = await fetch_all(
        """
        SELECT id::text, user_id, workspace_id::text, file_path, start_line, end_line, content
        FROM code_chunks
        WHERE workspace_id = $1::uuid AND user_id = $2 AND embedding_pending IS TRUE
        ORDER BY file_path, chunk_index
        LIMIT 500
        """,
        workspace_id,
        user_id,
        rls_context=rls,
    )
    if not rows:
        return {"embedded": 0, "message": "none_pending"}
    n = await cidx.embed_chunks_batch(list(rows), rls)
    return {"embedded": n, "batch_size": len(rows)}


@celery_app.task(bind=True, name="services.celery_tasks.code_workspace_tasks.embed_pending_code_chunks")
def embed_pending_code_chunks_task(self, workspace_id: str, user_id: str) -> Dict[str, Any]:
    """Drain pending code chunk embeddings for one workspace."""
    try:
        logger.info("embed_pending_code_chunks workspace=%s user=%s", workspace_id, user_id)
        return run_async(_embed_pending_async(workspace_id, user_id))
    except Exception as e:
        logger.error("embed_pending_code_chunks failed: %s", e)
        return {"embedded": 0, "error": str(e)}
