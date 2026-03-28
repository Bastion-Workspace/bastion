"""
User facts Celery tasks: async embedding of new facts and periodic purge of expired facts.
"""

import logging
from datetime import datetime
from typing import Dict, Any

from services.celery_app import celery_app
from services.celery_tasks.async_runner import run_async

logger = logging.getLogger(__name__)


async def _embed_one_fact(user_id: str, fact_key: str, value: str) -> bool:
    """Generate embedding for one fact and store in user_facts.embedding.
    Uses a new VectorServiceClient per call so the gRPC channel is bound to the
    current event loop (required in Celery workers where each task uses its own loop).
    """
    from clients.vector_service_client import VectorServiceClient
    from services.database_manager.database_helpers import execute

    text = f"{fact_key}: {value}"
    client = VectorServiceClient()
    try:
        await client.initialize(required=True)
        vectors = await client.generate_embeddings([text])
    finally:
        await client.close()

    if not vectors or len(vectors) == 0:
        logger.warning("embed_user_fact_task: no embedding returned for %s", fact_key)
        return False
    vec = vectors[0]
    await execute(
        "UPDATE user_facts SET embedding = $1 WHERE user_id = $2 AND fact_key = $3",
        vec,
        user_id,
        fact_key,
    )
    return True


@celery_app.task(bind=True, name="services.celery_tasks.fact_tasks.embed_user_fact_task")
def embed_user_fact_task(
    self,
    user_id: str,
    fact_key: str,
    value: str,
) -> Dict[str, Any]:
    """Background task: embed a single user fact and store in user_facts.embedding."""
    try:
        ok = run_async(_embed_one_fact(user_id, fact_key, value))
        return {
            "success": ok,
            "task_id": self.request.id,
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "fact_key": fact_key,
        }
    except Exception as e:
        logger.exception("embed_user_fact_task failed: %s", e)
        raise


async def _purge_expired_facts() -> int:
    """Delete user_facts where expires_at IS NOT NULL AND expires_at < NOW(). Returns deleted count."""
    from services.database_manager.database_helpers import fetch_value, execute

    deleted = await fetch_value(
        "SELECT COUNT(*) FROM user_facts WHERE expires_at IS NOT NULL AND expires_at < NOW()"
    )
    count = int(deleted) if deleted is not None else 0
    if count > 0:
        await execute("DELETE FROM user_facts WHERE expires_at IS NOT NULL AND expires_at < NOW()")
    return count


@celery_app.task(bind=True, name="services.celery_tasks.fact_tasks.purge_expired_facts_task")
def purge_expired_facts_task(self) -> Dict[str, Any]:
    """Celery Beat: purge user_facts with expires_at < NOW()."""
    try:
        deleted = run_async(_purge_expired_facts())
        if deleted > 0:
            logger.info("purge_expired_facts_task: deleted %s expired fact(s)", deleted)
        return {
            "success": True,
            "task_id": self.request.id,
            "timestamp": datetime.now().isoformat(),
            "deleted_count": deleted,
        }
    except Exception as e:
        logger.exception("purge_expired_facts_task failed: %s", e)
        raise
