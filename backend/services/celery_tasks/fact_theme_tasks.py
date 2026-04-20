"""
Periodic clustering of user_facts into user_fact_themes (embedding-based, no LLM).
"""

import logging
from datetime import datetime
from typing import Any, Dict, List

from services.celery_app import celery_app
from services.celery_tasks.async_runner import run_async

logger = logging.getLogger(__name__)


async def _cluster_all_users_async() -> Dict[str, Any]:
    from services.fact_theme_service import cluster_user_facts, list_users_with_embedded_facts

    user_ids = await list_users_with_embedded_facts()
    processed = 0
    errors: List[str] = []
    for uid in user_ids:
        try:
            await cluster_user_facts(uid)
            processed += 1
        except Exception as e:
            errors.append(f"{uid}: {e}")
            logger.warning("cluster_user_facts failed for %s: %s", uid, e)
    return {"users_seen": len(user_ids), "processed": processed, "errors": errors}


@celery_app.task(bind=True, name="services.celery_tasks.fact_theme_tasks.cluster_user_fact_themes_task")
def cluster_user_fact_themes_task(self) -> Dict[str, Any]:
    """Beat: rebuild fact themes for all users with enough embedded facts."""
    try:
        result = run_async(_cluster_all_users_async())
        return {
            "success": True,
            "task_id": self.request.id,
            "timestamp": datetime.now().isoformat(),
            **result,
        }
    except Exception as e:
        logger.exception("cluster_user_fact_themes_task failed: %s", e)
        return {
            "success": False,
            "task_id": self.request.id,
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
        }
