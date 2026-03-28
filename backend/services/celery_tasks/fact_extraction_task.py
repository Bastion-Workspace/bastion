"""
Legacy Celery entry point for per-turn user fact extraction (deprecated).

Facts and episodic memory are produced by post_session_analysis_task (session_analysis_task).
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from services.celery_app import celery_app

logger = logging.getLogger(__name__)

# Kept for callers that still import the constant (no longer used for scheduling).
FACT_EXTRACT_MESSAGE_INTERVAL = 20


@celery_app.task(bind=True, name="services.celery_tasks.fact_extraction_task.extract_user_facts_task")
def extract_user_facts_task(
    self,
    user_id: str,
    conversation_id: Optional[str],
    query: str,
    accumulated_response: str,
) -> Dict[str, Any]:
    """
    Deprecated: per-turn fact extraction replaced by post_session_analysis_task.
    No-op for in-flight Celery messages after deploy.
    """
    logger.warning(
        "extract_user_facts_task is deprecated; use post_session_analysis_task (user_id=%s conv=%s)",
        user_id,
        conversation_id,
    )
    return {
        "success": True,
        "deprecated": True,
        "task_id": self.request.id,
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "conversation_id": conversation_id,
        "extracted": 0,
        "skipped": "deprecated_no_op",
    }
