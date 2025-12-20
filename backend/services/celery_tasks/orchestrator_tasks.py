"""
Orchestrator Celery Tasks
Background processing for the "Big Stick" Orchestrator
"""

import logging
from datetime import datetime
from typing import Dict, Any

from services.celery_app import celery_app
from services.celery_utils import (
    safe_serialize_error, 
    clean_result_for_storage
)

logger = logging.getLogger(__name__)


# Deprecated functions removed:
# - _store_task_result_in_redis - only used by deprecated process_orchestrator_query
# - process_orchestrator_query - deprecated task that immediately returns error
# - _async_process_orchestrator_query - only used by deprecated process_orchestrator_query


@celery_app.task(bind=True, name="orchestrator.get_task_status")
def get_task_status(self, task_id: str) -> Dict[str, Any]:
    """Get the status of an orchestrator task"""
    try:
        # Get task result
        result = celery_app.AsyncResult(task_id)
        
        return clean_result_for_storage({
            "task_id": task_id,
            "status": result.status,
            "result": result.result,
            "ready": result.ready(),
            "successful": result.successful() if result.ready() else None,
            "failed": result.failed() if result.ready() else None,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting task status: {e}")
        error_data = safe_serialize_error(e, "Get task status")
        return clean_result_for_storage({
            "task_id": task_id,
            "status": "ERROR", 
            "error": error_data["error_message"],
            "error_type": error_data["error_type"],
            "timestamp": error_data["timestamp"]
        })


@celery_app.task(bind=True, name="orchestrator.cancel_task")
def cancel_orchestrator_task(self, task_id: str) -> Dict[str, Any]:
    """Cancel a running orchestrator task"""
    try:
        # Revoke the task
        celery_app.control.revoke(task_id, terminate=True)
        
        logger.info(f"🛑 TASK CANCELLED: {task_id}")
        
        return clean_result_for_storage({
            "task_id": task_id,
            "status": "CANCELLED",
            "message": "Task successfully cancelled",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Error cancelling task: {e}")
        error_data = safe_serialize_error(e, "Cancel task")
        return clean_result_for_storage({
            "task_id": task_id,
            "status": "ERROR",
            "error": error_data["error_message"],
            "error_type": error_data["error_type"],
            "message": "Failed to cancel task",
            "timestamp": error_data["timestamp"]
        })
