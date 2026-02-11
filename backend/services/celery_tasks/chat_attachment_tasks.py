"""
Chat Attachment Cleanup Tasks
Scheduled tasks for cleaning up old chat attachments
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

from services.celery_app import celery_app, TaskStatus

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="services.celery_tasks.chat_attachment_tasks.cleanup_old_chat_attachments_task")
def cleanup_old_chat_attachments_task(self, days: int = 7) -> Dict[str, Any]:
    """
    Background task for cleaning up old chat attachments
    
    Removes attachments older than specified days (default 7 days)
    """
    try:
        logger.info(f"🧹 CHAT ATTACHMENT TASK: Starting cleanup of attachments older than {days} days")

        # Create new event loop for this task
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def _cleanup():
                from services.chat_attachment_service import chat_attachment_service
                await chat_attachment_service.initialize()
                deleted_count = await chat_attachment_service.cleanup_old_attachments(days=days)
                return deleted_count
            
            deleted_count = loop.run_until_complete(_cleanup())
        finally:
            loop.close()

        logger.info(f"✅ CHAT ATTACHMENT TASK: Cleaned up {deleted_count} old attachment files")
        return {
            "success": True,
            "task_id": self.request.id,
            "timestamp": datetime.now().isoformat(),
            "deleted_count": deleted_count,
            "days": days,
            "message": f"Successfully cleaned up {deleted_count} old chat attachments"
        }

    except Exception as e:
        logger.error(f"❌ CHAT ATTACHMENT TASK ERROR: {e}")

        self.update_state(
            state=TaskStatus.FAILURE,
            meta={
                "error": str(e),
                "message": "Chat attachment cleanup failed",
                "timestamp": datetime.now().isoformat()
            }
        )

        return {
            "success": False,
            "error": str(e),
            "message": "Background chat attachment cleanup failed"
        }
