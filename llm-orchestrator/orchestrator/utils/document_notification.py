"""
Document Update Notification Utility
Allows any agent to notify the UI when documents are updated

Note: The backend document_service already sends WebSocket notifications when documents
are updated via update_document_content. This utility provides a way to explicitly
trigger notifications if needed, or can be called after updates to ensure notifications.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


async def notify_document_update(
    document_id: str,
    user_id: str,
    filename: Optional[str] = None,
    folder_id: Optional[str] = None
) -> None:
    """
    Notify the UI that a document has been updated.
    This will refresh any open editor tabs showing this document.
    
    Note: The backend document_service typically sends these notifications automatically
    when documents are updated. This function can be used to explicitly trigger
    notifications if needed.
    
    Args:
        document_id: Document ID that was updated
        user_id: User ID (for routing notifications)
        filename: Optional filename for UI display
        folder_id: Optional folder ID for folder refresh
    """
    try:
        # The backend document_service already sends WebSocket notifications
        # when update_document_content is called. However, we can trigger
        # an additional notification by calling get_document which may trigger
        # a status update, or we can rely on the backend's automatic notifications.
        
        # For now, we'll log that a notification should be sent
        # The actual notification happens in the backend when documents are updated
        logger.info(f"📄 Document update notification requested: {document_id} ({filename})")
        logger.info(f"📄 Backend should automatically send WebSocket notification via document_service")
        
        # TODO: If needed, we can add a gRPC call to explicitly trigger notification
        # For now, the backend's update_document_content already handles this
        
    except Exception as e:
        # Don't fail the save operation if notification fails
        logger.warning(f"⚠️ Failed to notify document update: {e}")

