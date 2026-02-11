"""
WebSocket Notification System for FileManager Service
Handles real-time updates for file operations
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class WebSocketNotifier:
    """Handles WebSocket notifications for file operations"""
    
    def __init__(self, websocket_manager=None):
        self.websocket_manager = websocket_manager
    
    async def notify_file_created(self, document_id: str, folder_id: str, user_id: Optional[str] = None, metadata: Dict[str, Any] = None):
        """Send notification when file is created"""
        try:
            if not self.websocket_manager:
                logger.debug("📡 WebSocket manager not available for file creation notification")
                return False
            
            # Send folder_update message to trigger frontend refresh
            message = {
                "type": "folder_update",
                "folder_id": folder_id,
                "action": "file_created",
                "document_id": document_id,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            if user_id:
                await self.websocket_manager.send_to_session(message, user_id)
                logger.info(f"📡 Sent folder update notification to user {user_id}: {document_id}")
            else:
                await self.websocket_manager.broadcast(message)
                logger.info(f"📡 Broadcasted folder update notification: {document_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send file creation notification: {e}")
            return False
    
    async def notify_file_processed(self, document_id: str, folder_id: str, user_id: Optional[str] = None, processing_info: Dict[str, Any] = None):
        """Send notification when file processing is completed"""
        try:
            if not self.websocket_manager:
                logger.debug("📡 WebSocket manager not available for file processing notification")
                return False
            
            # Send folder_update message to trigger frontend refresh
            message = {
                "type": "folder_update",
                "folder_id": folder_id,
                "action": "file_processed",
                "document_id": document_id,
                "timestamp": datetime.now().isoformat(),
                "processing_info": processing_info or {}
            }
            
            if user_id:
                await self.websocket_manager.send_to_session(message, user_id)
                logger.info(f"📡 Sent folder update notification to user {user_id}: {document_id}")
            else:
                await self.websocket_manager.broadcast(message)
                logger.info(f"📡 Broadcasted folder update notification: {document_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send file processing notification: {e}")
            return False
    
    async def notify_file_moved(self, document_id: str, old_folder_id: str, new_folder_id: str, user_id: Optional[str] = None):
        """Send notification when file is moved"""
        try:
            if not self.websocket_manager:
                logger.debug("📡 WebSocket manager not available for file move notification")
                return False
            
            message = {
                "type": "file_moved",
                "document_id": document_id,
                "old_folder_id": old_folder_id,
                "new_folder_id": new_folder_id,
                "timestamp": datetime.now().isoformat()
            }
            
            if user_id:
                await self.websocket_manager.send_to_session(message, user_id)
                logger.info(f"📡 Sent file move notification to user {user_id}: {document_id}")
            else:
                await self.websocket_manager.broadcast(message)
                logger.info(f"📡 Broadcasted file move notification: {document_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send file move notification: {e}")
            return False
    
    async def notify_file_deleted(self, document_id: str, folder_id: str, user_id: Optional[str] = None, items_deleted: int = 1):
        """Send notification when file is deleted"""
        try:
            if not self.websocket_manager:
                logger.debug("📡 WebSocket manager not available for file deletion notification")
                return False
            
            message = {
                "type": "file_deleted",
                "document_id": document_id,
                "folder_id": folder_id,
                "items_deleted": items_deleted,
                "timestamp": datetime.now().isoformat()
            }
            
            if user_id:
                await self.websocket_manager.send_to_session(message, user_id)
                logger.info(f"📡 Sent file deletion notification to user {user_id}: {document_id}")
            else:
                await self.websocket_manager.broadcast(message)
                logger.info(f"📡 Broadcasted file deletion notification: {document_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send file deletion notification: {e}")
            return False
    
    async def notify_folder_updated(
        self,
        folder_id: Optional[str] = None,
        user_id: Optional[str] = None,
        collection_type: Optional[str] = None
    ) -> bool:
        """Send notification that a folder/collection changed (e.g. after sidecar processing)."""
        try:
            if not self.websocket_manager:
                logger.debug("WebSocket manager not available for folder update notification")
                return False
            message = {
                "type": "folder_update",
                "action": "folder_updated",
                "folder_id": folder_id,
                "collection_type": collection_type or "user",
                "timestamp": datetime.now().isoformat()
            }
            if user_id:
                await self.websocket_manager.send_to_session(message, user_id)
            else:
                await self.websocket_manager.broadcast(message)
            return True
        except Exception as e:
            logger.error(f"Failed to send folder update notification: {e}")
            return False

    async def notify_folder_event(self, event_type: str, folder_data: dict, user_id: Optional[str] = None):
        """Send a single folder event notification with rich data"""
        try:
            if not self.websocket_manager:
                logger.debug("📡 WebSocket manager not available for folder event notification")
                return False
            
            message = {
                "type": "folder_event",
                "action": event_type,
                "folder": folder_data,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
            
            if user_id:
                await self.websocket_manager.send_to_session(message, user_id)
                logger.info(f"📡 Sent folder event notification to user {user_id}: {event_type} - {folder_data.get('folder_id', 'unknown')}")
            else:
                await self.websocket_manager.broadcast(message)
                logger.info(f"📡 Broadcasted folder event notification: {event_type} - {folder_data.get('folder_id', 'unknown')}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send folder event notification: {e}")
            return False
    
    # Single event system replaces multiple notification methods
    
    async def notify_processing_status_update(self, document_id: str, status: str, folder_id: str, user_id: Optional[str] = None, progress: Optional[float] = None):
        """Send notification when processing status changes"""
        try:
            if not self.websocket_manager:
                logger.debug("📡 WebSocket manager not available for processing status notification")
                return False
            
            message = {
                "type": "processing_status_update",
                "document_id": document_id,
                "status": status,
                "folder_id": folder_id,
                "progress": progress,
                "timestamp": datetime.now().isoformat()
            }
            
            if user_id:
                await self.websocket_manager.send_to_session(message, user_id)
                logger.info(f"📡 Sent processing status notification to user {user_id}: {document_id} -> {status}")
            else:
                await self.websocket_manager.broadcast(message)
                logger.info(f"📡 Broadcasted processing status notification: {document_id} -> {status}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send processing status notification: {e}")
            return False
    
    async def notify_error(self, document_id: str, error_message: str, user_id: Optional[str] = None):
        """Send notification when an error occurs"""
        try:
            if not self.websocket_manager:
                logger.debug("📡 WebSocket manager not available for error notification")
                return False
            
            message = {
                "type": "file_error",
                "document_id": document_id,
                "error_message": error_message,
                "timestamp": datetime.now().isoformat()
            }
            
            if user_id:
                await self.websocket_manager.send_to_session(message, user_id)
                logger.info(f"📡 Sent error notification to user {user_id}: {document_id}")
            else:
                await self.websocket_manager.broadcast(message)
                logger.info(f"📡 Broadcasted error notification: {document_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send error notification: {e}")
            return False
