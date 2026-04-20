"""
FileManager Service - Centralized file management for all agents and tools
"""

import logging
import hashlib
import re
from datetime import datetime
from typing import Optional, Dict, Any, List
import asyncio

from .models.file_placement_models import (
    FilePlacementRequest, FilePlacementResponse, SourceType,
    FileMoveRequest, FileMoveResponse,
    FileDeleteRequest, FileDeleteResponse,
    FileRenameRequest, FileRenameResponse,
    FolderStructureRequest, FolderStructureResponse
)
from .file_placement_strategies import FilePlacementStrategyFactory
from .websocket_notifier import WebSocketNotifier

from models.api_models import DocumentInfo, ProcessingStatus
from services.folder_service import FolderService
from utils.websocket_manager import get_websocket_manager

logger = logging.getLogger(__name__)


class FileManagerService:
    """Centralized file management service for all agents and tools"""
    
    def __init__(self):
        self.folder_service: Optional[FolderService] = None
        self.document_service = None
        self.websocket_notifier: Optional[WebSocketNotifier] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the FileManager service"""
        if self._initialized:
            return
        
        logger.info("🚀 Initializing FileManager Service...")
        
        # Lazy import to avoid circular dependency
        from services.service_container import service_container
        
        # Check if service container is initialized
        if service_container.is_initialized:
            # Get services from container
            self.document_service = service_container.document_service
            self.folder_service = service_container.folder_service
            logger.info("✅ Got services from initialized service container")
        else:
            # Service container not ready yet, initialize our own services
            logger.info("⚠️ Service container not ready, initializing own services")
            self.folder_service = FolderService()
            await self.folder_service.initialize()
            
            from repositories.document_repository import DocumentRepository
            from services.document_service_facade import DocumentServiceFacade
            from services.embedding_service_wrapper import get_embedding_service
            from utils.websocket_manager import get_websocket_manager

            repo = DocumentRepository()
            await repo.initialize()
            em = await get_embedding_service()
            _wm = get_websocket_manager()
            self.document_service = DocumentServiceFacade()
            await self.document_service.initialize(
                shared_document_repository=repo,
                shared_embedding_manager=em,
                shared_kg_service=None,
                websocket_manager=_wm,
                skip_incomplete_resume=True,
            )
            logger.info("Created document service facade for FileManager (pre-container bootstrap)")
        
        # Initialize WebSocket notifier
        websocket_manager = get_websocket_manager()
        self.websocket_notifier = WebSocketNotifier(websocket_manager)
        
        self._initialized = True
        logger.info("✅ FileManager Service initialized successfully")
    
    async def update_services_from_container(self):
        """Update services from service container when it becomes available"""
        if not self._initialized:
            return
        
        # Lazy import to avoid circular dependency
        from services.service_container import service_container
        
        if service_container.is_initialized:
            # Update services from container
            if service_container.document_service and self.document_service != service_container.document_service:
                # Close our own document service if we created one
                if hasattr(self.document_service, 'close'):
                    try:
                        await self.document_service.close()
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to close own document service: {e}")
                
                self.document_service = service_container.document_service
                logger.info("✅ Updated document service from container")
            
            if service_container.folder_service and self.folder_service != service_container.folder_service:
                self.folder_service = service_container.folder_service
                logger.info("✅ Updated folder service from container")
    
    async def place_file(self, request: FilePlacementRequest) -> FilePlacementResponse:
        """Place a file via document-service (same RPC as REST file-manager API)."""
        if not self._initialized:
            await self.initialize()
        if self.document_service is None:
            await self.update_services_from_container()

        logger.info("📁 Placing file: %s (source: %s)", request.title, request.source_type)
        from clients.document_service_client import get_document_service_client

        dsc = get_document_service_client()
        await dsc.initialize(required=True)
        uid = getattr(request, "user_id", None) or ""
        try:
            ok, data, err = await dsc.place_file_json(uid, request.model_dump(mode="json"))
            if not ok or not data:
                if self.websocket_notifier:
                    await self.websocket_notifier.notify_error(
                        "unknown", err or "place_file failed", request.user_id
                    )
                raise RuntimeError(err or "place_file failed")
            resp = FilePlacementResponse.model_validate(data)
            try:
                if self.websocket_notifier:
                    await self.websocket_notifier.notify_file_created(
                        resp.document_id,
                        resp.folder_id,
                        request.user_id,
                        metadata={
                            "source_type": request.source_type.value,
                            "filename": resp.filename,
                        },
                    )
            except Exception as ws_e:
                logger.debug("place_file websocket notify: %s", ws_e)
            return resp
        except Exception as e:
            logger.error("❌ Failed to place file: %s", e)
            if self.websocket_notifier:
                await self.websocket_notifier.notify_error(
                    "unknown", f"Failed to place file: {e}", request.user_id
                )
            raise

    async def move_file(self, request: FileMoveRequest) -> FileMoveResponse:
        """Move a file via document-service."""
        if not self._initialized:
            await self.initialize()
        logger.info(
            "📁 Moving file: %s to folder %s", request.document_id, request.new_folder_id
        )
        from clients.document_service_client import get_document_service_client

        dsc = get_document_service_client()
        await dsc.initialize(required=True)
        uid = getattr(request, "user_id", None) or ""
        try:
            ok, data, err = await dsc.move_file_json(uid, request.model_dump(mode="json"))
            if not ok or not data:
                if self.websocket_notifier:
                    await self.websocket_notifier.notify_error(
                        request.document_id, err or "move failed", request.user_id
                    )
                raise RuntimeError(err or "move_file failed")
            resp = FileMoveResponse.model_validate(data)
            try:
                if self.websocket_notifier:
                    await self.websocket_notifier.notify_file_moved(
                        request.document_id,
                        resp.old_folder_id,
                        request.new_folder_id,
                        request.user_id,
                    )
            except Exception as ws_e:
                logger.debug("move_file websocket: %s", ws_e)
            return resp
        except Exception as e:
            logger.error("❌ Failed to move file: %s", e)
            if self.websocket_notifier:
                await self.websocket_notifier.notify_error(
                    request.document_id, str(e), request.user_id
                )
            raise

    async def delete_file(self, request: FileDeleteRequest) -> FileDeleteResponse:
        """Delete a file via document-service."""
        if not self._initialized:
            await self.initialize()
        logger.info("🗑️ Deleting file: %s", request.document_id)
        from clients.document_service_client import get_document_service_client

        dsc = get_document_service_client()
        await dsc.initialize(required=True)
        uid = getattr(request, "user_id", None) or ""
        try:
            ok, data, err = await dsc.delete_file_json(uid, request.model_dump(mode="json"))
            if not ok or not data:
                if self.websocket_notifier:
                    await self.websocket_notifier.notify_error(
                        request.document_id, err or "delete failed", request.user_id
                    )
                raise RuntimeError(err or "delete_file failed")
            resp = FileDeleteResponse.model_validate(data)
            try:
                if self.websocket_notifier:
                    await self.websocket_notifier.notify_file_deleted(
                        request.document_id,
                        resp.folder_id,
                        request.user_id,
                        resp.items_deleted or 1,
                    )
            except Exception as ws_e:
                logger.debug("delete_file websocket: %s", ws_e)
            return resp
        except Exception as e:
            logger.error("❌ Failed to delete file: %s", e)
            if self.websocket_notifier:
                await self.websocket_notifier.notify_error(
                    request.document_id, str(e), request.user_id
                )
            raise

    async def rename_file(self, request: FileRenameRequest) -> FileRenameResponse:
        """Rename a file via document-service."""
        if not self._initialized:
            await self.initialize()
        logger.info("✏️ Renaming file: %s to %s", request.document_id, request.new_filename)
        from clients.document_service_client import get_document_service_client

        dsc = get_document_service_client()
        await dsc.initialize(required=True)
        uid = getattr(request, "user_id", None) or ""
        try:
            ok, data, err = await dsc.rename_file_json(uid, request.model_dump(mode="json"))
            if not ok or not data:
                if self.websocket_notifier:
                    await self.websocket_notifier.notify_error(
                        request.document_id, err or "rename failed", request.user_id
                    )
                raise RuntimeError(err or "rename_file failed")
            resp = FileRenameResponse.model_validate(data)
            try:
                if self.websocket_notifier:
                    await self.websocket_notifier.notify_document_status_update(
                        request.document_id,
                        "renamed",
                        resp.folder_id,
                        request.user_id,
                    )
            except Exception as ws_e:
                logger.debug("rename_file websocket: %s", ws_e)
            return resp
        except Exception as e:
            logger.error("❌ Failed to rename file: %s", e)
            if self.websocket_notifier:
                await self.websocket_notifier.notify_error(
                    request.document_id, str(e), request.user_id
                )
            raise

    async def create_folder_structure(self, request: FolderStructureRequest) -> FolderStructureResponse:
        """Create a folder structure"""
        if not self._initialized:
            await self.initialize()
        
        logger.info(f"📁 Creating folder structure: {'/'.join(request.folder_path)} (parent: {request.parent_folder_id})")
        
        try:
            # Create folder structure
            folder_id = await self._ensure_folder_structure(
                request.folder_path, 
                request.user_id, 
                request.collection_type,
                request.current_user_role,
                request.admin_user_id,
                request.parent_folder_id
            )
            
            # Send single WebSocket event notification
            folder_data = {
                "folder_id": folder_id,
                "name": request.folder_path[-1] if request.folder_path else "Unknown",
                "parent_folder_id": request.parent_folder_id,
                "user_id": request.user_id,
                "collection_type": request.collection_type,
                "created_at": datetime.now().isoformat()
            }
            websocket_sent = await self.websocket_notifier.notify_folder_event(
                "created", folder_data, request.user_id
            )
            
            response = FolderStructureResponse(
                folder_id=folder_id,
                folder_path=request.folder_path,
                parent_folder_id=request.parent_folder_id,
                creation_timestamp=datetime.now(),
                websocket_notification_sent=websocket_sent
            )
            
            logger.info(f"✅ Folder structure created successfully: {folder_id}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Failed to create folder structure: {e}")
            raise
    
    async def _ensure_folder_structure(self, folder_path: List[str], user_id: Optional[str], collection_type: str, current_user_role: str = "user", admin_user_id: str = None, parent_folder_id: Optional[str] = None) -> str:
        """Ensure folder structure exists and return the final folder ID"""
        if not folder_path:
            raise ValueError("Folder path cannot be empty")
        
        current_parent_id = parent_folder_id  # Start from the specified parent, not None
        created_folders = []  # Track folders for notifications
        
        for folder_name in folder_path:
            # Create or get folder
            folder_id = await self.folder_service.create_or_get_folder(
                folder_name, 
                parent_folder_id=current_parent_id,
                user_id=user_id,
                collection_type=collection_type,
                current_user_role=current_user_role,
                admin_user_id=admin_user_id
            )
            
            # Track folder for WebSocket notification
            # Note: We send notifications for all folders to ensure UI consistency
            # The frontend can handle duplicate notifications gracefully
            created_folders.append({
                "folder_id": folder_id,
                "name": folder_name,
                "parent_folder_id": current_parent_id,
                "user_id": user_id,
                "collection_type": collection_type,
                "created_at": datetime.now().isoformat()
            })
            
            current_parent_id = folder_id
        
        # Send WebSocket notifications for all folders in the path
        # This ensures the UI updates properly for RSS imports
        for folder_data in created_folders:
            try:
                await self.websocket_notifier.notify_folder_event(
                    "created", folder_data, user_id
                )
                logger.info(f"📡 Sent folder creation notification: {folder_data['name']} ({folder_data['folder_id']})")
            except Exception as e:
                logger.warning(f"⚠️ Failed to send folder creation notification for {folder_data['name']}: {e}")
        
        return current_parent_id
    
    def _generate_document_id(self, request: FilePlacementRequest) -> str:
        """Generate a unique document ID"""
        # Use content hash + timestamp for uniqueness
        if getattr(request, "content_bytes", None):
            content_hash = hashlib.md5(request.content_bytes).hexdigest()[:8]
        else:
            content_hash = hashlib.md5(request.content.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{content_hash}_{timestamp}"
    
    async def _start_processing(self, document_id: str, priority: int) -> Optional[str]:
        """Start document processing"""
        try:
            # Update status to processing
            await self.document_service.document_repository.update_status(document_id, ProcessingStatus.PROCESSING)
            
            # Send WebSocket notification
            await self.websocket_notifier.notify_processing_status_update(
                document_id, "processing", "unknown", None, 0.0
            )
            
            # Queue processing task (this would integrate with your existing processing system)
            # For now, we'll just return None as the task ID
            logger.info(f"🔄 Started processing for document: {document_id}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to start processing: {e}")
            return None

    async def place_file_concurrent(self, requests: List[FilePlacementRequest]) -> List[FilePlacementResponse]:
        """Place multiple files concurrently with optimal resource management"""
        if not self._initialized:
            await self.initialize()
        
        logger.info(f"📁 Placing {len(requests)} files concurrently")
        
        # Use semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(8)  # Limit to 8 concurrent file operations
        
        async def place_single_file(request: FilePlacementRequest) -> FilePlacementResponse:
            async with semaphore:
                try:
                    return await self.place_file(request)
                except Exception as e:
                    logger.error(f"❌ Failed to place file {request.title}: {e}")
                    # Return error response instead of raising
                    return FilePlacementResponse(
                        document_id="",
                        folder_id="",
                        filename=request.filename or "unknown",
                        processing_status=ProcessingStatus.FAILED,
                        placement_timestamp=datetime.now(),
                        websocket_notification_sent=False,
                        processing_task_id=None,
                        error=str(e)
                    )
        
        # Process all files concurrently
        tasks = [place_single_file(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error responses
        responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ Exception placing file {requests[i].title}: {result}")
                responses.append(FilePlacementResponse(
                    document_id="",
                    folder_id="",
                    filename=requests[i].filename or "unknown",
                    processing_status=ProcessingStatus.FAILED,
                    placement_timestamp=datetime.now(),
                    websocket_notification_sent=False,
                    processing_task_id=None,
                    error=str(result)
                ))
            else:
                responses.append(result)
        
        logger.info(f"✅ Completed concurrent placement of {len(requests)} files")
        return responses


# Global instance
_file_manager_instance = None


async def get_file_manager() -> FileManagerService:
    """Get the global FileManager instance"""
    global _file_manager_instance
    
    if _file_manager_instance is None:
        _file_manager_instance = FileManagerService()
        await _file_manager_instance.initialize()
    
    return _file_manager_instance