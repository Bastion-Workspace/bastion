"""
FileManager Service - Centralized file management for all agents and tools
"""

import logging
import hashlib
import re
import shutil
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
            
            # Initialize the most robust document service for parallel processing
            from services.parallel_document_service import ParallelDocumentService
            from utils.websocket_manager import get_websocket_manager
            
            self.document_service = ParallelDocumentService()
            
            # Set websocket manager before initialization to avoid AttributeError
            self.document_service.websocket_manager = get_websocket_manager()
            
            # Configure for optimal parallel processing
            from utils.parallel_document_processor import ProcessingConfig, ProcessingStrategy
            
            # Optimized processing config for concurrent file operations
            processing_config = ProcessingConfig(
                max_concurrent_documents=8,  # Handle multiple simultaneous file operations
                max_concurrent_chunks=16,    # Parallel chunk processing
                max_concurrent_embeddings=12, # Parallel embedding generation
                strategy=ProcessingStrategy.HYBRID,
                enable_document_level_parallelism=True,
                enable_chunk_level_parallelism=True,
                enable_io_parallelism=True,
                thread_pool_size=8,
                process_pool_size=2
            )
            
            # Note: Embedding configuration now handled by EmbeddingServiceWrapper
            # via USE_VECTOR_SERVICE flag in config
            
            await self.document_service.initialize(
                enable_parallel=True,
                processing_config=processing_config
            )
            logger.info("✅ Created optimized parallel document service for FileManager")
        
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
        """Place a file in the appropriate folder structure"""
        if not self._initialized:
            await self.initialize()
        
        # Try to update services from container if document service is not available
        if self.document_service is None:
            await self.update_services_from_container()
        
        logger.info(f"📁 Placing file: {request.title} (source: {request.source_type})")
        
        try:
            # Get placement strategy
            strategy = FilePlacementStrategyFactory.get_strategy(request.source_type)
            
            # Determine final folder
            if request.target_folder_id:
                folder_id = request.target_folder_id
            else:
                # Determine folder path
                folder_path = request.folder_path or strategy.get_folder_path(request)
                # Create folder structure if needed
                folder_id = await self._ensure_folder_structure(
                    folder_path, 
                    request.user_id, 
                    request.collection_type,
                    request.current_user_role,
                    request.admin_user_id
                )
            
            # Generate filename
            filename = request.filename or strategy.get_filename(request)
            # Respect user-provided extension for manual creation; validate allowed types and avoid double-ext
            try:
                from pathlib import Path
                from models.api_models import DocumentType
                if request.source_type == SourceType.MANUAL and filename:
                    suffix = Path(filename).suffix.lower()
                    if suffix:
                        if suffix not in [".md", ".org"]:
                            raise ValueError("Unsupported extension. Use .md or .org for manual files")
                        # Remove accidental duplicated extension like .md.md
                        for e in [".md", ".org"]:
                            if filename.lower().endswith(e + e):
                                filename = filename[: -len(e)]
                    else:
                        # No extension provided: infer from doc_type or default to .md
                        doc_type_name = str(getattr(request.doc_type, "name", request.doc_type)).upper()
                        desired = (".md" if doc_type_name == "MD" else 
                                   ".org" if doc_type_name == "ORG" else ".md")
                        filename = f"{filename}{desired}"
            except Exception as ve:
                # Re-raise validation errors; log others and continue
                if isinstance(ve, ValueError):
                    raise
                logger.warning(f"⚠️ Filename normalization warning: {ve}")
            
            # Generate document ID
            document_id = self._generate_document_id(request)
            
            # **ROOSEVELT FOLDER INHERITANCE AND COLLECTION TYPE**: Apply folder metadata and determine collection_type from folder
            final_tags = strategy.get_tags(request)
            final_category = request.category.value if request.category else "other"
            final_collection_type = request.collection_type  # Default to request's collection type
            final_user_id = request.user_id  # Default to request's user_id
            
            if folder_id:
                try:
                    # Get full folder details to determine collection_type
                    folder_details = await self.folder_service.get_folder(folder_id)
                    if folder_details:
                        # **ROOSEVELT FIX**: Documents inherit collection_type AND user_id from their folder!
                        folder_collection_type = folder_details.collection_type
                        if folder_collection_type:
                            final_collection_type = folder_collection_type
                            # Global folders → global documents (user_id=None)
                            if folder_collection_type == 'global':
                                final_user_id = None
                                logger.info(f"📋 COLLECTION TYPE INHERITANCE: Folder {folder_id} is GLOBAL, document will be global with user_id=None")
                            else:
                                logger.info(f"📋 COLLECTION TYPE INHERITANCE: Folder {folder_id} is {folder_collection_type}, document will be {final_collection_type}")
                    
                    folder_metadata = await self.folder_service.get_folder_metadata(folder_id)
                    
                    if folder_metadata.get('inherit_tags', True):
                        folder_category = folder_metadata.get('category')
                        folder_tags = folder_metadata.get('tags', [])
                        
                        # Apply folder category if document doesn't have one
                        if not request.category and folder_category:
                            final_category = folder_category
                        
                        # Merge folder tags with request tags
                        if folder_tags:
                            merged_tags = list(set(final_tags + folder_tags))
                            final_tags = merged_tags
                            logger.info(f"📋 FOLDER INHERITANCE (place_file): Merged tags - folder={folder_tags}, final={final_tags}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to apply folder metadata inheritance in place_file: {e}")
            
            # Create document metadata
            document_metadata = {
                "title": request.title,
                "category": final_category,
                "tags": final_tags,
                "description": strategy.get_description(request),
                "author": request.author,
                "language": request.language,
                "source_type": request.source_type.value,
                "source_metadata": request.source_metadata
            }
            
            # Save as physical file on disk for RSS, web sources, and manual creation
            markdown_file_path = None
            if request.source_type in [SourceType.RSS, SourceType.WEB_SCRAPING, SourceType.MANUAL]:
                markdown_file_path = await self._save_as_markdown_file(
                    request, document_id, filename, folder_id, document_metadata
                )
                logger.info(f"📝 Saved content as file: {markdown_file_path}")
            
            # Store document using document service
            if self.document_service is None:
                logger.error("❌ Document service is None - cannot create document")
                raise Exception("Document service not initialized")
            
            logger.info(f"📄 Creating document with service: {type(self.document_service).__name__}")
            success = await self.document_service.store_text_document(
                document_id, request.content, document_metadata, filename,
                user_id=final_user_id, collection_type=final_collection_type, folder_id=folder_id,
                file_path=markdown_file_path  # Pass file path for disk storage
            )
            
            if not success:
                raise Exception("Failed to store document via document service")
            
            # Send WebSocket notification
            websocket_sent = await self.websocket_notifier.notify_file_created(
                document_id, folder_id, final_user_id, 
                metadata={"source_type": request.source_type.value, "filename": filename}
            )
            
            # Start processing if requested
            processing_task_id = None
            if request.process_immediately:
                processing_task_id = await self._start_processing(document_id, request.priority)
            
            response = FilePlacementResponse(
                document_id=document_id,
                folder_id=folder_id,
                filename=filename,
                processing_status=ProcessingStatus.UPLOADING,
                placement_timestamp=datetime.now(),
                websocket_notification_sent=websocket_sent,
                processing_task_id=processing_task_id
            )
            
            logger.info(f"✅ File placed successfully: {document_id} in folder {folder_id}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Failed to place file: {e}")
            # Send error notification
            await self.websocket_notifier.notify_error(
                "unknown", f"Failed to place file: {str(e)}", request.user_id
            )
            raise
    
    async def move_file(self, request: FileMoveRequest) -> FileMoveResponse:
        """Move a file to a different folder"""
        if not self._initialized:
            await self.initialize()
        
        logger.info(f"📁 Moving file: {request.document_id} to folder {request.new_folder_id}")
        
        try:
            from pathlib import Path
            from config import settings
            
            # Get current document info
            current_doc = await self.document_service.document_repository.get_by_id(request.document_id)
            if not current_doc:
                raise ValueError(f"Document not found: {request.document_id}")
            
            old_folder_id = current_doc.folder_id
            filename = current_doc.filename
            user_id = current_doc.user_id
            collection_type = current_doc.collection_type if hasattr(current_doc, 'collection_type') else ("global" if not user_id else "user")
            
            # Find the current file on disk
            old_file_path = None
            
            # Try new folder structure first
            try:
                if old_folder_id:
                    old_file_path_str = await self.folder_service.get_document_file_path(
                        filename=filename,
                        folder_id=old_folder_id,
                        user_id=user_id,
                        collection_type=collection_type
                    )
                    old_file_path = Path(old_file_path_str)
                    
                    if not old_file_path.exists():
                        # Try with document_id prefix (legacy style)
                        filename_with_id = f"{request.document_id}_{filename}"
                        old_file_path_str = await self.folder_service.get_document_file_path(
                            filename=filename_with_id,
                            folder_id=old_folder_id,
                            user_id=user_id,
                            collection_type=collection_type
                        )
                        old_file_path = Path(old_file_path_str)
            except Exception as e:
                logger.warning(f"⚠️ Failed to find file in new structure: {e}")
            
            # Fall back to legacy flat structure if not found
            if not old_file_path or not old_file_path.exists():
                upload_dir = Path(settings.UPLOAD_DIR)
                legacy_paths = [
                    upload_dir / f"{request.document_id}_{filename}",
                    upload_dir / filename
                ]
                
                for legacy_path in legacy_paths:
                    if legacy_path.exists():
                        old_file_path = legacy_path
                        logger.info(f"📄 Found file in legacy location: {old_file_path}")
                        break
            
            # Get new folder's physical path
            # Pass user_id and role for RLS context (especially needed for team folders)
            new_folder_path = await self.folder_service.get_folder_physical_path(
                request.new_folder_id,
                user_id=request.user_id,
                user_role=request.current_user_role
            )
            if not new_folder_path:
                raise ValueError(f"Failed to get physical path for folder: {request.new_folder_id}")
            
            # Ensure new folder directory exists
            await self.folder_service._create_physical_directory(new_folder_path)
            
            # Move file on disk if it exists
            if old_file_path and old_file_path.exists():
                new_file_path = new_folder_path / filename
                
                # Handle case where file already exists at destination
                if new_file_path.exists() and new_file_path != old_file_path:
                    logger.warning(f"⚠️ File already exists at destination: {new_file_path}")
                    # Generate unique filename
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    name_parts = new_file_path.stem, new_file_path.suffix
                    new_file_path = new_folder_path / f"{name_parts[0]}_{timestamp}{name_parts[1]}"
                    logger.info(f"📝 Using unique filename: {new_file_path.name}")
                
                # Move the file
                shutil.move(str(old_file_path), str(new_file_path))
                logger.info(f"📦 Moved file on disk: {old_file_path} -> {new_file_path}")
            else:
                logger.warning(f"⚠️ File not found on disk: {old_file_path} (database will be updated anyway)")
            
            # Get destination folder info to determine collection_type and team_id
            dest_folder = await self.folder_service.get_folder(
                request.new_folder_id,
                user_id=request.user_id,
                current_user_role=request.current_user_role
            )
            if not dest_folder:
                raise ValueError(f"Destination folder not found: {request.new_folder_id}")
            
            # Prepare update fields
            update_fields = {"folder_id": request.new_folder_id}
            
            # Update collection_type, team_id, and user_id based on destination folder
            if dest_folder.collection_type == "team" and dest_folder.team_id:
                # Moving to team folder: set collection_type='team', team_id, and user_id=NULL
                update_fields["collection_type"] = "team"
                update_fields["team_id"] = str(dest_folder.team_id)
                update_fields["user_id"] = None
                logger.info(f"📦 Moving document to team folder - setting collection_type='team', team_id={dest_folder.team_id}, user_id=NULL")
            elif dest_folder.collection_type == "user" and dest_folder.user_id:
                # Moving to user folder: set collection_type='user', user_id, team_id=NULL
                update_fields["collection_type"] = "user"
                update_fields["user_id"] = str(dest_folder.user_id)
                update_fields["team_id"] = None
                logger.info(f"📦 Moving document to user folder - setting collection_type='user', user_id={dest_folder.user_id}")
            elif dest_folder.collection_type == "global":
                # Moving to global folder: set collection_type='global', user_id=NULL, team_id=NULL
                update_fields["collection_type"] = "global"
                update_fields["user_id"] = None
                update_fields["team_id"] = None
                logger.info(f"📦 Moving document to global folder - setting collection_type='global'")
            
            # Update document in database with all fields
            # Note: RLS context will be auto-detected by the repository using admin context
            success = await self.document_service.document_repository.update(
                request.document_id,
                **update_fields
            )
            if not success:
                raise ValueError(f"Failed to update document: {request.document_id}")
            
            # **ROOSEVELT FOLDER INHERITANCE**: Apply folder metadata if folder has metadata
            # This ensures documents inherit category/tags from their new folder and vector DB is updated
            if request.new_folder_id:
                try:
                    folder_metadata = await self.folder_service.get_folder_metadata(request.new_folder_id)
                    
                    if folder_metadata.get('inherit_tags', True):
                        folder_category = folder_metadata.get('category')
                        folder_tags = folder_metadata.get('tags', [])
                        
                        if folder_category or folder_tags:
                            from models.api_models import DocumentUpdateRequest, DocumentCategory
                            
                            # Parse category enum
                            doc_category = None
                            if folder_category:
                                try:
                                    doc_category = DocumentCategory(folder_category)
                                except ValueError:
                                    logger.warning(f"⚠️ Invalid folder category '{folder_category}'")
                            
                            # Update document with folder metadata
                            # This will update both PostgreSQL and Qdrant vector database
                            update_request = DocumentUpdateRequest(
                                category=doc_category,
                                tags=folder_tags if folder_tags else None
                            )
                            await self.document_service.update_document_metadata(request.document_id, update_request)
                            logger.info(f"📋 FOLDER INHERITANCE: Applied folder metadata to moved document {request.document_id} - category={folder_category}, tags={folder_tags}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to apply folder metadata inheritance during move: {e}")
                    # Don't fail the move operation if metadata inheritance fails
            
            # Send WebSocket notification
            websocket_sent = await self.websocket_notifier.notify_file_moved(
                request.document_id, old_folder_id, request.new_folder_id, request.user_id
            )
            
            response = FileMoveResponse(
                document_id=request.document_id,
                old_folder_id=old_folder_id,
                new_folder_id=request.new_folder_id,
                move_timestamp=datetime.now(),
                websocket_notification_sent=websocket_sent
            )
            
            logger.info(f"✅ File moved successfully: {request.document_id}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Failed to move file: {e}")
            await self.websocket_notifier.notify_error(
                request.document_id, f"Failed to move file: {str(e)}", request.user_id
            )
            raise
    
    async def delete_file(self, request: FileDeleteRequest) -> FileDeleteResponse:
        """Delete a file or folder"""
        if not self._initialized:
            await self.initialize()
        
        logger.info(f"🗑️ Deleting file: {request.document_id}")
        
        try:
            # Get document info
            doc = await self.document_service.document_repository.get_by_id(request.document_id)
            if not doc:
                raise ValueError(f"Document not found: {request.document_id}")
            
            folder_id = doc.folder_id
            items_deleted = 1
            
            # Delete document with proper user context
            success = await self.document_service.document_repository.delete(request.document_id, doc.user_id)
            if not success:
                raise ValueError(f"Failed to delete document: {request.document_id}")
            
            # Send WebSocket notification
            websocket_sent = await self.websocket_notifier.notify_file_deleted(
                request.document_id, folder_id, request.user_id, items_deleted
            )
            
            response = FileDeleteResponse(
                document_id=request.document_id,
                folder_id=folder_id,
                delete_timestamp=datetime.now(),
                websocket_notification_sent=websocket_sent,
                items_deleted=items_deleted
            )
            
            logger.info(f"✅ File deleted successfully: {request.document_id}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Failed to delete file: {e}")
            await self.websocket_notifier.notify_error(
                request.document_id, f"Failed to delete file: {str(e)}", request.user_id
            )
            raise

    async def rename_file(self, request: FileRenameRequest) -> FileRenameResponse:
        """Rename a file (updates filename/title and renames disk file if present)"""
        if not self._initialized:
            await self.initialize()
        logger.info(f"✏️ Renaming file: {request.document_id} to {request.new_filename}")
        try:
            from pathlib import Path
            from config import settings

            # Get current document info
            doc = await self.document_service.document_repository.get_by_id(request.document_id)
            if not doc:
                raise ValueError(f"Document not found: {request.document_id}")

            old_filename = doc.filename
            folder_id = doc.folder_id

            desired = request.new_filename.strip()
            if not desired:
                raise ValueError("New filename cannot be empty")

            # Preserve/normalize extension
            ext_map = { 'MD': '.md', 'ORG': '.org', 'HTML': '.html', 'TXT': '.txt' }
            doc_type_name = getattr(doc.doc_type, 'name', str(doc.doc_type)).upper()
            default_ext = ext_map.get(doc_type_name, Path(desired).suffix or '.txt')
            base = Path(desired).stem if Path(desired).suffix else desired
            if not base:
                base = Path(old_filename).stem
            new_filename = base + (Path(desired).suffix or default_ext)
            for e in ['.md', '.org', '.html', '.htm', '.txt']:
                if new_filename.lower().endswith(e + e):
                    new_filename = new_filename[:-len(e)]

            # Rename on disk using folder structure
            try:
                # Get the actual file path using folder service
                old_file_path = await self.folder_service.get_document_file_path(
                    filename=old_filename,
                    folder_id=folder_id,
                    user_id=doc.user_id,
                    collection_type=doc.collection_type
                )
                
                # Get the new file path with the new filename
                new_file_path = await self.folder_service.get_document_file_path(
                    filename=new_filename,
                    folder_id=folder_id,
                    user_id=doc.user_id,
                    collection_type=doc.collection_type
                )
                
                # Rename the file if paths are different and old file exists
                if old_file_path.exists() and old_file_path != new_file_path:
                    # Ensure parent directory exists for new path
                    new_file_path.parent.mkdir(parents=True, exist_ok=True)
                    old_file_path.rename(new_file_path)
                    logger.info(f"✅ Renamed file on disk: {old_file_path} -> {new_file_path}")
                elif not old_file_path.exists():
                    logger.warning(f"⚠️ Old file not found at: {old_file_path}")
                elif old_file_path == new_file_path:
                    logger.debug(f"File path unchanged (only case change?): {old_file_path}")
            except Exception as re:
                logger.warning(f"⚠️ Disk rename warning: {re}")
                import traceback
                logger.debug(traceback.format_exc())

            # Update DB
            updated = await self.document_service.document_repository.update(
                request.document_id,
                filename=new_filename,
                title=new_filename
            )
            if not updated:
                raise ValueError("Failed to update document metadata")

            # Update vector chunks with new filename metadata
            try:
                await self.document_service.embedding_manager.update_document_metadata_in_vectors(
                    document_id=request.document_id,
                    filename=new_filename,
                    title=new_filename
                )
                logger.info(f"✅ Updated vector chunk metadata for renamed file: {new_filename}")
            except Exception as vector_error:
                logger.warning(f"⚠️ Failed to update vector metadata (non-critical): {vector_error}")

            # Notify
            try:
                websocket_sent = await self.websocket_notifier.notify_document_status_update(
                    request.document_id, "renamed", folder_id, request.user_id
                )
            except Exception:
                websocket_sent = False

            return FileRenameResponse(
                document_id=request.document_id,
                old_filename=old_filename,
                new_filename=new_filename,
                folder_id=folder_id,
                rename_timestamp=datetime.now(),
                websocket_notification_sent=bool(websocket_sent)
            )
        except Exception as e:
            logger.error(f"❌ Failed to rename file: {e}")
            await self.websocket_notifier.notify_error(
                request.document_id, f"Failed to rename file: {str(e)}", request.user_id
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
    
    async def _save_as_markdown_file(
        self, 
        request: FilePlacementRequest, 
        document_id: str, 
        filename: str, 
        folder_id: str,
        document_metadata: Dict[str, Any]
    ) -> str:
        """
        Save content as file on disk
        
        Handles manual creation, RSS, and web scraping with proper folder structure
        """
        try:
            from services.markdown_formatter_service import get_markdown_formatter
            from pathlib import Path
            from config import settings
            
            # Get markdown formatter
            formatter = await get_markdown_formatter()
            
            # Determine file path structure
            upload_dir = Path(settings.UPLOAD_DIR)
            
            if request.source_type == SourceType.MANUAL:
                # Manual files use the new folder structure: uploads/Users/{username}/{folder}/
                # No need for ID prefix - folder isolation provides uniqueness
                file_path_str = await self.folder_service.get_document_file_path(
                    filename=filename,
                    folder_id=folder_id,
                    user_id=request.user_id,
                    collection_type=request.collection_type
                )
                file_path = Path(file_path_str)
                
                # Write content directly (no markdown formatting for manual files)
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(request.content)
                
                logger.info(f"📝 Saved manual file: {file_path} ({len(request.content)} characters)")
                return str(file_path)
                
            elif request.source_type == SourceType.RSS:
                # RSS articles go in web_sources/rss_articles/{feed_name}/
                feed_name = request.source_metadata.get('feed_name', 'unknown_feed')
                safe_feed_name = re.sub(r'[^\w\s-]', '', feed_name.lower())
                safe_feed_name = re.sub(r'[-\s]+', '_', safe_feed_name)
                
                file_dir = upload_dir / "web_sources" / "rss_articles" / safe_feed_name
            else:
                # Web scraped content goes in web_sources/scraped_content/{domain}/
                source_url = request.source_metadata.get('article_url', '')
                domain = formatter._extract_domain(source_url) if source_url else 'unknown'
                
                file_dir = upload_dir / "web_sources" / "scraped_content" / domain
            
            # Create directory if it doesn't exist
            file_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate markdown filename
            if request.source_type == SourceType.RSS:
                markdown_filename = formatter.generate_filename(
                    document_id, request.title, "rss", 
                    request.source_metadata.get('feed_name')
                )
            else:
                markdown_filename = formatter.generate_filename(
                    document_id, request.title, "web"
                )
            
            file_path = file_dir / markdown_filename
            
            # Format content as Markdown
            if request.source_type == SourceType.RSS:
                markdown_content = formatter.format_rss_article(
                    content=request.content,
                    metadata=document_metadata,
                    document_id=document_id,
                    title=request.title,
                    source_url=request.source_metadata.get('article_url'),
                    feed_name=request.source_metadata.get('feed_name'),
                    author=request.author,
                    published_date=datetime.fromisoformat(request.source_metadata.get('published_date')) if request.source_metadata.get('published_date') else None,
                    folder_id=folder_id,
                    images=request.source_metadata.get('images', [])
                )
            else:
                markdown_content = formatter.format_web_content(
                    content=request.content,
                    metadata=document_metadata,
                    document_id=document_id,
                    title=request.title,
                    source_url=request.source_metadata.get('source_url', ''),
                    folder_id=folder_id,
                    images=request.source_metadata.get('images', [])
                )
            
            # Write markdown file to disk
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            logger.info(f"✅ Saved Markdown file: {file_path} ({len(markdown_content)} characters)")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to save Markdown file: {e}")
            raise


# Global instance
_file_manager_instance = None


async def get_file_manager() -> FileManagerService:
    """Get the global FileManager instance"""
    global _file_manager_instance
    
    if _file_manager_instance is None:
        _file_manager_instance = FileManagerService()
        await _file_manager_instance.initialize()
    
    return _file_manager_instance
