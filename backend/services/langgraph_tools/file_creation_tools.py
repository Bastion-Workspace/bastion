"""
File Creation Tools for Agents
Allows agents to create files and folders in user's My Documents section
"""

import logging
from typing import Dict, Any, Optional
from uuid import uuid4
from datetime import datetime

logger = logging.getLogger(__name__)


async def create_user_file(
    filename: str,
    content: str,
    folder_id: Optional[str] = None,
    folder_path: Optional[str] = None,
    title: Optional[str] = None,
    tags: Optional[list] = None,
    category: Optional[str] = None,
    user_id: str = "system"
) -> Dict[str, Any]:
    """
    Create a file in the user's My Documents section
    
    **SECURITY**: Only creates files in user's collection (collection_type='user')
    Agents cannot create files in global collection
    
    Args:
        filename: Name of the file to create (e.g., "sensor_spec.md", "circuit_diagram.txt")
        content: File content as string
        folder_id: Optional folder ID to place file in (must be user's folder)
        folder_path: Optional folder path (e.g., "Projects/Electronics") - will create if needed
        title: Optional document title (defaults to filename)
        tags: Optional list of tags for the document
        category: Optional category for the document
        user_id: User ID (required - must match the user making the request)
    
    Returns:
        Dict with document_id, filename, folder_id, and status
    """
    try:
        logger.info(f"📝 Creating user file: {filename} for user {user_id}")
        
        # Import services
        from services.service_container import get_service_container
        from services.file_manager.file_manager_service import FileManagerService
        from services.file_manager.models.file_placement_models import (
            FilePlacementRequest, SourceType
        )
        
        # Get service container
        container = await get_service_container()
        
        # Get file manager - use get_file_manager if not in container yet
        if container.file_manager:
            file_manager: FileManagerService = container.file_manager
        else:
            # Fallback: get file manager directly
            from services.file_manager import get_file_manager
            file_manager = await get_file_manager()
        
        # Ensure file manager is initialized
        if not file_manager._initialized:
            await file_manager.initialize()
        
        # Determine folder_id if folder_path provided
        if folder_path and not folder_id:
            # Parse folder path and create/get folders
            folder_service = container.folder_service
            path_parts = [p.strip() for p in folder_path.split("/") if p.strip()]
            
            current_parent_id = None
            for folder_name in path_parts:
                # Get or create folder
                folder = await folder_service.get_or_create_subfolder(
                    parent_folder_id=current_parent_id,
                    folder_name=folder_name,
                    user_id=user_id,
                    collection_type="user",  # Always user collection
                    current_user_role="user",
                    admin_user_id=user_id
                )
                current_parent_id = folder.folder_id
            
            folder_id = current_parent_id
            logger.info(f"📁 Created/resolved folder path '{folder_path}' → folder_id: {folder_id}")
        
        # Prepare metadata
        metadata = {
            "title": title or filename,
            "tags": tags or [],
            "category": category or "other",
            "description": f"File created by agent for user {user_id}",
            "author": "agent",
            "source_type": "agent_created"
        }
        
        # Determine document type from filename
        from models.api_models import DocumentType, DocumentCategory
        
        doc_type = DocumentType.TXT
        if filename.lower().endswith('.md'):
            doc_type = DocumentType.MD
        elif filename.lower().endswith('.org'):
            doc_type = DocumentType.ORG
        elif filename.lower().endswith('.html') or filename.lower().endswith('.htm'):
            doc_type = DocumentType.HTML
        
        # Map category
        doc_category = DocumentCategory.OTHER
        if category:
            try:
                doc_category = DocumentCategory(category.lower())
            except ValueError:
                doc_category = DocumentCategory.OTHER
        
        # Create file placement request
        request = FilePlacementRequest(
            content=content,
            title=title or filename,
            filename=filename,
            source_type=SourceType.AGENT_CREATED,
            source_metadata=metadata,
            doc_type=doc_type,
            category=doc_category,
            tags=tags or [],
            description=metadata.get("description"),
            author=metadata.get("author"),
            user_id=user_id,
            collection_type="user",  # Always user collection - security restriction
            target_folder_id=folder_id,
            folder_path=None,  # Already resolved above if needed
            current_user_role="user",
            admin_user_id=user_id,
            process_immediately=True
        )
        
        # Place the file
        response = await file_manager.place_file(request)
        
        logger.info(f"✅ Created user file: {response.filename} (document_id: {response.document_id})")
        
        return {
            "success": True,
            "document_id": response.document_id,
            "filename": response.filename,
            "folder_id": response.folder_id,
            "status": "created",
            "message": f"File '{filename}' created successfully in user's My Documents"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to create user file '{filename}': {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to create file '{filename}': {str(e)}"
        }


async def create_user_folder(
    folder_name: str,
    parent_folder_id: Optional[str] = None,
    parent_folder_path: Optional[str] = None,
    user_id: str = "system"
) -> Dict[str, Any]:
    """
    Create a folder in the user's My Documents section
    
    **SECURITY**: Only creates folders in user's collection (collection_type='user')
    Agents cannot create folders in global collection
    
    Args:
        folder_name: Name of the folder to create (e.g., "Electronics Projects", "Components")
        parent_folder_id: Optional parent folder ID (must be user's folder)
        parent_folder_path: Optional parent folder path (e.g., "Projects") - will resolve to folder_id
        user_id: User ID (required - must match the user making the request)
    
    Returns:
        Dict with folder_id, folder_name, parent_folder_id, and status
    """
    try:
        logger.info(f"📁 Creating user folder: {folder_name} for user {user_id}")
        
        # Import services
        from services.service_container import get_service_container
        
        # Get service container
        container = await get_service_container()
        folder_service = container.folder_service
        
        # Resolve parent_folder_id from path if needed
        if parent_folder_path and not parent_folder_id:
            # Parse folder path and get/create folders
            path_parts = [p.strip() for p in parent_folder_path.split("/") if p.strip()]
            
            current_parent_id = None
            for folder_name_part in path_parts:
                # Get or create folder
                folder = await folder_service.get_or_create_subfolder(
                    parent_folder_id=current_parent_id,
                    folder_name=folder_name_part,
                    user_id=user_id,
                    collection_type="user",  # Always user collection
                    current_user_role="user",
                    admin_user_id=user_id
                )
                current_parent_id = folder.folder_id
            
            parent_folder_id = current_parent_id
            logger.info(f"📁 Resolved parent folder path '{parent_folder_path}' → folder_id: {parent_folder_id}")
        
        # Create folder
        folder = await folder_service.create_folder(
            name=folder_name,
            parent_folder_id=parent_folder_id,
            user_id=user_id,
            collection_type="user",  # Always user collection - security restriction
            current_user_role="user",
            admin_user_id=user_id
        )
        
        logger.info(f"✅ Created user folder: {folder.name} (folder_id: {folder.folder_id})")
        
        return {
            "success": True,
            "folder_id": folder.folder_id,
            "folder_name": folder.name,
            "parent_folder_id": folder.parent_folder_id,
            "status": "created",
            "message": f"Folder '{folder_name}' created successfully in user's My Documents"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to create user folder '{folder_name}': {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to create folder '{folder_name}': {str(e)}"
        }

