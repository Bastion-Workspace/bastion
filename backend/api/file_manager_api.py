"""
FileManager API - REST endpoints for centralized file management
"""

import logging
from fastapi import APIRouter, HTTPException, Depends
from typing import Optional

from services.file_manager import get_file_manager
from services.file_manager.models.file_placement_models import (
    FilePlacementRequest, FilePlacementResponse,
    FileMoveRequest, FileMoveResponse,
    FileDeleteRequest, FileDeleteResponse,
    FileRenameRequest, FileRenameResponse,
    FolderStructureRequest, FolderStructureResponse
)
from utils.auth_middleware import get_current_user
from api.document_api import check_document_access
from models.api_models import AuthenticatedUserResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["File Manager"])


@router.post("/api/file-manager/place-file", response_model=FilePlacementResponse)
async def place_file(
    request: FilePlacementRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Place a file in the appropriate folder structure
    
    SECURITY: Regular users cannot create files in Global folders (read-only access).
    """
    try:
        # SECURITY: Check folder access if target_folder_id is provided
        if request.target_folder_id:
            from services.service_container import get_service_container
            container = await get_service_container()
            folder_service = container.folder_service
            
            folder = await folder_service.get_folder(request.target_folder_id, current_user.user_id)
            if not folder:
                raise HTTPException(status_code=404, detail="Folder not found or access denied")
            
            # Prevent regular users from creating files in global folders
            if folder.collection_type == "global" and current_user.role != "admin":
                raise HTTPException(
                    status_code=403,
                    detail="Creating files in Global folders requires Admin privileges"
                )
        
        file_manager = await get_file_manager()
        
        # Set user_id and role from current user if not provided
        if not request.user_id and current_user:
            request.user_id = current_user.user_id
        if not hasattr(request, 'current_user_role') or not request.current_user_role:
            request.current_user_role = current_user.role
        
        response = await file_manager.place_file(request)
        logger.info(f"✅ File placed via API: {response.document_id}")
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to place file via API: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/file-manager/move-file", response_model=FileMoveResponse)
async def move_file(
    request: FileMoveRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Move a file to a different folder
    
    SECURITY: Requires write permission to the document.
    Regular users cannot move global documents (read-only access).
    """
    try:
        # Get document info first to check collection type for better error messages
        from services.service_container import get_service_container
        container = await get_service_container()
        doc_info = await container.document_service.get_document(request.document_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")
        
        collection_type = getattr(doc_info, 'collection_type', 'user')
        
        # SECURITY: Check if user has write access to this document
        try:
            doc_info = await check_document_access(request.document_id, current_user, "write")
        except HTTPException as e:
            # Provide more specific error message for move operation
            if e.status_code == 403 and collection_type == 'global':
                raise HTTPException(
                    status_code=403,
                    detail="Moving a Global file requires Admin privileges"
                )
            # Re-raise original exception for other cases
            raise
        
        file_manager = await get_file_manager()
        
        # Set user_id and role from current user if not provided
        if not request.user_id and current_user:
            request.user_id = current_user.user_id
            request.current_user_role = current_user.role
        
        response = await file_manager.move_file(request)
        logger.info(f"✅ File moved via API: {response.document_id}")
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to move file via API: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/file-manager/delete-file", response_model=FileDeleteResponse)
async def delete_file(
    request: FileDeleteRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Delete a file or folder
    
    SECURITY: Requires delete permission to the document.
    Regular users cannot delete global documents (read-only access).
    """
    try:
        # Get document info first to check collection type for better error messages
        from services.service_container import get_service_container
        container = await get_service_container()
        doc_info = await container.document_service.get_document(request.document_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")
        
        collection_type = getattr(doc_info, 'collection_type', 'user')
        
        # SECURITY: Check if user has delete access to this document
        try:
            doc_info = await check_document_access(request.document_id, current_user, "delete")
        except HTTPException as e:
            # Provide more specific error message for delete operation
            if e.status_code == 403 and collection_type == 'global':
                raise HTTPException(
                    status_code=403,
                    detail="Deleting a Global file requires Admin privileges"
                )
            # Re-raise original exception for other cases
            raise
        
        file_manager = await get_file_manager()
        
        # Set user_id from current user if not provided
        if not request.user_id and current_user:
            request.user_id = current_user.user_id
        
        response = await file_manager.delete_file(request)
        logger.info(f"✅ File deleted via API: {response.document_id}")
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete file via API: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/file-manager/rename-file", response_model=FileRenameResponse)
async def rename_file(
    request: FileRenameRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Rename a file (update filename/title and disk file if applicable)
    
    SECURITY: Requires write permission to the document.
    Regular users cannot rename global documents (read-only access).
    """
    try:
        # Get document info first to check collection type for better error messages
        from services.service_container import get_service_container
        container = await get_service_container()
        doc_info = await container.document_service.get_document(request.document_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")
        
        collection_type = getattr(doc_info, 'collection_type', 'user')
        
        # SECURITY: Check if user has write access to this document
        try:
            doc_info = await check_document_access(request.document_id, current_user, "write")
        except HTTPException as e:
            # Provide more specific error message for rename operation
            if e.status_code == 403 and collection_type == 'global':
                raise HTTPException(
                    status_code=403,
                    detail="Renaming a Global file requires Admin privileges"
                )
            # Re-raise original exception for other cases
            raise
        
        file_manager = await get_file_manager()
        if not request.user_id and current_user:
            request.user_id = current_user.user_id if hasattr(current_user, 'user_id') else current_user
        response = await file_manager.rename_file(request)
        logger.info(f"✅ File renamed via API: {response.document_id}")
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to rename file via API: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/file-manager/create-folder-structure", response_model=FolderStructureResponse)
async def create_folder_structure(
    request: FolderStructureRequest,
    current_user: Optional[str] = Depends(get_current_user)
):
    """Create a folder structure"""
    try:
        file_manager = await get_file_manager()
        
        # Set user_id from current user if not provided
        if not request.user_id and current_user:
            request.user_id = current_user
        
        response = await file_manager.create_folder_structure(request)
        logger.info(f"✅ Folder structure created via API: {response.folder_id}")
        return response
        
    except Exception as e:
        logger.error(f"❌ Failed to create folder structure via API: {e}")
        raise HTTPException(status_code=500, detail=str(e))
