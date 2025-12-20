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

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/file-manager", tags=["File Manager"])


@router.post("/place-file", response_model=FilePlacementResponse)
async def place_file(
    request: FilePlacementRequest,
    current_user: Optional[str] = Depends(get_current_user)
):
    """Place a file in the appropriate folder structure"""
    try:
        file_manager = await get_file_manager()
        
        # Set user_id from current user if not provided
        if not request.user_id and current_user:
            request.user_id = current_user
        
        response = await file_manager.place_file(request)
        logger.info(f"✅ File placed via API: {response.document_id}")
        return response
        
    except Exception as e:
        logger.error(f"❌ Failed to place file via API: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/move-file", response_model=FileMoveResponse)
async def move_file(
    request: FileMoveRequest,
    current_user = Depends(get_current_user)
):
    """Move a file to a different folder"""
    try:
        file_manager = await get_file_manager()
        
        # Set user_id and role from current user if not provided
        if not request.user_id and current_user:
            request.user_id = current_user.user_id if hasattr(current_user, 'user_id') else current_user
            request.current_user_role = current_user.role if hasattr(current_user, 'role') else 'user'
        
        response = await file_manager.move_file(request)
        logger.info(f"✅ File moved via API: {response.document_id}")
        return response
        
    except Exception as e:
        logger.error(f"❌ Failed to move file via API: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/delete-file", response_model=FileDeleteResponse)
async def delete_file(
    request: FileDeleteRequest,
    current_user: Optional[str] = Depends(get_current_user)
):
    """Delete a file or folder"""
    try:
        file_manager = await get_file_manager()
        
        # Set user_id from current user if not provided
        if not request.user_id and current_user:
            request.user_id = current_user
        
        response = await file_manager.delete_file(request)
        logger.info(f"✅ File deleted via API: {response.document_id}")
        return response
        
    except Exception as e:
        logger.error(f"❌ Failed to delete file via API: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rename-file", response_model=FileRenameResponse)
async def rename_file(
    request: FileRenameRequest,
    current_user: Optional[str] = Depends(get_current_user)
):
    """Rename a file (update filename/title and disk file if applicable)"""
    try:
        file_manager = await get_file_manager()
        if not request.user_id and current_user:
            request.user_id = current_user
        response = await file_manager.rename_file(request)
        logger.info(f"✅ File renamed via API: {response.document_id}")
        return response
    except Exception as e:
        logger.error(f"❌ Failed to rename file via API: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create-folder-structure", response_model=FolderStructureResponse)
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
