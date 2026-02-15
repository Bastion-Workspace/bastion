"""
Folder Management API endpoints
Extracted from main.py for better modularity
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import Response

from models.api_models import (
    FolderTreeResponse, FolderContentsResponse, DocumentFolder,
    FolderCreateRequest, FolderUpdateRequest, FolderMetadataUpdateRequest,
    AuthenticatedUserResponse
)
from services.service_container import get_service_container
from services.file_manager import get_file_manager
from services.file_manager.models.file_placement_models import FolderStructureRequest
from utils.auth_middleware import get_current_user
from utils.websocket_manager import get_websocket_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["folders"])


# Helper function to get folder service
async def _get_folder_service():
    """Get folder service from service container"""
    container = await get_service_container()
    return container.folder_service


# ===== FOLDER MANAGEMENT ENDPOINTS =====

@router.get("/api/folders/library/download")
async def download_library_zip(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Download the current user's library (My Documents) as a zip file."""
    folder_service = await _get_folder_service()
    try:
        zip_bytes = await folder_service.build_library_zip(current_user.user_id)
        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers={
                "Content-Disposition": 'attachment; filename="my-library.zip"',
            },
        )
    except Exception as e:
        logger.error(f"Failed to build library zip: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/folders/tree", response_model=FolderTreeResponse)
async def get_folder_tree(
    collection_type: str = "user",
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get the complete folder tree for the current user"""
    folder_service = await _get_folder_service()
    try:
        logger.debug(f"📁 Getting folder tree for user: {current_user.user_id}, collection_type: {collection_type}")
        folders = await folder_service.get_folder_tree(
            user_id=current_user.user_id, 
            collection_type=collection_type
        )
        logger.debug(f"📁 Found {len(folders)} folders")
        return FolderTreeResponse(
            folders=folders,
            total_folders=len(folders)
        )
    except Exception as e:
        logger.error(f"❌ Failed to get folder tree: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/folders/{folder_id}/contents", response_model=FolderContentsResponse)
async def get_folder_contents(
    folder_id: str,
    limit: int = 250,
    offset: int = 0,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get contents of a specific folder. Paginated for large folders (default limit 250)."""
    folder_service = await _get_folder_service()
    try:
        logger.debug(f"🔍 API: Getting folder contents for {folder_id} (user: {current_user.user_id}, limit: {limit}, offset: {offset})")
        contents = await folder_service.get_folder_contents(folder_id, current_user.user_id, limit=limit, offset=offset)
        if not contents:
            logger.warning(f"⚠️ API: Folder {folder_id} not found or access denied for user {current_user.user_id}")
            raise HTTPException(status_code=404, detail="Folder not found or access denied")
        
        logger.debug(f"✅ API: Returning folder contents for {folder_id}: {contents.total_documents} docs, {contents.total_subfolders} subfolders")
        return contents
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get folder contents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/folders", response_model=DocumentFolder)
async def create_folder(
    request: FolderCreateRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Create a new folder using FileManager for consistent WebSocket notifications"""
    try:
        logger.info(f"🔍 API: Create folder request from user {current_user.username} (role: {current_user.role})")
        logger.info(f"🔍 API: Request data: {request.dict()}")
        
        # Get FileManager service
        file_manager = await get_file_manager()
        
        folder_service = await _get_folder_service()
        # Determine collection type based on user role and request
        collection_type = "user"
        user_id = current_user.user_id
        team_id = None
        admin_user_id = None
        
        # Check if parent folder is provided - inherit collection type from parent
        if request.parent_folder_id:
            # Get parent folder to determine collection type and team_id
            parent_folder = await folder_service.get_folder(request.parent_folder_id, current_user.user_id, current_user.role)
            if parent_folder:
                parent_collection_type = parent_folder.collection_type or 'user'
                parent_team_id = parent_folder.team_id
                
                # Inherit collection type from parent
                if parent_collection_type == "team" and parent_team_id:
                    collection_type = "team"
                    team_id = str(parent_team_id) if parent_team_id else None
                    # For team folders, user_id must be NULL (per schema constraint)
                    user_id = None
                    # For team folders, use the creator's user_id as admin_user_id (for RLS and created_by)
                    admin_user_id = current_user.user_id
                    # Verify user is a team member
                    from api.teams_api import team_service
                    member_role = await team_service.check_team_access(team_id, current_user.user_id)
                    if not member_role:
                        raise HTTPException(status_code=403, detail="You must be a team member to create folders in team folders")
                    logger.info(f"🔍 API: Creating team folder - team_id: {team_id}, admin_user_id: {admin_user_id}")
                elif parent_collection_type == "global":
                    # Only admins can create folders in global folders
                    if current_user.role != "admin":
                        raise HTTPException(status_code=403, detail="Only admins can create folders in global folders")
                    collection_type = "global"
                    user_id = None
                    logger.info(f"🔍 API: Creating global folder")
                else:
                    # User folder - inherit user_id from parent or use current user
                    user_id = parent_folder.user_id or current_user.user_id
                    logger.info(f"🔍 API: Creating user folder - user_id: {user_id}")
                
                logger.info(f"🔍 API: Creating folder '{request.name}' in parent folder '{parent_folder.name}' (collection_type: {collection_type})")
            else:
                logger.warning(f"⚠️ Parent folder {request.parent_folder_id} not found, creating at root level")
        else:
            # No parent - check explicit collection_type in request
            if current_user.role == "admin" and getattr(request, 'collection_type', None) == "global":
                collection_type = "global"
                user_id = None  # Global folders have no user_id
                logger.info(f"🔍 API: Admin creating global folder - setting user_id to None")
            else:
                # Default to user folder
                logger.info(f"🔍 API: Creating folder '{request.name}' at root level (user folder)")
        
        logger.info(f"🔍 API: Final parameters - collection_type: {collection_type}, user_id: {user_id}, team_id: {team_id}")
        
        # Additional security validation
        if collection_type == "team":
            # Team folder creation - already validated team membership above
            pass
        elif current_user.role != "admin":
            # Regular users can only create folders for themselves
            if collection_type == "user" and user_id != current_user.user_id:
                raise HTTPException(status_code=403, detail="Regular users can only create folders for themselves")
            if collection_type == "global":
                raise HTTPException(status_code=403, detail="Regular users cannot create global folders")
        else:
            # Admins can only create folders for themselves or global folders
            if collection_type == "user" and user_id != current_user.user_id:
                raise HTTPException(status_code=403, detail="Admins can only create folders for themselves or global folders")
        
        # Build folder path
        folder_path = [request.name]
        
        # Create folder using FileManager for consistent WebSocket notifications
        # For team folders, admin_user_id should be the creator (for RLS context)
        # For user folders, admin_user_id should be the current user's ID (for validation)
        # For global folders, admin_user_id is only set if user is admin
        final_admin_user_id = None
        if collection_type == "team":
            # Team folders: admin_user_id is the creator (for RLS context)
            final_admin_user_id = admin_user_id or current_user.user_id
        elif collection_type == "user":
            # User folders: admin_user_id is the current user's ID (for validation in create_folder)
            final_admin_user_id = current_user.user_id
        elif current_user.role == "admin" and collection_type == "global":
            # Admin creating global folder
            final_admin_user_id = current_user.user_id
        
        folder_request = FolderStructureRequest(
            folder_path=folder_path,
            parent_folder_id=request.parent_folder_id,
            user_id=user_id,
            collection_type=collection_type,
            description=f"Folder created by {current_user.username}",
            current_user_role=current_user.role,
            admin_user_id=final_admin_user_id
        )
        
        logger.info(f"🔍 API: Creating folder via FileManager: {folder_request.dict()}")
        response = await file_manager.create_folder_structure(folder_request)
        
        # Get the created folder info to return
        # For team folders, use admin_user_id (creator) for RLS context
        # For user/global folders, use user_id (may be None for global)
        query_user_id = final_admin_user_id if collection_type == "team" else user_id
        folder = await folder_service.get_folder(response.folder_id, query_user_id, current_user.role)
        if not folder:
            raise HTTPException(status_code=500, detail="Folder created but could not retrieve folder info")
        
        # Single event system handles all notifications via FileManager
        logger.info(f"📡 Folder event notification handled by FileManager")
        
        logger.info(f"✅ API: Folder created successfully via FileManager: {folder.folder_id}")
        return folder
        
    except Exception as e:
        import traceback
        logger.error(f"❌ API: Failed to create folder via FileManager: {e}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/api/folders/{folder_id}/metadata")
async def update_folder_metadata(
    folder_id: str,
    request: FolderMetadataUpdateRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Update folder metadata (category, tags, inherit_tags)
    
    **ROOSEVELT FOLDER TAGGING PHASE 1**: Documents uploaded to this folder will inherit these tags!
    """
    try:
        folder_service = await _get_folder_service()
        logger.info(f"📋 Updating folder metadata: {folder_id} by user {current_user.username}")
        
        # Verify folder access
        folder = await folder_service.get_folder(folder_id, current_user.user_id, current_user.role)
        if not folder:
            raise HTTPException(status_code=404, detail="Folder not found or access denied")
        
        # Parse category value if provided
        category_value = request.category.value if request.category else None
        
        # Update metadata
        success = await folder_service.update_folder_metadata(
            folder_id,
            category=category_value,
            tags=request.tags,
            inherit_tags=request.inherit_tags
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update folder metadata")
        
        logger.info(f"✅ Folder metadata updated: {folder_id}")
        return {"success": True, "message": "Folder metadata updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to update folder metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/api/folders/{folder_id}", response_model=DocumentFolder)
async def update_folder(
    folder_id: str,
    request: FolderUpdateRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Update folder information"""
    try:
        folder_service = await _get_folder_service()
        updated_folder = await folder_service.update_folder(folder_id, request, current_user.user_id, current_user.role)
        if not updated_folder:
            raise HTTPException(status_code=404, detail="Folder not found or access denied")
        return updated_folder
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to update folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/api/folders/{folder_id}")
async def delete_folder(
    folder_id: str,
    recursive: bool = False,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Delete a folder"""
    try:
        folder_service = await _get_folder_service()
        # Get folder info before deletion for WebSocket notification
        folder = await folder_service.get_folder(folder_id, current_user.user_id, current_user.role)
        
        success = await folder_service.delete_folder(folder_id, current_user.user_id, recursive, current_user.role)
        if not success:
            raise HTTPException(status_code=404, detail="Folder not found or access denied")
        
        # Send WebSocket notification for folder deletion
        if folder:
            try:
                from utils.websocket_manager import get_websocket_manager
                websocket_manager = get_websocket_manager()
                if websocket_manager:
                    folder_data = {
                        "folder_id": folder_id,
                        "name": folder.name,
                        "parent_folder_id": folder.parent_folder_id,
                        "user_id": folder.user_id,
                        "collection_type": folder.collection_type,
                        "team_id": folder.team_id if hasattr(folder, 'team_id') else None,
                        "deleted_at": datetime.now().isoformat()
                    }
                    notification = {
                        "type": "folder_event",
                        "action": "deleted",
                        "folder": folder_data,
                        "user_id": current_user.user_id,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # For team folders, send to all team members
                    if folder.collection_type == "team" and folder.team_id:
                        from api.teams_api import team_service
                        members = await team_service.get_team_members(str(folder.team_id), current_user.user_id)
                        for member in members:
                            await websocket_manager.send_to_session(notification, member.get('user_id'))
                        logger.info(f"📡 Sent folder deletion notification to {len(members)} team members")
                    else:
                        # For user/global folders, send only to the owner
                        await websocket_manager.send_to_session(notification, current_user.user_id)
                        logger.info(f"📡 Sent folder deletion notification for user {current_user.user_id}")
                else:
                    logger.warning("⚠️ WebSocket manager not available for folder deletion notification")
            except Exception as e:
                logger.warning(f"⚠️ Failed to send WebSocket notification: {e}")
        
        return {"message": "Folder deleted successfully"}
    except PermissionError as e:
        logger.warning(f"⛔ Permission denied for folder deletion: {e}")
        raise HTTPException(status_code=403, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/folders/{folder_id}/move")
async def move_folder(
    folder_id: str,
    new_parent_id: str = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Move a folder to a new parent"""
    try:
        folder_service = await _get_folder_service()
        # Get current folder to capture old parent for notification
        old_folder = await folder_service.get_folder(folder_id, current_user.user_id, current_user.role)
        old_parent_id = getattr(old_folder, 'parent_folder_id', None) if old_folder else None

        success = await folder_service.move_folder(folder_id, new_parent_id, current_user.user_id, current_user.role)
        if not success:
            raise HTTPException(status_code=404, detail="Folder not found or access denied")

        # Get updated folder for notification payload
        updated_folder = await folder_service.get_folder(folder_id, current_user.user_id, current_user.role)

        # Notify via WebSocket so UI updates immediately
        try:
            from utils.websocket_manager import get_websocket_manager
            from datetime import datetime
            websocket_manager = get_websocket_manager()
            if websocket_manager and updated_folder:
                folder_data = {
                    "folder_id": updated_folder.folder_id,
                    "name": updated_folder.name,
                    "parent_folder_id": getattr(updated_folder, 'parent_folder_id', None),
                    "user_id": getattr(updated_folder, 'user_id', None),
                    "collection_type": getattr(updated_folder, 'collection_type', None),
                    "updated_at": datetime.now().isoformat()
                }
                await websocket_manager.send_to_session({
                    "type": "folder_event",
                    "action": "moved",
                    "folder": folder_data,
                    "old_parent_id": old_parent_id,
                    "new_parent_id": getattr(updated_folder, 'parent_folder_id', None),
                    "user_id": current_user.user_id,
                }, current_user.user_id)
        except Exception as ne:
            logger.warning(f"⚠️ Failed to send folder moved notification: {ne}")

        return {"message": "Folder moved successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to move folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/folders/default")
async def create_default_folders(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Create default folder structure for the current user"""
    try:
        folder_service = await _get_folder_service()
        folders = await folder_service.create_default_folders(current_user.user_id)
        return {
            "message": f"Default folders created successfully",
            "folders": [folder.dict() for folder in folders]
        }
    except Exception as e:
        logger.error(f"❌ Failed to create default folders: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/folders/{folder_id}/exempt")
async def exempt_folder_from_vectorization(
    folder_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Exempt a folder and all descendants from vectorization"""
    logger.info(f"🚫 API: Exempting folder {folder_id} for user {current_user.user_id}")
    try:
        from services.service_container import get_service_container
        container = await get_service_container()
        folder_service = container.folder_service

        success = await folder_service.exempt_folder_from_vectorization(
            folder_id,
            current_user.user_id
        )
        if success:
            logger.info(f"✅ API: Folder {folder_id} exempted successfully")
            return {"status": "success", "message": "Folder and descendants exempted from vectorization"}
        else:
            logger.error(f"❌ API: Failed to exempt folder {folder_id} - method returned false")
            raise HTTPException(status_code=500, detail="Failed to exempt folder")
    except Exception as e:
        logger.error(f"❌ Failed to exempt folder {folder_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/api/folders/{folder_id}/exempt")
async def remove_folder_exemption(
    folder_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Remove exemption from a folder (set to inherit from parent), re-process all documents"""
    try:
        from services.service_container import get_service_container
        container = await get_service_container()
        folder_service = container.folder_service
        
        success = await folder_service.remove_folder_exemption(
            folder_id,
            current_user.user_id
        )
        if success:
            return {"status": "success", "message": "Folder exemption removed - now inherits from parent"}
        else:
            raise HTTPException(status_code=500, detail="Failed to remove exemption")
    except Exception as e:
        logger.error(f"❌ Failed to remove exemption for folder {folder_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/folders/{folder_id}/exempt/override")
async def override_folder_exemption(
    folder_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Set folder to explicitly NOT exempt (override parent exemption)"""
    try:
        from services.service_container import get_service_container
        container = await get_service_container()
        folder_service = container.folder_service
        
        success = await folder_service.override_folder_exemption(
            folder_id,
            current_user.user_id
        )
        if success:
            return {"status": "success", "message": "Folder set to override parent exemption - not exempt"}
        else:
            raise HTTPException(status_code=500, detail="Failed to set override")
    except Exception as e:
        logger.error(f"❌ Failed to set override for folder {folder_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
