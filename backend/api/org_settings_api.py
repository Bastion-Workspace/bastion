"""
Org-Mode Settings API
CRUD operations for user org-mode configuration
"""

import logging
from fastapi import APIRouter, Depends, HTTPException

from pathlib import Path
from typing import List, Dict, Any

from utils.auth_middleware import get_current_user, AuthenticatedUserResponse
from services.org_settings_service import get_org_settings_service
from services.database_manager.database_helpers import fetch_one
from models.org_settings_models import (
    OrgModeSettings,
    OrgModeSettingsUpdate,
    OrgModeSettingsResponse
)
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Org Settings"])


@router.get("/api/org/settings")
async def get_org_settings(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> OrgModeSettingsResponse:
    """
    Get org-mode settings for the current user
    
    Retrieve the current user's org-mode configuration.
    
    Returns default settings if none exist yet.
    """
    try:
        logger.info("Fetching org settings for user %s", current_user.username)
        
        service = await get_org_settings_service()
        settings = await service.get_settings(current_user.user_id)
        
        return OrgModeSettingsResponse(
            success=True,
            settings=settings,
            message="Settings retrieved successfully"
        )
    
    except Exception as e:
        logger.error(f"❌ Failed to get org settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/org/settings")
async def update_org_settings(
    settings_update: OrgModeSettingsUpdate,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> OrgModeSettingsResponse:
    """
    Update org-mode settings for the current user
    
    Save org-mode configuration for the current user.
    
    Only provided fields will be updated. Omitted fields remain unchanged.
    """
    try:
        logger.info("Updating org settings for user %s", current_user.username)
        
        service = await get_org_settings_service()
        settings = await service.create_or_update_settings(
            current_user.user_id,
            settings_update
        )
        
        return OrgModeSettingsResponse(
            success=True,
            settings=settings,
            message="Settings updated successfully"
        )
    
    except Exception as e:
        logger.error(f"❌ Failed to update org settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/org/settings")
async def reset_org_settings(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> OrgModeSettingsResponse:
    """
    Reset org-mode settings to defaults
    
    Reset org-mode configuration to defaults.
    """
    try:
        logger.info("Resetting org settings for user %s", current_user.username)
        
        service = await get_org_settings_service()
        await service.delete_settings(current_user.user_id)
        
        # Return default settings
        default_settings = await service.get_settings(current_user.user_id)
        
        return OrgModeSettingsResponse(
            success=True,
            settings=default_settings,
            message="Settings reset to defaults"
        )
    
    except Exception as e:
        logger.error(f"❌ Failed to reset org settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/org/settings/todo-states")
async def get_todo_states(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> dict:
    """
    Get all TODO states for the current user
    
    Return TODO keyword/state definitions for the current user.
    
    Returns:
        {
            "active": ["TODO", "NEXT", "WAITING"],
            "done": ["DONE", "CANCELED"],
            "all": ["TODO", "NEXT", "WAITING", "DONE", "CANCELED"]
        }
    """
    try:
        service = await get_org_settings_service()
        states = await service.get_todo_states(current_user.user_id)
        
        return {
            "success": True,
            "states": states
        }
    
    except Exception as e:
        logger.error(f"❌ Failed to get TODO states: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/org/settings/tags")
async def get_tags(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> dict:
    """
    Get all predefined tags for the current user
    
    Return org tag definitions for the current user.
    """
    try:
        service = await get_org_settings_service()
        tags = await service.get_tags(current_user.user_id)
        
        return {
            "success": True,
            "tags": tags
        }
    
    except Exception as e:
        logger.error(f"❌ Failed to get tags: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/org/settings/journal-locations")
async def get_journal_locations(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get available filesystem directories for journal location
    
    Returns a tree of directories under the user's directory
    """
    try:
        # Get username
        row = await fetch_one("SELECT username FROM users WHERE user_id = $1", current_user.user_id)
        username = row['username'] if row else current_user.user_id
        
        # Build user directory path (logical; bytes live on document-service)
        upload_dir = Path(settings.UPLOAD_DIR)
        user_dir = upload_dir / "Users" / username

        from services import ds_upload_library_fs as dsf

        async def build_directory_tree(path: Path, relative_path: str = "") -> List[Dict[str, Any]]:
            """Recursively build directory tree via document-service listing."""
            directories: List[Dict[str, Any]] = []
            if not await dsf.is_dir(current_user.user_id, path):
                return directories
            try:
                for name in sorted(await dsf.list_dir_names(current_user.user_id, path)):
                    if name.startswith("."):
                        continue
                    item = path / name
                    if not await dsf.is_dir(current_user.user_id, item):
                        continue
                    item_relative = f"{relative_path}/{name}" if relative_path else name
                    children = await build_directory_tree(item, item_relative)
                    directories.append(
                        {"name": name, "path": item_relative, "children": children}
                    )
            except PermissionError:
                logger.warning("Permission denied accessing %s", path)

            return directories

        # Build tree starting from user directory
        tree = await build_directory_tree(user_dir)
        
        # Also include root option (empty path)
        result = {
            "success": True,
            "directories": tree,
            "root_path": str(user_dir.relative_to(upload_dir))
        }
        
        return result
    
    except Exception as e:
        logger.error(f"❌ Failed to get journal locations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

