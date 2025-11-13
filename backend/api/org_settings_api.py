"""
Org-Mode Settings API
CRUD operations for user org-mode configuration
"""

import logging
from fastapi import APIRouter, Depends, HTTPException

from utils.auth_middleware import get_current_user, AuthenticatedUserResponse
from services.org_settings_service import get_org_settings_service
from models.org_settings_models import (
    OrgModeSettings,
    OrgModeSettingsUpdate,
    OrgModeSettingsResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/org/settings", tags=["Org Settings"])


@router.get("")
async def get_org_settings(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> OrgModeSettingsResponse:
    """
    Get org-mode settings for the current user
    
    **BULLY!** Retrieve your org-mode configuration!
    
    Returns default settings if none exist yet.
    """
    try:
        logger.info(f"⚙️ ROOSEVELT: Fetching org settings for user {current_user.username}")
        
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


@router.put("")
async def update_org_settings(
    settings_update: OrgModeSettingsUpdate,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> OrgModeSettingsResponse:
    """
    Update org-mode settings for the current user
    
    **BULLY!** Save your org-mode configuration!
    
    Only provided fields will be updated. Omitted fields remain unchanged.
    """
    try:
        logger.info(f"⚙️ ROOSEVELT: Updating org settings for user {current_user.username}")
        
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


@router.delete("")
async def reset_org_settings(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> OrgModeSettingsResponse:
    """
    Reset org-mode settings to defaults
    
    **BULLY!** Start fresh with default configuration!
    """
    try:
        logger.info(f"⚙️ ROOSEVELT: Resetting org settings for user {current_user.username}")
        
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


@router.get("/todo-states")
async def get_todo_states(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> dict:
    """
    Get all TODO states for the current user
    
    **BULLY!** Retrieve your TODO state configuration!
    
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


@router.get("/tags")
async def get_tags(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> dict:
    """
    Get all predefined tags for the current user
    
    **BULLY!** Retrieve your tag definitions!
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

