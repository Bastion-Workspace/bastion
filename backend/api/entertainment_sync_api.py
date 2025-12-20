"""
Entertainment Sync API Endpoints
FastAPI endpoints for Sonarr/Radarr sync configuration management
"""

import logging
from typing import List
from uuid import UUID
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks

from models.entertainment_sync_models import (
    SyncConfigCreate, SyncConfigUpdate, SyncConfig, SyncItem,
    ItemFilters, SyncStatus, ConnectionTestResult, SyncTriggerResult
)
from models.api_models import AuthenticatedUserResponse
from services.entertainment_sync_service import EntertainmentSyncService, DuplicateConfigError
from services.radarr_service import RadarrService
from services.sonarr_service import SonarrService
from services.celery_tasks.entertainment_sync_tasks import sync_entertainment_source_task
from utils.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/entertainment/sync", tags=["entertainment_sync"])


@router.post("/config", response_model=SyncConfig)
async def create_sync_config(
    config: SyncConfigCreate,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> SyncConfig:
    """
    Create a new sync configuration for Radarr or Sonarr
    """
    try:
        logger.info(f"🎬 ENTERTAINMENT SYNC API: Creating config for user {current_user.user_id}")
        
        sync_service = EntertainmentSyncService()
        config_id = await sync_service.create_sync_config(current_user.user_id, config)
        
        # Get the created config
        created_config = await sync_service.get_config(config_id, current_user.user_id)
        if not created_config:
            raise HTTPException(status_code=500, detail="Failed to retrieve created configuration")
        
        logger.info(f"✅ ENTERTAINMENT SYNC API: Created config {config_id}")
        return created_config
        
    except DuplicateConfigError as e:
        logger.warning(f"⚠️ ENTERTAINMENT SYNC API: Duplicate config attempt: {e}")
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"❌ ENTERTAINMENT SYNC API ERROR: Failed to create config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create sync configuration: {str(e)}")


@router.get("/config", response_model=List[SyncConfig])
async def list_sync_configs(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> List[SyncConfig]:
    """
    Get all sync configurations for the current user
    """
    try:
        sync_service = EntertainmentSyncService()
        configs = await sync_service.get_user_configs(current_user.user_id)
        return configs
    except Exception as e:
        logger.error(f"❌ ENTERTAINMENT SYNC API ERROR: Failed to list configs: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve sync configurations")


@router.get("/config/{config_id}", response_model=SyncConfig)
async def get_sync_config(
    config_id: UUID,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> SyncConfig:
    """
    Get a specific sync configuration
    """
    try:
        sync_service = EntertainmentSyncService()
        config = await sync_service.get_config(config_id, current_user.user_id)
        if not config:
            raise HTTPException(status_code=404, detail="Configuration not found")
        return config
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ ENTERTAINMENT SYNC API ERROR: Failed to get config: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve configuration")


@router.put("/config/{config_id}", response_model=SyncConfig)
async def update_sync_config(
    config_id: UUID,
    updates: SyncConfigUpdate,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> SyncConfig:
    """
    Update a sync configuration
    """
    try:
        sync_service = EntertainmentSyncService()
        success = await sync_service.update_sync_config(config_id, current_user.user_id, updates)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update configuration")
        
        # Get updated config
        updated_config = await sync_service.get_config(config_id, current_user.user_id)
        if not updated_config:
            raise HTTPException(status_code=404, detail="Configuration not found after update")
        
        logger.info(f"✅ ENTERTAINMENT SYNC API: Updated config {config_id}")
        return updated_config
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ ENTERTAINMENT SYNC API ERROR: Failed to update config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}")


@router.delete("/config/{config_id}")
async def delete_sync_config(
    config_id: UUID,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Delete a sync configuration and all associated items
    """
    try:
        sync_service = EntertainmentSyncService()
        success = await sync_service.delete_sync_config(config_id, current_user.user_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete configuration")
        
        logger.info(f"✅ ENTERTAINMENT SYNC API: Deleted config {config_id}")
        return {"success": True, "message": "Configuration deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ ENTERTAINMENT SYNC API ERROR: Failed to delete config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete configuration: {str(e)}")


@router.post("/config/{config_id}/test", response_model=ConnectionTestResult)
async def test_connection(
    config_id: UUID,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> ConnectionTestResult:
    """
    Test connection to Radarr/Sonarr API
    """
    try:
        sync_service = EntertainmentSyncService()
        config = await sync_service.get_config_with_api_key(config_id, current_user.user_id)
        if not config:
            raise HTTPException(status_code=404, detail="Configuration not found")
        
        api_url = config['api_url']
        api_key = config['api_key']
        source_type = config['source_type']
        
        # Test connection
        if source_type == 'radarr':
            service = RadarrService(api_url, api_key)
        else:
            service = SonarrService(api_url, api_key)
        
        result = await service.test_connection()
        
        return ConnectionTestResult(
            success=result['success'],
            message=result['message'],
            version=result.get('version')
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ ENTERTAINMENT SYNC API ERROR: Connection test failed: {e}")
        return ConnectionTestResult(
            success=False,
            message=f"Connection test failed: {str(e)}"
        )


@router.post("/config/{config_id}/trigger", response_model=SyncTriggerResult)
async def trigger_manual_sync(
    config_id: UUID,
    background_tasks: BackgroundTasks,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> SyncTriggerResult:
    """
    Manually trigger a sync for a configuration
    """
    try:
        sync_service = EntertainmentSyncService()
        config = await sync_service.get_config(config_id, current_user.user_id)
        if not config:
            raise HTTPException(status_code=404, detail="Configuration not found")
        
        # Queue sync task
        task = sync_entertainment_source_task.delay(str(config_id))
        
        logger.info(f"✅ ENTERTAINMENT SYNC API: Triggered sync for config {config_id}, task {task.id}")
        return SyncTriggerResult(
            success=True,
            task_id=task.id,
            message=f"Sync task queued: {task.id}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ ENTERTAINMENT SYNC API ERROR: Failed to trigger sync: {e}")
        return SyncTriggerResult(
            success=False,
            message=f"Failed to trigger sync: {str(e)}"
        )


@router.get("/config/{config_id}/status", response_model=SyncStatus)
async def get_sync_status(
    config_id: UUID,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> SyncStatus:
    """
    Get sync status for a configuration
    """
    try:
        sync_service = EntertainmentSyncService()
        config = await sync_service.get_config(config_id, current_user.user_id)
        if not config:
            raise HTTPException(status_code=404, detail="Configuration not found")
        
        # Calculate next sync time
        next_sync_at = None
        if config.last_sync_at and config.sync_frequency_minutes:
            from datetime import timedelta
            next_sync_at = config.last_sync_at + timedelta(minutes=config.sync_frequency_minutes)
        
        return SyncStatus(
            config_id=config.config_id,
            enabled=config.enabled,
            last_sync_at=config.last_sync_at,
            last_sync_status=config.last_sync_status,
            items_synced=config.items_synced,
            sync_error=config.sync_error,
            next_sync_at=next_sync_at
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ ENTERTAINMENT SYNC API ERROR: Failed to get sync status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve sync status")


@router.get("/config/{config_id}/items", response_model=List[SyncItem])
async def list_synced_items(
    config_id: UUID,
    filters: ItemFilters = Depends(),
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> List[SyncItem]:
    """
    List synced items for a configuration
    """
    try:
        sync_service = EntertainmentSyncService()
        config = await sync_service.get_config(config_id, current_user.user_id)
        if not config:
            raise HTTPException(status_code=404, detail="Configuration not found")
        
        items = await sync_service.get_sync_items(config_id, filters)
        return items
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ ENTERTAINMENT SYNC API ERROR: Failed to list items: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve synced items")

