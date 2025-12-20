"""
Entertainment Sync Models
Pydantic models for Sonarr/Radarr API integration
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, HttpUrl
from datetime import datetime
from uuid import UUID


class SyncConfigCreate(BaseModel):
    """Request model for creating a new sync configuration"""
    source_type: Literal['radarr', 'sonarr'] = Field(..., description="Type of media manager")
    api_url: str = Field(..., description="Base URL of Radarr/Sonarr instance")
    api_key: str = Field(..., description="API key for authentication")
    enabled: bool = Field(default=True, description="Whether sync is enabled")
    sync_frequency_minutes: int = Field(default=60, description="Sync frequency in minutes")


class SyncConfigUpdate(BaseModel):
    """Request model for updating sync configuration"""
    api_url: Optional[str] = Field(None, description="Base URL of Radarr/Sonarr instance")
    api_key: Optional[str] = Field(None, description="API key for authentication")
    enabled: Optional[bool] = Field(None, description="Whether sync is enabled")
    sync_frequency_minutes: Optional[int] = Field(None, description="Sync frequency in minutes")


class SyncConfig(BaseModel):
    """Response model for sync configuration"""
    config_id: UUID = Field(..., description="Unique configuration ID")
    user_id: str = Field(..., description="User ID who owns this configuration")
    source_type: str = Field(..., description="Type of media manager: 'radarr' or 'sonarr'")
    api_url: str = Field(..., description="Base URL of Radarr/Sonarr instance")
    enabled: bool = Field(..., description="Whether sync is enabled")
    sync_frequency_minutes: int = Field(..., description="Sync frequency in minutes")
    last_sync_at: Optional[datetime] = Field(None, description="Last successful sync timestamp")
    last_sync_status: Optional[str] = Field(None, description="Last sync status: 'success', 'failed', 'running'")
    items_synced: int = Field(default=0, description="Number of items synced")
    sync_error: Optional[str] = Field(None, description="Error message from last sync if failed")
    created_at: datetime = Field(..., description="Configuration creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class SyncItem(BaseModel):
    """Response model for synced entertainment item"""
    item_id: UUID = Field(..., description="Unique item ID")
    config_id: UUID = Field(..., description="Configuration ID this item belongs to")
    external_id: str = Field(..., description="External ID from Radarr/Sonarr")
    external_type: str = Field(..., description="Item type: 'movie', 'series', or 'episode'")
    title: str = Field(..., description="Item title")
    tmdb_id: Optional[int] = Field(None, description="TMDB ID if available")
    tvdb_id: Optional[int] = Field(None, description="TVDB ID if available")
    season_number: Optional[int] = Field(None, description="Season number for episodes")
    episode_number: Optional[int] = Field(None, description="Episode number for episodes")
    parent_series_id: Optional[str] = Field(None, description="Parent series ID for episodes")
    metadata_hash: Optional[str] = Field(None, description="Hash of metadata for change detection")
    last_synced_at: datetime = Field(..., description="Last sync timestamp")
    vector_document_id: Optional[str] = Field(None, description="Vector document ID for tracking")


class ItemFilters(BaseModel):
    """Filters for querying synced items"""
    external_type: Optional[Literal['movie', 'series', 'episode']] = Field(None, description="Filter by item type")
    limit: int = Field(default=100, description="Maximum number of items to return")
    skip: int = Field(default=0, description="Number of items to skip")


class SyncStatus(BaseModel):
    """Status information for a sync configuration"""
    config_id: UUID = Field(..., description="Configuration ID")
    enabled: bool = Field(..., description="Whether sync is enabled")
    last_sync_at: Optional[datetime] = Field(None, description="Last successful sync timestamp")
    last_sync_status: Optional[str] = Field(None, description="Last sync status")
    items_synced: int = Field(default=0, description="Total items synced")
    sync_error: Optional[str] = Field(None, description="Error message if last sync failed")
    next_sync_at: Optional[datetime] = Field(None, description="Estimated next sync time")


class ConnectionTestResult(BaseModel):
    """Result of testing API connection"""
    success: bool = Field(..., description="Whether connection test succeeded")
    message: str = Field(..., description="Test result message")
    version: Optional[str] = Field(None, description="Radarr/Sonarr version if available")


class SyncTriggerResult(BaseModel):
    """Result of triggering a manual sync"""
    success: bool = Field(..., description="Whether sync was triggered")
    task_id: Optional[str] = Field(None, description="Celery task ID for tracking")
    message: str = Field(..., description="Result message")

