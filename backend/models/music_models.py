"""
Music Service Models
Pydantic models for SubSonic music service integration
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class MusicServiceConfigRequest(BaseModel):
    """Request to save media server configuration"""
    server_url: str = Field(..., description="Media server URL")
    username: str = Field(..., description="Server username (or API token for Audiobookshelf)")
    password: str = Field(..., description="Server password or API token")
    auth_type: str = Field(default="password", description="Authentication type: 'password' or 'token' (for SubSonic)")
    service_type: str = Field(default="subsonic", description="Service type: 'subsonic' or 'audiobookshelf'")
    service_name: Optional[str] = Field(default=None, description="User-friendly display name for this source")


class MusicServiceConfigResponse(BaseModel):
    """Response with music service configuration (without credentials)"""
    server_url: str
    username: str
    auth_type: str
    service_type: str = Field(default="subsonic", description="Service type: 'subsonic' or 'audiobookshelf'")
    service_name: Optional[str] = Field(default=None, description="User-friendly display name")
    is_active: bool = Field(default=True, description="Whether this source is active")
    has_config: bool = Field(default=True, description="Whether user has configured a music service")
    last_sync_at: Optional[datetime] = None
    sync_status: Optional[str] = None
    total_albums: int = 0
    total_artists: int = 0
    total_playlists: int = 0
    total_tracks: int = 0


class MediaSourceListResponse(BaseModel):
    """Response with list of all configured media sources"""
    sources: List[MusicServiceConfigResponse] = []


class MusicTrack(BaseModel):
    """Music track model"""
    id: str
    title: str
    artist: Optional[str] = None
    album: Optional[str] = None
    duration: Optional[int] = None  # Duration in seconds
    track_number: Optional[int] = None
    cover_art_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    service_type: Optional[str] = Field(default=None, description="Service type: 'subsonic' or 'audiobookshelf'")


class MusicAlbum(BaseModel):
    """Music album model"""
    id: str
    title: str
    artist: Optional[str] = None
    cover_art_id: Optional[str] = None
    track_count: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class MusicArtist(BaseModel):
    """Music artist model"""
    id: str
    name: str
    album_count: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class MusicPlaylist(BaseModel):
    """Music playlist model"""
    id: str
    name: str
    track_count: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class MusicLibraryResponse(BaseModel):
    """Response with music library data"""
    albums: List[MusicAlbum] = []
    artists: List[MusicArtist] = []
    playlists: List[MusicPlaylist] = []
    last_sync_at: Optional[datetime] = None


class MusicTracksResponse(BaseModel):
    """Response with tracks for an album or playlist"""
    tracks: List[MusicTrack] = []
    parent_id: str
    parent_type: str  # 'album' or 'playlist'


class StreamUrlResponse(BaseModel):
    """Response with authenticated stream URL"""
    stream_url: str
    expires_at: Optional[datetime] = None

