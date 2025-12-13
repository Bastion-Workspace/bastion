"""
Base Music Client Interface
Abstract base class for music service clients (SubSonic, Plex, Emby, etc.)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class BaseMusicClient(ABC):
    """Abstract base class for music service clients"""
    
    def __init__(self, server_url: str, username: str, password: str, **kwargs):
        """
        Initialize music client
        
        Args:
            server_url: Music server URL
            username: Username for authentication
            password: Password or token for authentication
            **kwargs: Additional service-specific parameters
        """
        self.server_url = server_url.rstrip('/')
        self.username = username
        self.password = password
    
    @abstractmethod
    async def test_connection(self) -> Dict[str, Any]:
        """
        Test connection to music server
        
        Returns:
            Dict with 'success' (bool) and optional 'error' (str) or 'auth_method_used' (str)
        """
        pass
    
    @abstractmethod
    async def get_albums(self) -> List[Dict[str, Any]]:
        """
        Fetch all albums from the music server
        
        Returns:
            List of album dicts with keys: id, title, artist, cover_art_id, metadata
        """
        pass
    
    @abstractmethod
    async def get_artists(self) -> List[Dict[str, Any]]:
        """
        Fetch all artists from the music server
        
        Returns:
            List of artist dicts with keys: id, name, metadata
        """
        pass
    
    @abstractmethod
    async def get_playlists(self) -> List[Dict[str, Any]]:
        """
        Fetch all playlists from the music server
        
        Returns:
            List of playlist dicts with keys: id, name, track_count, metadata
        """
        pass
    
    @abstractmethod
    async def get_album_tracks(self, album_id: str) -> List[Dict[str, Any]]:
        """
        Fetch tracks for an album
        
        Args:
            album_id: Album identifier
            
        Returns:
            List of track dicts with keys: id, title, artist, album, duration, 
            track_number, cover_art_id, metadata
        """
        pass
    
    @abstractmethod
    async def get_playlist_tracks(self, playlist_id: str) -> List[Dict[str, Any]]:
        """
        Fetch tracks for a playlist
        
        Args:
            playlist_id: Playlist identifier
            
        Returns:
            List of track dicts with keys: id, title, artist, album, duration, 
            track_number, cover_art_id, metadata
        """
        pass
    
    @abstractmethod
    async def get_stream_url(self, track_id: str) -> Optional[str]:
        """
        Generate authenticated stream URL for a track
        
        Args:
            track_id: Track identifier
            
        Returns:
            Authenticated stream URL or None if failed
        """
        pass
    
    async def add_to_playlist(self, playlist_id: str, track_ids: List[str]) -> Dict[str, Any]:
        """
        Add tracks to a playlist (optional - implement if service supports it)
        
        Args:
            playlist_id: Playlist identifier
            track_ids: List of track IDs to add
            
        Returns:
            Dict with 'success' (bool) and optional 'error' (str)
        """
        return {"success": False, "error": "Not implemented for this service"}
    
    async def remove_from_playlist(self, playlist_id: str, track_indices: List[int]) -> Dict[str, Any]:
        """
        Remove tracks from a playlist by index (optional - implement if service supports it)
        
        Args:
            playlist_id: Playlist identifier
            track_indices: List of track indices to remove
            
        Returns:
            Dict with 'success' (bool) and optional 'error' (str)
        """
        return {"success": False, "error": "Not implemented for this service"}
    
    async def search_tracks(self, query: str, limit: int = 25) -> List[Dict[str, Any]]:
        """
        Search for tracks (optional - for streaming services)
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of track dicts with keys: id, title, artist, album, duration, 
            track_number, cover_art_id, metadata
        """
        return []
    
    async def search_albums(self, query: str, limit: int = 25) -> List[Dict[str, Any]]:
        """
        Search for albums (optional - for streaming services)
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of album dicts with keys: id, title, artist, cover_art_id, metadata
        """
        return []
    
    async def search_artists(self, query: str, limit: int = 25) -> List[Dict[str, Any]]:
        """
        Search for artists (optional - for streaming services)
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of artist dicts with keys: id, name, metadata
        """
        return []
    
    def normalize_album(self, raw_album: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize album data to standard format for database storage
        
        Args:
            raw_album: Raw album data from service API
            
        Returns:
            Normalized album dict with: id, title, artist, cover_art_id, metadata
        """
        return {
            "id": str(raw_album.get("id", "")),
            "title": raw_album.get("title") or raw_album.get("name", ""),
            "artist": raw_album.get("artist") or raw_album.get("artistName", ""),
            "cover_art_id": raw_album.get("coverArt") or raw_album.get("cover_art_id") or raw_album.get("thumb", ""),
            "metadata": raw_album
        }
    
    def normalize_artist(self, raw_artist: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize artist data to standard format for database storage
        
        Args:
            raw_artist: Raw artist data from service API
            
        Returns:
            Normalized artist dict with: id, name, metadata
        """
        return {
            "id": str(raw_artist.get("id", "")),
            "name": raw_artist.get("name") or raw_artist.get("title", ""),
            "metadata": raw_artist
        }
    
    def normalize_playlist(self, raw_playlist: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize playlist data to standard format for database storage
        
        Args:
            raw_playlist: Raw playlist data from service API
            
        Returns:
            Normalized playlist dict with: id, name, track_count, metadata
        """
        return {
            "id": str(raw_playlist.get("id", "")),
            "name": raw_playlist.get("name") or raw_playlist.get("title", ""),
            "track_count": raw_playlist.get("songCount") or raw_playlist.get("track_count") or 0,
            "metadata": raw_playlist
        }
    
    def normalize_track(self, raw_track: Dict[str, Any], parent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Normalize track data to standard format for database storage
        
        Args:
            raw_track: Raw track data from service API
            parent_id: Optional parent album/playlist ID
            
        Returns:
            Normalized track dict with: id, title, artist, album, duration, 
            track_number, cover_art_id, parent_id, metadata
        """
        # Ensure parent_id is in metadata for AudioBookShelf streaming
        metadata = raw_track.copy() if isinstance(raw_track, dict) else {}
        if parent_id:
            metadata["parent_id"] = parent_id
        
        return {
            "id": str(raw_track.get("id", "")),
            "title": raw_track.get("title") or raw_track.get("name", ""),
            "artist": raw_track.get("artist") or raw_track.get("artistName", ""),
            "album": raw_track.get("album") or raw_track.get("albumName", ""),
            "duration": raw_track.get("duration") or raw_track.get("runtimeTicks", 0) // 10000000 if raw_track.get("runtimeTicks") else 0,
            "track_number": raw_track.get("track") or raw_track.get("trackNumber") or raw_track.get("IndexNumber"),
            "cover_art_id": raw_track.get("coverArt") or raw_track.get("cover_art_id") or raw_track.get("thumb", ""),
            "parent_id": parent_id,
            "metadata": metadata
        }

