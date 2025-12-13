"""
Deezer Music Client
Implementation of BaseMusicClient for Deezer streaming service
"""

import logging
from typing import Dict, Any, List, Optional
from urllib.parse import urlencode
import httpx

from .base_client import BaseMusicClient

logger = logging.getLogger(__name__)


class DeezerClient(BaseMusicClient):
    """Client for Deezer streaming service"""
    
    API_BASE_URL = "https://api.deezer.com"
    
    def __init__(self, server_url: str, username: str, password: str, **kwargs):
        """
        Initialize Deezer client
        
        Args:
            server_url: Not used for Deezer (API is fixed), but kept for interface compatibility
            username: Not used for Deezer, but kept for interface compatibility
            password: Deezer access token (ARL token or OAuth access token)
        """
        super().__init__(server_url, username, password, **kwargs)
        self.access_token = password  # For Deezer, password field contains the access token
        self.user_id = None  # Will be fetched during connection test
    
    def _build_url(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Build Deezer API URL with authentication"""
        base_url = f"{self.API_BASE_URL}/{endpoint.lstrip('/')}"
        
        query_params = params or {}
        
        # Add access token if available
        if self.access_token:
            query_params["access_token"] = self.access_token
        
        if query_params:
            return f"{base_url}?{urlencode(query_params)}"
        return base_url
    
    async def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Make HTTP request to Deezer API"""
        try:
            url = self._build_url(endpoint, params)
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                
                # Deezer API returns error dict if there's an error
                if isinstance(data, dict) and "error" in data:
                    error_info = data.get("error", {})
                    error_message = error_info.get("message", "Unknown error") if isinstance(error_info, dict) else str(error_info)
                    error_code = error_info.get("code", "unknown") if isinstance(error_info, dict) else "unknown"
                    logger.error(f"Deezer API error: {error_code} - {error_message}")
                    return None
                
                return data
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error calling Deezer API: {e.response.status_code} - {e.response.text[:200]}")
            return None
        except Exception as e:
            logger.error(f"Error calling Deezer API: {e}")
            return None
    
    async def _paginate_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None, limit: int = 25) -> List[Dict[str, Any]]:
        """Make paginated request to Deezer API"""
        all_items = []
        params = params or {}
        params["limit"] = min(limit, 100)  # Deezer max is 100 per page
        index = 0
        
        while len(all_items) < limit:
            params["index"] = index
            data = await self._make_request(endpoint, params)
            
            if not data:
                break
            
            # Deezer returns items in 'data' field for paginated endpoints
            items = data.get("data", [])
            if not isinstance(items, list):
                items = [items] if items else []
            
            if not items:
                break
            
            all_items.extend(items[:limit - len(all_items)])
            
            # Check if there's a next page
            if len(items) < params["limit"] or len(all_items) >= limit:
                break
            
            index += len(items)
        
        return all_items
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test connection to Deezer API and get user info"""
        try:
            data = await self._make_request("user/me")
            
            if not data:
                return {"success": False, "error": "Failed to authenticate with Deezer API"}
            
            # Store user ID for future requests
            self.user_id = str(data.get("id", ""))
            
            user_name = data.get("name", "Unknown")
            return {
                "success": True,
                "message": f"Connected to Deezer as {user_name}",
                "user_id": self.user_id,
                "user_name": user_name
            }
        except Exception as e:
            logger.error(f"Failed to test Deezer connection: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_albums(self) -> List[Dict[str, Any]]:
        """Fetch user's favorite albums from Deezer"""
        if not self.user_id:
            # Try to get user ID first
            test_result = await self.test_connection()
            if not test_result.get("success"):
                return []
        
        try:
            albums = await self._paginate_request(f"user/{self.user_id}/albums", limit=500)
            
            normalized = []
            for album in albums:
                normalized_album = self.normalize_album(album)
                normalized.append(normalized_album)
            
            logger.info(f"Fetched {len(normalized)} favorite albums from Deezer")
            return normalized
        except Exception as e:
            logger.error(f"Failed to fetch Deezer albums: {e}")
            return []
    
    async def get_artists(self) -> List[Dict[str, Any]]:
        """Fetch user's favorite artists from Deezer"""
        if not self.user_id:
            test_result = await self.test_connection()
            if not test_result.get("success"):
                return []
        
        try:
            artists = await self._paginate_request(f"user/{self.user_id}/artists", limit=500)
            
            normalized = []
            for artist in artists:
                normalized_artist = self.normalize_artist(artist)
                normalized.append(normalized_artist)
            
            logger.info(f"Fetched {len(normalized)} favorite artists from Deezer")
            return normalized
        except Exception as e:
            logger.error(f"Failed to fetch Deezer artists: {e}")
            return []
    
    async def get_playlists(self) -> List[Dict[str, Any]]:
        """Fetch user's playlists from Deezer"""
        if not self.user_id:
            test_result = await self.test_connection()
            if not test_result.get("success"):
                return []
        
        try:
            playlists = await self._paginate_request(f"user/{self.user_id}/playlists", limit=500)
            
            normalized = []
            for playlist in playlists:
                normalized_playlist = self.normalize_playlist(playlist)
                normalized.append(normalized_playlist)
            
            logger.info(f"Fetched {len(normalized)} playlists from Deezer")
            return normalized
        except Exception as e:
            logger.error(f"Failed to fetch Deezer playlists: {e}")
            return []
    
    async def get_album_tracks(self, album_id: str) -> List[Dict[str, Any]]:
        """Fetch tracks for an album"""
        try:
            # First get album details
            album_data = await self._make_request(f"album/{album_id}")
            if not album_data:
                return []
            
            # Get tracks from album
            tracks = await self._paginate_request(f"album/{album_id}/tracks", limit=500)
            
            normalized = []
            for track in tracks:
                normalized_track = self.normalize_track(track, parent_id=album_id)
                normalized.append(normalized_track)
            
            return normalized
        except Exception as e:
            logger.error(f"Failed to fetch Deezer album tracks: {e}")
            return []
    
    async def get_playlist_tracks(self, playlist_id: str) -> List[Dict[str, Any]]:
        """Fetch tracks for a playlist"""
        try:
            tracks = await self._paginate_request(f"playlist/{playlist_id}/tracks", limit=500)
            
            normalized = []
            for track in tracks:
                normalized_track = self.normalize_track(track, parent_id=playlist_id)
                normalized.append(normalized_track)
            
            return normalized
        except Exception as e:
            logger.error(f"Failed to fetch Deezer playlist tracks: {e}")
            return []
    
    async def get_stream_url(self, track_id: str) -> Optional[str]:
        """Generate streaming URL for a track"""
        try:
            # Get track details to check if streaming is available
            track_data = await self._make_request(f"track/{track_id}")
            if not track_data:
                return None
            
            # For Deezer, we need to use the preview URL or full stream URL
            # Premium accounts can access full streams via special endpoints
            # For now, return the preview URL (30 seconds) or try to get full stream
            
            # Deezer streaming URLs require special handling
            # The API doesn't directly provide stream URLs, but we can construct them
            # or use the preview URL
            
            preview_url = track_data.get("preview")
            if preview_url:
                return preview_url
            
            # For premium accounts, we might need to use a different approach
            # This would require additional Deezer API endpoints or proxy
            logger.warning(f"Full stream URL not available for track {track_id}, using preview")
            return preview_url
        except Exception as e:
            logger.error(f"Failed to generate Deezer stream URL: {e}")
            return None
    
    async def search_tracks(self, query: str, limit: int = 25) -> List[Dict[str, Any]]:
        """Search for tracks in Deezer catalog"""
        try:
            tracks = await self._paginate_request("search/track", {"q": query}, limit=limit)
            
            normalized = []
            for track in tracks:
                normalized_track = self.normalize_track(track)
                normalized.append(normalized_track)
            
            return normalized
        except Exception as e:
            logger.error(f"Failed to search Deezer tracks: {e}")
            return []
    
    async def search_albums(self, query: str, limit: int = 25) -> List[Dict[str, Any]]:
        """Search for albums in Deezer catalog"""
        try:
            albums = await self._paginate_request("search/album", {"q": query}, limit=limit)
            
            normalized = []
            for album in albums:
                normalized_album = self.normalize_album(album)
                normalized.append(normalized_album)
            
            return normalized
        except Exception as e:
            logger.error(f"Failed to search Deezer albums: {e}")
            return []
    
    async def search_artists(self, query: str, limit: int = 25) -> List[Dict[str, Any]]:
        """Search for artists in Deezer catalog"""
        try:
            artists = await self._paginate_request("search/artist", {"q": query}, limit=limit)
            
            normalized = []
            for artist in artists:
                normalized_artist = self.normalize_artist(artist)
                normalized.append(normalized_artist)
            
            return normalized
        except Exception as e:
            logger.error(f"Failed to search Deezer artists: {e}")
            return []
    
    def normalize_album(self, raw_album: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Deezer album data to standard format"""
        artist_name = ""
        if "artist" in raw_album:
            if isinstance(raw_album["artist"], dict):
                artist_name = raw_album["artist"].get("name", "")
            else:
                artist_name = str(raw_album["artist"])
        
        return {
            "id": str(raw_album.get("id", "")),
            "title": raw_album.get("title", ""),
            "artist": artist_name,
            "cover_art_id": raw_album.get("cover_medium") or raw_album.get("cover", ""),
            "metadata": raw_album
        }
    
    def normalize_artist(self, raw_artist: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Deezer artist data to standard format"""
        return {
            "id": str(raw_artist.get("id", "")),
            "name": raw_artist.get("name", ""),
            "metadata": raw_artist
        }
    
    def normalize_playlist(self, raw_playlist: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Deezer playlist data to standard format"""
        return {
            "id": str(raw_playlist.get("id", "")),
            "name": raw_playlist.get("title", ""),
            "track_count": raw_playlist.get("nb_tracks", 0),
            "metadata": raw_playlist
        }
    
    def normalize_track(self, raw_track: Dict[str, Any], parent_id: Optional[str] = None) -> Dict[str, Any]:
        """Normalize Deezer track data to standard format"""
        artist_name = ""
        if "artist" in raw_track:
            if isinstance(raw_track["artist"], dict):
                artist_name = raw_track["artist"].get("name", "")
            else:
                artist_name = str(raw_track["artist"])
        
        album_name = ""
        if "album" in raw_track:
            if isinstance(raw_track["album"], dict):
                album_name = raw_track["album"].get("title", "")
            else:
                album_name = str(raw_track["album"])
        
        # Deezer duration is in seconds
        duration = raw_track.get("duration", 0)
        
        return {
            "id": str(raw_track.get("id", "")),
            "title": raw_track.get("title", ""),
            "artist": artist_name,
            "album": album_name,
            "duration": duration,
            "track_number": raw_track.get("track_position", 0),
            "cover_art_id": raw_track.get("album", {}).get("cover_medium", "") if isinstance(raw_track.get("album"), dict) else raw_track.get("cover_medium", ""),
            "parent_id": parent_id,
            "metadata": raw_track
        }

