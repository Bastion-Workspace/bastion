"""
SubSonic Music Client
Implementation of BaseMusicClient for SubSonic-compatible servers
"""

import logging
import secrets
import hashlib
from typing import Dict, Any, List, Optional
from urllib.parse import urlencode
import httpx

from .base_client import BaseMusicClient

logger = logging.getLogger(__name__)


class SubSonicClient(BaseMusicClient):
    """Client for SubSonic-compatible music servers"""
    
    def __init__(self, server_url: str, username: str, password: str, auth_type: str = "password", **kwargs):
        """
        Initialize SubSonic client
        
        Args:
            server_url: SubSonic server URL
            username: SubSonic username
            password: SubSonic password or token
            auth_type: Authentication type ('password' or 'token')
        """
        super().__init__(server_url, username, password, **kwargs)
        self.auth_type = auth_type
    
    def _build_url(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        auth_type: Optional[str] = None
    ) -> str:
        """Build SubSonic API URL with authentication"""
        auth_method = auth_type or self.auth_type
        
        # Build base URL
        base_url = f"{self.server_url}/rest/{endpoint}"
        
        # Prepare parameters
        query_params = params or {}
        query_params["u"] = self.username
        query_params["v"] = "1.16.0"  # SubSonic API version
        query_params["c"] = "Bastion"  # Client name
        
        # Add authentication
        if auth_method == "token":
            # Token-based auth (preferred by Navidrome and modern servers)
            salt = secrets.token_hex(6)
            token_string = f"{self.password}{salt}"
            token = hashlib.md5(token_string.encode()).hexdigest()
            query_params["t"] = token
            query_params["s"] = salt
        else:
            # Password-based auth
            password_hash = hashlib.md5(self.password.encode()).hexdigest()
            query_params["p"] = password_hash
        
        query_params["f"] = "json"  # Response format
        
        return f"{base_url}?{urlencode(query_params)}"
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test connection to SubSonic server"""
        # Try both auth methods if needed
        auth_methods = []
        if self.auth_type == "token":
            auth_methods = ["token", "password"]
        else:
            auth_methods = ["password", "token"]
        
        last_error = None
        
        for auth_method in auth_methods:
            try:
                ping_url = self._build_url("ping", auth_type=auth_method)
                
                logger.info(f"Testing SubSonic connection with {auth_method} authentication")
                
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(ping_url)
                    response.raise_for_status()
                    data = response.json()
                    
                    subsonic_response = data.get("subsonic-response", {})
                    status = subsonic_response.get("status")
                    
                    if status == "ok":
                        return {"success": True, "auth_method_used": auth_method}
                    else:
                        error_info = subsonic_response.get("error", {})
                        if isinstance(error_info, dict):
                            error_code = error_info.get("code", "unknown")
                            error_message = error_info.get("message", "Unknown error")
                            last_error = f"SubSonic error {error_code}: {error_message}"
                            
                            if error_code == 40 and len(auth_methods) > 1 and auth_method != auth_methods[-1]:
                                logger.info(f"{auth_method} auth failed with error 40, trying next method...")
                                continue
                
                return {"success": False, "error": last_error or "Unknown SubSonic error"}
                
            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
                if e.response.status_code == 401 and len(auth_methods) > 1 and auth_method != auth_methods[-1]:
                    logger.info(f"{auth_method} auth failed with 401, trying next method...")
                    continue
            except httpx.TimeoutException:
                last_error = "Connection timeout"
                break
            except Exception as e:
                last_error = f"Connection error: {str(e)}"
                break
        
        return {"success": False, "error": last_error or "Connection failed"}
    
    async def get_albums(self) -> List[Dict[str, Any]]:
        """Fetch all albums from SubSonic server with pagination"""
        logger.info("SubSonicClient.get_albums() called")
        try:
            all_albums = []
            batch_size = 500  # SubSonic API typically limits to 500 per request
            offset = 0
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                while True:
                    albums_url = self._build_url("getAlbumList2", {
                        "type": "alphabeticalByName", 
                        "size": batch_size,
                        "offset": offset
                    })
                    logger.info(f"Fetching albums batch from SubSonic (offset: {offset}, size: {batch_size})")
                    
                    response = await client.get(albums_url)
                    logger.info(f"SubSonic albums API response status: {response.status_code}")
                    response.raise_for_status()
                    data = response.json()
                    
                    subsonic_response = data.get("subsonic-response", {})
                    
                    if subsonic_response.get("status") != "ok":
                        error_info = subsonic_response.get("error", {})
                        error_msg = error_info.get("message", "Unknown error") if isinstance(error_info, dict) else str(error_info)
                        error_code = error_info.get("code", "unknown") if isinstance(error_info, dict) else "unknown"
                        logger.error(f"SubSonic API error when fetching albums: {error_code} - {error_msg}")
                        break
                    
                    album_list2 = subsonic_response.get("albumList2", {})
                    album_list = album_list2.get("album", [])
                    if not isinstance(album_list, list):
                        album_list = [album_list] if album_list else []
                    
                    batch_count = len(album_list)
                    logger.info(f"SubSonic returned {batch_count} albums in this batch (offset: {offset})")
                    
                    # Normalize and add to collection
                    for album in album_list:
                        normalized = self.normalize_album(album)
                        all_albums.append(normalized)
                    
                    # Check if we got fewer than requested - means we're done
                    if batch_count < batch_size:
                        logger.info(f"Received {batch_count} albums (less than {batch_size}), pagination complete")
                        break
                    
                    # Move to next batch
                    offset += batch_size
                    
                    # Safety check to prevent infinite loops
                    if offset > 50000:
                        logger.warning(f"Reached safety limit of 50,000 albums, stopping pagination")
                        break
                
                logger.info(f"Total albums fetched from SubSonic: {len(all_albums)}")
                return all_albums
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching albums: {e.response.status_code} - {e.response.text[:500]}")
            return []
        except Exception as e:
            logger.error(f"Failed to fetch albums: {e}", exc_info=True)
            return []
    
    async def get_artists(self) -> List[Dict[str, Any]]:
        """Fetch all artists from SubSonic server"""
        try:
            artists_url = self._build_url("getArtists")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(artists_url)
                response.raise_for_status()
                data = response.json()
                
                subsonic_response = data.get("subsonic-response", {})
                if subsonic_response.get("status") != "ok":
                    error_info = subsonic_response.get("error", {})
                    error_msg = error_info.get("message", "Unknown error") if isinstance(error_info, dict) else str(error_info)
                    logger.error(f"SubSonic API error when fetching artists: {error_msg}")
                    return []
                
                artists_index = subsonic_response.get("artists", {}).get("index", [])
                if not isinstance(artists_index, list):
                    artists_index = [artists_index] if artists_index else []
                
                logger.info(f"SubSonic returned {len(artists_index)} artist index entries")
                
                artists = []
                for index_entry in artists_index:
                    artist_list = index_entry.get("artist", [])
                    if not isinstance(artist_list, list):
                        artist_list = [artist_list] if artist_list else []
                    
                    for artist in artist_list:
                        normalized = self.normalize_artist(artist)
                        artists.append(normalized)
                
                logger.info(f"Normalized {len(artists)} artists")
                return artists
        except Exception as e:
            logger.error(f"Failed to fetch artists: {e}", exc_info=True)
            return []
    
    async def get_playlists(self) -> List[Dict[str, Any]]:
        """Fetch all playlists from SubSonic server"""
        try:
            playlists_url = self._build_url("getPlaylists")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(playlists_url)
                response.raise_for_status()
                data = response.json()
                
                subsonic_response = data.get("subsonic-response", {})
                if subsonic_response.get("status") != "ok":
                    error_info = subsonic_response.get("error", {})
                    error_msg = error_info.get("message", "Unknown error") if isinstance(error_info, dict) else str(error_info)
                    logger.error(f"SubSonic API error when fetching playlists: {error_msg}")
                    return []
                
                playlist_list = subsonic_response.get("playlists", {}).get("playlist", [])
                if not isinstance(playlist_list, list):
                    playlist_list = [playlist_list] if playlist_list else []
                
                logger.info(f"SubSonic returned {len(playlist_list)} raw playlists")
                
                playlists = []
                for playlist in playlist_list:
                    normalized = self.normalize_playlist(playlist)
                    playlists.append(normalized)
                
                logger.info(f"Normalized {len(playlists)} playlists")
                return playlists
        except Exception as e:
            logger.error(f"Failed to fetch playlists: {e}", exc_info=True)
            return []
    
    async def get_album_tracks(self, album_id: str) -> List[Dict[str, Any]]:
        """Fetch tracks for an album"""
        try:
            album_url = self._build_url("getAlbum", {"id": album_id})
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(album_url)
                response.raise_for_status()
                data = response.json()
                
                album_data = data.get("subsonic-response", {}).get("album", {})
                track_list = album_data.get("song", [])
                if not isinstance(track_list, list):
                    track_list = [track_list] if track_list else []
                
                tracks = []
                for track in track_list:
                    normalized = self.normalize_track(track, parent_id=album_id)
                    tracks.append(normalized)
                
                return tracks
        except Exception as e:
            logger.error(f"Failed to fetch album tracks: {e}")
            return []
    
    async def get_playlist_tracks(self, playlist_id: str) -> List[Dict[str, Any]]:
        """Fetch tracks for a playlist"""
        try:
            playlist_url = self._build_url("getPlaylist", {"id": playlist_id})
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(playlist_url)
                response.raise_for_status()
                data = response.json()
                
                playlist_data = data.get("subsonic-response", {}).get("playlist", {})
                track_list = playlist_data.get("entry", [])
                if not isinstance(track_list, list):
                    track_list = [track_list] if track_list else []
                
                tracks = []
                for track in track_list:
                    normalized = self.normalize_track(track, parent_id=playlist_id)
                    tracks.append(normalized)
                
                return tracks
        except Exception as e:
            logger.error(f"Failed to fetch playlist tracks: {e}")
            return []
    
    async def get_stream_url(self, track_id: str) -> Optional[str]:
        """Generate authenticated stream URL for a track"""
        try:
            stream_url = self._build_url("stream", {"id": track_id})
            return stream_url
        except Exception as e:
            logger.error(f"Failed to generate stream URL: {e}")
            return None
    
    async def add_to_playlist(self, playlist_id: str, track_ids: List[str]) -> Dict[str, Any]:
        """Add tracks to a playlist"""
        try:
            params = {
                "playlistId": playlist_id,
                "songIdToAdd": track_ids  # SubSonic API accepts multiple songIdToAdd params
            }
            
            update_url = self._build_url("updatePlaylist", params)
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(update_url)
                response.raise_for_status()
                data = response.json()
                
                subsonic_response = data.get("subsonic-response", {})
                status = subsonic_response.get("status")
                
                if status == "ok":
                    logger.info(f"Successfully added {len(track_ids)} tracks to playlist {playlist_id}")
                    return {"success": True}
                else:
                    error = subsonic_response.get("error", {})
                    logger.error(f"Failed to add tracks: {error}")
                    return {"success": False, "error": error.get("message", "Unknown error")}
                    
        except Exception as e:
            logger.error(f"Failed to add tracks to playlist: {e}")
            return {"success": False, "error": str(e)}
    
    async def remove_from_playlist(self, playlist_id: str, track_indices: List[int]) -> Dict[str, Any]:
        """Remove tracks from a playlist by their index positions"""
        try:
            params = {
                "playlistId": playlist_id,
                "songIndexToRemove": track_indices  # SubSonic API uses track indices, not IDs
            }
            
            update_url = self._build_url("updatePlaylist", params)
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(update_url)
                response.raise_for_status()
                data = response.json()
                
                subsonic_response = data.get("subsonic-response", {})
                status = subsonic_response.get("status")
                
                if status == "ok":
                    logger.info(f"Successfully removed {len(track_indices)} tracks from playlist {playlist_id}")
                    return {"success": True}
                else:
                    error = subsonic_response.get("error", {})
                    logger.error(f"Failed to remove tracks: {error}")
                    return {"success": False, "error": error.get("message", "Unknown error")}
                    
        except Exception as e:
            logger.error(f"Failed to remove tracks from playlist: {e}")
            return {"success": False, "error": str(e)}

