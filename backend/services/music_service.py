"""
Music Service
SubSonic-compatible music streaming service with caching
"""

import logging
import base64
import secrets
import hashlib
import time
import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from urllib.parse import urlencode, urljoin, urlparse
import httpx
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

from services.database_manager.database_helpers import fetch_one, fetch_all, execute
from services.music_clients.client_factory import MusicClientFactory
from config import settings

logger = logging.getLogger(__name__)


class MusicService:
    """Service for managing SubSonic-compatible music streaming"""
    
    def __init__(self):
        self._master_key = None
        self._fernet = None
        self._initialized = False
    
    def _initialize_encryption(self):
        """Initialize encryption with master key from settings"""
        if self._initialized:
            return
        
        # Use SECRET_KEY as master key for encryption
        master_key_str = settings.SECRET_KEY
        
        if not master_key_str:
            logger.warning("SECRET_KEY not set! Generating temporary key...")
            self._master_key = Fernet.generate_key()
        else:
            try:
                # Derive a Fernet key from SECRET_KEY using PBKDF2
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=b'music_service_salt',
                    iterations=100000,
                    backend=default_backend()
                )
                derived_key = base64.urlsafe_b64encode(kdf.derive(master_key_str.encode()))
                self._master_key = derived_key
                self._fernet = Fernet(self._master_key)
            except Exception as e:
                logger.error(f"Failed to initialize encryption: {e}")
                self._master_key = Fernet.generate_key()
                self._fernet = Fernet(self._master_key)
        
        self._initialized = True
    
    def _encrypt_password(self, password: str, salt: str) -> str:
        """Encrypt password using Fernet"""
        self._initialize_encryption()
        encrypted_bytes = self._fernet.encrypt(password.encode('utf-8'))
        return base64.b64encode(encrypted_bytes).decode('utf-8')
    
    def _decrypt_password(self, encrypted_password: str) -> str:
        """Decrypt password using Fernet"""
        self._initialize_encryption()
        encrypted_bytes = base64.b64decode(encrypted_password.encode('utf-8'))
        decrypted_bytes = self._fernet.decrypt(encrypted_bytes)
        return decrypted_bytes.decode('utf-8')
    
    def _generate_salt(self) -> str:
        """Generate a random salt"""
        return secrets.token_urlsafe(32)
    
    async def save_config(
        self,
        user_id: str,
        server_url: str,
        username: str,
        password: str,
        auth_type: str = "password",
        service_type: str = "subsonic",
        service_name: Optional[str] = None
    ) -> bool:
        """Save music service configuration with encrypted password"""
        try:
            salt = self._generate_salt()
            encrypted_password = self._encrypt_password(password, salt)
            
            # Check if config exists for this user and service type
            rls_context = {"user_id": user_id}
            existing = await fetch_one(
                "SELECT id FROM music_service_configs WHERE user_id = $1 AND service_type = $2",
                user_id, service_type,
                rls_context=rls_context
            )
            
            if existing:
                # Update existing config
                await execute(
                    """UPDATE music_service_configs 
                    SET server_url = $1, username = $2, encrypted_password = $3, 
                        salt = $4, auth_type = $5, service_name = $6, updated_at = NOW()
                    WHERE user_id = $7 AND service_type = $8""",
                    server_url, username, encrypted_password, salt, auth_type, service_name, user_id, service_type,
                    rls_context=rls_context
                )
            else:
                # Insert new config
                await execute(
                    """INSERT INTO music_service_configs 
                    (user_id, server_url, username, encrypted_password, salt, auth_type, service_type, service_name, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW(), NOW())""",
                    user_id, server_url, username, encrypted_password, salt, auth_type, service_type, service_name,
                    rls_context=rls_context
                )
            
            logger.info(f"Saved music service config for user {user_id} (service: {service_type})")
            return True
        except Exception as e:
            logger.error(f"Failed to save music service config: {e}")
            return False
    
    async def get_config(self, user_id: str, service_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get music service configuration (without password)"""
        try:
            rls_context = {"user_id": user_id}
            if service_type:
                # Get specific service config
                row = await fetch_one(
                    """SELECT server_url, username, auth_type, service_type, service_name, is_active,
                       (SELECT last_sync_at FROM music_cache_metadata WHERE user_id = $1 AND service_type = $2) as last_sync_at,
                       (SELECT sync_status FROM music_cache_metadata WHERE user_id = $1 AND service_type = $2) as sync_status,
                       (SELECT total_albums FROM music_cache_metadata WHERE user_id = $1 AND service_type = $2) as total_albums,
                       (SELECT total_artists FROM music_cache_metadata WHERE user_id = $1 AND service_type = $2) as total_artists,
                       (SELECT total_playlists FROM music_cache_metadata WHERE user_id = $1 AND service_type = $2) as total_playlists,
                       (SELECT total_tracks FROM music_cache_metadata WHERE user_id = $1 AND service_type = $2) as total_tracks
                       FROM music_service_configs WHERE user_id = $1 AND service_type = $2""",
                    user_id, service_type,
                    rls_context=rls_context
                )
            else:
                # Get first active config (backward compatibility)
                row = await fetch_one(
                    """SELECT server_url, username, auth_type, service_type, service_name, is_active,
                       (SELECT last_sync_at FROM music_cache_metadata WHERE user_id = $1 LIMIT 1) as last_sync_at,
                       (SELECT sync_status FROM music_cache_metadata WHERE user_id = $1 LIMIT 1) as sync_status,
                       (SELECT total_albums FROM music_cache_metadata WHERE user_id = $1 LIMIT 1) as total_albums,
                       (SELECT total_artists FROM music_cache_metadata WHERE user_id = $1 LIMIT 1) as total_artists,
                       (SELECT total_playlists FROM music_cache_metadata WHERE user_id = $1 LIMIT 1) as total_playlists,
                       (SELECT total_tracks FROM music_cache_metadata WHERE user_id = $1 LIMIT 1) as total_tracks
                       FROM music_service_configs WHERE user_id = $1 AND (is_active IS NULL OR is_active = TRUE)
                       ORDER BY created_at LIMIT 1""",
                    user_id,
                    rls_context=rls_context
                )
            
            if not row:
                return None
            
            return {
                "server_url": row["server_url"],
                "username": row["username"],
                "auth_type": row["auth_type"],
                "service_type": row.get("service_type", "subsonic"),
                "service_name": row.get("service_name"),
                "is_active": row.get("is_active", True),
                "has_config": True,
                "last_sync_at": row.get("last_sync_at"),
                "sync_status": row.get("sync_status"),
                "total_albums": row.get("total_albums") or 0,
                "total_artists": row.get("total_artists") or 0,
                "total_playlists": row.get("total_playlists") or 0,
                "total_tracks": row.get("total_tracks") or 0
            }
        except Exception as e:
            logger.error(f"Failed to get music service config: {e}")
            return None
    
    async def get_user_sources(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all configured media sources for a user"""
        try:
            rls_context = {"user_id": user_id}
            rows = await fetch_all(
                """SELECT server_url, username, auth_type, service_type, service_name, is_active,
                   created_at, updated_at
                   FROM music_service_configs 
                   WHERE user_id = $1 AND (is_active IS NULL OR is_active = TRUE)
                   ORDER BY created_at""",
                user_id,
                rls_context=rls_context
            )
            
            sources = []
            for row in (rows or []):
                sources.append({
                    "server_url": row["server_url"],
                    "username": row["username"],
                    "auth_type": row["auth_type"],
                    "service_type": row.get("service_type", "subsonic"),
                    "service_name": row.get("service_name"),
                    "is_active": row.get("is_active", True),
                    "created_at": row.get("created_at"),
                    "updated_at": row.get("updated_at")
                })
            
            return sources
        except Exception as e:
            logger.error(f"Failed to get user sources: {e}")
            return []
    
    async def get_credentials(self, user_id: str, service_type: Optional[str] = None) -> Optional[Dict[str, str]]:
        """Get decrypted credentials for music service API calls"""
        try:
            rls_context = {"user_id": user_id}
            if service_type:
                row = await fetch_one(
                    """SELECT server_url, username, encrypted_password, auth_type, service_type 
                    FROM music_service_configs 
                    WHERE user_id = $1 AND service_type = $2 AND (is_active IS NULL OR is_active = TRUE)""",
                    user_id, service_type,
                    rls_context=rls_context
                )
            else:
                # Get first active config (backward compatibility)
                row = await fetch_one(
                    """SELECT server_url, username, encrypted_password, auth_type, service_type 
                    FROM music_service_configs 
                    WHERE user_id = $1 AND (is_active IS NULL OR is_active = TRUE)
                    ORDER BY created_at LIMIT 1""",
                    user_id,
                    rls_context=rls_context
                )
            
            if not row:
                return None
            
            password = self._decrypt_password(row["encrypted_password"])
            
            return {
                "server_url": row["server_url"],
                "username": row["username"],
                "password": password,
                "auth_type": row["auth_type"],
                "service_type": row.get("service_type", "subsonic")
            }
        except Exception as e:
            logger.error(f"Failed to get credentials: {e}")
            return None
    
    async def delete_config(self, user_id: str, service_type: Optional[str] = None) -> bool:
        """Delete music service configuration and cache"""
        try:
            rls_context = {"user_id": user_id}
            if service_type:
                # Delete specific service
                await execute(
                    "DELETE FROM music_service_configs WHERE user_id = $1 AND service_type = $2",
                    user_id, service_type,
                    rls_context=rls_context
                )
                await execute(
                    "DELETE FROM music_cache WHERE user_id = $1 AND service_type = $2",
                    user_id, service_type,
                    rls_context=rls_context
                )
                await execute(
                    "DELETE FROM music_cache_metadata WHERE user_id = $1 AND service_type = $2",
                    user_id, service_type,
                    rls_context=rls_context
                )
                logger.info(f"Deleted music service config for user {user_id} (service: {service_type})")
            else:
                # Delete all services (backward compatibility)
                await execute("DELETE FROM music_service_configs WHERE user_id = $1", user_id, rls_context=rls_context)
                await execute("DELETE FROM music_cache WHERE user_id = $1", user_id, rls_context=rls_context)
                await execute("DELETE FROM music_cache_metadata WHERE user_id = $1", user_id, rls_context=rls_context)
                logger.info(f"Deleted all music service configs for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete music service config: {e}")
            return False
    
    async def test_connection(self, user_id: str, service_type: Optional[str] = None) -> Dict[str, Any]:
        """Test connection to music server"""
        try:
            # Log which service type we're testing
            logger.info(f"Testing connection for service_type: {service_type}, user_id: {user_id}")
            
            creds = await self.get_credentials(user_id, service_type)
            if not creds:
                error_msg = f"No configuration found for service type: {service_type or 'default'}"
                logger.warning(error_msg)
                return {"success": False, "error": error_msg}
            
            actual_service_type = creds.get("service_type", "subsonic")
            logger.info(f"Creating {actual_service_type} client for connection test")
            
            # Create a fresh client instance for this test (ensures no state leakage)
            client = MusicClientFactory.create_client(
                service_type=actual_service_type,
                server_url=creds["server_url"],
                username=creds["username"],
                password=creds["password"],
                auth_type=creds.get("auth_type", "password")
            )
            
            if not client:
                error_msg = f"Unsupported service type: {actual_service_type}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
            
            # Test connection using client (each client instance is isolated)
            logger.info(f"Running test_connection() on {actual_service_type} client")
            result = await client.test_connection()
            
            if result.get("success"):
                logger.info(f"Connection test successful for {actual_service_type}")
            else:
                logger.warning(f"Connection test failed for {actual_service_type}: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            error_msg = f"Connection test failed: {str(e)}"
            logger.error(f"Connection test exception for service_type {service_type}: {e}", exc_info=True)
            return {"success": False, "error": error_msg}
    
    async def refresh_cache(self, user_id: str, service_type: Optional[str] = None) -> Dict[str, Any]:
        """Refresh music library cache from music server"""
        try:
            creds = await self.get_credentials(user_id, service_type)
            if not creds:
                return {"success": False, "error": "No configuration found"}
            
            service_type = creds.get("service_type", "subsonic")
            
            # Create client using factory
            client = MusicClientFactory.create_client(
                service_type=service_type,
                server_url=creds["server_url"],
                username=creds["username"],
                password=creds["password"],
                auth_type=creds["auth_type"]
            )
            
            if not client:
                return {"success": False, "error": f"Unsupported service type: {service_type}"}
            
            rls_context = {"user_id": user_id}
            
            # Update sync status (per-service)
            await execute(
                """INSERT INTO music_cache_metadata (user_id, service_type, sync_status, updated_at)
                VALUES ($1, $2, 'syncing', NOW())
                ON CONFLICT (user_id, service_type) DO UPDATE SET sync_status = 'syncing', updated_at = NOW()""",
                user_id, service_type,
                rls_context=rls_context
            )
            
            # Clear existing cache for this service
            await execute(
                "DELETE FROM music_cache WHERE user_id = $1 AND service_type = $2",
                user_id, service_type,
                rls_context=rls_context
            )
            
            logger.info(f"Starting cache refresh for user {user_id} using {service_type} service")
            logger.info(f"Client type: {type(client).__name__}, server: {creds['server_url']}")
            
            # Fetch albums, artists, playlists using client
            logger.info(f"Fetching albums from {service_type} client...")
            try:
                albums = await client.get_albums()
                logger.info(f"Albums fetched: {len(albums)}")
            except Exception as e:
                logger.error(f"Exception fetching albums: {e}", exc_info=True)
                albums = []
            
            logger.info(f"Fetching artists from {service_type} client...")
            try:
                artists = await client.get_artists()
                logger.info(f"Artists fetched: {len(artists)}")
            except Exception as e:
                logger.error(f"Exception fetching artists: {e}", exc_info=True)
                artists = []
            
            logger.info(f"Fetching playlists from {service_type} client...")
            try:
                playlists = await client.get_playlists()
                logger.info(f"Playlists fetched: {len(playlists)}")
            except Exception as e:
                logger.error(f"Exception fetching playlists: {e}", exc_info=True)
                playlists = []
            
            logger.info(f"Found {len(albums)} albums, {len(artists)} artists, {len(playlists)} playlists")
            
            # Store albums
            album_count = 0
            for album in albums:
                await execute(
                    """INSERT INTO music_cache 
                    (user_id, service_type, cache_type, item_id, title, artist, cover_art_id, metadata_json, created_at, updated_at)
                    VALUES ($1, $2, 'album', $3, $4, $5, $6, $7, NOW(), NOW())
                    ON CONFLICT (user_id, service_type, cache_type, item_id) DO UPDATE SET
                    title = $4, artist = $5, cover_art_id = $6, metadata_json = $7, updated_at = NOW()""",
                    user_id,
                    service_type,
                    album.get("id", ""),
                    album.get("title", ""),
                    album.get("artist", ""),
                    album.get("cover_art_id"),
                    json.dumps(album.get("metadata", {})),
                    rls_context=rls_context
                )
                album_count += 1
            
            # Store artists
            artist_count = 0
            for artist in artists:
                await execute(
                    """INSERT INTO music_cache 
                    (user_id, service_type, cache_type, item_id, title, metadata_json, created_at, updated_at)
                    VALUES ($1, $2, 'artist', $3, $4, $5, NOW(), NOW())
                    ON CONFLICT (user_id, service_type, cache_type, item_id) DO UPDATE SET
                    title = $4, metadata_json = $5, updated_at = NOW()""",
                    user_id,
                    service_type,
                    artist.get("id", ""),
                    artist.get("name", ""),
                    json.dumps(artist.get("metadata", {})),
                    rls_context=rls_context
                )
                artist_count += 1
            
            # Store playlists
            playlist_count = 0
            for playlist in playlists:
                await execute(
                    """INSERT INTO music_cache 
                    (user_id, service_type, cache_type, item_id, title, metadata_json, created_at, updated_at)
                    VALUES ($1, $2, 'playlist', $3, $4, $5, NOW(), NOW())
                    ON CONFLICT (user_id, service_type, cache_type, item_id) DO UPDATE SET
                    title = $4, metadata_json = $5, updated_at = NOW()""",
                    user_id,
                    service_type,
                    playlist.get("id", ""),
                    playlist.get("name", ""),
                    json.dumps(playlist.get("metadata", {})),
                    rls_context=rls_context
                )
                playlist_count += 1
            
            # Fetch tracks for each album (limit to avoid timeout)
            track_count = 0
            for album in albums[:100]:
                album_id = album.get("id")
                if not album_id:
                    continue
                
                try:
                    tracks = await client.get_album_tracks(album_id)
                    for track in tracks:
                        await execute(
                            """INSERT INTO music_cache 
                            (user_id, service_type, cache_type, item_id, parent_id, title, artist, album, 
                             duration, track_number, cover_art_id, metadata_json, created_at, updated_at)
                            VALUES ($1, $2, 'track', $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW(), NOW())
                            ON CONFLICT (user_id, service_type, cache_type, item_id) DO UPDATE SET
                            parent_id = $4, title = $5, artist = $6, album = $7, duration = $8, 
                            track_number = $9, cover_art_id = $10, metadata_json = $11, updated_at = NOW()""",
                            user_id,
                            service_type,
                            track.get("id", ""),
                            track.get("parent_id", album_id),
                            track.get("title", ""),
                            track.get("artist", ""),
                            track.get("album", ""),
                            track.get("duration"),
                            track.get("track_number"),
                            track.get("cover_art_id"),
                            json.dumps(track.get("metadata", {})),
                            rls_context=rls_context
                        )
                        track_count += 1
                except Exception as e:
                    logger.warning(f"Failed to fetch tracks for album {album_id}: {e}")
                    continue
            
            # Update sync metadata (per-service)
            await execute(
                """INSERT INTO music_cache_metadata 
                (user_id, service_type, last_sync_at, sync_status, total_albums, total_artists, total_playlists, total_tracks, updated_at)
                VALUES ($1, $2, NOW(), 'completed', $3, $4, $5, $6, NOW())
                ON CONFLICT (user_id, service_type) DO UPDATE SET
                last_sync_at = NOW(), sync_status = 'completed', 
                total_albums = $3, total_artists = $4, total_playlists = $5, total_tracks = $6, updated_at = NOW()""",
                user_id, service_type, album_count, artist_count, playlist_count, track_count,
                rls_context=rls_context
            )
            
            logger.info(f"Cache refresh completed: {album_count} albums, {artist_count} artists, {playlist_count} playlists, {track_count} tracks")
            
            return {
                "success": True,
                "albums": album_count,
                "artists": artist_count,
                "playlists": playlist_count,
                "tracks": track_count
            }
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"Failed to refresh cache: {e}")
            logger.error(f"Traceback: {error_trace}")
            rls_context = {"user_id": user_id}
            await execute(
                """INSERT INTO music_cache_metadata (user_id, service_type, sync_status, sync_error, updated_at)
                VALUES ($1, $2, 'failed', $3, NOW())
                ON CONFLICT (user_id, service_type) DO UPDATE SET sync_status = 'failed', sync_error = $3, updated_at = NOW()""",
                user_id, service_type, str(e),
                rls_context=rls_context
            )
            return {"success": False, "error": str(e)}
    
    async def get_tracks(self, user_id: str, parent_id: str, parent_type: str, service_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get tracks for an album or playlist"""
        try:
            if parent_type == "playlist":
                # For playlists, fetch from music server API
                creds = await self.get_credentials(user_id, service_type)
                if not creds:
                    return []
                
                service_type = creds.get("service_type", "subsonic")
                
                # Create client using factory
                client = MusicClientFactory.create_client(
                    service_type=service_type,
                    server_url=creds["server_url"],
                    username=creds["username"],
                    password=creds["password"],
                    auth_type=creds["auth_type"]
                )
                
                if not client:
                    logger.error(f"Unsupported service type: {service_type}")
                    return []
                
                tracks = await client.get_playlist_tracks(parent_id)
                
                # Convert to expected format
                result = []
                for track in tracks:
                    track_dict = {
                        "id": track.get("id", ""),
                        "title": track.get("title", ""),
                        "artist": track.get("artist", ""),
                        "album": track.get("album", ""),
                        "duration": track.get("duration"),
                        "track_number": track.get("track_number"),
                        "cover_art_id": track.get("cover_art_id"),
                        "metadata": track.get("metadata", {}),
                        "service_type": service_type  # Include service_type for frontend
                    }
                    # Include parent_id in metadata if available (for AudioBookShelf streaming)
                    if track.get("parent_id"):
                        if not isinstance(track_dict["metadata"], dict):
                            track_dict["metadata"] = {}
                        track_dict["metadata"]["parent_id"] = track.get("parent_id")
                    result.append(track_dict)
                return result
            else:
                # For albums, try cache first, then fall back to live API if no cached tracks
                rls_context = {"user_id": user_id}
                if service_type:
                    tracks_raw = await fetch_all(
                        """SELECT item_id as id, title, artist, album, duration, track_number, 
                           cover_art_id, metadata_json as metadata, service_type
                        FROM music_cache 
                        WHERE user_id = $1 AND service_type = $2 AND cache_type = 'track' AND parent_id = $3
                        ORDER BY track_number, title""",
                        user_id, service_type, parent_id,
                        rls_context=rls_context
                    )
                else:
                    tracks_raw = await fetch_all(
                        """SELECT item_id as id, title, artist, album, duration, track_number, 
                           cover_art_id, metadata_json as metadata, service_type
                        FROM music_cache 
                        WHERE user_id = $1 AND cache_type = 'track' AND parent_id = $2
                        ORDER BY track_number, title""",
                        user_id, parent_id,
                        rls_context=rls_context
                    )
                
                # If no cached tracks found, fetch live from API
                if not tracks_raw or len(tracks_raw) == 0:
                    logger.info(f"No cached tracks for album {parent_id}, fetching live from API")
                    creds = await self.get_credentials(user_id, service_type)
                    if creds:
                        service_type = creds.get("service_type", "subsonic")
                        client = MusicClientFactory.create_client(
                            service_type=service_type,
                            server_url=creds["server_url"],
                            username=creds["username"],
                            password=creds["password"],
                            auth_type=creds["auth_type"]
                        )
                        
                        if client:
                            tracks = await client.get_album_tracks(parent_id)
                            # Convert to expected format
                            result = []
                            for track in tracks:
                                result.append({
                                    "id": track.get("id", ""),
                                    "title": track.get("title", ""),
                                    "artist": track.get("artist", ""),
                                    "album": track.get("album", ""),
                                    "duration": track.get("duration"),
                                    "track_number": track.get("track_number"),
                                    "cover_art_id": track.get("cover_art_id"),
                                    "metadata": track.get("metadata", {}),
                                    "service_type": service_type  # Include service_type for frontend
                                })
                            return result
                    return []
                
                # Parse JSON strings back to dicts from cache
                tracks = []
                for track in tracks_raw:
                    track_dict = dict(track)
                    if isinstance(track_dict.get("metadata"), str):
                        try:
                            track_dict["metadata"] = json.loads(track_dict["metadata"])
                        except (json.JSONDecodeError, TypeError):
                            track_dict["metadata"] = {}
                    # Ensure service_type is set (fallback to provided service_type or None)
                    if not track_dict.get("service_type"):
                        track_dict["service_type"] = service_type
                    tracks.append(track_dict)
                
                return tracks
        except Exception as e:
            logger.error(f"Failed to get tracks: {e}")
            return []
    
    async def get_stream_url(self, user_id: str, track_id: str, service_type: Optional[str] = None, parent_id: Optional[str] = None) -> Optional[str]:
        """Generate authenticated stream URL for a track"""
        try:
            # If service_type is not provided, look up the track in cache to determine service_type
            rls_context = {"user_id": user_id}
            if not service_type:
                track_data = await fetch_one(
                    """SELECT service_type, metadata_json as metadata FROM music_cache 
                    WHERE user_id = $1 AND cache_type = 'track' AND item_id = $2""",
                    user_id, track_id,
                    rls_context=rls_context
                )
                if track_data:
                    service_type = track_data.get("service_type")
                    logger.info(f"Determined service_type '{service_type}' for track {track_id} from cache")
                    # Also extract parent_id from metadata if not provided
                    if not parent_id and track_data.get("metadata"):
                        metadata = track_data["metadata"]
                        if isinstance(metadata, str):
                            try:
                                metadata = json.loads(metadata)
                            except (json.JSONDecodeError, TypeError):
                                metadata = {}
                        parent_id = metadata.get("parent_id")
                        if parent_id:
                            logger.info(f"Extracted parent_id '{parent_id}' from track metadata")
                else:
                    logger.warning(f"Track {track_id} not found in cache, will default to subsonic")
            
            # If still no service_type, default to subsonic (backward compatibility)
            if not service_type:
                service_type = "subsonic"
            
            creds = await self.get_credentials(user_id, service_type)
            if not creds:
                logger.error(f"No credentials found for service_type: {service_type}")
                return None
            
            # Ensure we use the service_type from credentials (in case it differs)
            service_type = creds.get("service_type", service_type)
            
            # For AudioBookShelf, we need parent_id - try to get it from track metadata if not provided
            if service_type == "audiobookshelf" and not parent_id:
                # Try to get parent_id from cached track metadata
                track_data = await fetch_one(
                    """SELECT metadata_json as metadata FROM music_cache 
                    WHERE user_id = $1 AND service_type = $2 AND cache_type = 'track' AND item_id = $3""",
                    user_id, service_type, track_id,
                    rls_context=rls_context
                )
                if track_data and track_data.get("metadata"):
                    metadata = track_data["metadata"]
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except (json.JSONDecodeError, TypeError):
                            metadata = {}
                    parent_id = metadata.get("parent_id")
            
            # Create client using factory
            client = MusicClientFactory.create_client(
                service_type=service_type,
                server_url=creds["server_url"],
                username=creds["username"],
                password=creds["password"],
                auth_type=creds["auth_type"]
            )
            
            if not client:
                logger.error(f"Unsupported service type: {service_type}")
                return None
            
            # Pass parent_id to client if it's AudioBookShelf
            if service_type == "audiobookshelf" and hasattr(client, 'get_stream_url_with_parent'):
                stream_url = await client.get_stream_url_with_parent(track_id, parent_id)
            else:
                stream_url = await client.get_stream_url(track_id)
            return stream_url
        except Exception as e:
            logger.error(f"Failed to generate stream URL: {e}")
            return None
    
    async def search_tracks(self, user_id: str, query: str, service_type: str, limit: int = 25) -> List[Dict[str, Any]]:
        """Search for tracks in music service catalog"""
        try:
            if not service_type:
                logger.error("search_tracks requires service_type")
                return []
            
            creds = await self.get_credentials(user_id, service_type)
            if not creds:
                logger.error(f"No credentials found for service_type: {service_type}")
                return []
            
            # Create client using factory
            client = MusicClientFactory.create_client(
                service_type=service_type,
                server_url=creds.get("server_url", ""),
                username=creds.get("username", ""),
                password=creds.get("password", ""),
                auth_type=creds.get("auth_type", "password")
            )
            
            if not client:
                logger.error(f"Unsupported service type for search: {service_type}")
                return []
            
            # Call search method on client
            tracks = await client.search_tracks(query, limit)
            # Ensure service_type is included in each track
            for track in tracks:
                if "service_type" not in track:
                    track["service_type"] = service_type
            return tracks
        except Exception as e:
            logger.error(f"Failed to search tracks: {e}")
            return []
    
    async def search_albums(self, user_id: str, query: str, service_type: str, limit: int = 25) -> List[Dict[str, Any]]:
        """Search for albums in music service catalog"""
        try:
            if not service_type:
                logger.error("search_albums requires service_type")
                return []
            
            creds = await self.get_credentials(user_id, service_type)
            if not creds:
                logger.error(f"No credentials found for service_type: {service_type}")
                return []
            
            # Create client using factory
            client = MusicClientFactory.create_client(
                service_type=service_type,
                server_url=creds.get("server_url", ""),
                username=creds.get("username", ""),
                password=creds.get("password", ""),
                auth_type=creds.get("auth_type", "password")
            )
            
            if not client:
                logger.error(f"Unsupported service type for search: {service_type}")
                return []
            
            # Call search method on client
            albums = await client.search_albums(query, limit)
            return albums
        except Exception as e:
            logger.error(f"Failed to search albums: {e}")
            return []
    
    async def search_artists(self, user_id: str, query: str, service_type: str, limit: int = 25) -> List[Dict[str, Any]]:
        """Search for artists in music service catalog"""
        try:
            if not service_type:
                logger.error("search_artists requires service_type")
                return []
            
            creds = await self.get_credentials(user_id, service_type)
            if not creds:
                logger.error(f"No credentials found for service_type: {service_type}")
                return []
            
            # Create client using factory
            client = MusicClientFactory.create_client(
                service_type=service_type,
                server_url=creds.get("server_url", ""),
                username=creds.get("username", ""),
                password=creds.get("password", ""),
                auth_type=creds.get("auth_type", "password")
            )
            
            if not client:
                logger.error(f"Unsupported service type for search: {service_type}")
                return []
            
            # Call search method on client
            artists = await client.search_artists(query, limit)
            return artists
        except Exception as e:
            logger.error(f"Failed to search artists: {e}")
            return []
    
    async def get_library(self, user_id: str, service_type: Optional[str] = None) -> Dict[str, Any]:
        """Get cached music library"""
        try:
            # Require service_type to prevent accidentally combining data from multiple services
            if not service_type:
                logger.warning(f"get_library called without service_type for user {user_id} - returning empty library")
                return {
                    "albums": [],
                    "artists": [],
                    "playlists": [],
                    "last_sync_at": None
                }
            
            logger.info(f"Getting library for user {user_id}, service_type: {service_type}")
            
            rls_context = {"user_id": user_id}
            albums_raw = await fetch_all(
                """SELECT item_id as id, title, artist, cover_art_id, metadata_json as metadata
                FROM music_cache 
                WHERE user_id = $1 AND service_type = $2 AND cache_type = 'album'
                ORDER BY title""",
                user_id, service_type,
                rls_context=rls_context
            )
            
            # Parse JSON strings back to dicts
            albums = []
            for album in (albums_raw or []):
                album_dict = dict(album)
                if isinstance(album_dict.get("metadata"), str):
                    try:
                        album_dict["metadata"] = json.loads(album_dict["metadata"])
                    except (json.JSONDecodeError, TypeError):
                        album_dict["metadata"] = {}
                albums.append(album_dict)
            
            artists_raw = await fetch_all(
                """SELECT item_id as id, title as name, metadata_json as metadata
                FROM music_cache 
                WHERE user_id = $1 AND service_type = $2 AND cache_type = 'artist'
                ORDER BY title""",
                user_id, service_type,
                rls_context=rls_context
            )
            
            # Parse JSON strings back to dicts
            artists = []
            for artist in (artists_raw or []):
                artist_dict = dict(artist)
                if isinstance(artist_dict.get("metadata"), str):
                    try:
                        artist_dict["metadata"] = json.loads(artist_dict["metadata"])
                    except (json.JSONDecodeError, TypeError):
                        artist_dict["metadata"] = {}
                artists.append(artist_dict)
            
            playlists_raw = await fetch_all(
                """SELECT item_id as id, title as name, metadata_json as metadata
                FROM music_cache 
                WHERE user_id = $1 AND service_type = $2 AND cache_type = 'playlist'
                ORDER BY title""",
                user_id, service_type,
                rls_context=rls_context
            )
            
            # Parse JSON strings back to dicts
            playlists = []
            for playlist in (playlists_raw or []):
                playlist_dict = dict(playlist)
                if isinstance(playlist_dict.get("metadata"), str):
                    try:
                        playlist_dict["metadata"] = json.loads(playlist_dict["metadata"])
                    except (json.JSONDecodeError, TypeError):
                        playlist_dict["metadata"] = {}
                playlists.append(playlist_dict)
            
            metadata = await fetch_one(
                "SELECT last_sync_at FROM music_cache_metadata WHERE user_id = $1 AND service_type = $2",
                user_id, service_type,
                rls_context=rls_context
            )
            
            return {
                "albums": albums,
                "artists": artists,
                "playlists": playlists,
                "last_sync_at": metadata.get("last_sync_at") if metadata else None
            }
        except Exception as e:
            logger.error(f"Failed to get library: {e}")
            return {"albums": [], "artists": [], "playlists": [], "last_sync_at": None}
    
    async def get_albums_by_artist(self, user_id: str, artist_id: str, service_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get albums for a specific artist/author"""
        """Get albums for a specific artist"""
        try:
            rls_context = {"user_id": user_id}
            # First, get the artist name from the artist ID
            if service_type:
                artist = await fetch_one(
                    """SELECT title as name FROM music_cache 
                    WHERE user_id = $1 AND service_type = $2 AND cache_type = 'artist' AND item_id = $3""",
                    user_id, service_type, artist_id,
                    rls_context=rls_context
                )
            else:
                artist = await fetch_one(
                    """SELECT title as name FROM music_cache 
                    WHERE user_id = $1 AND cache_type = 'artist' AND item_id = $2""",
                    user_id, artist_id,
                    rls_context=rls_context
                )
            
            if not artist:
                logger.warning(f"Artist {artist_id} not found")
                return []
            
            artist_name = artist.get("name", "")
            if not artist_name:
                logger.warning(f"Artist {artist_id} has no name")
                return []
            
            logger.info(f"Getting albums for artist '{artist_name}' (ID: {artist_id})")
            
            # Get albums by matching artist name (case-insensitive)
            if service_type:
                albums_raw = await fetch_all(
                    """SELECT item_id as id, title, artist, cover_art_id, metadata_json as metadata
                    FROM music_cache 
                    WHERE user_id = $1 AND service_type = $2 AND cache_type = 'album' AND LOWER(artist) = LOWER($3)
                    ORDER BY title""",
                    user_id, service_type, artist_name,
                    rls_context=rls_context
                )
            else:
                albums_raw = await fetch_all(
                    """SELECT item_id as id, title, artist, cover_art_id, metadata_json as metadata
                    FROM music_cache 
                    WHERE user_id = $1 AND cache_type = 'album' AND LOWER(artist) = LOWER($2)
                    ORDER BY title""",
                    user_id, artist_name,
                    rls_context=rls_context
                )
            
            logger.info(f"Found {len(albums_raw)} albums for artist '{artist_name}'")
            
            # Parse JSON strings back to dicts
            albums = []
            for album in (albums_raw or []):
                album_dict = dict(album)
                if isinstance(album_dict.get("metadata"), str):
                    try:
                        album_dict["metadata"] = json.loads(album_dict["metadata"])
                    except (json.JSONDecodeError, TypeError):
                        album_dict["metadata"] = {}
                albums.append(album_dict)
            
            return albums
        except Exception as e:
            logger.error(f"Failed to get albums by artist: {e}")
            return []
    
    async def get_series_by_author(self, user_id: str, author_id: str, service_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get series for a specific author (Audiobookshelf only)"""
        try:
            if service_type != "audiobookshelf":
                # For non-Audiobookshelf services, return empty (no series concept)
                return []
            
            rls_context = {"user_id": user_id}
            # Get the author name from the author ID
            artist = await fetch_one(
                """SELECT title as name FROM music_cache 
                WHERE user_id = $1 AND service_type = $2 AND cache_type = 'artist' AND item_id = $3""",
                user_id, service_type, author_id,
                rls_context=rls_context
            )
            
            if not artist:
                logger.warning(f"Author {author_id} not found")
                return []
            
            author_name = artist.get("name", "")
            if not author_name:
                return []
            
            # Get all albums by this author
            albums_raw = await fetch_all(
                """SELECT item_id as id, title, artist, cover_art_id, metadata_json as metadata
                FROM music_cache 
                WHERE user_id = $1 AND service_type = $2 AND cache_type = 'album' AND artist = $3
                ORDER BY title""",
                user_id, service_type, author_name,
                rls_context=rls_context
            )
            
            # Extract unique series from albums
            series_map = {}
            series_extracted_count = 0
            for album in (albums_raw or []):
                album_dict = dict(album)
                metadata = album_dict.get("metadata", {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except (json.JSONDecodeError, TypeError):
                        metadata = {}
                
                # Get series from metadata
                # AudioBookShelf stores metadata in album["metadata"]["media"]["metadata"]
                # The structure is: album -> metadata (JSON) -> media -> metadata -> series
                # Series can be in various formats:
                # - Array of objects: "series": [{"name": "Series Name", "sequence": "1"}]
                # - Direct string: "series": "Series Name"
                # - Array of strings: "series": ["Series Name"]
                # - Series name field: "seriesName": "Series Name"
                series_name = None
                
                # Check if metadata has media.metadata structure (AudioBookShelf format)
                media_metadata = None
                if metadata and isinstance(metadata, dict):
                    if metadata.get("media") and isinstance(metadata["media"], dict):
                        media_metadata = metadata["media"].get("metadata", {})
                
                # Use media_metadata if available, otherwise fallback to metadata
                search_metadata = media_metadata if media_metadata else metadata
                
                if search_metadata:
                    # Check direct series field
                    series = search_metadata.get("series")
                    if series:
                        if isinstance(series, str):
                            series_name = series
                        elif isinstance(series, list) and len(series) > 0:
                            # Handle array format
                            first_series = series[0]
                            if isinstance(first_series, dict):
                                # Array of objects: get name field
                                series_name = first_series.get("name") or first_series.get("series")
                            elif isinstance(first_series, str):
                                # Array of strings: use first string
                                series_name = first_series
                    
                    # Fallback to seriesName field
                    if not series_name:
                        series_name = search_metadata.get("seriesName")
                    
                    # Check nested metadata (some APIs nest metadata) - only if we're not already in media_metadata
                    if not series_name and not media_metadata:
                        nested_metadata = search_metadata.get("metadata", {})
                        if isinstance(nested_metadata, dict):
                            nested_series = nested_metadata.get("series")
                            if nested_series:
                                if isinstance(nested_series, str):
                                    series_name = nested_series
                                elif isinstance(nested_series, list) and len(nested_series) > 0:
                                    first_nested = nested_series[0]
                                    if isinstance(first_nested, dict):
                                        series_name = first_nested.get("name") or first_nested.get("series")
                                    elif isinstance(first_nested, str):
                                        series_name = first_nested
                            if not series_name:
                                series_name = nested_metadata.get("seriesName")
                    
                    # Clean up series name
                    if series_name:
                        series_name = str(series_name).strip()
                        if not series_name or series_name.lower() == "unknown":
                            series_name = None
                
                if series_name and series_name.strip():
                    series_extracted_count += 1
                    series_clean = series_name.strip()
                    if series_clean not in series_map:
                        # Count books in this series
                        series_books = [a for a in albums_raw if self._album_in_series(a, series_clean)]
                        # Create series ID (avoid backslashes in f-string)
                        series_id_base = series_clean.lower().replace(' ', '_').replace(',', '').replace('.', '').replace('&', 'and').replace("'", "").replace('"', '')[:50]
                        series_map[series_clean] = {
                            "id": f"series_{series_id_base}",
                            "name": series_clean,
                            "author": author_name,
                            "book_count": len(series_books),
                            "metadata": {"series": series_clean, "author": author_name}
                        }
            
            logger.info(f"Extracted {len(series_map)} unique series from {len(albums_raw)} books ({series_extracted_count} books with series)")
            return list(series_map.values())
        except Exception as e:
            logger.error(f"Failed to get series by author: {e}")
            return []
    
    def _album_in_series(self, album: Dict[str, Any], series_name: str) -> bool:
        """Check if an album belongs to a series"""
        try:
            metadata = album.get("metadata", {})
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            
            # Check if metadata has media.metadata structure (AudioBookShelf format)
            media_metadata = None
            if metadata and isinstance(metadata, dict):
                if metadata.get("media") and isinstance(metadata["media"], dict):
                    media_metadata = metadata["media"].get("metadata", {})
            
            # Use media_metadata if available, otherwise fallback to metadata
            search_metadata = media_metadata if media_metadata else metadata
            
            if search_metadata:
                # Extract series using same logic as get_series_by_author
                series = search_metadata.get("series")
                album_series = None
                
                if series:
                    if isinstance(series, str):
                        album_series = series
                    elif isinstance(series, list) and len(series) > 0:
                        first_series = series[0]
                        if isinstance(first_series, dict):
                            album_series = first_series.get("name") or first_series.get("series")
                        elif isinstance(first_series, str):
                            album_series = first_series
                
                if not album_series:
                    album_series = search_metadata.get("seriesName")
                
                # Check nested metadata - only if we're not already in media_metadata
                if not album_series and not media_metadata:
                    nested_metadata = search_metadata.get("metadata", {})
                    if isinstance(nested_metadata, dict):
                        nested_series = nested_metadata.get("series")
                        if nested_series:
                            if isinstance(nested_series, str):
                                album_series = nested_series
                            elif isinstance(nested_series, list) and len(nested_series) > 0:
                                first_nested = nested_series[0]
                                if isinstance(first_nested, dict):
                                    album_series = first_nested.get("name") or first_nested.get("series")
                                elif isinstance(first_nested, str):
                                    album_series = first_nested
                        if not album_series:
                            album_series = nested_metadata.get("seriesName")
                
                if album_series:
                    album_series = str(album_series).strip()
                    return album_series.lower() == series_name.strip().lower()
            return False
        except Exception:
            return False
    
    async def get_albums_by_series(self, user_id: str, series_name: str, author_name: str, service_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get albums (books) for a specific series"""
        try:
            if service_type != "audiobookshelf":
                return []
            
            rls_context = {"user_id": user_id}
            # Get all albums by this author
            albums_raw = await fetch_all(
                """SELECT item_id as id, title, artist, cover_art_id, metadata_json as metadata
                FROM music_cache 
                WHERE user_id = $1 AND service_type = $2 AND cache_type = 'album' AND artist = $3
                ORDER BY title""",
                user_id, service_type, author_name,
                rls_context=rls_context
            )
            
            # Filter albums that belong to this series
            albums = []
            for album in (albums_raw or []):
                if self._album_in_series(album, series_name):
                    album_dict = dict(album)
                    if isinstance(album_dict.get("metadata"), str):
                        try:
                            album_dict["metadata"] = json.loads(album_dict["metadata"])
                        except (json.JSONDecodeError, TypeError):
                            album_dict["metadata"] = {}
                    albums.append(album_dict)
            
            return albums
        except Exception as e:
            logger.error(f"Failed to get albums by series: {e}")
            return []
    
    async def add_to_playlist(
        self, 
        user_id: str, 
        playlist_id: str, 
        track_ids: List[str], 
        service_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Add tracks to a playlist"""
        try:
            creds = await self.get_credentials(user_id, service_type)
            if not creds:
                return {"success": False, "error": "No music service configured"}
            
            service_type = creds.get("service_type", "subsonic")
            
            client = MusicClientFactory.create_client(
                service_type=service_type,
                server_url=creds["server_url"],
                username=creds["username"],
                password=creds["password"],
                auth_type=creds["auth_type"]
            )
            
            if not client:
                return {"success": False, "error": f"Unsupported service type: {service_type}"}
            
            result = await client.add_to_playlist(playlist_id, track_ids)
            return result
            
        except Exception as e:
            logger.error(f"Failed to add tracks to playlist: {e}")
            return {"success": False, "error": str(e)}
    
    async def remove_from_playlist(
        self, 
        user_id: str, 
        playlist_id: str, 
        track_ids: List[str], 
        service_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Remove tracks from a playlist by track IDs"""
        try:
            creds = await self.get_credentials(user_id, service_type)
            if not creds:
                return {"success": False, "error": "No music service configured"}
            
            service_type = creds.get("service_type", "subsonic")
            
            client = MusicClientFactory.create_client(
                service_type=service_type,
                server_url=creds["server_url"],
                username=creds["username"],
                password=creds["password"],
                auth_type=creds["auth_type"]
            )
            
            if not client:
                return {"success": False, "error": f"Unsupported service type: {service_type}"}
            
            # Get current playlist tracks to find indices
            playlist_tracks = await client.get_playlist_tracks(playlist_id)
            
            # Find indices of tracks to remove
            track_indices = []
            for idx, track in enumerate(playlist_tracks):
                if track.get("id") in track_ids:
                    track_indices.append(idx)
            
            if not track_indices:
                return {"success": False, "error": "No matching tracks found in playlist"}
            
            result = await client.remove_from_playlist(playlist_id, track_indices)
            return result
            
        except Exception as e:
            logger.error(f"Failed to remove tracks from playlist: {e}")
            return {"success": False, "error": str(e)}
    
# Global music service instance
music_service = MusicService()

