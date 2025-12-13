"""
Music API - SubSonic-compatible music streaming endpoints
"""

import logging
from typing import Dict, Any, Optional, AsyncIterator, List
from fastapi import APIRouter, HTTPException, Depends, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx

from services.music_service import music_service
from utils.auth_middleware import get_current_user
from models.api_models import AuthenticatedUserResponse
from models.music_models import (
    MusicServiceConfigRequest,
    MusicServiceConfigResponse,
    MediaSourceListResponse,
    MusicLibraryResponse,
    MusicTracksResponse,
    StreamUrlResponse,
    MusicAlbum,
    MusicArtist,
    MusicPlaylist,
    MusicTrack
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/music", tags=["Music"])


def _get_content_type_from_url(url: str) -> str:
    """Determine Content-Type from URL extension"""
    url_lower = url.lower()
    if url_lower.endswith('.mp3'):
        return 'audio/mpeg'
    elif url_lower.endswith('.flac'):
        return 'audio/flac'
    elif url_lower.endswith('.m4a') or url_lower.endswith('.alac'):
        return 'audio/mp4'
    elif url_lower.endswith('.ogg'):
        return 'audio/ogg'
    elif url_lower.endswith('.wav'):
        return 'audio/wav'
    elif url_lower.endswith('.aac'):
        return 'audio/aac'
    elif url_lower.endswith('.wma'):
        return 'audio/x-ms-wma'
    elif url_lower.endswith('.opus'):
        return 'audio/opus'
    else:
        # Default to MP3 if unknown
        return 'audio/mpeg'


@router.post("/config", response_model=Dict[str, Any])
async def save_music_config(
    request: MusicServiceConfigRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    """Save media server configuration"""
    try:
        success = await music_service.save_config(
            user_id=current_user.user_id,
            server_url=request.server_url,
            username=request.username,
            password=request.password,
            auth_type=request.auth_type,
            service_type=request.service_type,
            service_name=request.service_name
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save configuration")
        
        return {"success": True, "message": "Configuration saved"}
    except Exception as e:
        logger.error(f"Error saving music config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config", response_model=MusicServiceConfigResponse)
async def get_music_config(
    service_type: Optional[str] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> MusicServiceConfigResponse:
    """Get music service configuration (without credentials)"""
    try:
        config = await music_service.get_config(current_user.user_id, service_type)
        
        if not config:
            return MusicServiceConfigResponse(
                server_url="",
                username="",
                auth_type="password",
                service_type=service_type or "subsonic",
                has_config=False
            )
        
        return MusicServiceConfigResponse(**config)
    except Exception as e:
        logger.error(f"Error getting music config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sources", response_model=MediaSourceListResponse)
async def get_media_sources(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> MediaSourceListResponse:
    """Get all configured media sources for the user"""
    try:
        sources = await music_service.get_user_sources(current_user.user_id)
        
        # Convert to response models
        source_list = []
        for source in sources:
            # Get sync metadata for each source
            config = await music_service.get_config(current_user.user_id, source.get("service_type"))
            if config:
                source_list.append(MusicServiceConfigResponse(**config))
            else:
                # Fallback if no metadata
                source_list.append(MusicServiceConfigResponse(
                    server_url=source.get("server_url", ""),
                    username=source.get("username", ""),
                    auth_type=source.get("auth_type", "password"),
                    service_type=source.get("service_type", "subsonic"),
                    service_name=source.get("service_name"),
                    is_active=source.get("is_active", True),
                    has_config=True
                ))
        
        return MediaSourceListResponse(sources=source_list)
    except Exception as e:
        logger.error(f"Error getting media sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/config")
async def delete_music_config(
    service_type: Optional[str] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    """Delete music service configuration and cache"""
    try:
        success = await music_service.delete_config(current_user.user_id, service_type)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete configuration")
        
        message = f"Configuration deleted" if service_type else "All configurations deleted"
        return {"success": True, "message": message}
    except Exception as e:
        logger.error(f"Error deleting music config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test-connection")
async def test_connection(
    service_type: Optional[str] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    """Test connection to media server"""
    try:
        result = await music_service.test_connection(current_user.user_id, service_type)
        return result
    except Exception as e:
        logger.error(f"Error testing connection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/refresh")
async def refresh_cache(
    service_type: Optional[str] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    """Refresh music library cache from media server"""
    try:
        result = await music_service.refresh_cache(current_user.user_id, service_type)
        return result
    except Exception as e:
        logger.error(f"Error refreshing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/library", response_model=MusicLibraryResponse)
async def get_library(
    service_type: Optional[str] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> MusicLibraryResponse:
    """Get cached music library"""
    try:
        library = await music_service.get_library(current_user.user_id, service_type)
        
        # Convert to response models
        albums = [MusicAlbum(**album) for album in library.get("albums", [])]
        artists = [MusicArtist(**artist) for artist in library.get("artists", [])]
        playlists = [MusicPlaylist(**playlist) for playlist in library.get("playlists", [])]
        
        return MusicLibraryResponse(
            albums=albums,
            artists=artists,
            playlists=playlists,
            last_sync_at=library.get("last_sync_at")
        )
    except Exception as e:
        logger.error(f"Error getting library: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/albums/artist/{artist_id}", response_model=MusicLibraryResponse)
async def get_albums_by_artist(
    artist_id: str,
    service_type: Optional[str] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> MusicLibraryResponse:
    """Get albums for a specific artist"""
    try:
        albums = await music_service.get_albums_by_artist(current_user.user_id, artist_id, service_type)
        
        # Convert to response models
        albums_list = [MusicAlbum(**album) for album in albums]
        
        return MusicLibraryResponse(
            albums=albums_list,
            artists=[],
            playlists=[],
            last_sync_at=None
        )
    except Exception as e:
        logger.error(f"Error getting albums by artist: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/series/author/{author_id}")
async def get_series_by_author(
    author_id: str,
    service_type: Optional[str] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get series for a specific author (Audiobookshelf only)"""
    try:
        series = await music_service.get_series_by_author(current_user.user_id, author_id, service_type)
        return {"series": series}
    except Exception as e:
        logger.error(f"Error getting series by author: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/albums/series/{series_name}")
async def get_albums_by_series(
    series_name: str,
    author_name: str,
    service_type: Optional[str] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> MusicLibraryResponse:
    """Get albums (books) for a specific series"""
    try:
        albums = await music_service.get_albums_by_series(
            current_user.user_id, 
            series_name, 
            author_name, 
            service_type
        )
        
        albums_list = [MusicAlbum(**album) for album in albums]
        
        return MusicLibraryResponse(
            albums=albums_list,
            artists=[],
            playlists=[],
            last_sync_at=None
        )
    except Exception as e:
        logger.error(f"Error getting albums by series: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tracks/{parent_id}")
async def get_tracks(
    parent_id: str,
    parent_type: str = "album",
    service_type: Optional[str] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> MusicTracksResponse:
    """Get tracks for an album or playlist"""
    try:
        if parent_type not in ["album", "playlist"]:
            raise HTTPException(status_code=400, detail="parent_type must be 'album' or 'playlist'")
        
        tracks_data = await music_service.get_tracks(
            current_user.user_id,
            parent_id,
            parent_type,
            service_type
        )
        
        tracks = [MusicTrack(**track) for track in tracks_data]
        
        return MusicTracksResponse(
            tracks=tracks,
            parent_id=parent_id,
            parent_type=parent_type
        )
    except Exception as e:
        logger.error(f"Error getting tracks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stream/{track_id}", response_model=StreamUrlResponse)
async def get_stream_url(
    track_id: str,
    service_type: Optional[str] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> StreamUrlResponse:
    """Get authenticated stream URL for a track (legacy endpoint - use /stream-proxy/{track_id} for better format support)"""
    try:
        stream_url = await music_service.get_stream_url(current_user.user_id, track_id, service_type)
        
        if not stream_url:
            raise HTTPException(status_code=404, detail="Track not found or stream URL generation failed")
        
        return StreamUrlResponse(stream_url=stream_url)
    except Exception as e:
        logger.error(f"Error getting stream URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stream-proxy/{track_id}")
async def stream_proxy(
    track_id: str,
    request: Request,
    token: Optional[str] = None  # Allow token as query parameter for audio element compatibility
) -> StreamingResponse:
    """
    Proxy audio stream from SubSonic with proper headers and CORS support.
    This endpoint handles format detection and sets appropriate Content-Type headers.
    
    Supports authentication via:
    1. Bearer token in Authorization header (preferred)
    2. Token query parameter (for HTML5 audio element compatibility)
    """
    try:
        # Handle authentication - support both header and query parameter
        user_id = None
        
        # Try query parameter first (for HTML5 audio element compatibility)
        if token:
            try:
                from utils.auth_middleware import decode_jwt_token
                payload = decode_jwt_token(token)
                user_id = payload.get("user_id")
                if not user_id:
                    raise HTTPException(status_code=401, detail="Invalid token")
            except ValueError as e:
                logger.error(f"Token validation failed: {e}")
                raise HTTPException(status_code=401, detail="Invalid or expired token")
            except Exception as e:
                logger.error(f"Token validation error: {e}")
                raise HTTPException(status_code=401, detail="Invalid or expired token")
        else:
            # Try to get token from Authorization header
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
                try:
                    from utils.auth_middleware import decode_jwt_token
                    payload = decode_jwt_token(token)
                    user_id = payload.get("user_id")
                except ValueError:
                    raise HTTPException(status_code=401, detail="Invalid or expired token")
                except Exception:
                    raise HTTPException(status_code=401, detail="Invalid or expired token")
            else:
                raise HTTPException(status_code=401, detail="Authentication required")
        
        if not user_id:
            raise HTTPException(status_code=401, detail="User ID not found in token")
        
        # Get service_type from query parameter if provided
        service_type = request.query_params.get("service_type")
        
        # Get the stream URL from media server
        stream_url = await music_service.get_stream_url(user_id, track_id, service_type)
        
        if not stream_url:
            raise HTTPException(status_code=404, detail="Track not found or stream URL generation failed")
        
        # Determine Content-Type from URL
        content_type = _get_content_type_from_url(stream_url)
        
        logger.info(f"Proxying audio stream for track {track_id}, Content-Type: {content_type}, URL: {stream_url[:200]}")
        
        # Forward Range header for seeking support (case-insensitive check)
        range_header = None
        for header_name, header_value in request.headers.items():
            if header_name.lower() == "range":
                range_header = header_value
                break
        
        # Prepare request headers
        request_headers = {}
        if range_header:
            request_headers["Range"] = range_header
        
        # Base response headers (set before streaming)
        response_headers_dict = {
            "Content-Type": content_type,
            "Accept-Ranges": "bytes",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
            "Access-Control-Allow-Headers": "Range, Content-Type",
            "Cache-Control": "public, max-age=3600",
        }
        
        # Prepare final URL and headers (extract token from URL if present)
        final_stream_url = stream_url
        final_request_headers = request_headers.copy()
        
        # For AudioBookShelf, we need to add Bearer token to headers
        # Check if URL contains token query param (AudioBookShelf format)
        if "token=" in final_stream_url:
            # Extract token from URL and preserve other query params
            import urllib.parse
            parsed = urllib.parse.urlparse(final_stream_url)
            query_params = urllib.parse.parse_qs(parsed.query)
            token = query_params.get("token", [None])[0]
            if token:
                # Remove token from query params but keep others (like index for chapters)
                query_params.pop("token", None)
                # Rebuild URL without token
                new_query = urllib.parse.urlencode(query_params, doseq=True)
                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                if new_query:
                    clean_url += f"?{new_query}"
                final_request_headers["Authorization"] = f"Bearer {token}"
                final_stream_url = clean_url
                logger.debug(f"Cleaned stream URL: {final_stream_url[:200]}...")
        
        # Validate the stream URL before creating StreamingResponse
        # Make a HEAD request to check if resource exists and is accessible
        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                # Use HEAD request to validate without downloading content
                head_response = await client.head(final_stream_url, headers=final_request_headers)
                if head_response.status_code >= 400:
                    # Try to get error details
                    error_text = f"HTTP {head_response.status_code} error"
                    try:
                        if head_response.text:
                            error_text = head_response.text[:500]
                    except Exception:
                        pass
                    logger.error(f"Stream validation failed: {head_response.status_code} - {error_text}")
                    raise HTTPException(status_code=head_response.status_code, detail=f"Failed to access audio stream: {error_text}")
        except HTTPException:
            raise
        except Exception as e:
            # If HEAD fails, try GET with limited content to validate
            try:
                async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                    get_response = await client.get(final_stream_url, headers=final_request_headers)
                    if get_response.status_code >= 400:
                        error_text = f"HTTP {get_response.status_code} error"
                        try:
                            if get_response.text:
                                error_text = get_response.text[:500]
                        except Exception:
                            pass
                        logger.error(f"Stream validation failed: {get_response.status_code} - {error_text}")
                        raise HTTPException(status_code=get_response.status_code, detail=f"Failed to access audio stream: {error_text}")
            except HTTPException:
                raise
            except Exception as validation_error:
                logger.error(f"Stream validation error: {validation_error}")
                raise HTTPException(status_code=500, detail=f"Failed to validate stream URL: {str(validation_error)}")
        
        async def stream_audio() -> AsyncIterator[bytes]:
            """Stream audio data from music server"""
            try:
                logger.debug(f"Streaming from URL: {final_stream_url[:200]}... (headers: {list(final_request_headers.keys())})")
                
                async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
                    async with client.stream("GET", final_stream_url, headers=final_request_headers) as response:
                        # Log response status for debugging
                        logger.debug(f"Stream response status: {response.status_code}, content-type: {response.headers.get('content-type', 'unknown')}")
                        
                        # Status should be OK since we validated, but check anyway
                        if response.status_code >= 400:
                            logger.error(f"Unexpected error during streaming: {response.status_code}")
                            return
                        
                        # Stream chunks
                        chunk_count = 0
                        async for chunk in response.aiter_bytes():
                            chunk_count += 1
                            if chunk_count <= 3:  # Log first few chunks for debugging
                                logger.debug(f"Stream chunk {chunk_count}: {len(chunk)} bytes")
                            yield chunk
                            
            except Exception as e:
                logger.error(f"Error during audio streaming: {e}", exc_info=True)
                # Can't raise HTTPException in generator - just stop yielding
                return
        
        # Determine status code (206 for Range requests, 200 otherwise)
        status_code = 206 if range_header else 200
        
        return StreamingResponse(
            stream_audio(),
            status_code=status_code,
            media_type=content_type,
            headers=response_headers_dict
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting up stream proxy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class PlaylistModifyRequest(BaseModel):
    """Request for adding/removing tracks to/from playlist"""
    track_ids: List[str]


class SearchRequest(BaseModel):
    """Request for searching music catalog"""
    query: str
    service_type: str
    limit: int = 25


@router.post("/playlist/{playlist_id}/add-tracks")
async def add_tracks_to_playlist(
    playlist_id: str,
    request: PlaylistModifyRequest,
    service_type: Optional[str] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    """Add tracks to a playlist"""
    try:
        result = await music_service.add_to_playlist(
            current_user.user_id,
            playlist_id,
            request.track_ids,
            service_type
        )
        
        if not result.get("success"):
            raise HTTPException(
                status_code=400, 
                detail=result.get("error", "Failed to add tracks to playlist")
            )
        
        return {"success": True, "message": f"Added {len(request.track_ids)} tracks to playlist"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding tracks to playlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/playlist/{playlist_id}/remove-tracks")
async def remove_tracks_from_playlist(
    playlist_id: str,
    request: PlaylistModifyRequest,
    service_type: Optional[str] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    """Remove tracks from a playlist"""
    try:
        result = await music_service.remove_from_playlist(
            current_user.user_id,
            playlist_id,
            request.track_ids,
            service_type
        )
        
        if not result.get("success"):
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Failed to remove tracks from playlist")
            )
        
        return {"success": True, "message": f"Removed {len(request.track_ids)} tracks from playlist"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing tracks from playlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/tracks")
async def search_tracks(
    request: SearchRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> MusicTracksResponse:
    """Search for tracks in music service catalog"""
    try:
        tracks_data = await music_service.search_tracks(
            current_user.user_id,
            request.query,
            request.service_type,
            request.limit
        )
        
        tracks = [MusicTrack(**track) for track in tracks_data]
        
        return MusicTracksResponse(
            tracks=tracks,
            parent_id="",
            parent_type="search"
        )
    except Exception as e:
        logger.error(f"Error searching tracks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/albums")
async def search_albums(
    request: SearchRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> MusicLibraryResponse:
    """Search for albums in music service catalog"""
    try:
        albums_data = await music_service.search_albums(
            current_user.user_id,
            request.query,
            request.service_type,
            request.limit
        )
        
        albums = [MusicAlbum(**album) for album in albums_data]
        
        return MusicLibraryResponse(
            albums=albums,
            artists=[],
            playlists=[],
            last_sync_at=None
        )
    except Exception as e:
        logger.error(f"Error searching albums: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/artists")
async def search_artists(
    request: SearchRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> MusicLibraryResponse:
    """Search for artists in music service catalog"""
    try:
        artists_data = await music_service.search_artists(
            current_user.user_id,
            request.query,
            request.service_type,
            request.limit
        )
        
        artists = [MusicArtist(**artist) for artist in artists_data]
        
        return MusicLibraryResponse(
            albums=[],
            artists=artists,
            playlists=[],
            last_sync_at=None
        )
    except Exception as e:
        logger.error(f"Error searching artists: {e}")
        raise HTTPException(status_code=500, detail=str(e))

