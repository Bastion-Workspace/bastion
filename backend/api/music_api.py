"""
Music API - SubSonic-compatible music streaming endpoints
"""

import logging
from typing import Dict, Any, Optional, AsyncIterator, List
from fastapi import APIRouter, HTTPException, Depends, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx

from config import settings
from services.music_cover_cache import (
    compute_etag,
    etag_header_value,
    fetch_cover_upstream,
    get_cached_etag_only,
    get_or_fetch,
    if_none_match_matches,
)
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

router = APIRouter(tags=["Music"])


def _get_content_type_from_url(url: str) -> str:
    """Determine Content-Type from URL extension. Strips query string and fragment first."""
    path = url.split("?")[0].split("#")[0]
    url_lower = path.lower()
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


@router.post("/api/music/config", response_model=Dict[str, Any])
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


@router.get("/api/music/config", response_model=MusicServiceConfigResponse)
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


@router.get("/api/music/sources", response_model=MediaSourceListResponse)
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


@router.delete("/api/music/config")
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


@router.post("/api/music/test-connection")
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


@router.post("/api/music/refresh")
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


@router.get("/api/music/library", response_model=MusicLibraryResponse)
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


@router.get("/api/music/albums/artist/{artist_id}", response_model=MusicLibraryResponse)
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


@router.get("/api/music/series/author/{author_id}")
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


@router.get("/api/music/albums/series/{series_name}")
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


@router.get("/api/music/tracks/{parent_id}")
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


@router.get("/api/music/stream/{track_id}", response_model=StreamUrlResponse)
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


@router.get("/api/music/stream-proxy/{track_id}")
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
        # parent_id (e.g. podcast library item id) required for AudioBookShelf episode streaming
        parent_id = request.query_params.get("parent_id") or None

        # Get the stream URL from media server
        stream_url = await music_service.get_stream_url(user_id, track_id, service_type, parent_id=parent_id)
        
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
        
        # Open upstream stream first so we can mirror status and partial-content headers.
        # Partial responses (Content-Range) must also forward Content-Length for HTML5 media.
        logger.debug(
            "Streaming from URL: %s... (headers: %s)",
            final_stream_url[:200],
            list(final_request_headers.keys()),
        )
        client = httpx.AsyncClient(timeout=300.0, follow_redirects=True)
        try:
            upstream = await client.send(
                client.build_request("GET", final_stream_url, headers=final_request_headers),
                stream=True,
            )
        except Exception as e:
            await client.aclose()
            logger.error(f"Upstream stream request failed: {e}", exc_info=True)
            raise HTTPException(status_code=502, detail="Could not reach media server") from e

        if upstream.status_code >= 400:
            try:
                await upstream.aread()
            finally:
                await upstream.aclose()
                await client.aclose()
            logger.error(f"Upstream stream rejected: HTTP {upstream.status_code}")
            raise HTTPException(status_code=502, detail="Media server refused stream")

        out_headers = dict(response_headers_dict)
        uct = upstream.headers.get("content-type")
        if uct:
            out_headers["Content-Type"] = uct.split(";")[0].strip()
        # Only forward Content-Length together with Content-Range (partial content).
        # Forwarding upstream Content-Length on a full 200 while streaming bytes through
        # this proxy caused Uvicorn "Response content shorter than Content-Length" when
        # the upstream closed the socket early (timeout, proxy, etc.). Full responses use
        # chunked encoding to the client when we omit Content-Length.
        cr = upstream.headers.get("content-range")
        if cr:
            out_headers["Content-Range"] = cr
            cl = upstream.headers.get("content-length")
            if cl:
                out_headers["Content-Length"] = cl

        async def stream_audio() -> AsyncIterator[bytes]:
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            except (httpx.RemoteProtocolError, httpx.ReadError) as e:
                # Upstream dropped the connection mid-body; already logged at httpx layer.
                logger.warning("Upstream audio stream ended early: %s", e)
            except Exception as e:
                logger.error("Error during audio streaming: %s", e, exc_info=True)
            finally:
                await upstream.aclose()
                await client.aclose()

        return StreamingResponse(
            stream_audio(),
            status_code=upstream.status_code,
            headers=out_headers,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting up stream proxy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/music/cover-art/{cover_art_id}")
async def cover_art_proxy(
    cover_art_id: str,
    request: Request,
    token: Optional[str] = None,
) -> Response:
    """
    Proxy cover art from the media server (Subsonic getCoverArt).
    Authenticates like stream-proxy (Bearer header or token query param).
    """
    try:
        user_id = None
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

        query_service_type = request.query_params.get("service_type")
        creds = await music_service.get_credentials(
            user_id, query_service_type if query_service_type else None
        )
        if not creds:
            raise HTTPException(
                status_code=404, detail="Cover art not available for this source"
            )
        resolved_service_type = creds.get("service_type", "subsonic")

        size_raw = request.query_params.get("size", "300")
        try:
            size = int(size_raw)
        except ValueError:
            size = 300
        size = max(32, min(size, 1200))

        etag_hex = compute_etag(
            user_id, resolved_service_type, cover_art_id, size
        )
        cache_headers = {
            "Cache-Control": "public, max-age=31536000, immutable",
            "ETag": etag_header_value(etag_hex),
            "Vary": "Accept",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
        }

        # Only 304 when we have previously stored this key. Otherwise native image
        # stacks can send If-None-Match matching our deterministic ETag without a
        # usable cached body and would render blank art.
        rls_context = {"user_id": user_id}
        if if_none_match_matches(
            request.headers.get("If-None-Match"), etag_hex
        ):
            row_etag = await get_cached_etag_only(
                user_id,
                resolved_service_type,
                cover_art_id,
                size,
                rls_context,
            )
            if row_etag is not None:
                return Response(status_code=304, headers=cache_headers)

        if getattr(settings, "MUSIC_COVER_CACHE_ENABLED", True):

            async def fetch_body_from_upstream() -> tuple:
                url = await music_service.get_cover_art_url(
                    user_id, cover_art_id, resolved_service_type, size
                )
                if not url:
                    raise RuntimeError("no_cover_art_url")
                return await fetch_cover_upstream(url)

            def normalize_image_media_type(raw) -> str:
                s = str(raw) if raw is not None else "image/jpeg"
                return s.split(";")[0].strip() or "image/jpeg"

            try:
                body, media_type, etag_out = await get_or_fetch(
                    user_id,
                    resolved_service_type,
                    cover_art_id,
                    size,
                    rls_context,
                    fetch_body_from_upstream,
                )
            except RuntimeError as e:
                err = str(e)
                if err == "no_cover_art_url":
                    raise HTTPException(
                        status_code=404,
                        detail="Cover art not available for this source",
                    )
                if err.startswith("upstream_http_"):
                    code = err.replace("upstream_http_", "")
                    logger.error("Upstream cover art rejected: HTTP %s", code)
                    raise HTTPException(
                        status_code=502,
                        detail="Media server refused cover art",
                    )
                raise
            except Exception as cache_err:
                logger.warning(
                    "Cover art cache failed, fetching upstream only: %s",
                    cache_err,
                    exc_info=True,
                )
                try:
                    body, media_type = await fetch_body_from_upstream()
                except RuntimeError as e:
                    err = str(e)
                    if err == "no_cover_art_url":
                        raise HTTPException(
                            status_code=404,
                            detail="Cover art not available for this source",
                        )
                    if err.startswith("upstream_http_"):
                        code = err.replace("upstream_http_", "")
                        logger.error("Upstream cover art rejected: HTTP %s", code)
                        raise HTTPException(
                            status_code=502,
                            detail="Media server refused cover art",
                        )
                    raise
                media_type = normalize_image_media_type(media_type)
                return Response(
                    content=body,
                    media_type=media_type,
                    headers=cache_headers,
                )

            media_type = normalize_image_media_type(media_type)
            out_headers = {
                **cache_headers,
                "ETag": etag_header_value(etag_out),
            }
            return Response(
                content=body, media_type=media_type, headers=out_headers
            )

        art_url = await music_service.get_cover_art_url(
            user_id, cover_art_id, resolved_service_type, size
        )
        if not art_url:
            raise HTTPException(
                status_code=404, detail="Cover art not available for this source"
            )

        try:
            body, media_type = await fetch_cover_upstream(art_url)
        except RuntimeError as e:
            err = str(e)
            if err.startswith("upstream_http_"):
                code = err.replace("upstream_http_", "")
                logger.error("Upstream cover art rejected: HTTP %s", code)
                raise HTTPException(
                    status_code=502, detail="Media server refused cover art"
                )
            raise
        media_type = (
            str(media_type).split(";")[0].strip() if media_type else "image/jpeg"
        ) or "image/jpeg"

        return Response(
            content=body,
            media_type=media_type,
            headers=cache_headers,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error proxying cover art: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class PlaylistModifyRequest(BaseModel):
    """Request for adding/removing tracks to/from playlist"""
    track_ids: List[str]


class SearchRequest(BaseModel):
    """Request for searching music catalog"""
    query: str
    service_type: str
    limit: int = 25


@router.post("/api/music/playlist/{playlist_id}/add-tracks")
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


@router.post("/api/music/playlist/{playlist_id}/remove-tracks")
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


@router.post("/api/music/search/tracks")
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


@router.post("/api/music/search/albums")
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


@router.post("/api/music/search/artists")
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

