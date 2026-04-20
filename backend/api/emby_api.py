"""
Emby REST API: library browsing, images, video streaming proxy, playback reporting.
"""

import logging
from typing import Any, AsyncIterator, Dict, List, Optional
from urllib.parse import quote, unquote, urljoin, urlparse

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from models.api_models import AuthenticatedUserResponse
from services.music_clients.client_factory import MusicClientFactory
from services.music_clients.emby_client import EmbyClient
from services.music_service import music_service
from utils.auth_middleware import decode_jwt_token, get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Emby"])


def _content_length_from_content_range(content_range: str) -> Optional[str]:
    """Derive body length from ``Content-Range: bytes start-end/total`` when Content-Length is absent."""
    try:
        after = content_range.split(":", 1)[1].strip()
        range_spec, _total = after.split("/", 1)
        start_s, end_s = range_spec.split("-", 1)
        start, end = int(start_s), int(end_s)
        if end < start:
            return None
        return str(end - start + 1)
    except (ValueError, IndexError):
        return None


async def _stream_upstream_then_close(
    upstream: httpx.Response, client: httpx.AsyncClient
) -> AsyncIterator[bytes]:
    """Drain streamed body; always close upstream response and HTTP client (see video-stream route)."""
    try:
        async for chunk in upstream.aiter_bytes():
            if chunk:
                yield chunk
    finally:
        try:
            await upstream.aclose()
        except Exception:
            pass
        try:
            await client.aclose()
        except Exception:
            pass


async def _emby_client_for_user(user_id: str) -> EmbyClient:
    creds = await music_service.get_credentials(user_id, "emby")
    if not creds:
        raise HTTPException(status_code=404, detail="Emby is not configured")
    client = MusicClientFactory.create_client(
        service_type="emby",
        server_url=creds["server_url"],
        username=creds["username"],
        password=creds["password"],
        auth_type=creds.get("auth_type", "password"),
    )
    if not client or not isinstance(client, EmbyClient):
        raise HTTPException(status_code=500, detail="Failed to create Emby client")
    return client


def _user_id_from_jwt_request(request: Request, token: Optional[str]) -> str:
    if token:
        try:
            payload = decode_jwt_token(token)
            uid = payload.get("user_id")
            if uid:
                return uid
        except ValueError as e:
            logger.error("Emby JWT query token invalid: %s", e)
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        try:
            payload = decode_jwt_token(auth[7:])
            uid = payload.get("user_id")
            if uid:
                return uid
        except ValueError:
            pass
    raise HTTPException(status_code=401, detail="Authentication required")


def _validate_relay_path(path: str) -> str:
    path = path.strip()
    if ".." in path or path.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not path.startswith("Videos/"):
        raise HTTPException(status_code=400, detail="Invalid relay path")
    return path


def _content_type_from_path(path: str) -> str:
    lower = path.lower().split("?")[0]
    if lower.endswith(".m3u8"):
        return "application/vnd.apple.mpegurl"
    if lower.endswith(".ts"):
        return "video/mp2t"
    if lower.endswith(".mp4"):
        return "video/mp4"
    if lower.endswith(".vtt"):
        return "text/vtt"
    return "application/octet-stream"


def _path_after_emby(full_url: str, emby_root: str) -> Optional[str]:
    try:
        p = urlparse(full_url)
        root_p = urlparse(emby_root.rstrip("/"))
        if p.netloc and root_p.netloc and p.netloc != root_p.netloc:
            return None
        prefix = root_p.path.rstrip("/")
        if not p.path.startswith(prefix):
            return None
        rest = p.path[len(prefix) :].lstrip("/")
        if p.query:
            rest = f"{rest}?{p.query}"
        return rest
    except Exception:
        return None


def _rewrite_m3u8_text(
    body: str,
    upstream_base_url: str,
    bastion_base: str,
    emby_root: str,
    token: Optional[str],
) -> str:
    """
    Rewrite playlist lines to bastion relay URLs. upstream_base_url is the URL of the
    fetched playlist (for resolving relatives).
    """
    out_lines: List[str] = []
    token_q = f"&token={quote(token, safe='')}" if token else ""
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            out_lines.append(line)
            continue
        resolved = stripped
        if not stripped.startswith("http://") and not stripped.startswith("https://"):
            resolved = urljoin(upstream_base_url, stripped)
        sub = _path_after_emby(resolved, emby_root)
        if not sub:
            out_lines.append(line)
            continue
        try:
            _validate_relay_path(sub.split("?")[0])
        except HTTPException:
            out_lines.append(line)
            continue
        relay = f"{bastion_base}/api/emby/hls/relay?path={quote(sub, safe='')}{token_q}"
        out_lines.append(relay)
    return "\n".join(out_lines) + ("\n" if body.endswith("\n") else "")


@router.get("/api/emby/libraries")
async def emby_libraries(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    client = await _emby_client_for_user(current_user.user_id)
    return {"libraries": await client.get_libraries()}


@router.get("/api/emby/items")
async def emby_items(
    parent_id: Optional[str] = None,
    item_types: Optional[str] = None,
    sort_by: str = "SortName",
    sort_order: str = "Ascending",
    limit: int = 100,
    start_index: int = 0,
    recursive: bool = False,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    client = await _emby_client_for_user(current_user.user_id)
    data = await client.get_items(
        parent_id=parent_id,
        item_types=item_types,
        sort_by=sort_by,
        sort_order=sort_order,
        limit=limit,
        start_index=start_index,
        recursive=recursive,
    )
    return data


@router.get("/api/emby/items/latest")
async def emby_items_latest(
    parent_id: str,
    limit: int = 24,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    client = await _emby_client_for_user(current_user.user_id)
    items = await client.get_latest_items(parent_id, limit=limit)
    return {"Items": items}


@router.get("/api/emby/items/resume")
async def emby_items_resume(
    limit: int = 50,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    client = await _emby_client_for_user(current_user.user_id)
    items = await client.get_resume_items(limit=limit)
    return {"Items": items}


@router.get("/api/emby/items/{item_id}")
async def emby_item_detail(
    item_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    client = await _emby_client_for_user(current_user.user_id)
    return await client.get_item(item_id)


@router.get("/api/emby/shows/{series_id}/seasons")
async def emby_show_seasons(
    series_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    client = await _emby_client_for_user(current_user.user_id)
    items = await client.get_seasons(series_id)
    return {"Items": items}


@router.get("/api/emby/shows/{series_id}/episodes")
async def emby_show_episodes(
    series_id: str,
    season_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    client = await _emby_client_for_user(current_user.user_id)
    items = await client.get_episodes(series_id, season_id)
    return {"Items": items}


class PlaybackInfoBody(BaseModel):
    max_streaming_bitrate: int = Field(default=140_000_000)
    start_time_ticks: int = 0


@router.post("/api/emby/items/{item_id}/playback-info")
async def emby_playback_info(
    item_id: str,
    body: Optional[PlaybackInfoBody] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    client = await _emby_client_for_user(current_user.user_id)
    b = body or PlaybackInfoBody()
    payload = {
        "MaxStreamingBitrate": b.max_streaming_bitrate,
        "StartTimeTicks": b.start_time_ticks,
        "EnableDirectPlay": True,
        "EnableDirectStream": True,
        "EnableTranscoding": True,
    }
    return await client.post_playback_info(item_id, payload)


@router.get("/api/emby/image/{item_id}/{image_type}")
async def emby_image(
    item_id: str,
    image_type: str,
    request: Request,
    max_width: int = 400,
    index: int = 0,
    tag: Optional[str] = None,
    token: Optional[str] = None,
):
    user_id = _user_id_from_jwt_request(request, token)
    client = await _emby_client_for_user(user_id)
    await client._authenticate()
    url = client.build_upstream_image_url(
        item_id, image_type, max_width=max_width, image_index=index, tag=tag
    )
    headers = {"X-Emby-Token": client._access_token or ""}
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as hc:
        r = await hc.get(url, headers=headers)
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail="Image fetch failed")
    ct = r.headers.get("content-type", "image/jpeg")
    return Response(
        content=r.content,
        media_type=ct.split(";")[0].strip(),
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.get("/api/emby/video-stream/{item_id}")
async def emby_video_stream(
    item_id: str,
    request: Request,
    media_source_id: str,
    play_session_id: str,
    static: bool = True,
    start_time_ticks: int = 0,
    audio_stream_index: Optional[int] = None,
    token: Optional[str] = None,
):
    user_id = _user_id_from_jwt_request(request, token)
    client = await _emby_client_for_user(user_id)
    await client._authenticate()
    stream_url = client.build_upstream_video_stream_url(
        item_id,
        media_source_id=media_source_id,
        play_session_id=play_session_id,
        static=static,
        start_time_ticks=start_time_ticks or 0,
        audio_stream_index=audio_stream_index,
    )
    range_header = None
    for name, val in request.headers.items():
        if name.lower() == "range":
            range_header = val
            break
    req_headers: Dict[str, str] = {}
    if range_header:
        req_headers["Range"] = range_header
    token_hdr = {"X-Emby-Token": client._access_token or ""}
    req_headers = {**token_hdr, **req_headers}

    hc = httpx.AsyncClient(timeout=600.0, follow_redirects=True)
    try:
        upstream = await hc.send(
            hc.build_request("GET", stream_url, headers=req_headers),
            stream=True,
        )
    except Exception:
        await hc.aclose()
        raise

    if upstream.status_code >= 400:
        try:
            await upstream.aread()
        finally:
            await upstream.aclose()
            await hc.aclose()
        raise HTTPException(status_code=502, detail="Media server refused stream")

    # Keep hc open until the StreamingResponse body is fully read. Exiting ``async with``
    # immediately after ``send(stream=True)`` closes the client and corrupts progressive video.
    out_headers: Dict[str, str] = {
        "Accept-Ranges": "bytes",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
        "Access-Control-Allow-Headers": "Range, Content-Type",
        "Cache-Control": "public, max-age=3600",
    }
    uct = upstream.headers.get("content-type")
    if uct:
        out_headers["Content-Type"] = uct.split(";")[0].strip()
    else:
        out_headers["Content-Type"] = "video/mp4"
    cr = upstream.headers.get("content-range")
    if cr:
        out_headers["Content-Range"] = cr
    cl = upstream.headers.get("content-length")
    if not cl and cr:
        cl = _content_length_from_content_range(cr)
    if cl:
        out_headers["Content-Length"] = cl

    return StreamingResponse(
        _stream_upstream_then_close(upstream, hc),
        status_code=upstream.status_code,
        headers=out_headers,
    )


@router.get("/api/emby/hls/{item_id}/master.m3u8")
async def emby_hls_master(
    item_id: str,
    request: Request,
    media_source_id: str,
    play_session_id: str,
    audio_stream_index: Optional[int] = None,
    token: Optional[str] = None,
):
    user_id = _user_id_from_jwt_request(request, token)
    client = await _emby_client_for_user(user_id)
    await client._authenticate()
    upstream = client.build_upstream_hls_master_url(
        item_id,
        media_source_id=media_source_id,
        play_session_id=play_session_id,
        audio_stream_index=audio_stream_index,
    )
    hls_headers = {"X-Emby-Token": client._access_token or ""}
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as hc:
        r = await hc.get(upstream, headers=hls_headers)
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail="Failed to fetch HLS master")
    emby_root = client._root()
    bastion_base = str(request.base_url).rstrip("/")
    rewritten = _rewrite_m3u8_text(r.text, upstream, bastion_base, emby_root, token)
    return Response(
        content=rewritten,
        media_type="application/vnd.apple.mpegurl",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Cache-Control": "no-cache",
        },
    )


@router.get("/api/emby/hls/relay")
async def emby_hls_relay(
    request: Request,
    path: str,
    token: Optional[str] = None,
):
    user_id = _user_id_from_jwt_request(request, token)
    raw_path = unquote(path).strip()
    if "?" in raw_path:
        safe_part, query = raw_path.split("?", 1)
    else:
        safe_part, query = raw_path, ""
    _validate_relay_path(safe_part)
    client = await _emby_client_for_user(user_id)
    await client._authenticate()
    full = f"{client._root()}/{safe_part}"
    if query:
        full = f"{full}?{query}"
    if "api_key" not in full:
        sep = "&" if "?" in full else "?"
        full = f"{full}{sep}api_key={quote(client._access_token or '', safe='')}"

    range_header = None
    for name, val in request.headers.items():
        if name.lower() == "range":
            range_header = val
            break
    headers: Dict[str, str] = {}
    if range_header:
        headers["Range"] = range_header
    headers["X-Emby-Token"] = client._access_token or ""

    hc = httpx.AsyncClient(timeout=600.0, follow_redirects=True)
    try:
        upstream = await hc.send(hc.build_request("GET", full, headers=headers), stream=True)
    except Exception:
        await hc.aclose()
        raise

    if upstream.status_code >= 400:
        try:
            await upstream.aread()
        finally:
            await upstream.aclose()
            await hc.aclose()
        raise HTTPException(status_code=502, detail="Relay fetch failed")

    ct = _content_type_from_path(safe_part + ("?" + query if query else ""))
    uct = upstream.headers.get("content-type")
    if uct and ".m3u8" not in safe_part.lower():
        ct = uct.split(";")[0].strip()

    is_m3u8 = ".m3u8" in safe_part.lower() or (uct and "mpegurl" in uct)

    if is_m3u8:
        try:
            await upstream.aclose()
        finally:
            await hc.aclose()
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as hc2:
            r2 = await hc2.get(full, headers={"X-Emby-Token": client._access_token or ""})
        if r2.status_code >= 400:
            raise HTTPException(status_code=502, detail="Playlist fetch failed")
        emby_root = client._root()
        bastion_base = str(request.base_url).rstrip("/")
        self_relay = f"{bastion_base}/api/emby/hls/relay"
        token_q = f"&token={quote(token, safe='')}" if token else ""

        def rewrite_child(text: str) -> str:
            lines_out: List[str] = []
            for line in text.splitlines():
                st = line.strip()
                if not st or st.startswith("#"):
                    lines_out.append(line)
                    continue
                res = st
                if not st.startswith("http://") and not st.startswith("https://"):
                    res = urljoin(full, st)
                sub = _path_after_emby(res, emby_root)
                if not sub:
                    lines_out.append(line)
                    continue
                try:
                    _validate_relay_path(sub.split("?")[0])
                except HTTPException:
                    lines_out.append(line)
                    continue
                lines_out.append(
                    f"{self_relay}?path={quote(sub, safe='')}{token_q}"
                )
            return "\n".join(lines_out) + ("\n" if text.endswith("\n") else "")

        body = rewrite_child(r2.text)
        return Response(
            content=body,
            media_type="application/vnd.apple.mpegurl",
            headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "no-cache"},
        )

    out_headers = {
        "Content-Type": ct,
        "Access-Control-Allow-Origin": "*",
        "Cache-Control": "public, max-age=60" if ".ts" in safe_part.lower() else "no-cache",
    }
    cr = upstream.headers.get("content-range")
    if cr:
        out_headers["Content-Range"] = cr
    cl = upstream.headers.get("content-length")
    if not cl and cr:
        cl = _content_length_from_content_range(cr)
    if cl:
        out_headers["Content-Length"] = cl

    return StreamingResponse(
        _stream_upstream_then_close(upstream, hc),
        status_code=upstream.status_code,
        headers=out_headers,
    )


class PlaybackReportBody(BaseModel):
    """Request body matches Emby JSON property names (PascalCase)."""

    ItemId: str
    MediaSourceId: Optional[str] = None
    PlaySessionId: Optional[str] = None
    PositionTicks: int = 0
    PlayMethod: str = "DirectStream"
    IsPaused: bool = False
    AudioStreamIndex: Optional[int] = None
    SubtitleStreamIndex: Optional[int] = None

    def as_emby(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)


class PlaybackProgressReportBody(PlaybackReportBody):
    EventName: str = "TimeUpdate"


@router.post("/api/emby/playback/start")
async def emby_playback_start(
    body: PlaybackReportBody,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    client = await _emby_client_for_user(current_user.user_id)
    await client.report_playback_start(body.as_emby())
    return {"success": True}


@router.post("/api/emby/playback/progress")
async def emby_playback_progress(
    body: PlaybackProgressReportBody,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    client = await _emby_client_for_user(current_user.user_id)
    await client.report_playback_progress(body.model_dump(exclude_none=True))
    return {"success": True}


@router.post("/api/emby/playback/stopped")
async def emby_playback_stopped(
    body: PlaybackReportBody,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    client = await _emby_client_for_user(current_user.user_id)
    payload = {
        "ItemId": body.ItemId,
        "MediaSourceId": body.MediaSourceId,
        "PlaySessionId": body.PlaySessionId,
        "PositionTicks": body.PositionTicks,
    }
    await client.report_playback_stopped({k: v for k, v in payload.items() if v is not None})
    return {"success": True}


@router.get("/api/emby/search")
async def emby_search(
    q: str,
    limit: int = 40,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    client = await _emby_client_for_user(current_user.user_id)
    return await client.search_items(q, limit=limit)
