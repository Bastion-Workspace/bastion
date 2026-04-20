"""
Emby media client: BaseMusicClient for music cache/streaming plus video library helpers.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import httpx

from .base_client import BaseMusicClient

logger = logging.getLogger(__name__)


def _ticks_to_seconds(ticks: Optional[int]) -> int:
    if not ticks:
        return 0
    try:
        return int(ticks) // 10_000_000
    except (TypeError, ValueError):
        return 0


class EmbyClient(BaseMusicClient):
    """Client for Emby servers (music via Items API, video via extended methods)."""

    def __init__(self, server_url: str, username: str, password: str, **kwargs):
        super().__init__(server_url, username, password, **kwargs)
        self._access_token: Optional[str] = None
        self._user_id: Optional[str] = None
        self._device_id = kwargs.get("device_id") or f"bastion-{uuid.uuid4().hex[:16]}"

    def _root(self) -> str:
        base = self.server_url.rstrip("/")
        if base.lower().endswith("/emby"):
            return base
        return f"{base}/emby"

    def _url(self, path: str) -> str:
        path = path if path.startswith("/") else f"/{path}"
        return f"{self._root()}{path}"

    def _emby_auth_header(self) -> str:
        return (
            f'MediaBrowser Client="Bastion", Device="Server", '
            f'DeviceId="{self._device_id}", Version="1.0"'
        )

    async def _authenticate(self) -> None:
        if self._access_token and self._user_id:
            return
        url = self._url("/Users/AuthenticateByName")
        headers = {
            "Content-Type": "application/json",
            "X-Emby-Authorization": self._emby_auth_header(),
        }
        body = {"Username": self.username, "Pw": self.password}
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
        self._access_token = data.get("AccessToken")
        user = data.get("User") or {}
        self._user_id = user.get("Id")
        if not self._access_token or not self._user_id:
            raise ValueError("Emby authentication response missing token or user id")

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        retry_auth: bool = True,
    ) -> httpx.Response:
        await self._authenticate()
        headers = {
            "X-Emby-Token": self._access_token or "",
            "X-Emby-Authorization": self._emby_auth_header(),
        }
        url = self._url(path)
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            resp = await client.request(method, url, headers=headers, params=params, json=json_body)
        if resp.status_code == 401 and retry_auth:
            self._access_token = None
            self._user_id = None
            await self._authenticate()
            headers["X-Emby-Token"] = self._access_token or ""
            async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
                resp = await client.request(method, url, headers=headers, params=params, json=json_body)
        return resp

    async def test_connection(self) -> Dict[str, Any]:
        try:
            await self._authenticate()
            r = await self._request("GET", f"/Users/{self._user_id}")
            if r.status_code == 200:
                return {"success": True, "auth_method_used": "password"}
            return {"success": False, "error": f"HTTP {r.status_code}"}
        except Exception as e:
            logger.warning("Emby connection test failed: %s", e)
            return {"success": False, "error": str(e)}

    def _normalize_item_album(self, item: Dict[str, Any]) -> Dict[str, Any]:
        album_artist = item.get("AlbumArtist") or item.get("SeriesName") or ""
        return {
            "id": str(item.get("Id", "")),
            "title": item.get("Name") or "",
            "artist": album_artist,
            "cover_art_id": str(item.get("Id", "")),
            "metadata": item,
        }

    def _normalize_item_artist(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": str(item.get("Id", "")),
            "name": item.get("Name") or "",
            "metadata": item,
        }

    def _normalize_item_playlist(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": str(item.get("Id", "")),
            "name": item.get("Name") or "",
            "track_count": item.get("ChildCount") or 0,
            "metadata": item,
        }

    def _normalize_item_track(self, item: Dict[str, Any], parent_id: Optional[str]) -> Dict[str, Any]:
        ticks = item.get("RunTimeTicks")
        duration = _ticks_to_seconds(ticks) if ticks else item.get("RunTimeSeconds") or 0
        artists = item.get("Artists") or []
        artist_name = item.get("AlbumArtist") or (artists[0] if artists else "")
        return self.normalize_track(
            {
                "id": item.get("Id"),
                "title": item.get("Name"),
                "artist": artist_name,
                "album": item.get("Album") or "",
                "duration": duration,
                "track_number": item.get("IndexNumber"),
                "cover_art_id": item.get("AlbumId") or item.get("ParentId") or "",
            },
            parent_id=parent_id,
        )

    async def get_albums(self) -> List[Dict[str, Any]]:
        await self._authenticate()
        params = {
            "IncludeItemTypes": "MusicAlbum",
            "Recursive": "true",
            "Fields": "PrimaryImageTag,AlbumArtist,Overview",
            "SortBy": "SortName",
            "SortOrder": "Ascending",
            "Limit": "10000",
        }
        r = await self._request("GET", f"/Users/{self._user_id}/Items", params=params)
        r.raise_for_status()
        data = r.json()
        items = data.get("Items") or []
        return [self._normalize_item_album(x) for x in items]

    async def get_artists(self) -> List[Dict[str, Any]]:
        await self._authenticate()
        params = {
            "IncludeItemTypes": "MusicArtist",
            "Recursive": "true",
            "Fields": "PrimaryImageTag",
            "SortBy": "SortName",
            "Limit": "10000",
        }
        r = await self._request("GET", f"/Users/{self._user_id}/Items", params=params)
        r.raise_for_status()
        data = r.json()
        items = data.get("Items") or []
        return [self._normalize_item_artist(x) for x in items]

    async def get_playlists(self) -> List[Dict[str, Any]]:
        await self._authenticate()
        params = {
            "IncludeItemTypes": "Playlist",
            "Recursive": "true",
            "Fields": "ChildCount",
            "SortBy": "SortName",
            "Limit": "5000",
        }
        r = await self._request("GET", f"/Users/{self._user_id}/Items", params=params)
        r.raise_for_status()
        data = r.json()
        items = data.get("Items") or []
        return [self._normalize_item_playlist(x) for x in items]

    async def get_album_tracks(self, album_id: str) -> List[Dict[str, Any]]:
        await self._authenticate()
        params = {
            "ParentId": album_id,
            "IncludeItemTypes": "Audio",
            "Fields": "MediaSources,RunTimeTicks,AlbumArtist,Artists,Album,IndexNumber",
            "SortBy": "IndexNumber",
        }
        r = await self._request("GET", f"/Users/{self._user_id}/Items", params=params)
        r.raise_for_status()
        data = r.json()
        items = data.get("Items") or []
        return [self._normalize_item_track(x, album_id) for x in items]

    async def get_playlist_tracks(self, playlist_id: str) -> List[Dict[str, Any]]:
        return await self.get_album_tracks(playlist_id)

    async def get_stream_url(self, track_id: str) -> Optional[str]:
        await self._authenticate()
        q = urlencode({"static": "true", "api_key": self._access_token})
        return f"{self._root()}/Audio/{track_id}/stream?{q}"

    async def search_tracks(self, query: str, limit: int = 25) -> List[Dict[str, Any]]:
        await self._authenticate()
        params = {
            "SearchTerm": query,
            "IncludeItemTypes": "Audio",
            "Recursive": "true",
            "Limit": str(limit),
            "Fields": "RunTimeTicks,AlbumArtist,Album,IndexNumber",
        }
        r = await self._request("GET", f"/Users/{self._user_id}/Items", params=params)
        r.raise_for_status()
        data = r.json()
        items = data.get("Items") or []
        return [self._normalize_item_track(x, None) for x in items]

    # --- Video / library (used by emby_api) ---

    async def get_libraries(self) -> List[Dict[str, Any]]:
        await self._authenticate()
        r = await self._request("GET", f"/Users/{self._user_id}/Views")
        r.raise_for_status()
        data = r.json()
        items = data.get("Items") or []
        return [
            {
                "id": str(x.get("Id", "")),
                "name": x.get("Name", ""),
                "collection_type": x.get("CollectionType", ""),
                "item_id": str(x.get("Id", "")),
                "raw": x,
            }
            for x in items
        ]

    async def get_items(
        self,
        *,
        parent_id: Optional[str] = None,
        item_types: Optional[str] = None,
        sort_by: str = "SortName",
        sort_order: str = "Ascending",
        limit: int = 100,
        start_index: int = 0,
        recursive: bool = False,
    ) -> Dict[str, Any]:
        await self._authenticate()
        params: Dict[str, Any] = {
            "SortBy": sort_by,
            "SortOrder": sort_order,
            "Limit": str(limit),
            "StartIndex": str(start_index),
            "Fields": "PrimaryImageTag,BackdropImageTags,Overview,RunTimeTicks,UserData,ProductionYear,Type,SeriesName,IndexNumber,ParentIndexNumber",
        }
        if parent_id:
            params["ParentId"] = parent_id
        if item_types:
            params["IncludeItemTypes"] = item_types
        if recursive:
            params["Recursive"] = "true"
        r = await self._request("GET", f"/Users/{self._user_id}/Items", params=params)
        r.raise_for_status()
        return r.json()

    async def get_latest_items(self, parent_id: str, limit: int = 24) -> List[Dict[str, Any]]:
        await self._authenticate()
        params = {"ParentId": parent_id, "Limit": str(limit)}
        r = await self._request("GET", f"/Users/{self._user_id}/Items/Latest", params=params)
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else data.get("Items") or []

    async def get_resume_items(self, limit: int = 50) -> List[Dict[str, Any]]:
        await self._authenticate()
        params = {
            "Limit": str(limit),
            "IncludeItemTypes": "Movie,Episode",
            "Fields": "PrimaryImageTag,UserData,RunTimeTicks,SeriesName,ParentIndexNumber,IndexNumber",
        }
        r = await self._request("GET", f"/Users/{self._user_id}/Items/Resume", params=params)
        r.raise_for_status()
        data = r.json()
        return data.get("Items") or []

    async def get_item(self, item_id: str) -> Dict[str, Any]:
        await self._authenticate()
        params = {
            "Fields": "PrimaryImageTag,BackdropImageTags,Overview,RunTimeTicks,UserData,Path,MediaSources,Type,SeriesName,ParentIndexNumber,IndexNumber,SeasonName,SeriesId",
        }
        r = await self._request("GET", f"/Users/{self._user_id}/Items/{item_id}", params=params)
        r.raise_for_status()
        return r.json()

    async def get_seasons(self, series_id: str) -> List[Dict[str, Any]]:
        await self._authenticate()
        params = {"UserId": self._user_id, "Fields": "PrimaryImageTag,UserData,IndexNumber"}
        r = await self._request("GET", f"/Shows/{series_id}/Seasons", params=params)
        r.raise_for_status()
        data = r.json()
        return data.get("Items") or []

    async def get_episodes(self, series_id: str, season_id: str) -> List[Dict[str, Any]]:
        await self._authenticate()
        params = {
            "UserId": self._user_id,
            "SeasonId": season_id,
            "Fields": "PrimaryImageTag,Overview,RunTimeTicks,UserData,IndexNumber,ParentIndexNumber",
        }
        r = await self._request("GET", f"/Shows/{series_id}/Episodes", params=params)
        r.raise_for_status()
        data = r.json()
        return data.get("Items") or []

    async def post_playback_info(self, item_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        await self._authenticate()
        payload = dict(body)
        payload.setdefault("UserId", self._user_id)
        r = await self._request("POST", f"/Items/{item_id}/PlaybackInfo", json_body=payload)
        r.raise_for_status()
        return r.json()

    def build_upstream_video_stream_url(
        self,
        item_id: str,
        *,
        media_source_id: str,
        play_session_id: str,
        static: bool = True,
        start_time_ticks: int = 0,
        audio_stream_index: Optional[int] = None,
    ) -> str:
        """Direct progressive stream URL (api_key for upstream fetch from Bastion proxy)."""
        params: Dict[str, str] = {
            "Static": "true" if static else "false",
            "MediaSourceId": media_source_id,
            "PlaySessionId": play_session_id,
            "api_key": self._access_token or "",
        }
        if start_time_ticks:
            params["StartTimeTicks"] = str(start_time_ticks)
        if audio_stream_index is not None:
            params["AudioStreamIndex"] = str(int(audio_stream_index))
        q = urlencode(params)
        return f"{self._root()}/Videos/{item_id}/stream?{q}"

    def build_upstream_hls_master_url(
        self,
        item_id: str,
        *,
        media_source_id: str,
        play_session_id: str,
        audio_stream_index: Optional[int] = None,
    ) -> str:
        params = {
            "MediaSourceId": media_source_id,
            "PlaySessionId": play_session_id,
            "api_key": self._access_token or "",
        }
        if audio_stream_index is not None:
            params["AudioStreamIndex"] = str(int(audio_stream_index))
        q = urlencode(params)
        return f"{self._root()}/Videos/{item_id}/master.m3u8?{q}"

    def build_upstream_image_url(
        self,
        item_id: str,
        image_type: str = "Primary",
        *,
        max_width: int = 400,
        image_index: int = 0,
        tag: Optional[str] = None,
    ) -> str:
        params: Dict[str, str] = {"maxWidth": str(max_width), "api_key": self._access_token or ""}
        if tag:
            params["tag"] = tag
        q = urlencode(params)
        if image_index:
            return f"{self._root()}/Items/{item_id}/Images/{image_type}/{image_index}?{q}"
        return f"{self._root()}/Items/{item_id}/Images/{image_type}?{q}"

    async def report_playback_start(self, payload: Dict[str, Any]) -> None:
        await self._authenticate()
        r = await self._request("POST", "/Sessions/Playing", json_body=payload)
        if r.status_code >= 400:
            logger.warning("Emby playback start failed: %s %s", r.status_code, r.text[:200])

    async def report_playback_progress(self, payload: Dict[str, Any]) -> None:
        await self._authenticate()
        r = await self._request("POST", "/Sessions/Playing/Progress", json_body=payload)
        if r.status_code >= 400:
            logger.warning("Emby playback progress failed: %s", r.status_code)

    async def report_playback_stopped(self, payload: Dict[str, Any]) -> None:
        await self._authenticate()
        r = await self._request("POST", "/Sessions/Playing/Stopped", json_body=payload)
        if r.status_code >= 400:
            logger.warning("Emby playback stopped failed: %s", r.status_code)

    async def search_items(self, query: str, limit: int = 40) -> Dict[str, Any]:
        await self._authenticate()
        params = {
            "SearchTerm": query,
            "Recursive": "true",
            "Limit": str(limit),
            "Fields": "PrimaryImageTag,Type,RunTimeTicks,ProductionYear,UserData",
        }
        r = await self._request("GET", f"/Users/{self._user_id}/Items", params=params)
        r.raise_for_status()
        return r.json()
