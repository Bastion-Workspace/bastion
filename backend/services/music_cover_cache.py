"""Per-user LRU disk cache for proxied music cover art."""

import asyncio
import errno
import hashlib
import logging
import os
import shutil
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

import httpx

from config import settings
from services.database_manager.database_helpers import execute, fetch_all, fetch_one

logger = logging.getLogger(__name__)

_RETRYABLE_COVER_STATUS = frozenset({502, 503, 504})
_cover_upstream_sem: Optional[asyncio.Semaphore] = None


def _get_cover_upstream_semaphore() -> asyncio.Semaphore:
    global _cover_upstream_sem
    if _cover_upstream_sem is None:
        n = max(1, int(getattr(settings, "MUSIC_COVER_UPSTREAM_MAX_CONCURRENT", 8) or 8))
        _cover_upstream_sem = asyncio.Semaphore(n)
    return _cover_upstream_sem


async def fetch_cover_upstream(url: str) -> Tuple[bytes, str]:
    """
    Fetch cover bytes from Subsonic/Navidrome with bounded concurrency and retries.
    Browsers request many thumbnails at once; without this, upstream often returns 504.
    """
    attempts = max(1, int(getattr(settings, "MUSIC_COVER_UPSTREAM_RETRY_ATTEMPTS", 4) or 4))
    read_s = float(getattr(settings, "MUSIC_COVER_UPSTREAM_TIMEOUT_S", 45.0) or 45.0)
    connect_s = float(
        getattr(settings, "MUSIC_COVER_UPSTREAM_CONNECT_TIMEOUT_S", 15.0) or 15.0
    )
    timeout = httpx.Timeout(
        connect=connect_s,
        read=read_s,
        write=read_s,
        pool=read_s,
    )
    last_status = 0
    last_net_err: Optional[Exception] = None

    for attempt in range(attempts):
        async with _get_cover_upstream_semaphore():
            try:
                async with httpx.AsyncClient(
                    timeout=timeout, follow_redirects=True
                ) as client:
                    upstream = await client.get(url)
                if upstream.status_code == 200:
                    ct = upstream.headers.get("content-type", "image/jpeg")
                    return upstream.content, ct
                last_status = upstream.status_code
                if upstream.status_code not in _RETRYABLE_COVER_STATUS:
                    raise RuntimeError(f"upstream_http_{upstream.status_code}")
            except httpx.RequestError as e:
                last_net_err = e
                last_status = 0
            except RuntimeError:
                raise

        if attempt < attempts - 1:
            await asyncio.sleep(0.2 * (2**attempt))

    if last_net_err:
        logger.warning(
            "Cover upstream network error after %s attempts: %s",
            attempts,
            last_net_err,
        )
    raise RuntimeError(f"upstream_http_{last_status or 504}")


def compute_etag(user_id: str, service_type: str, cover_art_id: str, size: int) -> str:
    payload = f"{user_id}:{service_type}:{cover_art_id}:{size}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def etag_header_value(etag_hex: str) -> str:
    return '"' + etag_hex + '"'


def if_none_match_matches(header_val: Optional[str], etag_hex: str) -> bool:
    if not header_val:
        return False
    quoted = etag_header_value(etag_hex)
    for raw in header_val.split(","):
        part = raw.strip()
        if part == "*" or part == quoted:
            return True
        if part.startswith("W/") and len(part) > 2:
            inner = part[2:].strip()
            if inner == quoted:
                return True
    return False


def _fs_segment(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_")


def relative_blob_path(user_id: str, service_type: str, cover_art_id: str, size: int) -> str:
    h = hashlib.sha1(cover_art_id.encode("utf-8")).hexdigest()
    prefix = h[:2]
    safe_uid = _fs_segment(user_id)
    safe_st = _fs_segment(service_type)
    return f"{safe_uid}/{safe_st}/{prefix}/{h}_{size}.bin"


def abs_blob_path(rel_path: str) -> str:
    base = (settings.MUSIC_COVER_CACHE_DIR or "/app/music_cover_cache").rstrip(os.sep)
    return os.path.join(base, *rel_path.split("/"))


def _write_blob_atomic(abs_path: str, data: bytes) -> None:
    parent = os.path.dirname(abs_path)
    os.makedirs(parent, exist_ok=True)
    tmp = abs_path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, abs_path)


async def _read_blob(abs_path: str) -> bytes:
    def _read() -> bytes:
        with open(abs_path, "rb") as fh:
            return fh.read()

    return await asyncio.to_thread(_read)


async def _unlink_blob(abs_path: str) -> None:
    def _unlink() -> None:
        try:
            os.remove(abs_path)
        except FileNotFoundError:
            pass

    await asyncio.to_thread(_unlink)


async def evict_lru_if_needed(user_id: str, rls_context: Dict[str, str]) -> None:
    cap = int(getattr(settings, "MUSIC_COVER_CACHE_MAX_BYTES_PER_USER", 0) or 0)
    if cap <= 0:
        return

    row = await fetch_one(
        """SELECT COALESCE(SUM(bytes), 0)::bigint AS total FROM music_cover_cache_index
           WHERE user_id = $1""",
        user_id,
        rls_context=rls_context,
    )
    total = int(row["total"] if row else 0)

    while total > cap:
        victims = await fetch_all(
            """SELECT id, path, bytes FROM music_cover_cache_index
               WHERE user_id = $1 ORDER BY accessed_at ASC NULLS FIRST LIMIT 40""",
            user_id,
            rls_context=rls_context,
        )
        if not victims:
            break
        for v in victims:
            vid = v["id"]
            rel = v["path"]
            b = int(v["bytes"] or 0)
            abs_p = abs_blob_path(rel)
            await execute(
                "DELETE FROM music_cover_cache_index WHERE id = $1",
                vid,
                rls_context=rls_context,
            )
            await _unlink_blob(abs_p)
            await _unlink_blob(abs_p + ".meta.json")
            total -= b


async def _upsert_index_row(
    user_id: str,
    service_type: str,
    cover_art_id: str,
    size: int,
    bytes_len: int,
    etag_hex: str,
    content_type: str,
    rel_path: str,
    rls_context: Dict[str, str],
) -> None:
    await execute(
        """INSERT INTO music_cover_cache_index
           (user_id, service_type, cover_art_id, size, bytes, etag, content_type, path, created_at, accessed_at)
           VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW(), NOW())
           ON CONFLICT (user_id, service_type, cover_art_id, size) DO UPDATE SET
             bytes = EXCLUDED.bytes,
             etag = EXCLUDED.etag,
             content_type = EXCLUDED.content_type,
             path = EXCLUDED.path,
             accessed_at = NOW()""",
        user_id,
        service_type,
        cover_art_id,
        size,
        bytes_len,
        etag_hex,
        content_type,
        rel_path,
        rls_context=rls_context,
    )


async def get_cached_etag_only(
    user_id: str,
    service_type: str,
    cover_art_id: str,
    size: int,
    rls_context: Dict[str, str],
) -> Optional[str]:
    row = await fetch_one(
        """SELECT etag FROM music_cover_cache_index
           WHERE user_id = $1 AND service_type = $2 AND cover_art_id = $3 AND size = $4""",
        user_id,
        service_type,
        cover_art_id,
        size,
        rls_context=rls_context,
    )
    return row["etag"] if row else None


async def get_or_fetch(
    user_id: str,
    service_type: str,
    cover_art_id: str,
    size: int,
    rls_context: Dict[str, str],
    upstream_fetcher: Callable[[], Awaitable[Tuple[bytes, str]]],
) -> Tuple[bytes, str, str]:
    etag_hex = compute_etag(user_id, service_type, cover_art_id, size)

    row = await fetch_one(
        """SELECT id, path, content_type, etag FROM music_cover_cache_index
           WHERE user_id = $1 AND service_type = $2 AND cover_art_id = $3 AND size = $4""",
        user_id,
        service_type,
        cover_art_id,
        size,
        rls_context=rls_context,
    )

    if row:
        abs_p = abs_blob_path(row["path"])
        try:
            body = await _read_blob(abs_p)
        except OSError as e:
            if getattr(e, "errno", None) != errno.ENOENT:
                raise
            await execute(
                "DELETE FROM music_cover_cache_index WHERE id = $1",
                row["id"],
                rls_context=rls_context,
            )
        else:
            await execute(
                "UPDATE music_cover_cache_index SET accessed_at = NOW() WHERE id = $1",
                row["id"],
                rls_context=rls_context,
            )
            return body, row["content_type"], row["etag"]

    body, content_type = await upstream_fetcher()
    ct = (content_type or "image/jpeg").split(";")[0].strip() or "image/jpeg"
    rel = relative_blob_path(user_id, service_type, cover_art_id, size)
    abs_p = abs_blob_path(rel)
    await asyncio.to_thread(_write_blob_atomic, abs_p, body)
    await _upsert_index_row(
        user_id,
        service_type,
        cover_art_id,
        size,
        len(body),
        etag_hex,
        ct,
        rel,
        rls_context,
    )
    await evict_lru_if_needed(user_id, rls_context)
    return body, ct, etag_hex


async def purge_cover_cache_for_user_service(
    user_id: str, service_type: str, rls_context: Dict[str, str]
) -> None:
    rows = await fetch_all(
        """SELECT path FROM music_cover_cache_index WHERE user_id = $1 AND service_type = $2""",
        user_id,
        service_type,
        rls_context=rls_context,
    )
    await execute(
        "DELETE FROM music_cover_cache_index WHERE user_id = $1 AND service_type = $2",
        user_id,
        service_type,
        rls_context=rls_context,
    )
    for r in rows or []:
        rel = r.get("path") or ""
        if not rel:
            continue
        abs_p = abs_blob_path(rel)
        await _unlink_blob(abs_p)
        await _unlink_blob(abs_p + ".meta.json")
    safe_uid = _fs_segment(user_id)
    safe_st = _fs_segment(service_type)
    tree = abs_blob_path(f"{safe_uid}/{safe_st}")

    def _rm_tree() -> None:
        try:
            shutil.rmtree(tree, ignore_errors=True)
        except Exception as exc:
            logger.debug("purge_cover_cache_for_user_service rmtree %s: %s", tree, exc)

    await asyncio.to_thread(_rm_tree)


async def purge_cover_cache_for_user(user_id: str, rls_context: Dict[str, str]) -> None:
    rows = await fetch_all(
        """SELECT path FROM music_cover_cache_index WHERE user_id = $1""",
        user_id,
        rls_context=rls_context,
    )
    await execute(
        "DELETE FROM music_cover_cache_index WHERE user_id = $1",
        user_id,
        rls_context=rls_context,
    )
    for r in rows or []:
        rel = r.get("path") or ""
        if not rel:
            continue
        abs_p = abs_blob_path(rel)
        await _unlink_blob(abs_p)
        await _unlink_blob(abs_p + ".meta.json")
    tree = abs_blob_path(_fs_segment(user_id))

    def _rm_tree() -> None:
        try:
            shutil.rmtree(tree, ignore_errors=True)
        except Exception as exc:
            logger.debug("purge_cover_cache_for_user rmtree %s: %s", tree, exc)

    await asyncio.to_thread(_rm_tree)


def parse_warm_sizes(raw: Optional[str]) -> List[int]:
    out: List[int] = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            continue
    return out if out else [64, 200]
