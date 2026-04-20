"""HTTP client for koreader-sync-server (KoSync) JSON API."""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

KOSYNC_ACCEPT = "application/vnd.koreader.v1+json"
DEFAULT_TIMEOUT = httpx.Timeout(15.0, connect=8.0)


def md5_hex_password(plain: str) -> str:
    return hashlib.md5(plain.encode("utf-8")).hexdigest()


def _normalize_base(base_url: str) -> str:
    u = (base_url or "").strip().rstrip("/")
    if not u:
        return u
    return u


def _client(verify_ssl: bool) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, verify=verify_ssl, follow_redirects=True)


async def kosync_healthcheck(base_url: str, verify_ssl: bool = True) -> Tuple[bool, str]:
    base = _normalize_base(base_url)
    if not base:
        return False, "Missing base URL"
    url = f"{base}/healthcheck"
    try:
        async with _client(verify_ssl) as c:
            r = await c.get(url, headers={"Accept": KOSYNC_ACCEPT})
            if r.status_code == 200:
                return True, r.text[:500]
            return False, f"HTTP {r.status_code}"
    except Exception as e:
        logger.warning("KoSync healthcheck failed: %s", e)
        return False, str(e)


async def kosync_register(
    base_url: str, username: str, password_plain: str, verify_ssl: bool = True
) -> Tuple[int, Any]:
    base = _normalize_base(base_url)
    userkey = md5_hex_password(password_plain)
    url = f"{base}/users/create"
    async with _client(verify_ssl) as c:
        r = await c.post(
            url,
            headers={"Accept": KOSYNC_ACCEPT, "Content-Type": "application/json"},
            json={"username": username, "password": userkey},
        )
        try:
            body = r.json()
        except Exception:
            body = r.text
        return r.status_code, body


async def kosync_authorize(
    base_url: str, username: str, userkey: str, verify_ssl: bool = True
) -> Tuple[int, Any]:
    base = _normalize_base(base_url)
    url = f"{base}/users/auth"
    async with _client(verify_ssl) as c:
        r = await c.get(
            url,
            headers={
                "Accept": KOSYNC_ACCEPT,
                "x-auth-user": username,
                "x-auth-key": userkey,
            },
        )
        try:
            body = r.json()
        except Exception:
            body = r.text
        return r.status_code, body


async def kosync_get_progress(
    base_url: str,
    username: str,
    userkey: str,
    document: str,
    verify_ssl: bool = True,
) -> Tuple[int, Any]:
    base = _normalize_base(base_url)
    url = f"{base}/syncs/progress/{document}"
    async with _client(verify_ssl) as c:
        r = await c.get(
            url,
            headers={
                "Accept": KOSYNC_ACCEPT,
                "x-auth-user": username,
                "x-auth-key": userkey,
            },
        )
        try:
            body = r.json()
        except Exception:
            body = r.text
        return r.status_code, body


async def kosync_put_progress(
    base_url: str,
    username: str,
    userkey: str,
    payload: Dict[str, Any],
    verify_ssl: bool = True,
) -> Tuple[int, Any]:
    base = _normalize_base(base_url)
    url = f"{base}/syncs/progress"
    async with _client(verify_ssl) as c:
        r = await c.put(
            url,
            headers={
                "Accept": KOSYNC_ACCEPT,
                "Content-Type": "application/json",
                "x-auth-user": username,
                "x-auth-key": userkey,
            },
            json=payload,
        )
        try:
            body = r.json()
        except Exception:
            body = r.text
        return r.status_code, body
