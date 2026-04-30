"""
Expo Push API relay for mobile devices. Tokens stored in mobile_push_tokens.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import httpx

from services.database_manager.database_helpers import execute, fetch_all

logger = logging.getLogger(__name__)

EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"
MAX_TOKENS_PER_USER = 10
EXPO_TOKEN_PREFIX = "ExponentPushToken"


def _is_plausible_expo_token(token: str) -> bool:
    t = (token or "").strip()
    if len(t) < 22 or len(t) > 512:
        return False
    if t.startswith(EXPO_TOKEN_PREFIX):
        return True
    # Some clients may register FCM-style tokens when not using Expo Go
    return bool(re.match(r"^[A-Za-z0-9:_\-]+$", t))


async def register_push_token(
    user_id: str,
    token: str,
    platform: str,
    device_id: str,
    app_version: Optional[str] = None,
) -> None:
    """Upsert push token for user+device. Enforces max tokens per user (oldest revoked)."""
    uid = (user_id or "").strip()
    tok = (token or "").strip()
    plat = (platform or "unknown").strip().lower()[:20]
    did = (device_id or "").strip()[:255]
    if not uid or not tok or not did:
        raise ValueError("user_id, token, and device_id are required")
    if not _is_plausible_expo_token(tok):
        raise ValueError("Invalid push token format")

    rows = await fetch_all(
        """
        SELECT id FROM mobile_push_tokens
        WHERE user_id = $1 AND revoked_at IS NULL
        ORDER BY created_at ASC
        """,
        uid,
    )
    if rows and len(rows) >= MAX_TOKENS_PER_USER:
        excess = len(rows) - MAX_TOKENS_PER_USER + 1
        for r in rows[:excess]:
            await execute(
                "UPDATE mobile_push_tokens SET revoked_at = NOW() WHERE id = $1::uuid AND user_id = $2",
                r["id"],
                uid,
            )

    await execute(
        """
        INSERT INTO mobile_push_tokens (user_id, token, platform, device_id, app_version)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (user_id, device_id) DO UPDATE SET
            token = EXCLUDED.token,
            platform = EXCLUDED.platform,
            app_version = EXCLUDED.app_version,
            revoked_at = NULL,
            last_used_at = NOW()
        """,
        uid,
        tok,
        plat,
        did,
        (app_version or "")[:64] or None,
    )


async def revoke_push_token(user_id: str, device_id: str) -> bool:
    uid = (user_id or "").strip()
    did = (device_id or "").strip()
    if not uid or not did:
        return False
    await execute(
        """
        UPDATE mobile_push_tokens SET revoked_at = NOW()
        WHERE user_id = $1 AND device_id = $2 AND revoked_at IS NULL
        """,
        uid,
        did,
    )
    return True


async def list_active_tokens(user_id: str) -> List[str]:
    uid = (user_id or "").strip()
    if not uid:
        return []
    rows = await fetch_all(
        """
        SELECT token FROM mobile_push_tokens
        WHERE user_id = $1 AND revoked_at IS NULL
        ORDER BY last_used_at DESC NULLS LAST, created_at DESC
        """,
        uid,
    )
    return [r["token"] for r in (rows or []) if r.get("token")]


async def send_expo_push(
    tokens: List[str],
    title: str,
    body: str,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """Send one Expo push message to many device tokens. Returns parsed API JSON."""
    if not tokens:
        return {"data": []}
    messages = []
    for t in tokens:
        messages.append(
            {
                "to": t,
                "title": (title or "")[:256],
                "body": (body or "")[:512],
                "data": {k: str(v)[:500] for k, v in (data or {}).items()},
                "sound": "default",
            }
        )
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                EXPO_PUSH_URL,
                json=messages,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.warning("Expo push send failed: %s", e)
        return {"data": [], "error": str(e)}


async def handle_expo_response_and_revoke_dead(user_id: str, expo_json: Dict[str, Any]) -> None:
    """Revoke tokens that Expo reports as DeviceNotRegistered."""
    uid = (user_id or "").strip()
    if not uid:
        return
    for item in (expo_json or {}).get("data") or []:
        if not isinstance(item, dict):
            continue
        status = (item.get("status") or "").strip()
        if status != "error":
            continue
        details = item.get("details") or {}
        err = (details.get("error") if isinstance(details, dict) else None) or ""
        if "DeviceNotRegistered" not in str(err):
            continue
        tok = (item.get("to") or "").strip()
        if not tok:
            continue
        await execute(
            """
            UPDATE mobile_push_tokens SET revoked_at = NOW()
            WHERE user_id = $1 AND token = $2 AND revoked_at IS NULL
            """,
            uid,
            tok,
        )


async def send_push_for_user(
    user_id: str,
    title: str,
    body: str,
    data: Dict[str, Any],
) -> bool:
    """Send push to all active tokens for user. Returns True if at least one ticket succeeded."""
    tokens = await list_active_tokens(user_id)
    if not tokens:
        return False
    result = await send_expo_push(tokens, title, body, data)
    await handle_expo_response_and_revoke_dead(user_id, result)
    for item in result.get("data") or []:
        if isinstance(item, dict) and item.get("status") == "ok":
            return True
    return False
