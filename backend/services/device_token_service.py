"""
Device token service for Bastion Local Proxy daemon authentication.
"""

import hashlib
import logging
import secrets
import uuid
from typing import List, Optional, Tuple

from services.database_manager.database_helpers import fetch_all, fetch_one, execute

logger = logging.getLogger(__name__)


MAX_DEVICE_TOKENS_PER_USER = 10


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


async def create_device_token(user_id: str, device_name: str) -> Tuple[str, str]:
    """
    Create a new device token. Returns (token_id, raw_token).
    The raw token is shown only once; store the hash in the DB.
    Raises ValueError if the user already has MAX_DEVICE_TOKENS_PER_USER active tokens.
    """
    count = await count_active_tokens(user_id)
    if count >= MAX_DEVICE_TOKENS_PER_USER:
        raise ValueError(
            f"Maximum device tokens per user ({MAX_DEVICE_TOKENS_PER_USER}) reached. "
            "Revoke an existing token before creating a new one."
        )
    raw_token = secrets.token_urlsafe(32)
    token_hash = _hash_token(raw_token)
    token_id = str(uuid.uuid4())
    await execute(
        """
        INSERT INTO device_tokens (id, user_id, token_hash, device_name)
        VALUES ($1::uuid, $2, $3, $4)
        """,
        token_id,
        user_id,
        token_hash,
        device_name,
    )
    return token_id, raw_token


async def resolve_token(raw_token: str) -> Optional[dict]:
    """
    Validate a device token and return the token row (with user_id) if valid.
    Optionally updates last_connected_at and last_ip.
    """
    token_hash = _hash_token(raw_token)
    row = await fetch_one(
        """
        SELECT id, user_id, device_name, last_connected_at
        FROM device_tokens
        WHERE token_hash = $1 AND revoked_at IS NULL
        """,
        token_hash,
    )
    return dict(row) if row else None


async def count_active_tokens(user_id: str) -> int:
    """Return the number of non-revoked device tokens for the user."""
    row = await fetch_one(
        """
        SELECT COUNT(*) AS n FROM device_tokens
        WHERE user_id = $1 AND revoked_at IS NULL
        """,
        user_id,
    )
    return int(row["n"]) if row else 0


async def update_last_connected(token_id: str, ip: Optional[str] = None) -> None:
    """Update last_connected_at and optionally last_ip for a device token."""
    if ip:
        await execute(
            """
            UPDATE device_tokens
            SET last_connected_at = NOW(), last_ip = $2
            WHERE id = $1::uuid
            """,
            token_id,
            ip,
        )
    else:
        await execute(
            """
            UPDATE device_tokens
            SET last_connected_at = NOW()
            WHERE id = $1::uuid
            """,
            token_id,
        )


async def list_device_tokens(user_id: str) -> List[dict]:
    """List all device tokens for a user (without raw token)."""
    rows = await fetch_all(
        """
        SELECT id, device_name, last_connected_at, last_ip, created_at,
               (revoked_at IS NOT NULL) AS revoked
        FROM device_tokens
        WHERE user_id = $1
        ORDER BY created_at DESC
        """,
        user_id,
    )
    return [dict(r) for r in rows]


async def revoke_device_token(token_id: str, user_id: str) -> bool:
    """Revoke a device token. Returns True if a row was updated."""
    await execute(
        """
        UPDATE device_tokens
        SET revoked_at = NOW()
        WHERE id = $1::uuid AND user_id = $2 AND revoked_at IS NULL
        """,
        token_id,
        user_id,
    )
    return True
