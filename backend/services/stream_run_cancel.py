"""
Server-side cancel flags for interactive chat streaming runs.

Stop uses POST /api/v2/chat/unified/job/{run_id}/cancel which sets a short-lived Redis key.
The gRPC stream proxy polls this key and cancels the orchestrator RPC when set.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

logger = logging.getLogger(__name__)

STREAM_CANCEL_KEY_PREFIX = "bastion:stream_cancel:"
STREAM_CANCEL_TTL_SEC = 7200

_redis_client = None


async def _get_redis():
    global _redis_client
    if _redis_client is None:
        import redis.asyncio as redis
        from config import get_settings

        settings = get_settings()
        _redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis_client


def is_stream_run_id(job_id: str) -> bool:
    """True if job_id is a canonical UUID string (server-issued stream run_id)."""
    if not job_id or not isinstance(job_id, str):
        return False
    s = job_id.strip()
    if len(s) != 36:
        return False
    try:
        uuid.UUID(s)
        return True
    except (ValueError, AttributeError, TypeError):
        return False


async def set_stream_cancel_requested(run_id: str) -> bool:
    """Record user Stop for this stream run. Returns False if Redis is unavailable."""
    rid = (run_id or "").strip()
    if not is_stream_run_id(rid):
        return False
    try:
        r = await _get_redis()
        key = f"{STREAM_CANCEL_KEY_PREFIX}{rid}"
        await r.setex(key, STREAM_CANCEL_TTL_SEC, "1")
        logger.info("Stream cancel flag set for run_id=%s", rid)
        return True
    except Exception as e:
        logger.warning("Failed to set stream cancel flag: %s", e)
        return False


async def is_stream_cancel_requested(run_id: str) -> bool:
    try:
        r = await _get_redis()
        key = f"{STREAM_CANCEL_KEY_PREFIX}{(run_id or '').strip()}"
        v = await r.get(key)
        return bool(v)
    except Exception as e:
        logger.warning("Stream cancel check failed (treating as not cancelled): %s", e)
        return False
