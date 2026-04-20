"""Shared helpers for gRPC handler mixins."""

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import grpc

logger = logging.getLogger(__name__)


async def grpc_metadata_json_string_list(
    context: grpc.aio.ServicerContext,
    key: str,
) -> Optional[List[str]]:
    """
    If metadata entry `key` (case-insensitive) is present, parse value as JSON array of strings.
    Returns None if the key is absent (caller should treat as unrestricted).
    Returns [] if present but empty or invalid JSON.
    """
    try:
        inv = await context.invocation_metadata()
    except Exception:
        return None
    if not inv:
        return None
    want = (key or "").lower()
    for m in inv:
        mk = getattr(m, "key", None)
        if mk is None and isinstance(m, (tuple, list)) and len(m) >= 1:
            mk = m[0]
        if isinstance(mk, bytes):
            mk = mk.decode("utf-8", errors="replace")
        if (mk or "").lower() != want:
            continue
        mv = getattr(m, "value", None)
        if mv is None and isinstance(m, (tuple, list)) and len(m) >= 2:
            mv = m[1]
        if isinstance(mv, bytes):
            mv = mv.decode("utf-8", errors="replace")
        try:
            parsed = json.loads(mv or "[]")
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
            return []
        except json.JSONDecodeError:
            return []
    return None


def jsonb_list(val: Any, fallback: Optional[List[Any]] = None) -> List[Any]:
    """Safely convert a JSONB value (may be str, list, or None) to a Python list.

    asyncpg may return JSONB as str; ``list(str)`` would iterate characters and break
    allowlists (e.g. allowed_connections).
    """
    if fallback is None:
        fallback = []
    if val is None:
        return fallback
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else fallback
        except (json.JSONDecodeError, TypeError):
            return fallback
    return fallback


def jsonb_dict(val: Any, fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Safely convert a JSONB value (may be str, dict, or None) to a Python dict."""
    if fallback is None:
        fallback = {}
    if val is None:
        return fallback
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, dict) else fallback
        except (json.JSONDecodeError, TypeError):
            return fallback
    return fallback


def json_default(value: Any) -> str:
    """JSON serializer for non-primitive values returned from DB/service layers."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    return str(value)
