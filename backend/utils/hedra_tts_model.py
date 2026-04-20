"""
Hedra TTS engine selection: UUID model_id from Hedra GET /models (text_to_speech), not ElevenLabs API ids.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional
from uuid import UUID

import httpx

from services.user_settings_kv_service import get_user_setting
from services.user_voice_provider_service import (
    SETTING_USER_ADMIN_HEDRA_TTS_MODEL_ID,
    SETTING_USER_HEDRA_TTS_MODEL_ID,
)

logger = logging.getLogger(__name__)

HEDRA_PUBLIC_MODELS_URL = "https://api.hedra.com/web-app/public/models"


def validated_hedra_tts_model_id(raw: Optional[str]) -> str:
    """Return empty string (Hedra default engine) or a UUID string; raise ValueError if invalid."""
    s = (raw or "").strip()
    if not s:
        return ""
    try:
        return str(UUID(s))
    except ValueError as e:
        raise ValueError(
            "Invalid Hedra TTS model id. Pick a model from the list, or leave empty for the platform default."
        ) from e


async def resolve_hedra_tts_model_id_for_user(
    user_id: str,
    request: Any,
    use_admin_tts: bool,
) -> str:
    """Resolve model_id from request override (if set) else user KV; validate UUID."""
    req_mid = (getattr(request, "model_id", "") or "").strip()
    if req_mid:
        return validated_hedra_tts_model_id(req_mid)
    if use_admin_tts:
        stored = (
            await get_user_setting(user_id, SETTING_USER_ADMIN_HEDRA_TTS_MODEL_ID) or ""
        ).strip()
    else:
        stored = (
            await get_user_setting(user_id, SETTING_USER_HEDRA_TTS_MODEL_ID) or ""
        ).strip()
    return validated_hedra_tts_model_id(stored)


def _is_text_to_speech_model(entry: dict) -> bool:
    t = str(entry.get("type") or "").strip().lower().replace("-", "_")
    return t == "text_to_speech"


async def fetch_hedra_text_to_speech_models(api_key: str) -> List[dict[str, Any]]:
    """Return text_to_speech models from Hedra public API (names include ElevenLabs vs Minimax, etc.)."""
    key = (api_key or "").strip()
    if not key:
        return []
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            r = await client.get(
                HEDRA_PUBLIC_MODELS_URL,
                headers={"X-API-Key": key},
                params=[("types", "text_to_speech")],
            )
            if r.status_code == 422:
                r = await client.get(
                    HEDRA_PUBLIC_MODELS_URL,
                    headers={"X-API-Key": key},
                )
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        logger.warning("Hedra list models failed: %s", e)
        raise RuntimeError(f"Could not list Hedra TTS models: {e}") from e

    if not isinstance(data, list):
        return []
    out: List[dict[str, Any]] = []
    for m in data:
        if not isinstance(m, dict):
            continue
        if not _is_text_to_speech_model(m):
            continue
        mid = str(m.get("id") or "").strip()
        if not mid:
            continue
        out.append(
            {
                "id": mid,
                "name": str(m.get("name") or mid).strip() or mid,
                "type": str(m.get("type") or "text_to_speech"),
                "description": (m.get("description") or "") if m.get("description") is not None else "",
                "display_order": m.get("display_order"),
            }
        )
    out.sort(
        key=lambda x: (
            x["display_order"] is None,
            x["display_order"] if x["display_order"] is not None else 0,
            x["name"].lower(),
        )
    )
    return out
