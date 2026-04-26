"""
OpenRouter TTS model discovery (OpenAI-compatible /audio/speech; model slugs from Models API).
"""

from __future__ import annotations

import logging
import re
from typing import Any, List, Optional

import httpx

from services.user_settings_kv_service import get_user_setting
from services.user_voice_provider_service import (
    SETTING_USER_ADMIN_OPENROUTER_TTS_MODEL_ID,
    SETTING_USER_OPENROUTER_TTS_MODEL_ID,
)

logger = logging.getLogger(__name__)

OPENROUTER_DEFAULT_V1 = "https://openrouter.ai/api/v1"
_SLUG_RE = re.compile(r"^[a-zA-Z0-9_./\-]+$")


def normalize_openrouter_v1_root(base_url: Optional[str]) -> str:
    """Return API root ending with /v1."""
    root = (base_url or "").strip().rstrip("/")
    if not root:
        root = OPENROUTER_DEFAULT_V1.rstrip("/")
    if not root.endswith("/v1"):
        root = f"{root}/v1"
    return root


def validated_openrouter_tts_model_id(raw: Optional[str]) -> str:
    """Return empty (use voice-service deployer default) or a model slug; raise ValueError if invalid."""
    s = (raw or "").strip()
    if not s:
        return ""
    if len(s) > 128 or not _SLUG_RE.match(s):
        raise ValueError(
            "Invalid OpenRouter TTS model id. Pick a model from the list, or leave empty to use the server default."
        )
    return s


async def resolve_openrouter_tts_model_id_for_user(
    user_id: str,
    request: Any,
    use_admin_tts: bool,
) -> str:
    """Resolve model_id from request override (if set) else user KV."""
    req_mid = (getattr(request, "model_id", "") or "").strip()
    if req_mid:
        return validated_openrouter_tts_model_id(req_mid)
    if use_admin_tts:
        stored = (
            await get_user_setting(user_id, SETTING_USER_ADMIN_OPENROUTER_TTS_MODEL_ID) or ""
        ).strip()
    else:
        stored = (
            await get_user_setting(user_id, SETTING_USER_OPENROUTER_TTS_MODEL_ID) or ""
        ).strip()
    return validated_openrouter_tts_model_id(stored)


def _model_has_speech_output(entry: dict) -> bool:
    modalities = entry.get("output_modalities")
    if isinstance(modalities, list):
        if not modalities:
            return False
        return any(str(m).strip().lower() == "speech" for m in modalities)
    # Filtered Models API may omit the field on each row; treat as speech-capable.
    return True


async def fetch_openrouter_tts_models(
    api_key: str, base_url: Optional[str] = None
) -> List[dict[str, Any]]:
    """Return TTS-capable models from OpenRouter Models API."""
    key = (api_key or "").strip()
    if not key:
        return []
    root = normalize_openrouter_v1_root(base_url)
    url = f"{root}/models"
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            r = await client.get(
                url,
                headers={"Authorization": f"Bearer {key}"},
                params={"output_modalities": "speech"},
            )
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        logger.warning("OpenRouter list models failed: %s", e)
        raise RuntimeError(f"Could not list OpenRouter TTS models: {e}") from e

    raw_list: List[dict] = []
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        raw_list = [x for x in data["data"] if isinstance(x, dict)]
    elif isinstance(data, list):
        raw_list = [x for x in data if isinstance(x, dict)]

    out: List[dict[str, Any]] = []
    for m in raw_list:
        mid = str(m.get("id") or "").strip()
        if not mid:
            continue
        if not _model_has_speech_output(m):
            continue
        out.append(
            {
                "id": mid,
                "name": str(m.get("name") or mid).strip() or mid,
            }
        )
    return out
