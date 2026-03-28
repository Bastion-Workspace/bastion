"""
ElevenLabs TTS synthesis model allowlist and validation for user settings / API.
"""

from typing import Any, Optional

ALLOWED_ELEVENLABS_TTS_MODEL_IDS = frozenset(
    {
        "eleven_multilingual_v2",
        "eleven_turbo_v2_5",
        "eleven_flash_v2_5",
        "eleven_v3",
    }
)


def validated_elevenlabs_model_id(raw: Optional[str]) -> str:
    """Return empty string (platform default) or a known model id; raise ValueError if invalid."""
    s = (raw or "").strip()
    if not s:
        return ""
    if s not in ALLOWED_ELEVENLABS_TTS_MODEL_IDS:
        raise ValueError(
            "Invalid ElevenLabs synthesis model. Use a supported model id or leave empty for the platform default."
        )
    return s


async def resolve_elevenlabs_tts_model_id_for_user(
    user_id: str,
    request: Any,
    use_admin_tts: bool,
) -> str:
    """Resolve model_id from request override (if set) else user KV; validate allowlist."""
    from services.user_settings_kv_service import get_user_setting
    from services.user_voice_provider_service import (
        SETTING_USER_ADMIN_ELEVENLABS_TTS_MODEL_ID,
        SETTING_USER_ELEVENLABS_TTS_MODEL_ID,
    )

    req_mid = (getattr(request, "model_id", "") or "").strip()
    if req_mid:
        return validated_elevenlabs_model_id(req_mid)
    if use_admin_tts:
        stored = (
            await get_user_setting(user_id, SETTING_USER_ADMIN_ELEVENLABS_TTS_MODEL_ID) or ""
        ).strip()
    else:
        stored = (
            await get_user_setting(user_id, SETTING_USER_ELEVENLABS_TTS_MODEL_ID) or ""
        ).strip()
    return validated_elevenlabs_model_id(stored)
