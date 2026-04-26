"""
User voice provider API (BYOK TTS/STT).
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException

from models.api_models import AuthenticatedUserResponse
from config import settings
from services.user_voice_provider_service import (
    SETTING_USE_ADMIN_STT,
    SETTING_USE_ADMIN_TTS,
    SETTING_USER_ADMIN_ELEVENLABS_TTS_MODEL_ID,
    SETTING_USER_ADMIN_HEDRA_TTS_MODEL_ID,
    SETTING_USER_ADMIN_OPENROUTER_TTS_MODEL_ID,
    SETTING_USER_ADMIN_TTS_PROVIDER,
    SETTING_USER_ADMIN_TTS_VOICE_ID,
    SETTING_USER_ADMIN_TTS_VOICE_PIPER,
    SETTING_USER_ADMIN_TTS_VOICE_SERVER,
    SETTING_USER_BYOK_TTS_ENGINE,
    SETTING_USER_BYOK_TTS_VOICE_PIPER,
    SETTING_USER_ELEVENLABS_TTS_MODEL_ID,
    SETTING_USER_HEDRA_TTS_MODEL_ID,
    SETTING_USER_OPENROUTER_TTS_MODEL_ID,
    SETTING_USER_STT_PROVIDER_ID,
    SETTING_USER_TTS_PROVIDER_ID,
    SETTING_USER_TTS_VOICE_ID,
    user_voice_provider_service,
)
from utils.elevenlabs_tts_model import validated_elevenlabs_model_id
from utils.hedra_tts_model import (
    fetch_hedra_text_to_speech_models,
    validated_hedra_tts_model_id,
)
from utils.openrouter_tts_model import (
    fetch_openrouter_tts_models,
    validated_openrouter_tts_model_id,
)
from services.user_settings_kv_service import get_user_setting, set_user_setting
from utils.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/user", tags=["User Voice Providers"])


def _truthy(raw: Optional[str]) -> bool:
    if raw is None:
        return True
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


@router.get("/voice-providers")
async def list_voice_providers(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    try:
        providers = await user_voice_provider_service.list_providers(
            current_user.user_id
        )
        return {"providers": providers}
    except Exception as e:
        logger.exception("list_voice_providers failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/voice-providers")
async def add_voice_provider(
    request: Dict[str, Any],
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    try:
        provider_type = request.get("provider_type")
        provider_role = request.get("provider_role")
        if not provider_type or not provider_role:
            raise HTTPException(
                status_code=400,
                detail="provider_type and provider_role required",
            )
        provider_id = await user_voice_provider_service.add_provider(
            current_user.user_id,
            provider_type=str(provider_type).strip().lower(),
            provider_role=str(provider_role).strip().lower(),
            api_key=request.get("api_key"),
            base_url=request.get("base_url"),
            display_name=request.get("display_name"),
        )
        providers = await user_voice_provider_service.list_providers(
            current_user.user_id
        )
        return {"provider_id": provider_id, "status": "success", "providers": providers}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("add_voice_provider failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/voice-providers/{provider_id:int}")
async def remove_voice_provider(
    provider_id: int,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    try:
        await user_voice_provider_service.remove_provider(
            current_user.user_id, provider_id
        )
        return {"status": "success"}
    except Exception as e:
        logger.exception("remove_voice_provider failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/voice-providers/{provider_id:int}/voices")
async def get_voice_provider_voices(
    provider_id: int,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    try:
        voices = await user_voice_provider_service.list_provider_voices(
            current_user.user_id, provider_id
        )
        return {"success": True, "voices": voices}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        logger.exception("get_voice_provider_voices failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


async def _voice_settings_payload(user_id: str) -> Dict[str, Any]:
    use_admin_tts = _truthy(await get_user_setting(user_id, SETTING_USE_ADMIN_TTS))
    user_tts_voice = (await get_user_setting(user_id, SETTING_USER_TTS_VOICE_ID) or "").strip()
    admin_tts_voice_legacy = (
        await get_user_setting(user_id, SETTING_USER_ADMIN_TTS_VOICE_ID) or ""
    ).strip()
    admin_server_v = (
        await get_user_setting(user_id, SETTING_USER_ADMIN_TTS_VOICE_SERVER) or ""
    ).strip()
    admin_piper_v = (
        await get_user_setting(user_id, SETTING_USER_ADMIN_TTS_VOICE_PIPER) or ""
    ).strip()
    admin_tts_provider = (
        await get_user_setting(user_id, SETTING_USER_ADMIN_TTS_PROVIDER) or ""
    ).strip().lower()
    raw_byok_engine = (
        await get_user_setting(user_id, SETTING_USER_BYOK_TTS_ENGINE) or "cloud"
    ).strip().lower()
    if raw_byok_engine not in ("cloud", "piper", "browser"):
        raw_byok_engine = "cloud"
    user_tts_provider_id = await get_user_setting(user_id, SETTING_USER_TTS_PROVIDER_ID) or ""
    byok_piper_v = (
        await get_user_setting(user_id, SETTING_USER_BYOK_TTS_VOICE_PIPER) or ""
    ).strip()

    # Legacy single admin voice: map into per-backend keys for display when new keys empty
    if not admin_server_v and not admin_piper_v and admin_tts_voice_legacy:
        if admin_tts_provider == "piper":
            admin_piper_v = admin_tts_voice_legacy
        else:
            admin_server_v = admin_tts_voice_legacy
    elif not admin_server_v and admin_tts_voice_legacy and admin_tts_provider != "piper":
        admin_server_v = admin_tts_voice_legacy
    elif not admin_piper_v and admin_tts_voice_legacy and admin_tts_provider == "piper":
        admin_piper_v = admin_tts_voice_legacy

    prefer_browser_tts = False
    effective_voice = ""
    effective_provider = ""

    if use_admin_tts:
        # Missing / blank = browser for new deployments; "server" = deployer voice-service default.
        if admin_tts_provider in ("browser", ""):
            prefer_browser_tts = True
            effective_voice = ""
            effective_provider = ""
        elif admin_tts_provider == "piper":
            effective_voice = admin_piper_v
            effective_provider = "piper"
        else:
            effective_voice = admin_server_v
            effective_provider = (
                "" if admin_tts_provider == "server" else admin_tts_provider
            )
    else:
        if raw_byok_engine == "browser":
            prefer_browser_tts = True
            effective_voice = ""
            effective_provider = ""
        elif raw_byok_engine == "piper":
            effective_voice = byok_piper_v or user_tts_voice
            effective_provider = "piper"
        else:
            effective_voice = user_tts_voice
            effective_provider = await user_voice_provider_service.get_linked_tts_provider_type_name(
                user_id
            )

    admin_voice_mirror = (
        admin_piper_v if admin_tts_provider == "piper" else admin_server_v
    )

    user_el_model = (
        await get_user_setting(user_id, SETTING_USER_ELEVENLABS_TTS_MODEL_ID) or ""
    ).strip()
    admin_el_model = (
        await get_user_setting(user_id, SETTING_USER_ADMIN_ELEVENLABS_TTS_MODEL_ID) or ""
    ).strip()
    user_hedra_model = (
        await get_user_setting(user_id, SETTING_USER_HEDRA_TTS_MODEL_ID) or ""
    ).strip()
    admin_hedra_model = (
        await get_user_setting(user_id, SETTING_USER_ADMIN_HEDRA_TTS_MODEL_ID) or ""
    ).strip()
    user_openrouter_model = (
        await get_user_setting(user_id, SETTING_USER_OPENROUTER_TTS_MODEL_ID) or ""
    ).strip()
    admin_openrouter_model = (
        await get_user_setting(user_id, SETTING_USER_ADMIN_OPENROUTER_TTS_MODEL_ID) or ""
    ).strip()

    return {
        "use_admin_tts": use_admin_tts,
        "use_admin_stt": _truthy(await get_user_setting(user_id, SETTING_USE_ADMIN_STT)),
        "user_tts_provider_id": user_tts_provider_id or "",
        "user_tts_voice_id": user_tts_voice,
        "user_stt_provider_id": await get_user_setting(
            user_id, SETTING_USER_STT_PROVIDER_ID
        )
        or "",
        "user_admin_tts_voice_id": admin_voice_mirror,
        "user_admin_tts_voice_server": admin_server_v,
        "user_admin_tts_voice_piper": admin_piper_v,
        "user_admin_tts_provider": admin_tts_provider,
        "user_byok_tts_engine": raw_byok_engine,
        "user_byok_tts_voice_piper": byok_piper_v,
        "prefer_browser_tts": prefer_browser_tts,
        "effective_server_voice_id": effective_voice,
        "effective_server_provider": effective_provider,
        "user_elevenlabs_tts_model_id": user_el_model,
        "user_admin_elevenlabs_tts_model_id": admin_el_model,
        "user_hedra_tts_model_id": user_hedra_model,
        "user_admin_hedra_tts_model_id": admin_hedra_model,
        "user_openrouter_tts_model_id": user_openrouter_model,
        "user_admin_openrouter_tts_model_id": admin_openrouter_model,
    }


@router.get("/hedra-tts-models")
async def list_hedra_tts_models(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """List Hedra text-to-speech engines (e.g. ElevenLabs vs Minimax) for the user's API key context."""
    uid = current_user.user_id
    try:
        use_admin = _truthy(await get_user_setting(uid, SETTING_USE_ADMIN_TTS))
        api_key = ""
        if use_admin:
            api_key = (getattr(settings, "HEDRA_API_KEY", "") or "").strip()
            if not api_key:
                raise HTTPException(
                    status_code=400,
                    detail="Set HEDRA_API_KEY on the backend (same value as voice-service) to list Hedra TTS engines.",
                )
        else:
            ctx = await user_voice_provider_service.get_voice_context(uid, "tts")
            if not ctx or str(ctx.get("provider_type") or "").lower() != "hedra":
                raise HTTPException(
                    status_code=400,
                    detail="Select a Hedra cloud TTS provider to list engines.",
                )
            api_key = (ctx.get("api_key") or "").strip()
            if not api_key:
                raise HTTPException(
                    status_code=400,
                    detail="Hedra API key is missing for this provider.",
                )
        models = await fetch_hedra_text_to_speech_models(api_key)
        return {"success": True, "models": models}
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        logger.exception("list_hedra_tts_models failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/openrouter-tts-models")
async def list_openrouter_tts_models(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """List OpenRouter TTS models (speech output modality) for admin or BYOK OpenRouter key."""
    uid = current_user.user_id
    try:
        use_admin = _truthy(await get_user_setting(uid, SETTING_USE_ADMIN_TTS))
        api_key = ""
        base_url: Optional[str] = None
        if use_admin:
            api_key = (getattr(settings, "OPENROUTER_API_KEY", "") or "").strip()
            if not api_key:
                raise HTTPException(
                    status_code=400,
                    detail="Set OPENROUTER_API_KEY on the backend to list OpenRouter TTS models for built-in TTS.",
                )
            base_url = (getattr(settings, "OPENROUTER_BASE_URL", "") or "").strip() or None
        else:
            ctx = await user_voice_provider_service.get_voice_context(uid, "tts")
            if not ctx or str(ctx.get("provider_type") or "").lower() != "openrouter":
                raise HTTPException(
                    status_code=400,
                    detail="Select an OpenRouter cloud TTS provider to list models.",
                )
            api_key = (ctx.get("api_key") or "").strip()
            base_url = (ctx.get("base_url") or "").strip() or None
            if not api_key:
                raise HTTPException(
                    status_code=400,
                    detail="OpenRouter API key is missing for this provider.",
                )
        models = await fetch_openrouter_tts_models(api_key, base_url)
        return {"success": True, "models": models}
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        logger.exception("list_openrouter_tts_models failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/voice-settings")
async def get_voice_settings(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    try:
        return await _voice_settings_payload(current_user.user_id)
    except Exception as e:
        logger.exception("get_voice_settings failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.put("/voice-settings")
async def put_voice_settings(
    request: Dict[str, Any],
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    uid = current_user.user_id
    try:
        if "use_admin_tts" in request:
            v = request["use_admin_tts"]
            await set_user_setting(
                uid,
                SETTING_USE_ADMIN_TTS,
                "true" if bool(v) else "false",
                "string",
            )
        if "use_admin_stt" in request:
            v = request["use_admin_stt"]
            await set_user_setting(
                uid,
                SETTING_USE_ADMIN_STT,
                "true" if bool(v) else "false",
                "string",
            )
        if "user_tts_provider_id" in request:
            await set_user_setting(
                uid,
                SETTING_USER_TTS_PROVIDER_ID,
                str(request["user_tts_provider_id"] or "").strip(),
                "string",
            )
        if "user_tts_voice_id" in request:
            await set_user_setting(
                uid,
                SETTING_USER_TTS_VOICE_ID,
                str(request["user_tts_voice_id"] or "").strip(),
                "string",
            )
        if "user_stt_provider_id" in request:
            await set_user_setting(
                uid,
                SETTING_USER_STT_PROVIDER_ID,
                str(request["user_stt_provider_id"] or "").strip(),
                "string",
            )
        if "user_admin_tts_voice_id" in request:
            await set_user_setting(
                uid,
                SETTING_USER_ADMIN_TTS_VOICE_ID,
                str(request["user_admin_tts_voice_id"] or "").strip(),
                "string",
            )
        if "user_admin_tts_voice_server" in request:
            await set_user_setting(
                uid,
                SETTING_USER_ADMIN_TTS_VOICE_SERVER,
                str(request["user_admin_tts_voice_server"] or "").strip(),
                "string",
            )
        if "user_admin_tts_voice_piper" in request:
            await set_user_setting(
                uid,
                SETTING_USER_ADMIN_TTS_VOICE_PIPER,
                str(request["user_admin_tts_voice_piper"] or "").strip(),
                "string",
            )
        if "user_byok_tts_voice_piper" in request:
            await set_user_setting(
                uid,
                SETTING_USER_BYOK_TTS_VOICE_PIPER,
                str(request["user_byok_tts_voice_piper"] or "").strip(),
                "string",
            )
        if "user_admin_tts_provider" in request:
            await set_user_setting(
                uid,
                SETTING_USER_ADMIN_TTS_PROVIDER,
                str(request["user_admin_tts_provider"] or "").strip(),
                "string",
            )
        if "user_byok_tts_engine" in request:
            eng = str(request["user_byok_tts_engine"] or "").strip().lower()
            if eng not in ("cloud", "piper", "browser"):
                eng = "cloud"
            await set_user_setting(
                uid,
                SETTING_USER_BYOK_TTS_ENGINE,
                eng,
                "string",
            )
        if "user_elevenlabs_tts_model_id" in request:
            v = str(request["user_elevenlabs_tts_model_id"] or "").strip()
            try:
                validated_elevenlabs_model_id(v)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            await set_user_setting(
                uid,
                SETTING_USER_ELEVENLABS_TTS_MODEL_ID,
                v,
                "string",
            )
        if "user_admin_elevenlabs_tts_model_id" in request:
            v = str(request["user_admin_elevenlabs_tts_model_id"] or "").strip()
            try:
                validated_elevenlabs_model_id(v)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            await set_user_setting(
                uid,
                SETTING_USER_ADMIN_ELEVENLABS_TTS_MODEL_ID,
                v,
                "string",
            )
        if "user_hedra_tts_model_id" in request:
            v = str(request["user_hedra_tts_model_id"] or "").strip()
            try:
                validated_hedra_tts_model_id(v)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            await set_user_setting(
                uid,
                SETTING_USER_HEDRA_TTS_MODEL_ID,
                v,
                "string",
            )
        if "user_admin_hedra_tts_model_id" in request:
            v = str(request["user_admin_hedra_tts_model_id"] or "").strip()
            try:
                validated_hedra_tts_model_id(v)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            await set_user_setting(
                uid,
                SETTING_USER_ADMIN_HEDRA_TTS_MODEL_ID,
                v,
                "string",
            )
        if "user_openrouter_tts_model_id" in request:
            v = str(request["user_openrouter_tts_model_id"] or "").strip()
            try:
                validated_openrouter_tts_model_id(v)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            await set_user_setting(
                uid,
                SETTING_USER_OPENROUTER_TTS_MODEL_ID,
                v,
                "string",
            )
        if "user_admin_openrouter_tts_model_id" in request:
            v = str(request["user_admin_openrouter_tts_model_id"] or "").strip()
            try:
                validated_openrouter_tts_model_id(v)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            await set_user_setting(
                uid,
                SETTING_USER_ADMIN_OPENROUTER_TTS_MODEL_ID,
                v,
                "string",
            )

        return await _voice_settings_payload(uid)
    except Exception as e:
        logger.exception("put_voice_settings failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
