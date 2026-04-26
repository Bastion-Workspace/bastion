"""
Voice API routes for text-to-speech synthesis.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from clients.voice_service_client import get_voice_service_client
from services.user_voice_provider_service import (
    SETTING_USE_ADMIN_TTS,
    SETTING_USER_ADMIN_TTS_PROVIDER,
    SETTING_USER_ADMIN_TTS_VOICE_ID,
    SETTING_USER_ADMIN_TTS_VOICE_PIPER,
    SETTING_USER_ADMIN_TTS_VOICE_SERVER,
    SETTING_USER_BYOK_TTS_ENGINE,
    SETTING_USER_BYOK_TTS_VOICE_PIPER,
    SETTING_USER_TTS_VOICE_ID,
    user_voice_provider_service,
)
from services.user_settings_kv_service import get_user_setting
from utils.auth_middleware import AuthenticatedUserResponse, get_current_user
from utils.elevenlabs_tts_model import resolve_elevenlabs_tts_model_id_for_user
from utils.hedra_tts_model import resolve_hedra_tts_model_id_for_user
from utils.openrouter_tts_model import resolve_openrouter_tts_model_id_for_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Voice"])


def _truthy_use_admin_tts(raw: Optional[str]) -> bool:
    if raw is None:
        return True
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000)
    voice_id: Optional[str] = ""
    provider: Optional[str] = ""
    output_format: Optional[str] = "mp3"
    model_id: Optional[str] = Field(default="", max_length=128)


async def _elevenlabs_model_id_resolved(
    user_id: str, request: SynthesizeRequest, use_admin_tts: bool
) -> str:
    try:
        return await resolve_elevenlabs_tts_model_id_for_user(
            user_id, request, use_admin_tts
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


async def _openrouter_model_id_resolved(
    user_id: str, request: SynthesizeRequest, use_admin_tts: bool
) -> str:
    try:
        return await resolve_openrouter_tts_model_id_for_user(
            user_id, request, use_admin_tts
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


async def _resolve_tts_synthesis_params(
    user_id: str,
    request: SynthesizeRequest,
) -> tuple[str, str, str, str, str]:
    """Resolve voice_id, provider, api_key, base_url, model_id (BYOK vs admin vs browser)."""
    voice_id = (request.voice_id or "").strip()
    provider = (request.provider or "").strip()
    api_key = ""
    base_url = ""

    use_admin_tts = _truthy_use_admin_tts(
        await get_user_setting(user_id, SETTING_USE_ADMIN_TTS)
    )

    if use_admin_tts:
        raw_admin = (
            await get_user_setting(user_id, SETTING_USER_ADMIN_TTS_PROVIDER) or ""
        ).strip().lower()
        # Missing / blank = browser (greenfield). Explicit "server" = voice-service deployer default.
        if raw_admin in ("browser", ""):
            raise HTTPException(
                status_code=400,
                detail="Browser TTS is selected in settings; use client speech synthesis.",
            )
        admin_prov = raw_admin
        if not voice_id:
            if admin_prov == "piper":
                voice_id = (
                    await get_user_setting(user_id, SETTING_USER_ADMIN_TTS_VOICE_PIPER) or ""
                ).strip()
            else:
                voice_id = (
                    await get_user_setting(user_id, SETTING_USER_ADMIN_TTS_VOICE_SERVER) or ""
                ).strip()
            if not voice_id:
                voice_id = (
                    await get_user_setting(user_id, SETTING_USER_ADMIN_TTS_VOICE_ID) or ""
                ).strip()
        if not provider:
            provider = (
                await get_user_setting(user_id, SETTING_USER_ADMIN_TTS_PROVIDER) or ""
            ).strip()
        if admin_prov == "server":
            provider = ""
        elif admin_prov == "piper":
            provider = "piper"
        prov_lc = (provider or "").strip().lower()
        model_id = ""
        if prov_lc == "elevenlabs":
            model_id = await _elevenlabs_model_id_resolved(
                user_id, request, use_admin_tts=True
            )
        elif prov_lc == "hedra":
            try:
                model_id = await resolve_hedra_tts_model_id_for_user(
                    user_id, request, use_admin_tts=True
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
        elif prov_lc == "openrouter":
            model_id = await _openrouter_model_id_resolved(
                user_id, request, use_admin_tts=True
            )
        return voice_id, provider, api_key, base_url, model_id

    # BYOK: honor user_byok_tts_engine before get_voice_context. Otherwise a saved
    # user_tts_provider_id would always force cloud credentials and ignore Piper/Browser.
    engine = (
        await get_user_setting(user_id, SETTING_USER_BYOK_TTS_ENGINE) or "cloud"
    ).strip().lower()
    if engine not in ("cloud", "piper", "browser"):
        engine = "cloud"
    if engine == "browser":
        raise HTTPException(
            status_code=400,
            detail="Browser TTS is selected in settings; use client speech synthesis.",
        )
    if engine == "piper":
        vid = (
            await get_user_setting(user_id, SETTING_USER_BYOK_TTS_VOICE_PIPER) or ""
        ).strip()
        if not vid:
            vid = (await get_user_setting(user_id, SETTING_USER_TTS_VOICE_ID) or "").strip()
        if not voice_id:
            voice_id = vid
        return voice_id, "piper", "", "", ""

    ctx = await user_voice_provider_service.get_voice_context(user_id, "tts")
    if ctx:
        provider = ctx.get("provider_type") or provider
        api_key = ctx.get("api_key") or ""
        base_url = ctx.get("base_url") or ""
        if not voice_id and ctx.get("voice_id"):
            voice_id = str(ctx["voice_id"]).strip()
        prov_lc = (provider or "").strip().lower()
        model_id = ""
        if prov_lc == "elevenlabs":
            model_id = await _elevenlabs_model_id_resolved(
                user_id, request, use_admin_tts=False
            )
        elif prov_lc == "hedra":
            try:
                model_id = await resolve_hedra_tts_model_id_for_user(
                    user_id, request, use_admin_tts=False
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
        elif prov_lc == "openrouter":
            model_id = await _openrouter_model_id_resolved(
                user_id, request, use_admin_tts=False
            )
            if not (model_id or "").strip():
                raise HTTPException(
                    status_code=400,
                    detail="Select an OpenRouter TTS model in voice settings (BYOK).",
                )
        return voice_id, provider, api_key, base_url, model_id

    vid = (await get_user_setting(user_id, SETTING_USER_TTS_VOICE_ID) or "").strip()
    if not voice_id:
        voice_id = vid
    linked = await user_voice_provider_service.get_linked_tts_provider_type_name(
        user_id
    )
    prov = provider or linked or ""
    prov_lc = (prov or "").strip().lower()
    model_id = ""
    if prov_lc == "elevenlabs":
        model_id = await _elevenlabs_model_id_resolved(
            user_id, request, use_admin_tts=False
        )
    elif prov_lc == "hedra":
        try:
            model_id = await resolve_hedra_tts_model_id_for_user(
                user_id, request, use_admin_tts=False
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    elif prov_lc == "openrouter":
        model_id = await _openrouter_model_id_resolved(
            user_id, request, use_admin_tts=False
        )
        if not (model_id or "").strip():
            raise HTTPException(
                status_code=400,
                detail="Select an OpenRouter TTS model in voice settings.",
            )
    return voice_id, prov, "", "", model_id


@router.post("/api/voice/synthesize")
async def synthesize_voice(
    request: SynthesizeRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Synthesize text with voice-service and return audio bytes."""
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text is required")

        voice_id, provider, api_key, base_url, model_id = (
            await _resolve_tts_synthesis_params(current_user.user_id, request)
        )

        client = await get_voice_service_client()
        result = await client.synthesize(
            text=request.text.strip(),
            voice_id=voice_id,
            provider=provider,
            output_format=(request.output_format or "mp3").strip().lower(),
            api_key=api_key,
            base_url=base_url,
            model_id=model_id,
        )

        if result.get("error"):
            raise HTTPException(status_code=502, detail=result["error"])

        audio_data = result.get("audio_data") or b""
        if not audio_data:
            raise HTTPException(status_code=502, detail="Voice service returned empty audio")

        audio_format = (result.get("audio_format") or request.output_format or "mp3").lower()
        media_type = {
            "mp3": "audio/mpeg",
            "ogg": "audio/ogg",
            "wav": "audio/wav",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "flac": "audio/flac",
            "pcm": "audio/L16",
        }.get(audio_format, "application/octet-stream")

        return Response(
            content=audio_data,
            media_type=media_type,
            headers={
                "X-Audio-Format": audio_format,
                "Cache-Control": "no-store",
                "Content-Disposition": f'inline; filename="tts.{audio_format}"',
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Voice synthesis failed for user %s: %s", current_user.user_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/voice/synthesize/stream")
async def synthesize_voice_stream(
    request: SynthesizeRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Stream synthesized audio for low time-to-first-byte (default OGG/Opus)."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    voice_id, provider, api_key, base_url, model_id = (
        await _resolve_tts_synthesis_params(current_user.user_id, request)
    )
    out_fmt = (request.output_format or "ogg").strip().lower()
    prov_lc = (provider or "").strip().lower()
    stream_fmt = (
        "mp3"
        if prov_lc in ("elevenlabs", "hedra", "openrouter") and out_fmt != "wav"
        else out_fmt
    )
    client = await get_voice_service_client()

    async def audio_bytes():
        try:
            async for chunk in client.synthesize_stream(
                text=request.text.strip(),
                voice_id=voice_id,
                provider=provider,
                output_format=out_fmt,
                api_key=api_key,
                base_url=base_url,
                model_id=model_id,
            ):
                if chunk:
                    yield chunk
        except Exception as e:
            logger.error(
                "Voice synthesis stream failed for user %s: %s",
                current_user.user_id,
                e,
            )
            raise

    media_type = {
        "mp3": "audio/mpeg",
        "ogg": "audio/ogg",
        "wav": "audio/wav",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "pcm": "audio/L16",
    }.get(stream_fmt, "audio/ogg")

    return StreamingResponse(
        audio_bytes(),
        media_type=media_type,
        headers={
            "X-Audio-Format": stream_fmt,
            "Cache-Control": "no-store",
            "Content-Disposition": f'inline; filename="tts-stream.{stream_fmt}"',
        },
    )


@router.get("/api/voice/voices")
async def list_tts_voices(
    provider: str = "",
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> Dict[str, Any]:
    """List available TTS voices from voice-service."""
    try:
        ctx = await user_voice_provider_service.get_voice_context(
            current_user.user_id, "tts"
        )
        api_key = ""
        base_url = ""
        prov = (provider or "").strip()
        # Honor explicit ?provider= (e.g. piper for Settings); do not replace with BYOK row.
        if ctx and not prov:
            prov = ctx.get("provider_type") or ""
            api_key = ctx.get("api_key") or ""
            base_url = ctx.get("base_url") or ""

        client = await get_voice_service_client()
        result = await client.list_voices(
            provider=prov, api_key=api_key, base_url=base_url
        )
        if result.get("error"):
            raise HTTPException(status_code=502, detail=result["error"])
        return {"success": True, "voices": result.get("voices", [])}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("List voices failed for user %s: %s", current_user.user_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/voice/status")
async def voice_status(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return service health and available providers for frontend capability checks."""
    try:
        ctx = await user_voice_provider_service.get_voice_context(
            current_user.user_id, "tts"
        )
        api_key = ""
        base_url = ""
        prov = ""
        if ctx:
            prov = ctx.get("provider_type") or ""
            api_key = ctx.get("api_key") or ""
            base_url = ctx.get("base_url") or ""

        client = await get_voice_service_client()
        health = await client.health_check()

        voices_result = await client.list_voices(
            provider=prov, api_key=api_key, base_url=base_url
        )
        voices = voices_result.get("voices", [])
        providers = sorted({v.get("provider", "").strip().lower() for v in voices if v.get("provider")})

        return {
            "success": True,
            "available": health.get("status") == "healthy",
            "status": health.get("status"),
            "service_version": health.get("service_version"),
            "device": health.get("device"),
            "providers": providers,
            "voice_count": len(voices),
        }
    except Exception as e:
        logger.warning("Voice status unavailable for user %s: %s", current_user.user_id, e)
        return {
            "success": False,
            "available": False,
            "status": "unavailable",
            "providers": [],
            "voice_count": 0,
            "error": str(e),
        }
