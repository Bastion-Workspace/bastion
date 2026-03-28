"""
User LLM Provider API - per-user providers (OpenAI, OpenRouter, Ollama, vLLM) and enabled models.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Depends

from models.api_models import AuthenticatedUserResponse
from models.provider_models import PROVIDER_TYPES
from services.user_llm_provider_service import user_llm_provider_service
from services.user_settings_kv_service import get_user_setting, set_user_setting
from utils.auth_middleware import get_current_user
from models.api_models import AuthenticatedUserResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/user", tags=["User LLM Providers"])


@router.get("/llm-providers")
async def list_llm_providers(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """List current user's LLM providers (no API keys returned)."""
    try:
        providers = await user_llm_provider_service.list_providers(current_user.user_id)
        return {"providers": providers}
    except Exception as e:
        logger.exception("list_llm_providers failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/llm-providers")
async def add_llm_provider(
    request: Dict[str, Any],
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Add a provider. Validates connection before saving. Body: provider_type, api_key?, base_url?, display_name?."""
    try:
        provider_type = request.get("provider_type")
        if not provider_type or provider_type not in PROVIDER_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"provider_type must be one of: {', '.join(PROVIDER_TYPES)}",
            )
        api_key = request.get("api_key")
        base_url = request.get("base_url")
        display_name = request.get("display_name")
        if provider_type in ("ollama", "vllm") and not base_url:
            raise HTTPException(status_code=400, detail="base_url required for ollama and vllm")
        provider_id = await user_llm_provider_service.add_provider(
            current_user.user_id,
            provider_type=provider_type,
            api_key=api_key,
            base_url=base_url,
            display_name=display_name,
        )
        providers = await user_llm_provider_service.list_providers(current_user.user_id)
        if len(providers) == 1:
            await set_user_setting(
                current_user.user_id,
                "use_admin_models",
                "false",
                "string",
            )
        return {"provider_id": provider_id, "status": "success"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("add_llm_provider failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/llm-providers/{provider_id:int}")
async def remove_llm_provider(
    provider_id: int,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Remove a provider and its enabled models."""
    try:
        await user_llm_provider_service.remove_provider(current_user.user_id, provider_id)
        return {"status": "success"}
    except Exception as e:
        logger.exception("remove_llm_provider failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/llm-providers/{provider_id:int}/models")
async def get_provider_models(
    provider_id: int,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Fetch available models from this provider (proxies to provider /v1/models)."""
    try:
        models = await user_llm_provider_service.fetch_provider_models(current_user.user_id, provider_id)
        return {"models": models}
    except Exception as e:
        logger.exception("get_provider_models failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/enabled")
async def get_user_enabled_models(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Get current user's enabled model list (for My Providers UI)."""
    try:
        models = await user_llm_provider_service.get_user_enabled_models(current_user.user_id)
        return {"enabled_models": models}
    except Exception as e:
        logger.exception("get_user_enabled_models failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/models/enabled")
async def set_user_enabled_models(
    request: Dict[str, Any],
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Set enabled models for a provider. Body: provider_id, model_ids (list of model id strings)."""
    try:
        provider_id = request.get("provider_id")
        model_ids = request.get("model_ids", [])
        if provider_id is None:
            raise HTTPException(status_code=400, detail="provider_id required")
        if not isinstance(model_ids, list):
            model_ids = []
        await user_llm_provider_service.set_user_enabled_models(
            current_user.user_id, provider_id, [str(m) for m in model_ids]
        )
        return {"status": "success"}
    except Exception as e:
        logger.exception("set_user_enabled_models failed")
        raise HTTPException(status_code=500, detail=str(e))


MODEL_ROLE_KEYS = ("user_chat_model", "user_fast_model", "user_image_gen_model", "user_image_analysis_model")


@router.get("/models/roles")
async def get_user_model_roles(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Return all four per-user model role overrides (empty string = use admin default)."""
    try:
        roles = {}
        for key in MODEL_ROLE_KEYS:
            val = await get_user_setting(current_user.user_id, key)
            roles[key] = val or ""
        return roles
    except Exception as e:
        logger.exception("get_user_model_roles failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/models/roles")
async def set_user_model_roles(
    request: Dict[str, str],
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Set one or more per-user model role keys. Body: user_chat_model?, user_fast_model?, user_image_gen_model?, user_image_analysis_model?."""
    try:
        for key in MODEL_ROLE_KEYS:
            if key in request:
                value = request[key]
                await set_user_setting(
                    current_user.user_id,
                    key,
                    (value or "").strip(),
                    "string",
                )
        roles = {}
        for key in MODEL_ROLE_KEYS:
            val = await get_user_setting(current_user.user_id, key)
            roles[key] = val or ""
        return {"status": "success", "roles": roles}
    except Exception as e:
        logger.exception("set_user_model_roles failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/settings/use-admin-models")
async def get_use_admin_models(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Get use_admin_models toggle (true = use admin-enabled models, false = use my providers)."""
    try:
        value = await get_user_setting(current_user.user_id, "use_admin_models")
        return {"use_admin_models": (value or "true").lower() == "true"}
    except Exception as e:
        logger.exception("get_use_admin_models failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/settings/use-admin-models")
async def set_use_admin_models(
    request: Dict[str, Any],
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Set use_admin_models toggle. Body: value (boolean)."""
    try:
        value = request.get("value", True)
        await set_user_setting(
            current_user.user_id,
            "use_admin_models",
            "true" if value else "false",
            "string",
        )
        return {"use_admin_models": bool(value), "status": "success"}
    except Exception as e:
        logger.exception("set_use_admin_models failed")
        raise HTTPException(status_code=500, detail=str(e))
