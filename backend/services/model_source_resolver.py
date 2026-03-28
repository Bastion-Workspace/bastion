"""
Model source resolver: single place for available models, enabled models, and credential resolution.
User mode = user providers only; admin mode = env-backed admin providers.
"""

import logging
from typing import Any, Dict, List, Optional

from models.api_models import ModelInfo

from services.admin_provider_registry import admin_provider_registry
from services.settings_service import settings_service
from services.user_llm_provider_service import user_llm_provider_service
from services.user_settings_kv_service import get_user_setting

logger = logging.getLogger(__name__)


async def _use_admin_models(user_id: str) -> bool:
    """True if user should see admin-enabled models and credentials."""
    val = await get_user_setting(user_id, "use_admin_models")
    return (val or "true").lower() == "true"


async def get_available_models(user_id: Optional[str]) -> List[ModelInfo]:
    """Available models for the user. User mode = user providers only; admin mode = admin registry."""
    if user_id:
        use_admin = await _use_admin_models(user_id)
        if not use_admin:
            return await user_llm_provider_service.get_available_models_for_user(user_id)
    return await admin_provider_registry.get_all_admin_models()


async def get_enabled_models(user_id: Optional[str]) -> List[str]:
    """Enabled model IDs for dropdown/validation. User mode = user enabled list; admin mode = org enabled list."""
    if user_id:
        use_admin = await _use_admin_models(user_id)
        if not use_admin:
            rows = await user_llm_provider_service.get_user_enabled_models(user_id)
            return [m["model_id"] for m in rows]
    return await settings_service.get_enabled_models()


async def resolve_model_context(
    user_id: str, model_id: str
) -> Optional[Dict[str, Any]]:
    """
    Resolve credentials for a model. Returns dict with api_key, base_url, real_model_id.
    For user mode uses user provider; for admin mode uses admin provider that owns the model.
    """
    use_admin = await _use_admin_models(user_id)
    if use_admin:
        creds = await admin_provider_registry.resolve_admin_credentials(model_id)
        if creds:
            api_key, base_url, provider_type = creds
            return {"api_key": api_key, "base_url": base_url, "real_model_id": model_id, "provider_type": provider_type}
        return None
    ctx = await user_llm_provider_service.get_llm_context_for_model(user_id, model_id)
    return ctx


async def try_soft_retarget(user_id: str, model_id: str) -> Dict[str, Any]:
    """
    Resolve model_id for current user. If not in current enabled list, try same id in available (soft retarget).
    Returns { model_id, retargeted: bool, available: bool }.
    """
    enabled = await get_enabled_models(user_id)
    available = await get_available_models(user_id)
    available_ids = {m.id for m in available}
    if model_id in enabled and model_id in available_ids:
        return {"model_id": model_id, "retargeted": False, "available": True}
    if model_id in available_ids:
        return {"model_id": model_id, "retargeted": False, "available": True}
    if not enabled:
        fallback = next((m.id for m in available), None)
        return {"model_id": fallback or model_id, "retargeted": bool(fallback), "available": bool(fallback)}
    fallback = next((m.id for m in available if m.id in enabled), None)
    return {"model_id": fallback or model_id, "retargeted": bool(fallback and fallback != model_id), "available": bool(fallback)}


async def pick_fallback_model_id(user_id: Optional[str]) -> Optional[str]:
    """First enabled model that is still on the provider catalog, else any available model."""
    if not user_id:
        return None
    enabled = await get_enabled_models(user_id)
    available = await get_available_models(user_id)
    avail_ids = {m.id for m in available}
    for mid in enabled:
        if mid in avail_ids:
            return mid
    if available:
        return available[0].id
    return None


async def _catalog_slice_byok(user_id: str, image_generation_model_id: str) -> Dict[str, Any]:
    """On-demand catalog slice for BYOK users (per-user catalog, not cached on registry)."""
    enabled = await get_enabled_models(user_id)
    available = await get_available_models(user_id)
    available_ids = {m.id for m in available}
    catalog_verified = len(available_ids) > 0
    img = (image_generation_model_id or "").strip() or (await settings_service.get_image_generation_model() or "").strip()

    if catalog_verified:
        orphaned_enabled_models = [mid for mid in enabled if mid not in available_ids]
        effective_enabled_models = [mid for mid in enabled if mid in available_ids]
    else:
        orphaned_enabled_models = []
        effective_enabled_models = list(enabled)

    selectable_chat_models = [mid for mid in effective_enabled_models if mid != img]

    orphaned_role_models: Dict[str, str] = {}
    if catalog_verified:
        image_analysis = (await settings_service.get_image_analysis_model() or "").strip()
        classification = (await settings_service.get_classification_model() or "").strip()
        text_completion_raw = await settings_service.get_text_completion_model()
        text_completion = (text_completion_raw or "").strip()
        role_pairs = [
            ("image_generation_model", img),
            ("image_analysis_model", image_analysis),
            ("classification_model", classification),
            ("text_completion_model", text_completion),
        ]
        for key, val in role_pairs:
            if val and val not in available_ids:
                orphaned_role_models[key] = val

    return {
        "orphaned_enabled_models": orphaned_enabled_models,
        "effective_enabled_models": effective_enabled_models,
        "selectable_chat_models": selectable_chat_models,
        "catalog_verified": catalog_verified,
        "orphaned_role_models": orphaned_role_models,
    }


async def get_enabled_models_catalog_slice(
    user_id: Optional[str],
    image_generation_model_id: str = "",
) -> Dict[str, Any]:
    """
    Compare persisted enabled model IDs to the current provider catalog.

    Org/admin path uses cached slice on AdminProviderRegistry (recomputed on cache refresh or invalidate_slice).

    When the catalog fetch yields no models (provider outage / misconfig), do not mark
    every enabled ID as orphaned; omit orphan detection until the catalog loads.

    Returns dict with orphaned_enabled_models, effective_enabled_models, selectable_chat_models,
    catalog_verified, orphaned_role_models.
    """
    img = image_generation_model_id or (await settings_service.get_image_generation_model() or "")

    if user_id is None or await _use_admin_models(user_id):
        return await admin_provider_registry.get_org_catalog_slice()

    return await _catalog_slice_byok(user_id, img)


async def get_chat_selectable_model_ids_for_user(user_id: Optional[str]) -> List[str]:
    """Model IDs allowed in chat dropdowns: catalog intersection when available; excludes image gen model."""
    sl = await get_enabled_models_catalog_slice(user_id, "")
    return sl["selectable_chat_models"]
