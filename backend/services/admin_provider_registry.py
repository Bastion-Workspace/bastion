"""
Admin provider registry: env-backed API keys and normalized model lists for OpenAI, OpenRouter, Groq.
Used when use_admin_models=true; ModelSourceResolver aggregates admin vs user models.

Caches the org-level catalog slice (enabled vs catalog intersection, role-model orphans) alongside raw model list.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx

from config import settings
from models.api_models import ModelInfo
from models.provider_models import (
    CLOUD_BASE_URLS,
    needs_openrouter_headers,
)

logger = logging.getLogger(__name__)

CACHE_DURATION = 3600  # 1 hour

# If a fetch returns fewer than this fraction of the last trusted count, treat catalog as unverified.
PARTIAL_CATALOG_RATIO = 0.5
# Minimum previous count before ratio check applies (avoid noise on tiny catalogs).
PARTIAL_CATALOG_MIN_PREV = 10


def _admin_base_url(provider_type: str) -> str:
    """Base URL for admin provider; env override or CLOUD_BASE_URLS default."""
    override = ""
    if provider_type == "openai":
        override = getattr(settings, "OPENAI_BASE_URL", "") or ""
    elif provider_type == "openrouter":
        override = getattr(settings, "OPENROUTER_BASE_URL", "") or ""
    elif provider_type == "groq":
        override = getattr(settings, "GROQ_BASE_URL", "") or ""
    if override:
        return override.rstrip("/")
    return CLOUD_BASE_URLS.get(provider_type, "")


def get_enabled_admin_providers() -> List[Tuple[str, str, str]]:
    """Return list of (provider_type, api_key, base_url) for admin providers that have a key set."""
    out: List[Tuple[str, str, str]] = []
    if getattr(settings, "OPENAI_API_KEY", ""):
        out.append(("openai", settings.OPENAI_API_KEY, _admin_base_url("openai")))
    if getattr(settings, "OPENROUTER_API_KEY", ""):
        out.append(("openrouter", settings.OPENROUTER_API_KEY, _admin_base_url("openrouter")))
    if getattr(settings, "GROQ_API_KEY", ""):
        out.append(("groq", settings.GROQ_API_KEY, _admin_base_url("groq")))
    return out


async def _fetch_models_for_provider(
    provider_type: str, api_key: str, base_url: str
) -> List[ModelInfo]:
    """Fetch /models from one admin provider and return normalized ModelInfo list with source=admin."""
    url = f"{base_url}/v1/models" if not base_url.endswith("/v1") else f"{base_url.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    if needs_openrouter_headers(provider_type):
        headers["HTTP-Referer"] = getattr(settings, "SITE_URL", "https://localhost")
        headers["X-Title"] = "Bastion AI Workspace"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers)
            if response.status_code != 200:
                logger.warning(
                    "Admin provider %s models API returned %s: %s",
                    provider_type,
                    response.status_code,
                    (response.text or "")[:200],
                )
                return []
            data = response.json()
    except Exception as e:
        logger.warning("Admin provider %s fetch error: %s", provider_type, e)
        return []
    raw_list = data.get("data", []) if isinstance(data, dict) else []
    provider_label = provider_type.replace("-", " ").title()
    models: List[ModelInfo] = []
    for m in raw_list:
        if not isinstance(m, dict):
            continue
        mid = m.get("id")
        if not mid:
            continue
        name = m.get("name", mid)
        context_length = 0
        input_cost: Optional[float] = None
        output_cost: Optional[float] = None
        if provider_type == "openrouter":
            context_length = int(m.get("context_length", 0) or 0)
            pricing = m.get("pricing") or {}
            if isinstance(pricing, dict):
                try:
                    input_cost = float(pricing["prompt"]) if pricing.get("prompt") is not None else None
                except (TypeError, ValueError):
                    input_cost = None
                try:
                    output_cost = float(pricing["completion"]) if pricing.get("completion") is not None else None
                except (TypeError, ValueError):
                    output_cost = None
        elif provider_type == "groq":
            context_length = int(m.get("context_window", 0) or m.get("context_length", 0) or 0)
        else:
            context_length = int(m.get("context_length", 0) or 0)
        provider_display = mid.split("/")[0] if "/" in str(mid) else provider_label
        provider_display = provider_display.replace("-", " ").title()
        models.append(
            ModelInfo(
                id=str(mid),
                name=name or str(mid),
                provider=provider_display,
                context_length=context_length,
                input_cost=input_cost,
                output_cost=output_cost,
                description=m.get("description"),
                output_modalities=m.get("output_modalities") if isinstance(m.get("output_modalities"), list) else None,
                input_modalities=m.get("input_modalities") if isinstance(m.get("input_modalities"), list) else None,
                source="admin",
                provider_type=provider_type,
                provider_id=None,
            )
        )
    return models


@dataclass
class OrgCatalogSlice:
    """Cached intersection of org enabled_models and admin provider catalog."""

    orphaned_enabled_models: List[str]
    effective_enabled_models: List[str]
    selectable_chat_models: List[str]
    catalog_verified: bool
    orphaned_role_models: Dict[str, str]
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "orphaned_enabled_models": self.orphaned_enabled_models,
            "effective_enabled_models": self.effective_enabled_models,
            "selectable_chat_models": self.selectable_chat_models,
            "catalog_verified": self.catalog_verified,
            "orphaned_role_models": dict(self.orphaned_role_models),
        }


class AdminProviderRegistry:
    """Registry of admin (env-backed) LLM providers and their model lists."""

    def __init__(self) -> None:
        self._cache: Optional[List[ModelInfo]] = None
        self._cache_timestamp: Optional[float] = None
        self._catalog_slice: Optional[OrgCatalogSlice] = None
        self._last_known_catalog_size: int = 0
        self._fetch_trusted: bool = True

    def invalidate_slice(self) -> None:
        """Drop cached org catalog slice; raw model cache unchanged. Next get_org_catalog_slice recomputes."""
        self._catalog_slice = None

    async def get_org_catalog_slice(self) -> Dict[str, Any]:
        """Ensure raw catalog is loaded, then return cached org slice (recomputes if invalidated)."""
        await self.get_all_admin_models()
        if self._catalog_slice is None:
            await self._recompute_org_catalog_slice()
        assert self._catalog_slice is not None
        return self._catalog_slice.to_dict()

    async def _recompute_org_catalog_slice(self) -> None:
        from services.settings_service import settings_service

        models = self._cache if self._cache is not None else []
        available_ids = {m.id for m in models}
        catalog_nonempty = len(available_ids) > 0
        catalog_verified = catalog_nonempty and self._fetch_trusted

        enabled = await settings_service.get_enabled_models()
        img = (await settings_service.get_image_generation_model() or "").strip()
        image_analysis = (await settings_service.get_image_analysis_model() or "").strip()
        classification = (await settings_service.get_classification_model() or "").strip()
        text_completion_raw = await settings_service.get_text_completion_model()
        text_completion = (text_completion_raw or "").strip()

        if catalog_verified:
            orphaned_enabled_models = [mid for mid in enabled if mid not in available_ids]
            effective_enabled_models = [mid for mid in enabled if mid in available_ids]
        else:
            orphaned_enabled_models = []
            effective_enabled_models = list(enabled)

        selectable_chat_models = [mid for mid in effective_enabled_models if mid != img]

        orphaned_role_models: Dict[str, str] = {}
        if catalog_verified:
            role_pairs = [
                ("image_generation_model", img),
                ("image_analysis_model", image_analysis),
                ("classification_model", classification),
                ("text_completion_model", text_completion),
            ]
            for key, val in role_pairs:
                if val and val not in available_ids:
                    orphaned_role_models[key] = val

        now = time.time()
        self._catalog_slice = OrgCatalogSlice(
            orphaned_enabled_models=orphaned_enabled_models,
            effective_enabled_models=effective_enabled_models,
            selectable_chat_models=selectable_chat_models,
            catalog_verified=catalog_verified,
            orphaned_role_models=orphaned_role_models,
            timestamp=now,
        )

    async def get_all_admin_models(self) -> List[ModelInfo]:
        """Aggregate models from all enabled admin providers. Cached for CACHE_DURATION."""
        now = time.time()
        if (
            self._cache is not None
            and self._cache_timestamp is not None
            and (now - self._cache_timestamp) < CACHE_DURATION
        ):
            if self._catalog_slice is None:
                await self._recompute_org_catalog_slice()
            return self._cache

        providers = get_enabled_admin_providers()
        if not providers:
            self._cache = []
            self._cache_timestamp = now
            self._fetch_trusted = True
            await self._recompute_org_catalog_slice()
            return self._cache

        all_models: List[ModelInfo] = []
        for provider_type, api_key, base_url in providers:
            models = await _fetch_models_for_provider(provider_type, api_key, base_url)
            all_models.extend(models)
        all_models.sort(key=lambda m: (m.provider_type or "", m.provider, m.name))

        n = len(all_models)
        prev = self._last_known_catalog_size
        if n > 0 and prev >= PARTIAL_CATALOG_MIN_PREV and n < int(prev * PARTIAL_CATALOG_RATIO):
            self._fetch_trusted = False
            logger.warning(
                "Admin registry: catalog size dropped sharply (%s -> %s); orphan detection disabled for this fetch",
                prev,
                n,
            )
        else:
            self._fetch_trusted = True

        if self._fetch_trusted and n > 0:
            self._last_known_catalog_size = n

        self._cache = all_models
        self._cache_timestamp = now
        logger.info("Admin registry: loaded %s models from %s provider(s)", len(all_models), len(providers))
        await self._recompute_org_catalog_slice()
        return all_models

    def refresh(self) -> None:
        """Invalidate raw cache and org slice so next get_all_admin_models fetches fresh data."""
        self._cache = None
        self._cache_timestamp = None
        self._catalog_slice = None

    async def resolve_admin_credentials(self, model_id: str) -> Optional[Tuple[str, str, str]]:
        """Return (api_key, base_url, provider_type) for the admin provider that owns this model_id, or None."""
        providers = get_enabled_admin_providers()
        if not providers:
            return None
        models = await self.get_all_admin_models()
        provider_type: Optional[str] = None
        for m in models:
            if m.id == model_id:
                provider_type = m.provider_type
                break
        if not provider_type:
            return None
        for ptype, api_key, base_url in providers:
            if ptype == provider_type:
                return (api_key, base_url, ptype)
        return None


admin_provider_registry = AdminProviderRegistry()
