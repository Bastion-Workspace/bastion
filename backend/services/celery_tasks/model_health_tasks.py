"""
Periodic LLM catalog health check: refresh admin registry, detect orphans, notify admins.
"""

import logging
from typing import Any, Dict

from services.celery_app import celery_app
from services.celery_tasks.async_runner import run_async

logger = logging.getLogger(__name__)


@celery_app.task(name="services.celery_tasks.model_health_tasks.check_admin_llm_catalog_health_task")
def check_admin_llm_catalog_health_task() -> Dict[str, Any]:
    """Refresh admin model catalog and notify admins if enabled or role models are orphaned."""
    try:
        return run_async(_async_check_admin_llm_catalog_health())
    except Exception as e:
        logger.error("check_admin_llm_catalog_health_task failed: %s", e)
        return {"success": False, "error": str(e)}


async def _async_check_admin_llm_catalog_health() -> Dict[str, Any]:
    from services.admin_provider_registry import admin_provider_registry
    from services.model_configuration_notifier import notify_active_admins_catalog_health
    from services.settings_service import settings_service

    if not getattr(settings_service, "_initialized", False):
        await settings_service.initialize()

    admin_provider_registry.refresh()
    slice_data = await admin_provider_registry.get_org_catalog_slice()

    orphans = slice_data.get("orphaned_enabled_models") or []
    role_orphans = slice_data.get("orphaned_role_models") or {}
    verified = slice_data.get("catalog_verified")

    if not verified:
        preview = (
            "The admin LLM provider catalog could not be verified (empty or partial response). "
            "Check API keys and use Refresh in Settings > Models."
        )
        sent = await notify_active_admins_catalog_health(
            preview=preview,
            dedupe_key="catalog_unverified",
        )
        return {
            "success": True,
            "catalog_verified": False,
            "notifications_sent": sent,
        }

    if not orphans and not role_orphans:
        return {
            "success": True,
            "catalog_verified": True,
            "orphaned_enabled_count": 0,
            "notifications_sent": 0,
        }

    parts = []
    if orphans:
        parts.append(f"{len(orphans)} enabled chat model ID(s) not in the live catalog")
    if role_orphans:
        parts.append(
            "stale role settings: "
            + ", ".join(f"{k}={v}" for k, v in list(role_orphans.items())[:6])
        )
    preview = "OpenRouter/catalog mismatch: " + "; ".join(parts) + ". Open Settings > Models to fix."
    dedupe = f"orphans:{len(orphans)}:{len(role_orphans)}:{hash(tuple(sorted(role_orphans.items()))) % 100000}"
    sent = await notify_active_admins_catalog_health(preview=preview, dedupe_key=dedupe)

    return {
        "success": True,
        "catalog_verified": True,
        "orphaned_enabled_count": len(orphans),
        "orphaned_role_keys": list(role_orphans.keys()),
        "notifications_sent": sent,
    }
