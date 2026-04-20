"""
Periodic federation outbox sync (pull from remote peers + prune local outbox).
"""

import logging
from typing import Any, Dict

from config import settings
from services.celery_app import celery_app
from services.celery_tasks.async_runner import run_async

logger = logging.getLogger(__name__)

ADMIN_RLS = {"user_id": "", "user_role": "admin"}


async def _async_federation_sync_outbox() -> Dict[str, Any]:
    if not getattr(settings, "FEDERATION_ENABLED", False):
        return {"skipped": True, "reason": "federation_disabled"}
    from services.federation_service import federation_service

    return await federation_service.sync_pull_for_local_instance(ADMIN_RLS)


async def _async_federation_presence_sync() -> Dict[str, Any]:
    if not getattr(settings, "FEDERATION_ENABLED", False):
        return {"skipped": True, "reason": "federation_disabled"}
    from services.federation_service import federation_service

    return await federation_service.sync_federation_presence_outbound(ADMIN_RLS)


@celery_app.task(name="services.celery_tasks.federation_tasks.federation_sync_outbox_beat")
def federation_sync_outbox_beat() -> Dict[str, Any]:
    """Celery Beat: pull remote outbox for asymmetric peers; prune local outbox."""
    try:
        return run_async(_async_federation_sync_outbox())
    except Exception as e:
        logger.error("federation_sync_outbox_beat failed: %s", e)
        return {"success": False, "error": str(e)}


@celery_app.task(name="services.celery_tasks.federation_tasks.federation_presence_sync_beat")
def federation_presence_sync_beat() -> Dict[str, Any]:
    """Celery Beat: push federated presence batches to peers."""
    try:
        return run_async(_async_federation_presence_sync())
    except Exception as e:
        logger.error("federation_presence_sync_beat failed: %s", e)
        return {"success": False, "error": str(e)}
