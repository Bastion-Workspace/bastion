"""
Post-heartbeat delivery: optional workspace publish and notifications from heartbeat_config.delivery.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_WS_PUBLISH_MAX = 500_000


async def apply_post_heartbeat_delivery(
    line_id: str,
    user_id: str,
    team_name: str,
    heartbeat_config: Optional[Dict[str, Any]],
    full_response: str,
    ceo_agent_profile_id: Optional[str],
    success: bool,
    error: Optional[str],
) -> None:
    """Publish brief to workspace when configured; optional success notification."""
    if not heartbeat_config or not isinstance(heartbeat_config, dict):
        return
    delivery = heartbeat_config.get("delivery")
    if not isinstance(delivery, dict):
        return

    from services import agent_workspace_service
    from services.celery_tasks.team_heartbeat_utils import _send_team_notification

    if success:
        key = (delivery.get("publish_workspace_key") or "").strip()
        if key:
            overwrite = bool(delivery.get("publish_workspace_overwrite"))
            should_write = True
            if not overwrite:
                ent = await agent_workspace_service.get_workspace_entry(line_id, key, user_id)
                existing = (ent or {}).get("value") if isinstance(ent, dict) else None
                if existing and str(existing).strip():
                    should_write = False
            if should_write:
                text = (full_response or "")[:_WS_PUBLISH_MAX]
                if text.strip():
                    try:
                        await agent_workspace_service.set_workspace_entry(
                            line_id,
                            key,
                            text,
                            user_id,
                            updated_by_agent_id=ceo_agent_profile_id,
                        )
                    except Exception as e:
                        logger.warning("publish_workspace_key write failed: %s", e)

        if bool(delivery.get("notify_on_success")):
            try:
                await _send_team_notification(
                    user_id,
                    line_id,
                    team_name,
                    "heartbeat_completed",
                    message="Line heartbeat completed successfully.",
                )
            except Exception as e:
                logger.debug("notify_on_success failed: %s", e)

    if not success and not bool(delivery.get("notify_on_failure", True)):
        return
    # Failure notifications are handled by the caller when notify_on_failure is True (default).
