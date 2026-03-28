"""
WebSocket notifications and persistence when saved LLM model IDs are stale or unavailable.
Uses the same agent_notification path as schedule pause / execution events.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

NOTIFICATION_SUBTYPE = "model_configuration"
DEDUPE_KEY_PREFIX = "model_cfg_notify:"
DEDUPE_HOURS = 24


def _dedupe_storage_key(dedupe_key: str) -> str:
    return f"{DEDUPE_KEY_PREFIX}{dedupe_key}"


async def maybe_notify_model_configuration_issue(
    user_id: str,
    *,
    title: str,
    preview: str,
    dedupe_key: str,
    agent_name: str = "AI models",
    agent_profile_id: Optional[str] = None,
    playbook_id: Optional[str] = None,
    step_names: Optional[List[str]] = None,
    requested_model: Optional[str] = None,
    effective_model: Optional[str] = None,
) -> None:
    """Send in-app notification once per dedupe_key per DEDUPE_HOURS for this user."""
    try:
        from services.user_settings_kv_service import get_user_setting, set_user_setting

        storage_key = _dedupe_storage_key(dedupe_key)
        last_raw = await get_user_setting(user_id, storage_key)
        now = datetime.now(timezone.utc)
        if last_raw:
            try:
                last = datetime.fromisoformat(last_raw.replace("Z", "+00:00"))
                if now - last < timedelta(hours=DEDUPE_HOURS):
                    return
            except (TypeError, ValueError):
                pass

        from utils.websocket_manager import get_websocket_manager

        ws = get_websocket_manager()
        if not ws:
            return

        payload: Dict[str, Any] = {
            "type": "agent_notification",
            "subtype": NOTIFICATION_SUBTYPE,
            "agent_name": agent_name,
            "title": (title or "Model configuration")[:200],
            "preview": (preview or "")[:500],
            "timestamp": now.isoformat(),
        }
        if agent_profile_id:
            payload["agent_profile_id"] = agent_profile_id
        if playbook_id:
            payload["playbook_id"] = playbook_id
        if step_names:
            payload["step_names"] = step_names[:20]
        if requested_model:
            payload["requested_model"] = requested_model[:300]
        if effective_model:
            payload["effective_model"] = effective_model[:300]

        await ws.send_to_session(payload, user_id)
        await set_user_setting(user_id, storage_key, now.isoformat(), "string")
    except Exception as e:
        logger.debug("model configuration notification skipped: %s", e)


async def persist_conversation_user_chat_model(
    conversation_id: str,
    user_id: str,
    effective_model_id: Optional[str],
) -> None:
    """Set or remove user_chat_model in conversation metadata_json."""
    try:
        from services.database_manager.database_helpers import execute, fetch_one

        rls = {"user_id": user_id, "user_role": "user"}
        row = await fetch_one(
            "SELECT metadata_json FROM conversations WHERE conversation_id = $1 AND user_id = $2",
            conversation_id,
            user_id,
            rls_context=rls,
        )
        if not row:
            return
        raw = row.get("metadata_json")
        if isinstance(raw, dict):
            meta = dict(raw)
        else:
            meta = json.loads(raw or "{}")
        if effective_model_id:
            meta["user_chat_model"] = effective_model_id
        else:
            meta.pop("user_chat_model", None)
        await execute(
            "UPDATE conversations SET metadata_json = $1, updated_at = NOW() "
            "WHERE conversation_id = $2 AND user_id = $3",
            json.dumps(meta),
            conversation_id,
            user_id,
            rls_context=rls,
        )
    except Exception as e:
        logger.warning("persist_conversation_user_chat_model failed: %s", e)


async def persist_user_kv_model(user_id: str, setting_key: str, effective_model_id: Optional[str]) -> None:
    """Set or clear a user_settings KV key used for model selection (empty string clears for OR-chain)."""
    try:
        from services.user_settings_kv_service import set_user_setting

        val = effective_model_id if effective_model_id else ""
        await set_user_setting(user_id, setting_key, val, "string")
    except Exception as e:
        logger.warning("persist_user_kv_model %s failed: %s", setting_key, e)


async def persist_agent_profile_model_preference(
    user_id: str,
    agent_profile_id: str,
    effective_model_id: Optional[str],
) -> None:
    """Update or clear model_preference for an agent profile owned by user_id."""
    try:
        from services.database_manager.database_helpers import execute

        rls = {"user_id": user_id, "user_role": "user"}
        await execute(
            "UPDATE agent_profiles SET model_preference = $1, updated_at = NOW() "
            "WHERE id = $2::uuid AND user_id = $3",
            effective_model_id,
            agent_profile_id,
            user_id,
            rls_context=rls,
        )
    except Exception as e:
        logger.warning("persist_agent_profile_model_preference failed: %s", e)


_KV_KEYS = {
    "chat": "user_chat_model",
    "fast": "user_fast_model",
    "image": "user_image_gen_model",
}


async def handle_user_model_retarget_flow(
    user_id: str,
    raw_model_id: Optional[str],
    *,
    role: str,
    source: str,
    conversation_id: str,
) -> Optional[str]:
    """
    Apply try_soft_retarget, notify, and persist for chat/classification/image model selection.
    source: request | conversation | user_kv | org — only conversation/user_kv are persisted.
    Returns model id to use with resolve_model_context (may be emergency fallback).
    """
    if not raw_model_id or not str(raw_model_id).strip():
        return None

    from services.model_source_resolver import pick_fallback_model_id, try_soft_retarget

    raw_model_id = str(raw_model_id).strip()
    retarget = await try_soft_retarget(user_id, raw_model_id)
    effective = retarget.get("model_id") or raw_model_id
    available = bool(retarget.get("available"))
    retargeted = bool(retarget.get("retargeted"))

    if available and not retargeted and raw_model_id == effective:
        return effective

    emergency = None
    if not available:
        emergency = await pick_fallback_model_id(user_id)
    effective_for_use = effective if available else (emergency or raw_model_id)

    role_labels = {
        "chat": "Chat",
        "fast": "Classification",
        "image": "Image generation",
    }
    label = role_labels.get(role, "Model")

    if not available and not emergency:
        preview = (
            f"{label} model «{raw_model_id}» is not available and no fallback could be selected. "
            "Update enabled models in Settings > AI Models."
        )
    elif not available and emergency:
        preview = (
            f"{label} model «{raw_model_id}» is not available. Using «{emergency}» for this session; saved preference was cleared or updated."
        )
    else:
        preview = f"{label} model «{raw_model_id}» is not valid for current settings. Using «{effective_for_use}»."

    dedupe = f"{role}:{source}:{raw_model_id}->{effective_for_use}"
    await maybe_notify_model_configuration_issue(
        user_id,
        title=f"{label} model adjusted",
        preview=preview[:500],
        dedupe_key=dedupe,
    )

    persist_val: Optional[str]
    if available:
        persist_val = effective
    else:
        persist_val = emergency

    if role == "chat" and source == "conversation" and conversation_id:
        await persist_conversation_user_chat_model(conversation_id, user_id, persist_val)
    elif source == "user_kv" and role in _KV_KEYS:
        await persist_user_kv_model(user_id, _KV_KEYS[role], persist_val)

    return effective_for_use


async def notify_active_admins_catalog_health(*, preview: str, dedupe_key: str) -> int:
    """Notify all active admin users of catalog/orphan issues (WebSocket, deduped per user)."""
    from services.database_manager.database_helpers import fetch_all

    try:
        rows = await fetch_all(
            "SELECT user_id FROM users WHERE role = $1 AND is_active = true",
            "admin",
        )
    except Exception as e:
        logger.warning("notify_active_admins_catalog_health: failed to list admins: %s", e)
        return 0

    count = 0
    for row in rows:
        uid = row.get("user_id")
        if not uid:
            continue
        await maybe_notify_model_configuration_issue(
            uid,
            title="LLM catalog health",
            preview=preview[:500],
            dedupe_key=f"admin_catalog:{dedupe_key}",
            agent_name="System",
        )
        count += 1
    return count
