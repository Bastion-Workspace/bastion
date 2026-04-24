"""
Notify connected users that Agent Factory @mention handles should be refreshed.

Used when profiles are shared/revoked/updated so recipients see changes without reload.
"""

from __future__ import annotations

import logging
from typing import List, Sequence

logger = logging.getLogger(__name__)


async def collect_users_for_agent_profile_handle_notifications(profile_id: str) -> List[str]:
    """Return distinct user_ids to notify: profile owner + direct (non-transitive) share recipients."""
    from services.database_manager.database_helpers import fetch_all, fetch_one

    out: List[str] = []
    owner_row = await fetch_one(
        "SELECT user_id::text AS user_id FROM agent_profiles WHERE id = $1::uuid",
        profile_id,
    )
    if owner_row and owner_row.get("user_id"):
        out.append(str(owner_row["user_id"]))

    rows = await fetch_all(
        """
        SELECT DISTINCT shared_with_user_id::text AS user_id
        FROM agent_artifact_shares
        WHERE artifact_type = 'agent_profile'
          AND artifact_id = $1::uuid
          AND COALESCE(is_transitive, false) = false
        """,
        profile_id,
    )
    for r in rows or []:
        uid = r.get("user_id")
        if uid and uid not in out:
            out.append(uid)
    return out


async def notify_agent_handles_changed(user_ids: Sequence[str]) -> None:
    """Broadcast to user-level WebSockets (same channel as messaging room updates)."""
    uids = [u for u in dict.fromkeys(user_ids) if u]
    if not uids:
        return
    try:
        from utils.websocket_manager import get_websocket_manager

        ws = get_websocket_manager()
        await ws.broadcast_to_users(
            list(uids),
            {"type": "agent_handles_changed", "source": "agent_factory"},
        )
    except Exception as e:
        logger.warning("agent_handles_changed broadcast failed: %s", e)
