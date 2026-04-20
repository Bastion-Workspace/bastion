"""Minimal team membership lookup for document search (document-service)."""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


async def list_user_teams(user_id: str) -> Optional[List[Dict[str, Any]]]:
    """Return [{'team_id': ...}, ...] for teams the user belongs to."""
    if not user_id or user_id == "system":
        return None
    try:
        from ds_db.database_manager.database_helpers import fetch_all

        rows = await fetch_all(
            """
            SELECT t.team_id::text AS team_id
            FROM teams t
            INNER JOIN team_members tm ON tm.team_id = t.team_id AND tm.user_id = $1
            """,
            user_id,
        )
        if not rows:
            return []
        return [{"team_id": str(r.get("team_id"))} for r in rows]
    except Exception as e:
        logger.warning("team_lookup failed for %s: %s", user_id, e)
        return None
