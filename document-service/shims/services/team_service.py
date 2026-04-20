"""Team membership for folder tree — document-service shares Postgres with the backend."""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class TeamService:
    async def initialize(self) -> None:
        return None

    async def list_user_teams(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Return team ids for teams the user belongs to.

        FolderService only needs ``team_id`` keys; shape matches backend usage.
        """
        if not user_id:
            return []
        try:
            from ds_db.database_manager.database_helpers import fetch_all

            rls_context = {"user_id": user_id, "user_role": "user"}
            rows = await fetch_all(
                """
                SELECT team_id::text AS team_id
                FROM team_members
                WHERE user_id = $1
                ORDER BY team_id
                """,
                user_id,
                rls_context=rls_context,
            )
            out: List[Dict[str, Any]] = []
            for row in rows or []:
                tid = row.get("team_id")
                if tid:
                    out.append({"team_id": tid})
            return out
        except Exception as e:
            logger.warning("list_user_teams failed for user_id=%s: %s", user_id, e)
            return []
