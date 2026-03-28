"""
Episode Service - Episodic memory: conversation-derived events for "remember what we worked on?"
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from services.database_manager.database_helpers import execute, fetch_all, fetch_one

logger = logging.getLogger(__name__)

VALID_EPISODE_TYPES = (
    "chat",
    "research",
    "editing",
    "coding",
    "automation",
    "file_management",
    "general",
)


def _json_list(items: List[Any]) -> str:
    """Serialize list to JSON array string for PostgreSQL jsonb."""
    return json.dumps(items or [])


class EpisodeService:
    """CRUD and formatting for user_episodes (episodic memory)."""

    async def get_user_episodes(
        self,
        user_id: str,
        limit: int = 50,
        days: Optional[int] = 30,
    ) -> List[Dict[str, Any]]:
        """Load recent episodes for a user. If days is None, no date filter."""
        try:
            if days is not None:
                since = datetime.now(timezone.utc) - timedelta(days=days)
                rows = await fetch_all(
                    """SELECT id, user_id, conversation_id, summary, episode_type, agent_used,
                       tools_used, key_topics, outcome, embedding, is_aged, created_at
                       FROM user_episodes
                       WHERE user_id = $1 AND created_at >= $2
                       ORDER BY created_at DESC
                       LIMIT $3""",
                    user_id,
                    since,
                    limit,
                )
            else:
                rows = await fetch_all(
                    """SELECT id, user_id, conversation_id, summary, episode_type, agent_used,
                       tools_used, key_topics, outcome, embedding, is_aged, created_at
                       FROM user_episodes
                       WHERE user_id = $1
                       ORDER BY created_at DESC
                       LIMIT $2""",
                    user_id,
                    limit,
                )
            return [dict(r) for r in (rows or [])]
        except Exception as e:
            logger.warning("Failed to get user episodes for %s: %s", user_id, e)
            return []

    async def create_episode(
        self,
        user_id: str,
        summary: str,
        episode_type: str = "general",
        conversation_id: Optional[str] = None,
        agent_used: Optional[str] = None,
        tools_used: Optional[List[str]] = None,
        key_topics: Optional[List[str]] = None,
        outcome: str = "completed",
        is_aged: bool = False,
    ) -> Optional[int]:
        """Insert one episode. Returns episode id or None on failure."""
        try:
            tools_json = tools_used if tools_used is not None else []
            topics_json = key_topics if key_topics is not None else []
            if episode_type not in VALID_EPISODE_TYPES:
                episode_type = "general"
            row = await fetch_one(
                """INSERT INTO user_episodes (user_id, conversation_id, summary, episode_type, agent_used, tools_used, key_topics, outcome, is_aged)
                   VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8, $9)
                   RETURNING id""",
                user_id,
                conversation_id,
                summary,
                episode_type,
                agent_used,
                _json_list(tools_json),
                _json_list(topics_json),
                outcome,
                is_aged,
            )
            return int(row["id"]) if row else None
        except Exception as e:
            logger.warning("Failed to create episode for user %s: %s", user_id, e)
            return None

    async def update_episode_embedding(self, episode_id: int, embedding: List[float]) -> bool:
        """Set embedding for an episode (used after async embed task)."""
        try:
            await execute(
                "UPDATE user_episodes SET embedding = $1 WHERE id = $2",
                embedding,
                episode_id,
            )
            return True
        except Exception as e:
            logger.warning("Failed to update episode embedding for id %s: %s", episode_id, e)
            return False

    async def delete_episode(self, user_id: str, episode_id: int) -> bool:
        """Delete one episode. Returns True if a row was deleted."""
        try:
            result = await execute(
                "DELETE FROM user_episodes WHERE user_id = $1 AND id = $2",
                user_id,
                episode_id,
            )
            return "DELETE" in (result or "")
        except Exception as e:
            logger.warning("Failed to delete episode %s for user %s: %s", episode_id, user_id, e)
            return False

    async def mark_episodes_aged_before(self, cutoff) -> int:
        """Set is_aged=true for episodes created before cutoff. Returns rows updated."""
        try:
            result = await execute(
                """UPDATE user_episodes SET is_aged = TRUE
                   WHERE created_at < $1 AND is_aged = FALSE""",
                cutoff,
            )
            if result and "UPDATE" in result:
                try:
                    return int(result.split()[-1])
                except (ValueError, IndexError):
                    pass
            return 0
        except Exception as e:
            logger.warning("mark_episodes_aged_before failed: %s", e)
            return 0

    async def fetch_episodes_for_graduation(self, older_than, limit: int = 30) -> List[Dict[str, Any]]:
        """Episodes older than cutoff, for fact graduation before delete."""
        try:
            rows = await fetch_all(
                """SELECT id, user_id, summary, created_at FROM user_episodes
                   WHERE created_at < $1 ORDER BY created_at ASC LIMIT $2""",
                older_than,
                limit,
            )
            return [dict(r) for r in (rows or [])]
        except Exception as e:
            logger.warning("fetch_episodes_for_graduation failed: %s", e)
            return []

    async def delete_episode_by_id(self, episode_id: int) -> bool:
        try:
            result = await execute("DELETE FROM user_episodes WHERE id = $1", episode_id)
            return "DELETE" in (result or "")
        except Exception as e:
            logger.warning("delete_episode_by_id failed: %s", e)
            return False

    async def delete_all_episodes(self, user_id: str) -> int:
        """Delete all episodes for a user. Returns count deleted."""
        try:
            result = await execute("DELETE FROM user_episodes WHERE user_id = $1", user_id)
            if result and "DELETE" in result:
                try:
                    return int(result.split()[-1])
                except (ValueError, IndexError):
                    pass
            return 0
        except Exception as e:
            logger.warning("Failed to delete all episodes for user %s: %s", user_id, e)
            return 0

    @staticmethod
    def format_episodes_for_prompt(episodes: List[Dict[str, Any]]) -> str:
        """Format episodes for LLM system prompt: 'RECENT ACTIVITY:\\n- [date] [type]: summary'."""
        if not episodes:
            return ""
        lines = []
        for ep in episodes:
            created = ep.get("created_at")
            if hasattr(created, "strftime"):
                date_str = created.strftime("%Y-%m-%d")
            elif isinstance(created, str):
                date_str = created[:10] if len(created) >= 10 else created
            else:
                date_str = ""
            ep_type = ep.get("episode_type") or "general"
            summary = (ep.get("summary") or "").strip()
            if summary:
                lines.append(f"- [{date_str}] [{ep_type}]: {summary}")
        if not lines:
            return ""
        return "RECENT ACTIVITY:\n" + "\n".join(lines)


episode_service = EpisodeService()
