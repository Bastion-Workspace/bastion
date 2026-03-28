"""
Idempotent PostgreSQL column/index ensures shared by the API (lifespan) and Celery workers.

Workers do not run FastAPI lifespan; without these guards, scheduled tasks can hit
UndefinedColumnError on databases provisioned before newer migrations.
"""

import logging

import asyncpg

from config import settings

logger = logging.getLogger(__name__)


async def ensure_user_memory_schema_columns() -> None:
    """Ensure session summary flag and episode tier columns (migration 108)."""
    try:
        conn = await asyncpg.connect(settings.DATABASE_URL)
        try:
            await conn.execute(
                "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS needs_session_summary BOOLEAN NOT NULL DEFAULT FALSE"
            )
            await conn.execute(
                "ALTER TABLE user_episodes ADD COLUMN IF NOT EXISTS is_aged BOOLEAN NOT NULL DEFAULT FALSE"
            )
            await conn.execute(
                """CREATE INDEX IF NOT EXISTS idx_conversations_needs_summary_updated
                   ON conversations (needs_session_summary, updated_at)
                   WHERE needs_session_summary = TRUE"""
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_user_episodes_created_aged ON user_episodes (created_at, is_aged)"
            )
            logger.info("Ensured user memory schema columns (needs_session_summary, is_aged)")
        finally:
            await conn.close()
    except Exception as e:
        logger.warning("Could not ensure user memory schema columns: %s", e)
