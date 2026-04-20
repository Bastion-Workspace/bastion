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
    """Ensure session summary flag and episode tier columns (migration 108).

    Skips DDL when columns and indexes already exist. App role (e.g. bastion_user) is not
    table owner: ALTER/CREATE INDEX require owner/superuser, so running them every startup
    spams errors even when migration 108 was applied as postgres.
    """
    try:
        conn = await asyncpg.connect(settings.DATABASE_URL)
        try:
            has_conv_col = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = 'conversations'
                      AND column_name = 'needs_session_summary'
                )
                """
            )
            has_ep_col = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = 'user_episodes'
                      AND column_name = 'is_aged'
                )
                """
            )
            has_idx_conv = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1 FROM pg_indexes
                    WHERE schemaname = 'public'
                      AND indexname = 'idx_conversations_needs_summary_updated'
                )
                """
            )
            has_idx_ep = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1 FROM pg_indexes
                    WHERE schemaname = 'public'
                      AND indexname = 'idx_user_episodes_created_aged'
                )
                """
            )

            if has_conv_col and has_ep_col and has_idx_conv and has_idx_ep:
                logger.debug(
                    "User memory schema (108) already present; skipping DDL as non-owner"
                )
                return

            if not has_conv_col:
                await conn.execute(
                    "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS "
                    "needs_session_summary BOOLEAN NOT NULL DEFAULT FALSE"
                )
            if not has_ep_col:
                await conn.execute(
                    "ALTER TABLE user_episodes ADD COLUMN IF NOT EXISTS "
                    "is_aged BOOLEAN NOT NULL DEFAULT FALSE"
                )
            if not has_idx_conv:
                await conn.execute(
                    """CREATE INDEX IF NOT EXISTS idx_conversations_needs_summary_updated
                       ON conversations (needs_session_summary, updated_at)
                       WHERE needs_session_summary = TRUE"""
                )
            if not has_idx_ep:
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_user_episodes_created_aged "
                    "ON user_episodes (created_at, is_aged)"
                )
            logger.info("Ensured user memory schema columns (needs_session_summary, is_aged)")
        finally:
            await conn.close()
    except Exception as e:
        logger.warning("Could not ensure user memory schema columns: %s", e)
