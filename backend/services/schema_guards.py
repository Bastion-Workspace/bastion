"""
Read-only PostgreSQL schema checks shared by the API (lifespan) and Celery workers.

DDL (ALTER TABLE, CREATE INDEX) is not run as the app role: tables are owned by the
bootstrap superuser. Greenfield installs get columns and indexes from
`backend/postgres_init/01_init.sql`; legacy DBs should run migration 108 (or full
`migrations/108_memory_session_summary.sql`) as a privileged user.
"""

import logging

import asyncpg

from config import settings

logger = logging.getLogger(__name__)


async def ensure_user_memory_schema_columns() -> None:
    """Verify session summary flag and episode tier columns (migration 108).

    Logs at debug when complete; warns once if anything is missing so operators can
    apply migration 108 as superuser. Does not execute DDL as bastion_user.
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
                logger.debug("User memory schema (108) present")
                return

            logger.warning(
                "User memory schema (108) incomplete "
                "(conversations.needs_session_summary=%s, user_episodes.is_aged=%s, "
                "idx_conversations_needs_summary_updated=%s, idx_user_episodes_created_aged=%s). "
                "Apply migration 108 as a table owner or superuser.",
                has_conv_col,
                has_ep_col,
                has_idx_conv,
                has_idx_ep,
            )
        finally:
            await conn.close()
    except Exception as e:
        logger.warning("Could not verify user memory schema: %s", e)
