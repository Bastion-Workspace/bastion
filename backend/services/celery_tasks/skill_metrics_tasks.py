"""Periodic tasks for skill execution metrics: refresh materialized view + prune old events."""

import asyncio
import logging

from services.celery_app import celery_app
from services.database_manager.celery_database_helpers import run_async_db_task

logger = logging.getLogger(__name__)

RETENTION_DAYS = 90


@celery_app.task(name="services.celery_tasks.skill_metrics_tasks.refresh_skill_usage_stats")
def refresh_skill_usage_stats():
    """Refresh the skill_usage_stats materialized view and prune events older than RETENTION_DAYS."""

    async def _run():
        from utils.shared_db_pool import execute

        try:
            await execute("REFRESH MATERIALIZED VIEW CONCURRENTLY skill_usage_stats")
            logger.info("Refreshed skill_usage_stats materialized view")
        except Exception as e:
            if "has not been populated" in str(e):
                await execute("REFRESH MATERIALIZED VIEW skill_usage_stats")
                logger.info("Refreshed skill_usage_stats (initial, non-concurrent)")
            else:
                logger.warning("Failed to refresh skill_usage_stats: %s", e)

        try:
            result = await execute(
                "DELETE FROM skill_execution_events WHERE created_at < NOW() - INTERVAL '%s days'" % RETENTION_DAYS,
            )
            logger.info("Pruned skill execution events older than %d days", RETENTION_DAYS)
        except Exception as e:
            logger.warning("Failed to prune old skill execution events: %s", e)

    run_async_db_task(_run)
