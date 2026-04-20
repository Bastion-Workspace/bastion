"""Weekly task to generate skill promotion/demotion recommendations based on usage stats."""

import logging

from services.celery_app import celery_app
from services.database_manager.celery_database_helpers import run_async_db_task

logger = logging.getLogger(__name__)

PROMOTE_MIN_USES_30D = 20
PROMOTE_MIN_UNIQUE_AGENTS = 3
PROMOTE_MIN_SUCCESS_RATE = 0.70

DEMOTE_MAX_USES_30D = 2
DEMOTE_LOOKBACK_DAYS = 60


@celery_app.task(name="services.celery_tasks.skill_promotion_tasks.generate_skill_promotion_recommendations")
def generate_skill_promotion_recommendations():
    """Analyze skill_usage_stats and insert promotion/demotion recommendations."""

    async def _run():
        from utils.shared_db_pool import fetch_all, execute, fetch_one

        try:
            await execute("REFRESH MATERIALIZED VIEW CONCURRENTLY skill_usage_stats")
        except Exception:
            try:
                await execute("REFRESH MATERIALIZED VIEW skill_usage_stats")
            except Exception as e:
                logger.warning("Cannot refresh stats before promotion analysis: %s", e)
                return

        promote_rows = await fetch_all(
            """
            SELECT s.skill_id, s.skill_slug, s.total_uses, s.unique_agents,
                   s.success_rate, s.uses_last_30d
            FROM skill_usage_stats s
            JOIN agent_skills a ON a.id = s.skill_id
            WHERE a.is_core = false
              AND a.is_candidate = false
              AND s.uses_last_30d >= $1
              AND s.unique_agents >= $2
              AND s.success_rate >= $3
              AND NOT EXISTS (
                  SELECT 1 FROM skill_promotion_recommendations r
                  WHERE r.skill_id = s.skill_id AND r.action = 'promote'
                    AND r.status = 'pending'
              )
            """,
            PROMOTE_MIN_USES_30D, PROMOTE_MIN_UNIQUE_AGENTS, PROMOTE_MIN_SUCCESS_RATE,
        )

        for row in promote_rows or []:
            reason = (
                f"Used {row['uses_last_30d']}x in 30d by {row['unique_agents']} agents "
                f"with {row['success_rate']:.0%} success rate"
            )
            await execute(
                """
                INSERT INTO skill_promotion_recommendations
                    (skill_id, skill_slug, action, reason, evidence)
                VALUES ($1, $2, 'promote', $3, $4::jsonb)
                """,
                str(row["skill_id"]),
                row["skill_slug"],
                reason,
                '{"uses_30d": %d, "unique_agents": %d, "success_rate": %.3f}' % (
                    row["uses_last_30d"], row["unique_agents"], float(row["success_rate"] or 0),
                ),
            )
        if promote_rows:
            logger.info("Generated %d promotion recommendations", len(promote_rows))

        demote_rows = await fetch_all(
            """
            SELECT a.id AS skill_id, a.slug AS skill_slug,
                   COALESCE(s.uses_last_30d, 0) AS uses_last_30d,
                   COALESCE(s.total_uses, 0) AS total_uses
            FROM agent_skills a
            LEFT JOIN skill_usage_stats s ON s.skill_id = a.id
            WHERE a.is_core = true
              AND a.is_builtin = false
              AND a.is_candidate = false
              AND COALESCE(s.uses_last_30d, 0) <= $1
              AND a.created_at < NOW() - INTERVAL '%s days'
              AND NOT EXISTS (
                  SELECT 1 FROM skill_promotion_recommendations r
                  WHERE r.skill_id = a.id AND r.action = 'demote'
                    AND r.status = 'pending'
              )
            """ % DEMOTE_LOOKBACK_DAYS,
            DEMOTE_MAX_USES_30D,
        )

        for row in demote_rows or []:
            reason = (
                f"Only {row['uses_last_30d']} uses in last 30d "
                f"(total lifetime: {row['total_uses']})"
            )
            await execute(
                """
                INSERT INTO skill_promotion_recommendations
                    (skill_id, skill_slug, action, reason, evidence)
                VALUES ($1, $2, 'demote', $3, $4::jsonb)
                """,
                str(row["skill_id"]),
                row["skill_slug"],
                reason,
                '{"uses_30d": %d, "total_uses": %d}' % (
                    row["uses_last_30d"], row["total_uses"],
                ),
            )
        if demote_rows:
            logger.info("Generated %d demotion recommendations", len(demote_rows))

    run_async_db_task(_run)
