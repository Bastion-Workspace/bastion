"""
Twitter Ingestion Tasks
Background polling of a user's Following timelines and related expansions
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from services.celery_app import celery_app, update_task_progress, TaskStatus
from services.user_settings_kv_service import get_user_setting, set_user_setting

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="services.celery_tasks.twitter_tasks.scheduled_twitter_poll_task")
def scheduled_twitter_poll_task(self) -> Dict[str, Any]:
    """
    Scheduled task to poll Twitter for all users who enabled ingestion.
    """
    try:
        update_task_progress(self, 1, 3, "Checking users with Twitter enabled...")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(_async_poll_enabled_users(self))
        finally:
            loop.close()

        update_task_progress(self, 3, 3, "Twitter polling initiated for enabled users")
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"❌ TWITTER SCHEDULED TASK ERROR: {e}")
        self.update_state(state=TaskStatus.FAILURE, meta={"error": str(e)})
        return {"success": False, "error": str(e)}


async def _async_poll_enabled_users(task) -> Dict[str, Any]:
    """Poll all users with twitter.enabled=true. Placeholder enumerator uses recent audit of sessions/users."""
    try:
        # Minimal safe enumerator: query users table for active users, then check per-user setting
        from services.database_manager.database_helpers import fetch_all

        users = await fetch_all("SELECT user_id FROM users WHERE is_active = true")
        polled = 0
        for row in users:
            user_id = row["user_id"]
            enabled = (await get_user_setting(user_id, "twitter.enabled")) == "true"
            if not enabled:
                continue
            # Dispatch per-user poll as separate task for isolation
            poll_twitter_for_user_task.delay(user_id=user_id)
            polled += 1

        return {"users_polled": polled}
    except Exception as e:
        logger.error(f"❌ Failed to enumerate enabled users: {e}")
        raise


@celery_app.task(bind=True, name="services.celery_tasks.twitter_tasks.poll_twitter_for_user")
def poll_twitter_for_user_task(self, user_id: str) -> Dict[str, Any]:
    """Poll Twitter for a specific user based on their settings."""
    try:
        update_task_progress(self, 1, 4, f"Polling Twitter for user {user_id}...")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(_async_poll_user_twitter(self, user_id))
        finally:
            loop.close()

        return result
    except Exception as e:
        logger.error(f"❌ TWITTER USER POLL ERROR: {e}")
        self.update_state(state=TaskStatus.FAILURE, meta={"error": str(e)})
        return {"success": False, "error": str(e)}


async def _async_poll_user_twitter(task, user_id: str) -> Dict[str, Any]:
    """
    Placeholder ingestion logic:
    - Reads user settings (bearer_token, backfill_days, include_*)
    - Updates last_poll_at and returns a stub result
    Real fetch/normalize/embed will be added next.
    """
    bearer = await get_user_setting(user_id, "twitter.bearer_token")
    if not bearer:
        return {"success": False, "message": "No bearer token configured"}

    poll_interval = int((await get_user_setting(user_id, "twitter.poll_interval_minutes")) or 5)
    backfill_days = int((await get_user_setting(user_id, "twitter.backfill_days")) or 60)
    include_replies = ((await get_user_setting(user_id, "twitter.include_replies")) or "true") == "true"
    include_retweets = ((await get_user_setting(user_id, "twitter.include_retweets")) or "true") == "true"
    include_quotes = ((await get_user_setting(user_id, "twitter.include_quotes")) or "true") == "true"
    x_user_id = await get_user_setting(user_id, "twitter.user_id")

    since_dt = datetime.utcnow() - timedelta(days=backfill_days)

    from services.twitter_ingestion_service import TwitterIngestionService, TwitterIngestionConfig

    svc = TwitterIngestionService()
    await svc.initialize()
    try:
        cfg = TwitterIngestionConfig(
            bearer_token=bearer,
            user_id=x_user_id or user_id,
            backfill_days=backfill_days,
            include_replies=include_replies,
            include_retweets=include_retweets,
            include_quotes=include_quotes,
        )

        since_id = await get_user_setting(user_id, "twitter.since_id")
        tweets, newest_id = await svc.fetch_following_tweets(cfg, since_id)
        embedded = await svc.embed_tweets(tweets, user_id=user_id)

        if newest_id:
            await set_user_setting(user_id, "twitter.since_id", newest_id)
        await set_user_setting(user_id, "twitter.last_poll_at", datetime.utcnow().isoformat())
    finally:
        await svc.close()

    logger.info(
        f"🐦 TWITTER POLL: user={user_id} interval={poll_interval}m backfill_days={backfill_days} "
        f"replies={include_replies} retweets={include_retweets} quotes={include_quotes} since={since_dt.isoformat()}"
    )

    update_task_progress(task, 4, 4, "Twitter poll complete")
    return {"success": True, "polled_since": since_dt.isoformat(), "embedded": embedded}


