"""
Browser Session Health Tasks - Periodic check of saved browser session states.

check_browser_session_health: Beat task that runs every 30 minutes. For each
  saved browser_session_state (user_id, site_domain), restores the session,
  navigates to the site root, and marks the session invalid if the page
  looks like a login screen (redirect or title).
"""

import logging
from typing import Any, Dict

from services.celery_app import celery_app
from services.celery_tasks.async_runner import run_async
from services.database_manager.database_helpers import fetch_all
from services.browser_session_state_service import get_browser_session_state_service

logger = logging.getLogger(__name__)

LOGIN_INDICATORS = ("login", "signin", "sign-in", "auth", "log-in", "sso")


def _looks_like_login(url: str, page_title: str) -> bool:
    """Heuristic: redirect or title suggests we are on a login page."""
    lower_url = (url or "").lower()
    lower_title = (page_title or "").lower()
    for token in LOGIN_INDICATORS:
        if token in lower_url or token in lower_title:
            return True
    return False


async def _async_check_browser_session_health() -> Dict[str, Any]:
    """Load saved sessions, probe each with a quick navigate; invalidate if login page detected."""
    from clients.crawl_service_client import CrawlServiceClient

    state_svc = get_browser_session_state_service()
    rows = await fetch_all(
        "SELECT user_id, site_domain FROM browser_session_states WHERE is_valid = true"
    )
    if not rows:
        return {"checked": 0, "invalidated": 0}

    client = CrawlServiceClient()
    try:
        await client.initialize()
    except Exception as e:
        logger.warning("Browser session health check: could not init crawl client: %s", e)
        return {"checked": 0, "invalidated": 0, "error": str(e)}

    try:
        invalidated = 0
        for row in rows:
            user_id = row["user_id"]
            site_domain = (row["site_domain"] or "").strip().strip(".")
            if not site_domain:
                continue
            state_json = await state_svc.load_session_state(user_id, site_domain)
            if not state_json:
                continue
            session_id = None
            try:
                session_id = await client.browser_create_session(
                    timeout_seconds=15,
                    storage_state_json=state_json,
                )
                if not session_id:
                    continue
                result = await client.browser_execute_action(
                    session_id, "navigate", url=f"https://{site_domain}"
                )
                if not result.get("success"):
                    await client.browser_destroy_session(session_id)
                    invalidated += 1
                    await state_svc.invalidate_session_state(user_id, site_domain)
                    logger.info("Browser session invalidated (navigate failed): %s / %s", user_id, site_domain)
                    continue
                extract_result = await client.browser_execute_action(
                    session_id, "extract", selector="title"
                )
                page_title = (extract_result.get("extracted_content") or "").strip() if extract_result.get("success") else ""
                await client.browser_destroy_session(session_id)
                session_id = None
                if _looks_like_login("", page_title):
                    await state_svc.invalidate_session_state(user_id, site_domain)
                    invalidated += 1
                    logger.info("Browser session invalidated (login page detected): %s / %s", user_id, site_domain)
            except Exception as e:
                logger.warning("Browser session health check failed for %s / %s: %s", user_id, site_domain, e)
                if session_id:
                    try:
                        await client.browser_destroy_session(session_id)
                    except Exception:
                        pass

        if invalidated or rows:
            logger.info("Browser session health check: %s checked, %s invalidated", len(rows), invalidated)
        return {"checked": len(rows), "invalidated": invalidated}
    finally:
        await client.close()


@celery_app.task(bind=True, name="services.celery_tasks.browser_session_health_tasks.check_browser_session_health")
def check_browser_session_health(self) -> Dict[str, Any]:
    """Celery Beat: every 30 minutes, check saved browser sessions and invalidate expired logins."""
    try:
        return run_async(_async_check_browser_session_health())
    except Exception as e:
        logger.exception("check_browser_session_health failed: %s", e)
        return {"checked": 0, "invalidated": 0, "error": str(e)}
