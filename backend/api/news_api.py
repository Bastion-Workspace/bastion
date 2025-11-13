"""
News API Endpoints
Balanced synthesized news headlines and articles.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends

from models.news_models import NewsHeadlinesResponse, NewsArticleSynth
from models.api_models import AuthenticatedUserResponse
from utils.auth_middleware import get_current_user
from services.capabilities_service import capabilities_service
from services.service_container import get_service_container
from utils.auth_middleware import require_admin

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/news", tags=["News"])


@router.get("/headlines", response_model=NewsHeadlinesResponse)
async def get_headlines(
    limit: int = 20,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    try:
        # Capability gate: feature.news.view
        if not await capabilities_service.user_has_feature(current_user.dict(), "feature.news.view"):
            raise HTTPException(status_code=403, detail="News feature is disabled for this user")
        service_container = await get_service_container()
        news_service = getattr(service_container, 'news_service', None)
        if not news_service:
            raise HTTPException(status_code=503, detail="News service not available")
        headlines = await news_service.get_latest_headlines(user_id=current_user.user_id, limit=limit)
        return NewsHeadlinesResponse(headlines=headlines)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ NEWS API ERROR: Failed to get headlines: {e}")
        raise HTTPException(status_code=500, detail="Failed to get headlines")


@router.get("/{news_id}", response_model=NewsArticleSynth)
async def get_article(
    news_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    try:
        # Capability gate: feature.news.view
        if not await capabilities_service.user_has_feature(current_user.dict(), "feature.news.view"):
            raise HTTPException(status_code=403, detail="News feature is disabled for this user")
        service_container = await get_service_container()
        news_service = getattr(service_container, 'news_service', None)
        if not news_service:
            raise HTTPException(status_code=503, detail="News service not available")
        article = await news_service.get_article(news_id, user_id=current_user.user_id)
        if not article:
            raise HTTPException(status_code=404, detail="News article not found")
        return article
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ NEWS API ERROR: Failed to get article {news_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get article")


@router.delete("/purge")
async def purge_all_news(
    current_user: AuthenticatedUserResponse = Depends(require_admin())
):
    """Delete all news articles and their markdown files (admin only)."""
    try:
        service_container = await get_service_container()
        news_service = getattr(service_container, 'news_service', None)
        if not news_service:
            raise HTTPException(status_code=503, detail="News service not available")

        # Fetch file paths to delete from disk first
        from services.database_manager.database_helpers import fetch_all, execute
        rows = await fetch_all("SELECT file_path FROM news_articles")
        import os
        deleted_files = 0
        for row in rows or []:
            p = row.get("file_path") if isinstance(row, dict) else None
            try:
                if p and os.path.exists(p):
                    os.remove(p)
                    deleted_files += 1
            except Exception:
                pass

        # Delete all rows
        await execute("DELETE FROM news_articles")

        return {"status": "ok", "deleted_files": deleted_files}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ NEWS API ERROR: Failed to purge news: {e}")
        raise HTTPException(status_code=500, detail="Failed to purge news")


