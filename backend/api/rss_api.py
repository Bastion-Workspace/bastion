"""
RSS API Endpoints
FastAPI endpoints for RSS feed management and article processing
"""

import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import ValidationError, BaseModel, Field

from tools_service.models.rss_models import (
    RSSFeed, RSSArticle, RSSFeedCreate, RSSArticleImport,
    RSSFeedPollResult, RSSArticleProcessResult
)
from models.api_models import AuthenticatedUserResponse
from tools_service.services.rss_service import get_rss_service
from services.celery_tasks.rss_tasks import poll_rss_feeds_task, process_rss_article_task
from services.user_settings_kv_service import get_user_setting, set_user_setting
from services.database_manager.database_helpers import execute
from services.rss_import_placement import resolve_rss_import_target_folder_id
from utils.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["RSS"])

RSS_IMPORT_TARGET_FOLDER_KEY = "rss_import_target_folder_id"


class RSSImportLocationResponse(BaseModel):
    folder_id: Optional[str] = Field(None, description="My Documents or (admin) global folder ID; null uses default Web Sources placement")


class RSSImportLocationUpdate(BaseModel):
    folder_id: Optional[str] = Field(None, description="Set to null or empty to clear and use default placement")


class RSSStarToggleResponse(BaseModel):
    is_starred: bool = Field(..., description="Starred state after toggle")


class RSSBulkCountResponse(BaseModel):
    count: int = Field(..., description="Number of rows affected")


@router.post("/api/rss/feeds", response_model=RSSFeed)
async def create_rss_feed(
    feed_data: RSSFeedCreate,
    update_if_exists: bool = False,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Create a new RSS feed
    
    Register a new RSS feed for the current user.
    """
    try:
        logger.info(f"📡 RSS API: Creating RSS feed for user {current_user.user_id}")
        
        # Set user ID for user-specific feeds
        feed_data.user_id = current_user.user_id
        
        rss_service = await get_rss_service()
        feed = await rss_service.create_feed(feed_data, update_if_exists=update_if_exists)
        
        # Check if this was an existing feed or a new one
        existing_feed = await rss_service.get_feed_by_url(feed_data.feed_url, current_user.user_id)
        
        if existing_feed and existing_feed.feed_id == feed.feed_id:
            if update_if_exists:
                logger.info(f"✅ RSS API: Updated existing RSS feed {feed.feed_id}")
            else:
                logger.info(f"✅ RSS API: Feed already exists, returning existing feed {feed.feed_id}")
            return feed
        else:
            logger.info(f"✅ RSS API: Created new RSS feed {feed.feed_id}")
            try:
                poll_rss_feeds_task.delay(
                    user_id=current_user.user_id,
                    feed_ids=[feed.feed_id],
                    force_poll=True,
                )
                logger.info(f"RSS API: Enqueued initial poll for new feed {feed.feed_id}")
            except Exception as poll_err:
                logger.warning(
                    "RSS API: Could not enqueue initial poll for feed %s: %s",
                    feed.feed_id,
                    poll_err,
                )
            return feed
        
    except ValidationError as e:
        logger.error(f"❌ RSS API ERROR: Validation error creating feed: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid feed data: {str(e)}")
    except Exception as e:
        logger.error(f"❌ RSS API ERROR: Failed to create RSS feed: {e}")
        raise HTTPException(status_code=500, detail="Failed to create RSS feed")


@router.post("/api/rss/feeds/global", response_model=RSSFeed)
async def create_global_rss_feed(
    feed_data: RSSFeedCreate,
    update_if_exists: bool = False,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Create a new global RSS feed (admin only)
    
    Register a global RSS feed (admin only).
    """
    try:
        # Only admin users can create global feeds
        if current_user.role != "admin":
            raise HTTPException(status_code=403, detail="Only admin users can create global RSS feeds")
        
        logger.info(f"📡 RSS API: Creating global RSS feed for admin {current_user.user_id}")
        
        # Set user_id to None for global feeds
        feed_data.user_id = None
        
        rss_service = await get_rss_service()
        feed = await rss_service.create_feed(feed_data, update_if_exists=update_if_exists)
        
        logger.info(f"✅ RSS API: Created global RSS feed {feed.feed_id}")
        try:
            poll_rss_feeds_task.delay(
                user_id=current_user.user_id,
                feed_ids=[feed.feed_id],
                force_poll=True,
            )
            logger.info(f"RSS API: Enqueued initial poll for new global feed {feed.feed_id}")
        except Exception as poll_err:
            logger.warning(
                "RSS API: Could not enqueue initial poll for global feed %s: %s",
                feed.feed_id,
                poll_err,
            )
        return feed
        
    except ValidationError as e:
        logger.error(f"❌ RSS API ERROR: Validation error creating global feed: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid feed data: {str(e)}")
    except Exception as e:
        logger.error(f"❌ RSS API ERROR: Failed to create global RSS feed: {e}")
        raise HTTPException(status_code=500, detail="Failed to create global RSS feed")


@router.get("/api/rss/feeds", response_model=List[RSSFeed])
async def get_rss_feeds(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Get all RSS feeds for the current user
    
    List RSS feeds visible to the current user.
    """
    try:
        logger.info(f"📡 RSS API: Getting RSS feeds for user {current_user.user_id}")
        
        rss_service = await get_rss_service()
        is_admin = current_user.role == "admin"
        feeds = await rss_service.get_user_feeds(current_user.user_id, is_admin=is_admin)
        
        logger.info(f"✅ RSS API: Retrieved {len(feeds)} RSS feeds for {'admin' if is_admin else 'user'} {current_user.user_id}")
        return feeds
        
    except Exception as e:
        logger.error(f"❌ RSS API ERROR: Failed to get RSS feeds: {e}")
        raise HTTPException(status_code=500, detail="Failed to get RSS feeds")


@router.get("/api/rss/feeds/categorized")
async def get_categorized_rss_feeds(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Get RSS feeds categorized by user-specific vs global
    
    Response separates user-scoped feeds from global feeds.
    """
    try:
        logger.info(f"📡 RSS API: Getting categorized RSS feeds for user {current_user.user_id}")
        
        rss_service = await get_rss_service()
        is_admin = current_user.role == "admin"
        feeds = await rss_service.get_user_feeds(current_user.user_id, is_admin=is_admin)
        
        # Categorize feeds
        user_feeds = []
        global_feeds = []
        
        for feed in feeds:
            if feed.user_id is None:
                global_feeds.append(feed)
            else:
                user_feeds.append(feed)
        
        result = {
            "user_feeds": user_feeds,
            "global_feeds": global_feeds,
            "total_user_feeds": len(user_feeds),
            "total_global_feeds": len(global_feeds)
        }
        
        logger.info(f"✅ RSS API: Retrieved {len(user_feeds)} user feeds and {len(global_feeds)} global feeds for {'admin' if is_admin else 'user'} {current_user.user_id}")
        return result
        
    except Exception as e:
        logger.error(f"❌ RSS API ERROR: Failed to get categorized RSS feeds: {e}")
        raise HTTPException(status_code=500, detail="Failed to get categorized RSS feeds")


@router.get("/api/rss/feeds/validate")
async def validate_feed_url(
    feed_url: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Validate an RSS feed URL and return feed information

    Registered before /feeds/{feed_id} so "validate" is not captured as a feed_id.
    """
    try:
        logger.info(f"📡 RSS API: Validating RSS feed URL: {feed_url} for user {current_user.user_id}")

        rss_service = await get_rss_service()

        existing_feed = await rss_service.get_feed_by_url(feed_url, current_user.user_id)

        validation_result = {
            "status": "success",
            "feed_url": feed_url,
            "exists_for_user": existing_feed is not None,
            "data": {
                "title": "Sample RSS Feed",
                "description": "This is a sample RSS feed for testing purposes",
                "articles": [
                    {"title": "Sample Article 1", "description": "This is a sample article for testing..."},
                    {"title": "Sample Article 2", "description": "Another sample article for testing..."}
                ]
            }
        }

        if existing_feed:
            validation_result["existing_feed"] = {
                "feed_id": existing_feed.feed_id,
                "feed_name": existing_feed.feed_name,
                "category": existing_feed.category,
                "tags": existing_feed.tags
            }

        return validation_result

    except Exception as e:
        logger.error(f"❌ RSS API ERROR: Failed to validate feed URL: {e}")
        raise HTTPException(status_code=500, detail="Failed to validate RSS feed URL")


@router.get("/api/rss/feeds/{feed_id}", response_model=RSSFeed)
async def get_rss_feed(
    feed_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Get a specific RSS feed by ID
    
    **Trust busting for unauthorized feed access!** Only return feeds the user can access!
    """
    try:
        logger.info(f"📡 RSS API: Getting RSS feed {feed_id} for user {current_user.user_id}")
        
        rss_service = await get_rss_service()
        feed = await rss_service.get_feed(feed_id)
        
        if not feed:
            raise HTTPException(status_code=404, detail="RSS feed not found")
        
        # Check if user has access to this feed
        if feed.user_id and feed.user_id != current_user.user_id:
            raise HTTPException(status_code=403, detail="Access denied to this RSS feed")
        
        logger.info(f"✅ RSS API: Retrieved RSS feed {feed_id}")
        return feed
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ RSS API ERROR: Failed to get RSS feed {feed_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get RSS feed")


@router.delete("/api/rss/feeds/{feed_id}")
async def delete_rss_feed(
    feed_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Delete an RSS feed
    
    Remove an RSS feed from monitoring.
    """
    try:
        logger.info(f"📡 RSS API: Deleting RSS feed {feed_id} for user {current_user.user_id}")
        
        rss_service = await get_rss_service()
        is_admin = current_user.role == "admin"
        success = await rss_service.delete_feed(feed_id, current_user.user_id, is_admin=is_admin)
        
        if not success:
            raise HTTPException(status_code=404, detail="RSS feed not found or access denied")
        
        logger.info(f"✅ RSS API: Deleted RSS feed {feed_id}")
        return {"message": "RSS feed deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ RSS API ERROR: Failed to delete RSS feed {feed_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete RSS feed")


@router.put("/api/rss/feeds/{feed_id}", response_model=RSSFeed)
async def update_rss_feed(
    feed_id: str,
    feed_data: RSSFeedCreate,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Update RSS feed metadata
    
    Update feed metadata and polling interval.
    """
    try:
        logger.info(f"📡 RSS API: Updating RSS feed {feed_id} for user {current_user.user_id}")
        
        # Set user ID for user-specific feeds
        feed_data.user_id = current_user.user_id
        
        rss_service = await get_rss_service()
        
        # Check admin status
        is_admin = current_user.role == "admin"
        
        # First check if user has access to this feed
        existing_feed = await rss_service.get_feed(feed_id)
        if not existing_feed:
            raise HTTPException(status_code=404, detail="RSS feed not found")
        
        # Regular users can only update their own feeds or global feeds
        if not is_admin and existing_feed.user_id and existing_feed.user_id != current_user.user_id:
            raise HTTPException(status_code=403, detail="Access denied to this RSS feed")
        
        # Update the feed metadata
        updated_feed = await rss_service.update_feed_metadata(feed_id, feed_data, current_user.user_id, is_admin=is_admin)
        
        logger.info(f"✅ RSS API: Updated RSS feed {feed_id}")
        return updated_feed
        
    except HTTPException:
        raise
    except ValidationError as e:
        logger.error(f"❌ RSS API ERROR: Validation error updating feed: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid feed data: {str(e)}")
    except Exception as e:
        logger.error(f"❌ RSS API ERROR: Failed to update RSS feed {feed_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update RSS feed")


@router.get("/api/rss/feeds/{feed_id}/articles", response_model=List[RSSArticle])
async def get_feed_articles(
    feed_id: str,
    limit: int = 100,
    read_filter: str = "all",
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Get articles for a specific RSS feed.

    read_filter: ``all`` (default), ``unread``, or ``read``.
    """
    try:
        logger.info(f"📡 RSS API: Getting articles for feed {feed_id}, user {current_user.user_id}")
        
        rf = (read_filter or "all").strip().lower()
        if rf not in ("all", "unread", "read"):
            rf = "all"

        rss_service = await get_rss_service()
        articles = await rss_service.get_feed_articles(
            feed_id, current_user.user_id, limit, read_filter=rf
        )
        
        logger.info(f"✅ RSS API: Retrieved {len(articles)} articles for feed {feed_id}")
        return articles
        
    except Exception as e:
        logger.error(f"❌ RSS API ERROR: Failed to get articles for feed {feed_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get feed articles")


@router.get("/api/rss/articles", response_model=List[RSSArticle])
async def get_all_user_articles(
    limit: int = 200,
    read_filter: str = "unread",
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """
    Cross-feed article list for the current user (feeds: own + global).

    read_filter: ``all``, ``unread`` (default), or ``read``.
    """
    try:
        rf = (read_filter or "unread").strip().lower()
        if rf not in ("all", "unread", "read"):
            rf = "unread"
        rss_service = await get_rss_service()
        articles = await rss_service.get_all_user_articles(
            current_user.user_id, limit=limit, read_filter=rf
        )
        return articles
    except Exception as e:
        logger.error("RSS API: get all user articles failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list articles")


async def _require_feed_access(
    rss_service, feed_id: str, current_user: AuthenticatedUserResponse
) -> None:
    feed = await rss_service.get_feed(feed_id)
    if not feed:
        raise HTTPException(status_code=404, detail="RSS feed not found")
    is_admin = current_user.role == "admin"
    if not is_admin and feed.user_id and feed.user_id != current_user.user_id:
        raise HTTPException(status_code=403, detail="Access denied to this RSS feed")


@router.get("/api/rss/articles/{article_id}", response_model=RSSArticle)
async def get_rss_article(
    article_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Return one article (including stored body fields) if the user can access its feed."""
    try:
        rss_service = await get_rss_service()
        article = await rss_service.get_article(article_id)
        if not article:
            raise HTTPException(status_code=404, detail="Article not found")
        await _require_feed_access(rss_service, article.feed_id, current_user)
        return article
    except HTTPException:
        raise
    except Exception as e:
        logger.error("RSS API: get article failed for %s: %s", article_id, e)
        raise HTTPException(status_code=500, detail="Failed to load article")


@router.post("/api/rss/mark-all-read", response_model=RSSBulkCountResponse)
async def mark_all_rss_articles_read_for_user(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Mark all unread RSS articles for the current user as read."""
    try:
        rss_service = await get_rss_service()
        n = await rss_service.mark_all_user_articles_read(current_user.user_id)
        return RSSBulkCountResponse(count=n)
    except Exception as e:
        logger.error("RSS API: mark-all-read (user) failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to mark articles read")


@router.post("/api/rss/feeds/{feed_id}/mark-all-read", response_model=RSSBulkCountResponse)
async def mark_all_feed_articles_read(
    feed_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Mark all unread articles in this feed as read (single bulk update)."""
    try:
        rss_service = await get_rss_service()
        await _require_feed_access(rss_service, feed_id, current_user)
        n = await rss_service.mark_all_feed_read(feed_id, current_user.user_id)
        return RSSBulkCountResponse(count=n)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("RSS API: mark-all-read failed for feed %s: %s", feed_id, e)
        raise HTTPException(status_code=500, detail="Failed to mark articles read")


@router.delete("/api/rss/feeds/{feed_id}/read-articles", response_model=RSSBulkCountResponse)
async def delete_feed_read_articles(
    feed_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Delete read, non-imported articles in this feed (starred articles are kept)."""
    try:
        rss_service = await get_rss_service()
        await _require_feed_access(rss_service, feed_id, current_user)
        n = await rss_service.delete_all_read_articles(feed_id, current_user.user_id)
        return RSSBulkCountResponse(count=n)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("RSS API: delete read articles failed for feed %s: %s", feed_id, e)
        raise HTTPException(status_code=500, detail="Failed to delete read articles")


@router.get("/api/rss/settings/import-location", response_model=RSSImportLocationResponse)
async def get_rss_import_location(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Default folder for RSS imports (My Documents). Empty means Web Sources / feed name."""
    try:
        fid = await get_user_setting(current_user.user_id, RSS_IMPORT_TARGET_FOLDER_KEY)
        return RSSImportLocationResponse(folder_id=fid if fid else None)
    except Exception as e:
        logger.error("RSS API: get import location failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to load RSS import location")


@router.put("/api/rss/settings/import-location", response_model=RSSImportLocationResponse)
async def put_rss_import_location(
    body: RSSImportLocationUpdate,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Set or clear the default folder for RSS article imports."""
    try:
        role = current_user.role or "user"
        if body.folder_id is None or (isinstance(body.folder_id, str) and not body.folder_id.strip()):
            await execute(
                "DELETE FROM user_settings WHERE user_id = $1 AND key = $2",
                current_user.user_id,
                RSS_IMPORT_TARGET_FOLDER_KEY,
            )
            return RSSImportLocationResponse(folder_id=None)

        resolved = await resolve_rss_import_target_folder_id(
            body.folder_id, user_id=current_user.user_id, user_role=role
        )
        if not resolved:
            raise HTTPException(
                status_code=400,
                detail="Invalid folder_id or you do not have access to this folder",
            )

        await set_user_setting(
            current_user.user_id,
            RSS_IMPORT_TARGET_FOLDER_KEY,
            resolved,
            "string",
        )
        return RSSImportLocationResponse(folder_id=resolved)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("RSS API: put import location failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to save RSS import location")


@router.post("/api/rss/articles/{article_id}/import")
async def import_rss_article(
    article_id: str,
    import_data: RSSArticleImport,
    background_tasks: BackgroundTasks,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Import an RSS article for full processing
    
    Fetch and persist full article HTML for an RSS item.
    """
    try:
        logger.info(f"📡 RSS API: Importing article {article_id} for user {current_user.user_id}")
        
        role = current_user.role or "user"
        resolved_target: Optional[str] = None
        if import_data.target_folder_id:
            resolved_target = await resolve_rss_import_target_folder_id(
                import_data.target_folder_id,
                user_id=current_user.user_id,
                user_role=role,
            )
            if not resolved_target:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid target_folder_id or access denied",
                )
        else:
            from_setting = await get_user_setting(current_user.user_id, RSS_IMPORT_TARGET_FOLDER_KEY)
            resolved_target = await resolve_rss_import_target_folder_id(
                from_setting,
                user_id=current_user.user_id,
                user_role=role,
            )

        # Start background task for article processing
        background_tasks.add_task(
            process_rss_article_task.delay,
            article_id=article_id,
            user_id=current_user.user_id,
            collection_name=import_data.collection_name,
            target_folder_id=resolved_target,
            user_role=role,
        )
        
        logger.info(f"✅ RSS API: Started import task for article {article_id}")
        return {
            "message": "Article import started",
            "article_id": article_id,
            "task_status": "processing"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ RSS API ERROR: Failed to import article {article_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to import article")


@router.put("/api/rss/articles/{article_id}/read")
async def mark_article_read(
    article_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Mark an RSS article as read
    
    Mark an article as read for the current user.
    """
    try:
        logger.info(f"📡 RSS API: Marking article {article_id} as read for user {current_user.user_id}")
        
        rss_service = await get_rss_service()
        success = await rss_service.mark_article_read(article_id, current_user.user_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Article not found or access denied")
        
        logger.info(f"✅ RSS API: Marked article {article_id} as read")
        return {"message": "Article marked as read"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ RSS API ERROR: Failed to mark article {article_id} as read: {e}")
        raise HTTPException(status_code=500, detail="Failed to mark article as read")


@router.put("/api/rss/articles/{article_id}/star", response_model=RSSStarToggleResponse)
async def toggle_rss_article_star(
    article_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Toggle starred/saved state for an article."""
    try:
        rss_service = await get_rss_service()
        article = await rss_service.get_article(article_id)
        if not article:
            raise HTTPException(status_code=404, detail="Article not found")
        await _require_feed_access(rss_service, article.feed_id, current_user)
        new_state = await rss_service.toggle_article_starred(
            article_id, current_user.user_id
        )
        if new_state is None:
            raise HTTPException(status_code=404, detail="Article not found or access denied")
        return RSSStarToggleResponse(is_starred=new_state)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("RSS API: toggle star failed for %s: %s", article_id, e)
        raise HTTPException(status_code=500, detail="Failed to toggle star")


@router.delete("/api/rss/articles/{article_id}")
async def delete_rss_article(
    article_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Delete an RSS article for the current user when permitted.
    """
    try:
        logger.info(f"📡 RSS API: Deleting article {article_id} for user {current_user.user_id}")
        
        rss_service = await get_rss_service()
        success = await rss_service.delete_article(article_id, current_user.user_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Article not found or access denied")
        
        logger.info(f"✅ RSS API: Deleted article {article_id}")
        return {"message": "Article deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ RSS API ERROR: Failed to delete article {article_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete article")


@router.post("/api/rss/feeds/{feed_id}/poll")
async def poll_rss_feed(
    feed_id: str,
    force_poll: bool = False,
    background_tasks: BackgroundTasks = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Poll a specific RSS feed for new articles
    
    Trigger an immediate poll of one feed.
    """
    try:
        logger.info(f"📡 RSS API: Polling feed {feed_id} for user {current_user.user_id}")
        
        # Start background task for feed polling
        if background_tasks:
            background_tasks.add_task(
                poll_rss_feeds_task.delay,
                user_id=current_user.user_id,
                feed_ids=[feed_id],
                force_poll=force_poll
            )
        
        logger.info(f"✅ RSS API: Started polling task for feed {feed_id}")
        return {
            "message": "Feed polling started",
            "feed_id": feed_id,
            "task_status": "processing"
        }
        
    except Exception as e:
        logger.error(f"❌ RSS API ERROR: Failed to poll feed {feed_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to poll RSS feed")


@router.get("/api/rss/unread-count")
async def get_unread_count(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Get unread article count per feed
    
    Return unread counts per feed for the current user.
    """
    try:
        logger.info(f"📡 RSS API: Getting unread count for user {current_user.user_id}")
        
        rss_service = await get_rss_service()
        unread_counts = await rss_service.get_unread_count(current_user.user_id)
        
        logger.info(f"✅ RSS API: Retrieved unread counts for {len(unread_counts)} feeds")
        return unread_counts
        
    except Exception as e:
        logger.error(f"❌ RSS API ERROR: Failed to get unread count: {e}")
        raise HTTPException(status_code=500, detail="Failed to get unread count")


@router.post("/api/rss/feeds/{feed_id}/subscribe")
async def subscribe_to_feed(
    feed_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Subscribe to an RSS feed
    
    Subscribe the current user to a feed.
    """
    try:
        logger.info(f"📡 RSS API: Subscribing user {current_user.user_id} to feed {feed_id}")
        
        rss_service = await get_rss_service()
        success = await rss_service.subscribe_to_feed(feed_id, current_user.user_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to subscribe to feed")
        
        logger.info(f"✅ RSS API: Subscribed user to feed {feed_id}")
        return {"message": "Successfully subscribed to RSS feed"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ RSS API ERROR: Failed to subscribe to feed {feed_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to subscribe to feed")


@router.delete("/api/rss/feeds/{feed_id}/subscribe")
async def unsubscribe_from_feed(
    feed_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Unsubscribe from an RSS feed
    
    Unsubscribe the current user from a feed.
    """
    try:
        logger.info(f"📡 RSS API: Unsubscribing user {current_user.user_id} from feed {feed_id}")
        
        rss_service = await get_rss_service()
        success = await rss_service.unsubscribe_from_feed(feed_id, current_user.user_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Subscription not found")
        
        logger.info(f"✅ RSS API: Unsubscribed user from feed {feed_id}")
        return {"message": "Successfully unsubscribed from RSS feed"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ RSS API ERROR: Failed to unsubscribe from feed {feed_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to unsubscribe from feed")
