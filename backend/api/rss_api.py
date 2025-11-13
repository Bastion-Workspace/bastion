"""
RSS API Endpoints
FastAPI endpoints for RSS feed management and article processing
"""

import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import ValidationError

from models.rss_models import (
    RSSFeed, RSSArticle, RSSFeedCreate, RSSArticleImport,
    RSSFeedPollResult, RSSArticleProcessResult
)
from models.api_models import AuthenticatedUserResponse
from services.service_container import get_service_container
from services.celery_tasks.rss_tasks import poll_rss_feeds_task, process_rss_article_task, extract_full_content_task
from utils.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rss", tags=["RSS"])


@router.post("/feeds", response_model=RSSFeed)
async def create_rss_feed(
    feed_data: RSSFeedCreate,
    update_if_exists: bool = False,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Create a new RSS feed
    
    **BULLY!** Add a new RSS feed to monitor for articles!
    """
    try:
        logger.info(f"📡 RSS API: Creating RSS feed for user {current_user.user_id}")
        
        # Set user ID for user-specific feeds
        feed_data.user_id = current_user.user_id
        
        service_container = await get_service_container()
        feed = await service_container.rss_service.create_feed(feed_data, update_if_exists=update_if_exists)
        
        # Check if this was an existing feed or a new one
        existing_feed = await service_container.rss_service.get_feed_by_url(feed_data.feed_url, current_user.user_id)
        
        if existing_feed and existing_feed.feed_id == feed.feed_id:
            if update_if_exists:
                logger.info(f"✅ RSS API: Updated existing RSS feed {feed.feed_id}")
            else:
                logger.info(f"✅ RSS API: Feed already exists, returning existing feed {feed.feed_id}")
            return feed
        else:
            logger.info(f"✅ RSS API: Created new RSS feed {feed.feed_id}")
            return feed
        
    except ValidationError as e:
        logger.error(f"❌ RSS API ERROR: Validation error creating feed: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid feed data: {str(e)}")
    except Exception as e:
        logger.error(f"❌ RSS API ERROR: Failed to create RSS feed: {e}")
        raise HTTPException(status_code=500, detail="Failed to create RSS feed")


@router.post("/feeds/global", response_model=RSSFeed)
async def create_global_rss_feed(
    feed_data: RSSFeedCreate,
    update_if_exists: bool = False,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Create a new global RSS feed (admin only)
    
    **BULLY!** Add a new global RSS feed that all users can access!
    """
    try:
        # Only admin users can create global feeds
        if current_user.role != "admin":
            raise HTTPException(status_code=403, detail="Only admin users can create global RSS feeds")
        
        logger.info(f"📡 RSS API: Creating global RSS feed for admin {current_user.user_id}")
        
        # Set user_id to None for global feeds
        feed_data.user_id = None
        
        service_container = await get_service_container()
        feed = await service_container.rss_service.create_feed(feed_data, update_if_exists=update_if_exists)
        
        logger.info(f"✅ RSS API: Created global RSS feed {feed.feed_id}")
        return feed
        
    except ValidationError as e:
        logger.error(f"❌ RSS API ERROR: Validation error creating global feed: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid feed data: {str(e)}")
    except Exception as e:
        logger.error(f"❌ RSS API ERROR: Failed to create global RSS feed: {e}")
        raise HTTPException(status_code=500, detail="Failed to create global RSS feed")


@router.get("/feeds", response_model=List[RSSFeed])
async def get_rss_feeds(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Get all RSS feeds for the current user
    
    **By George!** Retrieve all RSS feeds the user has access to!
    """
    try:
        logger.info(f"📡 RSS API: Getting RSS feeds for user {current_user.user_id}")
        
        service_container = await get_service_container()
        is_admin = current_user.role == "admin"
        feeds = await service_container.rss_service.get_user_feeds(current_user.user_id, is_admin=is_admin)
        
        logger.info(f"✅ RSS API: Retrieved {len(feeds)} RSS feeds for {'admin' if is_admin else 'user'} {current_user.user_id}")
        return feeds
        
    except Exception as e:
        logger.error(f"❌ RSS API ERROR: Failed to get RSS feeds: {e}")
        raise HTTPException(status_code=500, detail="Failed to get RSS feeds")


@router.get("/feeds/categorized")
async def get_categorized_rss_feeds(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Get RSS feeds categorized by user-specific vs global
    
    **BULLY!** This separates "My Docs" feeds from "Global" feeds!
    """
    try:
        logger.info(f"📡 RSS API: Getting categorized RSS feeds for user {current_user.user_id}")
        
        service_container = await get_service_container()
        is_admin = current_user.role == "admin"
        feeds = await service_container.rss_service.get_user_feeds(current_user.user_id, is_admin=is_admin)
        
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


@router.get("/feeds/{feed_id}", response_model=RSSFeed)
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
        
        service_container = await get_service_container()
        feed = await service_container.rss_service.get_feed(feed_id)
        
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


@router.get("/feeds/validate")
async def validate_feed_url(
    feed_url: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Validate an RSS feed URL and return feed information
    
    **BULLY!** Test if an RSS feed URL is valid and get its metadata!
    """
    try:
        logger.info(f"📡 RSS API: Validating RSS feed URL: {feed_url} for user {current_user.user_id}")
        
        service_container = await get_service_container()
        
        # Check if feed already exists for this user
        existing_feed = await service_container.rss_service.get_feed_by_url(feed_url, current_user.user_id)
        
        # This would validate the RSS feed URL and return feed metadata
        # For now, return a mock response for testing
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


@router.delete("/feeds/{feed_id}")
async def delete_rss_feed(
    feed_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Delete an RSS feed
    
    **BULLY!** Remove an RSS feed from monitoring!
    """
    try:
        logger.info(f"📡 RSS API: Deleting RSS feed {feed_id} for user {current_user.user_id}")
        
        service_container = await get_service_container()
        is_admin = current_user.role == "admin"
        success = await service_container.rss_service.delete_feed(feed_id, current_user.user_id, is_admin=is_admin)
        
        if not success:
            raise HTTPException(status_code=404, detail="RSS feed not found or access denied")
        
        logger.info(f"✅ RSS API: Deleted RSS feed {feed_id}")
        return {"message": "RSS feed deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ RSS API ERROR: Failed to delete RSS feed {feed_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete RSS feed")


@router.put("/feeds/{feed_id}", response_model=RSSFeed)
async def update_rss_feed(
    feed_id: str,
    feed_data: RSSFeedCreate,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Update RSS feed metadata
    
    **BULLY!** Update the feed's name, category, tags, and check interval!
    """
    try:
        logger.info(f"📡 RSS API: Updating RSS feed {feed_id} for user {current_user.user_id}")
        
        # Set user ID for user-specific feeds
        feed_data.user_id = current_user.user_id
        
        service_container = await get_service_container()
        
        # Check admin status
        is_admin = current_user.role == "admin"
        
        # First check if user has access to this feed
        existing_feed = await service_container.rss_service.get_feed(feed_id)
        if not existing_feed:
            raise HTTPException(status_code=404, detail="RSS feed not found")
        
        # Regular users can only update their own feeds or global feeds
        if not is_admin and existing_feed.user_id and existing_feed.user_id != current_user.user_id:
            raise HTTPException(status_code=403, detail="Access denied to this RSS feed")
        
        # Update the feed metadata
        updated_feed = await service_container.rss_service.update_feed_metadata(feed_id, feed_data, current_user.user_id, is_admin=is_admin)
        
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


@router.get("/feeds/{feed_id}/articles", response_model=List[RSSArticle])
async def get_feed_articles(
    feed_id: str,
    limit: int = 100,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Get articles for a specific RSS feed
    
    **By George!** Retrieve articles from the specified RSS feed!
    """
    try:
        logger.info(f"📡 RSS API: Getting articles for feed {feed_id}, user {current_user.user_id}")
        
        service_container = await get_service_container()
        articles = await service_container.rss_service.get_feed_articles(feed_id, current_user.user_id, limit)
        
        logger.info(f"✅ RSS API: Retrieved {len(articles)} articles for feed {feed_id}")
        return articles
        
    except Exception as e:
        logger.error(f"❌ RSS API ERROR: Failed to get articles for feed {feed_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get feed articles")


@router.post("/articles/{article_id}/import")
async def import_rss_article(
    article_id: str,
    import_data: RSSArticleImport,
    background_tasks: BackgroundTasks,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Import an RSS article for full processing
    
    **BULLY!** Download and process the full article content!
    """
    try:
        logger.info(f"📡 RSS API: Importing article {article_id} for user {current_user.user_id}")
        
        # Start background task for article processing
        background_tasks.add_task(
            process_rss_article_task.delay,
            article_id=article_id,
            user_id=current_user.user_id,
            collection_name=import_data.collection_name
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


@router.put("/articles/{article_id}/read")
async def mark_article_read(
    article_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Mark an RSS article as read
    
    **By George!** Mark this article as read by the user!
    """
    try:
        logger.info(f"📡 RSS API: Marking article {article_id} as read for user {current_user.user_id}")
        
        service_container = await get_service_container()
        success = await service_container.rss_service.mark_article_read(article_id, current_user.user_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Article not found or access denied")
        
        logger.info(f"✅ RSS API: Marked article {article_id} as read")
        return {"message": "Article marked as read"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ RSS API ERROR: Failed to mark article {article_id} as read: {e}")
        raise HTTPException(status_code=500, detail="Failed to mark article as read")


@router.delete("/articles/{article_id}")
async def delete_rss_article(
    article_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Delete an RSS article
    
    **Trust busting for unwanted articles!** Remove this article from the user's feed!
    """
    try:
        logger.info(f"📡 RSS API: Deleting article {article_id} for user {current_user.user_id}")
        
        service_container = await get_service_container()
        success = await service_container.rss_service.delete_article(article_id, current_user.user_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Article not found or access denied")
        
        logger.info(f"✅ RSS API: Deleted article {article_id}")
        return {"message": "Article deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ RSS API ERROR: Failed to delete article {article_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete article")


@router.post("/feeds/{feed_id}/poll")
async def poll_rss_feed(
    feed_id: str,
    force_poll: bool = False,
    background_tasks: BackgroundTasks = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Poll a specific RSS feed for new articles
    
    **BULLY!** Force a poll of the specified RSS feed!
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

@router.post("/articles/extract-full-content")
async def extract_full_content_for_existing_articles(
    background_tasks: BackgroundTasks,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Extract full content for existing articles with truncated descriptions
    
    **By George!** Use Crawl4AI to extract full content for articles that need it!
    """
    try:
        logger.info(f"📡 RSS API: Starting full content extraction for user {current_user.user_id}")
        
        # Get articles that need full content extraction
        service_container = await get_service_container()
        articles_needing_content = await service_container.rss_service.get_articles_needing_full_content(limit=20)
        
        if not articles_needing_content:
            return {
                "message": "No articles found that need full content extraction",
                "articles_processed": 0
            }
        
        # Start background task for content extraction
        background_tasks.add_task(
            extract_full_content_task.delay,
            user_id=current_user.user_id,
            article_ids=[article.article_id for article in articles_needing_content]
        )
        
        logger.info(f"✅ RSS API: Started full content extraction for {len(articles_needing_content)} articles")
        return {
            "message": "Full content extraction started",
            "articles_found": len(articles_needing_content),
            "task_status": "processing"
        }
        
    except Exception as e:
        logger.error(f"❌ RSS API ERROR: Failed to start full content extraction: {e}")
        raise HTTPException(status_code=500, detail="Failed to start full content extraction")
    """
    Manually poll an RSS feed for new articles
    
    **BULLY!** Force a poll of the RSS feed to check for new articles!
    """
    try:
        logger.info(f"📡 RSS API: Manual poll of feed {feed_id} for user {current_user.user_id}")
        
        # Start background task for feed polling
        if background_tasks:
            background_tasks.add_task(
                poll_rss_feeds_task.delay,
                user_id=current_user.user_id,
                feed_ids=[feed_id],
                force_poll=force_poll
            )
        
        logger.info(f"✅ RSS API: Started poll task for feed {feed_id}")
        return {
            "message": "RSS feed polling started",
            "feed_id": feed_id,
            "force_poll": force_poll,
            "task_status": "processing"
        }
        
    except Exception as e:
        logger.error(f"❌ RSS API ERROR: Failed to poll feed {feed_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to poll RSS feed")


@router.get("/unread-count")
async def get_unread_count(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Get unread article count per feed
    
    **By George!** Get the count of unread articles for each RSS feed!
    """
    try:
        logger.info(f"📡 RSS API: Getting unread count for user {current_user.user_id}")
        
        service_container = await get_service_container()
        unread_counts = await service_container.rss_service.get_unread_count(current_user.user_id)
        
        logger.info(f"✅ RSS API: Retrieved unread counts for {len(unread_counts)} feeds")
        return unread_counts
        
    except Exception as e:
        logger.error(f"❌ RSS API ERROR: Failed to get unread count: {e}")
        raise HTTPException(status_code=500, detail="Failed to get unread count")


@router.post("/feeds/{feed_id}/subscribe")
async def subscribe_to_feed(
    feed_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Subscribe to an RSS feed
    
    **BULLY!** Subscribe the user to receive articles from this feed!
    """
    try:
        logger.info(f"📡 RSS API: Subscribing user {current_user.user_id} to feed {feed_id}")
        
        service_container = await get_service_container()
        success = await service_container.rss_service.subscribe_to_feed(feed_id, current_user.user_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to subscribe to feed")
        
        logger.info(f"✅ RSS API: Subscribed user to feed {feed_id}")
        return {"message": "Successfully subscribed to RSS feed"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ RSS API ERROR: Failed to subscribe to feed {feed_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to subscribe to feed")


@router.delete("/feeds/{feed_id}/subscribe")
async def unsubscribe_from_feed(
    feed_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Unsubscribe from an RSS feed
    
    **By George!** Unsubscribe the user from this RSS feed!
    """
    try:
        logger.info(f"📡 RSS API: Unsubscribing user {current_user.user_id} from feed {feed_id}")
        
        service_container = await get_service_container()
        success = await service_container.rss_service.unsubscribe_from_feed(feed_id, current_user.user_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Subscription not found")
        
        logger.info(f"✅ RSS API: Unsubscribed user from feed {feed_id}")
        return {"message": "Successfully unsubscribed from RSS feed"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ RSS API ERROR: Failed to unsubscribe from feed {feed_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to unsubscribe from feed")
