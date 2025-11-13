"""
RSS Service
Database operations and business logic for RSS feed management

**BULLY!** Now using the centralized DatabaseManager for all database operations!
"""

import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import os
import hashlib

from models.rss_models import (
    RSSFeed, RSSArticle, RSSFeedSubscription, RSSFeedCreate,
    RSSFeedPollResult, RSSArticleProcessResult, RSSFeedHealth
)
from services.database_manager import get_database_manager
from services.database_manager.database_helpers import (
    fetch_all, fetch_one, fetch_value, execute, insert_and_return_id,
    update_by_id, delete_by_id, count_records, record_exists
)

logger = logging.getLogger(__name__)


class RSSService:
    """
    RSS Service for feed and article management
    
    **BULLY!** This service handles all RSS database operations and business logic!
    """
    
    def __init__(self):
        # **BULLY!** No more connection pool chaos - using centralized DatabaseManager!
        self._database_manager = None
        logger.info("🔧 RSS Service initialized with centralized DatabaseManager")
    
    # **BULLY!** No more connection pool methods - DatabaseManager handles everything!
    
    async def initialize(self, shared_db_pool=None):
        """Initialize the RSS service with centralized DatabaseManager"""
        try:
            logger.info("🔧 Initializing RSS service with DatabaseManager...")
            
            # Get the centralized database manager
            self._database_manager = await get_database_manager()
            logger.info("✅ RSS service using centralized DatabaseManager")
            
        except Exception as e:
            logger.error(f"❌ RSS SERVICE ERROR: Failed to initialize: {e}")
            raise e
    
    async def close(self):
        """Close RSS service - DatabaseManager handles connection cleanup"""
        logger.info("🔄 RSS service closed - DatabaseManager handles cleanup")
    
    # **BULLY!** Celery pool method removed - DatabaseManager handles all environments!
    
    # RSS Feed Operations
    async def get_feed_by_url(self, feed_url: str, user_id: str) -> Optional[RSSFeed]:
        """Get RSS feed by URL and user ID"""
        try:
            # **BULLY!** Using DatabaseManager - no more pool management!
            # Handle global feeds (user_id = None) and user-specific feeds
            if user_id is None:
                # For global feeds, look for feeds with user_id = NULL
                row_data = await fetch_one("""
                    SELECT * FROM rss_feeds WHERE feed_url = $1 AND user_id IS NULL
                """, feed_url)
            else:
                # For user-specific feeds, look for feeds with the specific user_id
                row_data = await fetch_one("""
                    SELECT * FROM rss_feeds WHERE feed_url = $1 AND user_id = $2
                """, feed_url, user_id)
            
            if row_data:
                # Convert JSON tags back to list for Pydantic model
                if 'tags' in row_data and isinstance(row_data['tags'], str):
                    try:
                        row_data['tags'] = json.loads(row_data['tags'])
                    except json.JSONDecodeError:
                        row_data['tags'] = []
                return RSSFeed(**row_data)
            
            return None
                
        except Exception as e:
            logger.error(f"❌ RSS SERVICE ERROR: Failed to get feed by URL: {e}")
            raise e

    async def create_feed(self, feed_data: RSSFeedCreate, update_if_exists: bool = False) -> RSSFeed:
        """Create a new RSS feed"""
        try:
            # Check if feed already exists for this user (or global)
            existing_feed = await self.get_feed_by_url(feed_data.feed_url, feed_data.user_id)
            
            if existing_feed:
                if update_if_exists:
                    logger.info(f"🔄 RSS SERVICE: Feed already exists for user {feed_data.user_id}, updating metadata")
                    return await self.update_feed_metadata(existing_feed.feed_id, feed_data, feed_data.user_id, is_admin=False)
                else:
                    logger.info(f"🔄 RSS SERVICE: Feed already exists for user {feed_data.user_id}, returning existing feed")
                    return existing_feed
            
            # Generate feed ID from URL hash (handle global feeds properly)
            if feed_data.user_id is None:
                # For global feeds, use just the URL for ID generation
                feed_id = hashlib.sha256(f"{feed_data.feed_url}_global".encode()).hexdigest()[:32]
            else:
                # For user-specific feeds, include user_id in ID generation
                feed_id = hashlib.sha256(f"{feed_data.feed_url}_{feed_data.user_id}".encode()).hexdigest()[:32]
            
            # Prepare tags as JSON string
            tags_json = json.dumps(feed_data.tags) if feed_data.tags else '[]'
            
            # **BULLY!** Using DatabaseManager for feed creation!
            row_data = await fetch_one("""
                INSERT INTO rss_feeds (feed_id, feed_url, feed_name, category, tags, check_interval, user_id, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, NOW(), NOW())
                RETURNING *
            """, feed_id, feed_data.feed_url, feed_data.feed_name, 
                 feed_data.category, tags_json, feed_data.check_interval, feed_data.user_id)
            
            if row_data:
                # Convert JSON tags back to list for Pydantic model
                if 'tags' in row_data and isinstance(row_data['tags'], str):
                    try:
                        row_data['tags'] = json.loads(row_data['tags'])
                    except json.JSONDecodeError:
                        row_data['tags'] = []
                
                new_feed = RSSFeed(**row_data)
                
                # Trigger immediate RSS feed refresh for the new feed
                await self._trigger_immediate_feed_refresh(new_feed.feed_id, feed_data.user_id)
                
                logger.info(f"✅ RSS SERVICE: Created new feed {feed_id} for user {feed_data.user_id}")
                return new_feed
            else:
                raise Exception("Failed to create RSS feed")
                    
        except Exception as e:
            logger.error(f"❌ RSS SERVICE ERROR: Failed to create feed: {e}")
            raise

    async def update_feed_metadata(self, feed_id: str, feed_data: RSSFeedCreate, user_id: str = None, is_admin: bool = False) -> RSSFeed:
        """Update RSS feed metadata (name, category, tags, check_interval)"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                from .database_manager.database_helpers import fetch_one
                
                # Convert tags list to JSON string for database storage
                tags_json = json.dumps(feed_data.tags) if feed_data.tags else '[]'
                
                if is_admin:
                    # Admin users can update any feed
                    row_data = await fetch_one("""
                        UPDATE rss_feeds 
                        SET feed_name = $2, category = $3, tags = $4, check_interval = $5, updated_at = NOW()
                        WHERE feed_id = $1
                        RETURNING *
                    """, feed_id, feed_data.feed_name, feed_data.category, 
                         tags_json, feed_data.check_interval)
                else:
                    # Regular users can only update their own feeds or global feeds
                    row_data = await fetch_one("""
                        UPDATE rss_feeds 
                        SET feed_name = $2, category = $3, tags = $4, check_interval = $5, updated_at = NOW()
                        WHERE feed_id = $1 AND (user_id = $6 OR user_id IS NULL)
                        RETURNING *
                    """, feed_id, feed_data.feed_name, feed_data.category, 
                         tags_json, feed_data.check_interval, user_id)
                
                if not row_data:
                    raise ValueError(f"Feed {feed_id} not found or access denied")
                
                # Convert JSON tags back to list for Pydantic model
                if 'tags' in row_data and isinstance(row_data['tags'], str):
                    try:
                        row_data['tags'] = json.loads(row_data['tags'])
                    except json.JSONDecodeError:
                        row_data['tags'] = []
                
                logger.info(f"✅ RSS SERVICE: Updated RSS feed {feed_id} metadata")
                return RSSFeed(**row_data)
                
            except Exception as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ["another operation is in progress", "connection was closed", "connection does not exist"]) and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 1  # Exponential backoff
                    logger.warning(f"⚠️ RSS SERVICE: Database connection issue updating feed metadata {feed_id} (attempt {attempt + 1}/{max_retries}), retrying in {wait_time} seconds: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"❌ RSS SERVICE ERROR: Failed to update feed metadata {feed_id} after {max_retries} attempts: {e}")
                raise e
        
        raise Exception(f"Failed to update feed metadata {feed_id} after {max_retries} attempts")
    
    async def get_feed(self, feed_id: str) -> Optional[RSSFeed]:
        """Get RSS feed by ID using centralized DatabaseManager"""
        try:
            # **BULLY!** Simple, reliable database query - DatabaseManager handles retries!
            row_data = await fetch_one("SELECT * FROM rss_feeds WHERE feed_id = $1", feed_id)
            
            if row_data:
                # Convert JSON tags back to list for Pydantic model
                if 'tags' in row_data and isinstance(row_data['tags'], str):
                    try:
                        row_data['tags'] = json.loads(row_data['tags'])
                    except json.JSONDecodeError:
                        row_data['tags'] = []
                return RSSFeed(**row_data)
            
            return None
            
        except Exception as e:
            logger.error(f"❌ RSS SERVICE ERROR: Failed to get feed {feed_id}: {e}")
            return None
    
    async def get_user_feeds(self, user_id: str, is_admin: bool = False) -> List[RSSFeed]:
        """Get all RSS feeds for a user"""
        try:
            # **BULLY!** Using DatabaseManager for user feeds!
            if is_admin:
                # Admin users can see their own feeds + global feeds (user_id IS NULL)
                # But NOT other users' feeds to maintain proper separation
                rows = await fetch_all("""
                    SELECT * FROM rss_feeds 
                    WHERE user_id = $1 OR user_id IS NULL
                    ORDER BY created_at DESC
                """, user_id)
            else:
                # Regular users can only see their own feeds + global feeds
                rows = await fetch_all("""
                    SELECT * FROM rss_feeds 
                    WHERE user_id = $1 OR user_id IS NULL
                    ORDER BY created_at DESC
                """, user_id)
            
            feeds = []
            for row_dict in rows:
                # Convert JSON tags back to list for Pydantic model
                if 'tags' in row_dict and isinstance(row_dict['tags'], str):
                    try:
                        row_dict['tags'] = json.loads(row_dict['tags'])
                    except json.JSONDecodeError:
                        row_dict['tags'] = []
                feeds.append(RSSFeed(**row_dict))
            
            return feeds
                
        except Exception as e:
            logger.error(f"❌ RSS SERVICE ERROR: Failed to get user feeds: {e}")
            return []
    
    async def get_all_feeds_admin(self) -> List[RSSFeed]:
        """Get ALL RSS feeds for admin management purposes"""
        try:
            rows = await fetch_all("""
                SELECT * FROM rss_feeds 
                ORDER BY created_at DESC
            """)
            
            feeds = []
            for row in rows:
                row_dict = dict(row)
                # Convert JSON tags back to list for Pydantic model
                if 'tags' in row_dict and isinstance(row_dict['tags'], str):
                    try:
                        row_dict['tags'] = json.loads(row_dict['tags'])
                    except json.JSONDecodeError:
                        row_dict['tags'] = []
                feeds.append(RSSFeed(**row_dict))
            
            return feeds
                
        except Exception as e:
            logger.error(f"❌ RSS SERVICE ERROR: Failed to get all feeds for admin: {e}")
            return []
    
    async def get_feeds_needing_poll(self, user_id: Optional[str] = None) -> List[RSSFeed]:
        """Get RSS feeds that need polling based on their check intervals with concurrency control"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                from .database_manager.database_helpers import fetch_all
                
                # Build query based on user_id parameter with better concurrency control
                if user_id:
                    query = """
                        SELECT * FROM rss_feeds 
                        WHERE user_id = $1 
                        AND (last_check IS NULL OR 
                             last_check + (check_interval || ' seconds')::interval < NOW())
                            AND (is_polling IS NULL OR is_polling = false)
                        ORDER BY last_check ASC NULLS FIRST
                            LIMIT 10
                    """
                    params = [user_id]
                else:
                    # Get all feeds that need polling (for global polling) with concurrency control
                    query = """
                        SELECT * FROM rss_feeds 
                        WHERE (last_check IS NULL OR 
                               last_check + (check_interval || ' seconds')::interval < NOW())
                            AND (is_polling IS NULL OR is_polling = false)
                        ORDER BY last_check ASC NULLS FIRST
                            LIMIT 10
                    """
                    params = []
                
                rows = await fetch_all(query, *params)

                feeds = []
                for row_dict in rows:
                    # Convert JSON tags back to list for Pydantic model
                    if 'tags' in row_dict and isinstance(row_dict['tags'], str):
                        try:
                            row_dict['tags'] = json.loads(row_dict['tags'])
                        except json.JSONDecodeError:
                            row_dict['tags'] = []
                    feeds.append(RSSFeed(**row_dict))

                logger.info(f"📡 RSS SERVICE: Found {len(feeds)} feeds needing poll")
                return feeds
                    
            except Exception as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ["another operation is in progress", "connection was closed", "connection does not exist"]) and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 1  # Exponential backoff
                    logger.warning(f"⚠️ RSS SERVICE: Database connection issue getting feeds needing poll (attempt {attempt + 1}/{max_retries}), retrying in {wait_time} seconds: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"❌ RSS SERVICE ERROR: Failed to get feeds needing poll after {max_retries} attempts: {e}")
                    return []
        
            return []
    
    async def update_feed_last_check(self, feed_id: str) -> bool:
        """Update feed last_check timestamp with optimistic locking"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Use optimistic locking to prevent race conditions
                result = await execute("""
                    UPDATE rss_feeds 
                        SET last_check = NOW(), updated_at = NOW(), is_polling = false
                    WHERE feed_id = $1
                """, feed_id)
                
                if result == "UPDATE 1":
                    logger.info(f"✅ RSS SERVICE: Updated last_check for feed {feed_id}")
                    return True
                else:
                    logger.warning(f"⚠️ RSS SERVICE: No rows updated for feed {feed_id}")
                    return False
                
            except Exception as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ["another operation is in progress", "connection was closed", "connection does not exist"]) and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 1  # Exponential backoff
                    logger.warning(f"⚠️ RSS SERVICE: Database connection issue updating feed last_check {feed_id} (attempt {attempt + 1}/{max_retries}), retrying in {wait_time} seconds: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"❌ RSS SERVICE ERROR: Failed to update feed last_check {feed_id} after {max_retries} attempts: {e}")
                    return False
        
        return False

    async def mark_feed_polling(self, feed_id: str, is_polling: bool = True) -> bool:
        """Mark feed as polling to prevent concurrent polling"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                from .database_manager.database_helpers import execute
                
                result = await execute("""
                    UPDATE rss_feeds 
                    SET is_polling = $2, updated_at = NOW()
                    WHERE feed_id = $1
                """, feed_id, is_polling)
                
                if result == "UPDATE 1":
                    status = "polling" if is_polling else "not polling"
                    logger.info(f"✅ RSS SERVICE: Marked feed {feed_id} as {status}")
                    return True
                else:
                    logger.warning(f"⚠️ RSS SERVICE: No rows updated for feed {feed_id}")
                    return False
                    
            except Exception as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ["another operation is in progress", "connection was closed", "connection does not exist"]) and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 1  # Exponential backoff
                    logger.warning(f"⚠️ RSS SERVICE: Database connection issue marking feed {feed_id} as polling (attempt {attempt + 1}/{max_retries}), retrying in {wait_time} seconds: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"❌ RSS SERVICE ERROR: Failed to mark feed {feed_id} as polling after {max_retries} attempts: {e}")
                    return False
        
        return False
    
    async def delete_feed(self, feed_id: str, user_id: str, is_admin: bool = False) -> bool:
        """Delete RSS feed (only if user owns it or is admin)"""
        try:
            from .database_manager.database_helpers import execute, fetch_one
            
            # Permission check
            if is_admin:
                can_delete = await fetch_one("SELECT 1 FROM rss_feeds WHERE feed_id = $1", feed_id) is not None
            else:
                can_delete = await fetch_one(
                    "SELECT 1 FROM rss_feeds WHERE feed_id = $1 AND (user_id = $2 OR user_id IS NULL)",
                    feed_id, user_id
                ) is not None
            if not can_delete:
                return False

            # Remove associated News articles synthesized from this feed's RSS articles
            await execute(
                """
                DELETE FROM news_articles 
                WHERE id IN (
                    SELECT article_id FROM rss_articles WHERE feed_id = $1
                )
                """,
                feed_id,
            )

            # Delete the feed (will cascade delete rss_articles via FK)
            if is_admin:
                result = await execute("DELETE FROM rss_feeds WHERE feed_id = $1", feed_id)
            else:
                result = await execute(
                    "DELETE FROM rss_feeds WHERE feed_id = $1 AND (user_id = $2 OR user_id IS NULL)",
                    feed_id, user_id
                )
            
            return result == "DELETE 1"
            
        except Exception as e:
            logger.error(f"❌ RSS SERVICE ERROR: Failed to delete feed: {e}")
            return False
    
    # RSS Article Operations
    async def save_article(self, article: RSSArticle) -> bool:
        """Save RSS article to database"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                from .database_manager.database_helpers import execute
                
                await execute("""
                    INSERT INTO rss_articles 
                    (article_id, feed_id, title, description, full_content, full_content_html, images, link, published_date, 
                     is_processed, is_read, content_hash, user_id)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    ON CONFLICT (article_id) DO NOTHING
                """, article.article_id, article.feed_id, article.title, 
                     article.description, article.full_content, article.full_content_html, json.dumps(article.images) if article.images else None, 
                     article.link, article.published_date,
                     article.is_processed, article.is_read, article.content_hash, article.user_id)
                
                logger.info(f"✅ RSS SERVICE: Successfully saved article '{article.title}' with ID {article.article_id}")
                return True
                
            except Exception as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ["another operation is in progress", "connection was closed", "connection does not exist"]) and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 1  # Exponential backoff
                    logger.warning(f"⚠️ RSS SERVICE: Database connection issue saving article {article.article_id} (attempt {attempt + 1}/{max_retries}), retrying in {wait_time} seconds: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"❌ RSS SERVICE ERROR: Failed to save article {article.article_id} after {max_retries} attempts: {e}")
                    return False
        
        return False
    
    async def get_article(self, article_id: str) -> Optional[RSSArticle]:
        """Get RSS article by ID using centralized DatabaseManager"""
        try:
            # **BULLY!** Simple, reliable database query - DatabaseManager handles retries!
            row_data = await fetch_one("SELECT * FROM rss_articles WHERE article_id = $1", article_id)
            
            if row_data:
                # Convert JSON images back to list for Pydantic model
                if 'images' in row_data and isinstance(row_data['images'], str):
                    try:
                        row_data['images'] = json.loads(row_data['images'])
                    except json.JSONDecodeError:
                        row_data['images'] = []
                return RSSArticle(**row_data)
            
            return None
            
        except Exception as e:
            logger.error(f"❌ RSS SERVICE ERROR: Failed to get article {article_id}: {e}")
            return None
    
    async def get_feed_articles(self, feed_id: str, user_id: str, limit: int = 100) -> List[RSSArticle]:
        """Get articles for a specific feed"""
        try:
            # **BULLY!** Using DatabaseManager for feed articles!
            rows = await fetch_all("""
                SELECT * FROM rss_articles 
                WHERE feed_id = $1 AND (user_id = $2 OR user_id IS NULL)
                ORDER BY published_date DESC NULLS LAST, created_at DESC
                LIMIT $3
            """, feed_id, user_id, limit)
            
            articles = []
            for row_dict in rows:
                # Convert JSON images back to list for Pydantic model
                if 'images' in row_dict and isinstance(row_dict['images'], str):
                    try:
                        row_dict['images'] = json.loads(row_dict['images'])
                    except json.JSONDecodeError:
                        row_dict['images'] = []
                articles.append(RSSArticle(**row_dict))
            
            return articles
                
        except Exception as e:
            logger.error(f"❌ RSS SERVICE ERROR: Failed to get feed articles: {e}")
            return []
    
    async def is_duplicate_article(self, article: RSSArticle) -> bool:
        """Check if article is a duplicate based on content hash"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Check for duplicate by content hash and also by URL to be more robust
                row = await fetch_one("""
                    SELECT 1 FROM rss_articles 
                    WHERE (content_hash = $1 AND feed_id = $2) OR (link = $3 AND feed_id = $2)
                """, article.content_hash, article.feed_id, article.link)
                
                is_duplicate = row is not None
                logger.info(f"🔍 RSS SERVICE: Duplicate check for '{article.title}' - Hash: {article.content_hash[:16]}... - Is duplicate: {is_duplicate}")
                
                return is_duplicate
                    
            except Exception as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ["another operation is in progress", "connection was closed", "connection does not exist"]) and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 1  # Exponential backoff
                    logger.warning(f"⚠️ RSS SERVICE: Database connection issue checking duplicate for {article.article_id} (attempt {attempt + 1}/{max_retries}), retrying in {wait_time} seconds: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"❌ RSS SERVICE ERROR: Failed to check duplicate for {article.article_id} after {max_retries} attempts: {e}")
                    return False
        
            return False
    
    async def mark_article_read(self, article_id: str, user_id: str) -> bool:
        """Mark article as read"""
        try:
            await execute("""
                UPDATE rss_articles 
                SET is_read = true, updated_at = NOW()
                WHERE article_id = $1 AND (user_id = $2 OR user_id IS NULL)
            """, article_id, user_id)
            
            return True
                
        except Exception as e:
            logger.error(f"❌ RSS SERVICE ERROR: Failed to mark article read: {e}")
            return False

    async def mark_article_processed(self, article_id: str) -> bool:
        """Mark article as processed (imported)"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                from .database_manager.database_helpers import execute
                
                result = await execute("""
                    UPDATE rss_articles 
                    SET is_processed = true, updated_at = NOW()
                    WHERE article_id = $1
                """, article_id)
                
                if result == "UPDATE 1":
                    logger.info(f"✅ RSS SERVICE: Marked article {article_id} as processed")
                    return True
                else:
                    logger.warning(f"⚠️ RSS SERVICE: No rows updated for article {article_id}")
                    return False
                    
            except Exception as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ["another operation is in progress", "connection was closed", "connection does not exist"]) and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 1  # Exponential backoff
                    logger.warning(f"⚠️ RSS SERVICE: Database connection issue marking article {article_id} as processed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time} seconds: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"❌ RSS SERVICE ERROR: Failed to mark article {article_id} as processed after {max_retries} attempts: {e}")
                    return False
        
                return False
    
    async def update_article_full_content(self, article_id: str, full_content: str, images: Optional[List[Dict[str, Any]]] = None, full_content_html: Optional[str] = None) -> bool:
        """Update article with full content (text + HTML) and images extracted by Crawl4AI"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if images is not None and full_content_html is not None:
                    await execute("""
                        UPDATE rss_articles 
                        SET full_content = $2, full_content_html = $3, images = $4, updated_at = NOW()
                        WHERE article_id = $1
                    """, article_id, full_content, full_content_html, json.dumps(images))
                elif images is not None:
                    await execute("""
                        UPDATE rss_articles 
                        SET full_content = $2, images = $3, updated_at = NOW()
                        WHERE article_id = $1
                    """, article_id, full_content, json.dumps(images))
                elif full_content_html is not None:
                    await execute("""
                        UPDATE rss_articles 
                        SET full_content = $2, full_content_html = $3, updated_at = NOW()
                        WHERE article_id = $1
                    """, article_id, full_content, full_content_html)
                else:
                    await execute("""
                        UPDATE rss_articles 
                        SET full_content = $2, updated_at = NOW()
                        WHERE article_id = $1
                    """, article_id, full_content)
                
                logger.info(f"✅ RSS SERVICE: Updated full content for article {article_id}")
                return True
                    
            except Exception as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ["another operation is in progress", "connection was closed", "connection does not exist"]) and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 1  # Exponential backoff
                    logger.warning(f"⚠️ RSS SERVICE: Database connection issue updating full content for article {article_id} (attempt {attempt + 1}/{max_retries}), retrying in {wait_time} seconds: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"❌ RSS SERVICE ERROR: Failed to update full content for article {article_id} after {max_retries} attempts: {e}")
                    return False
        
            return False
    
    async def get_articles_needing_full_content(self, limit: int = 50) -> List[RSSArticle]:
        """Get articles that have truncated descriptions but no full content"""
        try:
            rows = await fetch_all("""
                SELECT * FROM rss_articles 
                WHERE full_content IS NULL 
                AND description IS NOT NULL
                AND (
                    description LIKE '%read more%' OR
                    description LIKE '%continue reading%' OR
                    description LIKE '%...%' OR
                    LENGTH(description) < 200
                )
                ORDER BY created_at DESC
                LIMIT $1
            """, limit)
            
            articles = []
            for row in rows:
                row_dict = dict(row)
                # Convert JSON images back to list for Pydantic model
                if 'images' in row_dict and isinstance(row_dict['images'], str):
                    try:
                        row_dict['images'] = json.loads(row_dict['images'])
                    except json.JSONDecodeError:
                        row_dict['images'] = []
                articles.append(RSSArticle(**row_dict))
            
            return articles
                
        except Exception as e:
            logger.error(f"❌ RSS SERVICE ERROR: Failed to get articles needing full content: {e}")
            return []
    
    async def delete_article(self, article_id: str, user_id: str) -> bool:
        """Delete RSS article"""
        try:
            # Check ownership/access
            row = await fetch_one("SELECT user_id FROM rss_articles WHERE article_id = $1", article_id)
            if not row:
                return False
            owner_id = row.get("user_id") if isinstance(row, dict) else row[0]
            if owner_id is not None and owner_id != user_id:
                return False

            # Delete associated News article first
            await execute("DELETE FROM news_articles WHERE id = $1", article_id)

            # Delete RSS article
            result = await execute(
                "DELETE FROM rss_articles WHERE article_id = $1 AND (user_id = $2 OR user_id IS NULL)",
                article_id, user_id
            )
            return result == "DELETE 1"
                
        except Exception as e:
            logger.error(f"❌ RSS SERVICE ERROR: Failed to delete article: {e}")
            return False
    
    async def get_unread_count(self, user_id: str) -> Dict[str, int]:
        """Get unread article count per feed"""
        try:
            # **BULLY!** Using DatabaseManager for unread count!
            rows = await fetch_all("""
                SELECT feed_id, COUNT(*) as unread_count
                FROM rss_articles 
                WHERE is_read = false AND (user_id = $1 OR user_id IS NULL)
                GROUP BY feed_id
            """, user_id)
            
            return {row['feed_id']: row['unread_count'] for row in rows}
                
        except Exception as e:
            logger.error(f"❌ RSS SERVICE ERROR: Failed to get unread count: {e}")
            return {}
    
    # RSS Feed Subscription Operations
    async def subscribe_to_feed(self, feed_id: str, user_id: str) -> bool:
        """Subscribe user to RSS feed"""
        try:
            await execute("""
                INSERT INTO rss_feed_subscriptions (subscription_id, feed_id, user_id)
                VALUES ($1, $2, $3)
                ON CONFLICT (feed_id, user_id) DO NOTHING
            """, f"{feed_id}_{user_id}", feed_id, user_id)
            
            return True
                
        except Exception as e:
            logger.error(f"❌ RSS SERVICE ERROR: Failed to subscribe to feed: {e}")
            return False
    
    async def unsubscribe_from_feed(self, feed_id: str, user_id: str) -> bool:
        """Unsubscribe user from RSS feed"""
        try:
            result = await execute("""
                DELETE FROM rss_feed_subscriptions 
                WHERE feed_id = $1 AND user_id = $2
            """, feed_id, user_id)
            
            return result == "DELETE 1"
                
        except Exception as e:
            logger.error(f"❌ RSS SERVICE ERROR: Failed to unsubscribe from feed: {e}")
            return False
    
    # RSS Feed Health Operations
    async def get_feed_health(self, feed_id: str) -> Optional[RSSFeedHealth]:
        """Get RSS feed health metrics"""
        try:
            pool = await self._get_pool()
            
            async with pool.acquire() as conn:
                # Get basic feed info
                feed_row = await conn.fetchrow("""
                    SELECT * FROM rss_feeds WHERE feed_id = $1
                """, feed_id)
                
                if not feed_row:
                    return None
                
                # Get article statistics
                stats_row = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_articles,
                        COUNT(CASE WHEN published_date >= NOW() - INTERVAL '7 days' THEN 1 END) as recent_articles,
                        MAX(published_date) as last_article_date
                    FROM rss_articles 
                    WHERE feed_id = $1
                """, feed_id)
                
                # Calculate health metrics
                total_articles = stats_row['total_articles'] if stats_row else 0
                recent_articles = stats_row['recent_articles'] if stats_row else 0
                last_article_date = stats_row['last_article_date'] if stats_row else None
                
                # Determine if feed is healthy
                is_healthy = (
                    feed_row['is_active'] and 
                    recent_articles > 0 and
                    (feed_row['last_check'] is None or 
                     feed_row['last_check'] >= datetime.utcnow() - timedelta(days=1))
                )
                
                return RSSFeedHealth(
                    feed_id=feed_id,
                    is_healthy=is_healthy,
                    last_successful_poll=feed_row['last_check'],
                    consecutive_failures=0,  # TODO: Implement failure tracking
                    average_response_time=None,  # TODO: Implement response time tracking
                    articles_per_day=recent_articles / 7 if recent_articles > 0 else 0,
                    last_article_date=last_article_date
                )
                
        except Exception as e:
            logger.error(f"❌ RSS SERVICE ERROR: Failed to get feed health: {e}")
            return None

    async def _trigger_immediate_feed_refresh(self, feed_id: str, user_id: str):
        """
        Triggers an immediate poll task for a specific feed.
        This is typically done after a feed is created or updated.
        """
        try:
            logger.info(f"🔄 RSS SERVICE: Triggering immediate poll for feed {feed_id} for user {user_id}")
            
            # Import Celery task for RSS polling
            from services.celery_tasks.rss_tasks import poll_rss_feeds_task
            
            # Trigger immediate poll for the specific feed with force_poll=True
            task = poll_rss_feeds_task.delay(
                user_id=user_id,
                feed_ids=[feed_id],
                force_poll=True
            )
            
            logger.info(f"✅ RSS SERVICE: Immediate poll task queued with ID {task.id} for feed {feed_id}")
            
        except Exception as e:
            logger.error(f"❌ RSS SERVICE ERROR: Failed to trigger immediate poll for feed {feed_id}: {e}")
            # Don't raise the exception - feed creation should still succeed even if immediate poll fails

    async def cleanup_stuck_polling_feeds(self) -> int:
        """Clean up feeds that have been stuck in polling state for too long"""
        try:
            # **BULLY!** Using DatabaseManager for cleanup operations!
            result = await execute("""
                UPDATE rss_feeds 
                SET is_polling = false, updated_at = NOW()
                WHERE is_polling = true 
                AND updated_at < NOW() - INTERVAL '30 minutes'
            """)
            
            # Extract number of updated rows
            if result.startswith("UPDATE "):
                updated_count = int(result.split()[1])
                if updated_count > 0:
                    logger.info(f"🧹 RSS SERVICE: Cleaned up {updated_count} stuck polling feeds")
                return updated_count
            else:
                logger.warning(f"⚠️ RSS SERVICE: Unexpected result from cleanup: {result}")
                return 0
                    
        except Exception as e:
            logger.error(f"❌ RSS SERVICE ERROR: Failed to cleanup stuck polling feeds: {e}")
            return 0


# Global RSS service instance
_rss_service_instance = None

async def get_rss_service(shared_db_pool=None) -> RSSService:
    """Get global RSS service instance"""
    global _rss_service_instance
    if _rss_service_instance is None:
        _rss_service_instance = RSSService()
        if shared_db_pool:
            await _rss_service_instance.initialize(shared_db_pool)
    return _rss_service_instance
