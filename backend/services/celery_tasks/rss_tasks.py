"""
RSS Celery Tasks
Background processing for RSS feed monitoring and article processing
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List

from services.celery_app import celery_app, update_task_progress, TaskStatus
from celery.exceptions import SoftTimeLimitExceeded
from services.celery_utils import (
    safe_serialize_error, 
    safe_update_task_state, 
    clean_result_for_storage,
    create_progress_meta,
    safe_task_wrapper
)

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="services.celery_tasks.rss_tasks.extract_full_content_task")
def extract_full_content_task(
    self,
    user_id: str,
    article_ids: List[str]
) -> Dict[str, Any]:
    """
    Background task for extracting full content from RSS articles using Crawl4AI
    
    **BULLY!** This task uses Crawl4AI to extract complete article content!
    """
    try:
        logger.info(f"🕷️ RSS TASK: Starting full content extraction for {len(article_ids)} articles")
        
        update_task_progress(self, 1, 4, "Initializing Crawl4AI extraction...")
        
        # Create new event loop for this task to avoid "Event loop is closed" errors
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(_async_extract_full_content(
                self, user_id, article_ids
            ))
        finally:
            loop.close()
        
        logger.info(f"✅ RSS TASK: Completed full content extraction successfully")
        return result
        
    except Exception as e:
        logger.error(f"❌ RSS TASK ERROR: {e}")
        
        self.update_state(
            state=TaskStatus.FAILURE,
            meta={
                "error": str(e),
                "message": "Full content extraction failed",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return {
            "success": False,
            "error": str(e),
            "message": "Background full content extraction failed"
        }


async def _cleanup_stuck_polling_feeds():
    """Clean up feeds that have been stuck in polling state for too long"""
    try:
        from tools_service.services.rss_service import get_rss_service
        rss_service = await get_rss_service()

        cleaned_count = await rss_service.cleanup_stuck_polling_feeds()
        if cleaned_count > 0:
            logger.info(f"🧹 RSS TASK: Cleaned up {cleaned_count} stuck polling feeds")

        return cleaned_count
    except Exception as e:
        logger.error(f"❌ RSS TASK ERROR: Failed to cleanup stuck polling feeds: {e}")
        return 0


async def _get_feeds_summary():
    """Get a summary of RSS feeds status for monitoring"""
    try:
        from tools_service.services.rss_service import get_rss_service
        rss_service = await get_rss_service()

        # Get total feeds count
        from services.database_manager.database_helpers import fetch_value
        try:
            total_feeds = await fetch_value("SELECT COUNT(*) FROM rss_feeds")
        except Exception:
            total_feeds = 0

        # Get feeds needing poll
        feeds_needing_poll = await rss_service.get_feeds_needing_poll()
        feeds_needing_poll_count = len(feeds_needing_poll)

        # Get feeds currently being polled
        try:
            polling_feeds = await fetch_value("SELECT COUNT(*) FROM rss_feeds WHERE is_polling = true")
        except Exception:
            polling_feeds = 0

        # Get feeds by check interval distribution
        interval_counts = {}
        from services.database_manager.database_helpers import fetch_all
        rows = await fetch_all("SELECT check_interval, COUNT(*) as count FROM rss_feeds GROUP BY check_interval ORDER BY check_interval")
        for row in rows:
            interval_minutes = row['check_interval'] // 60
            interval_counts[f"{interval_minutes}min"] = row['count']

        return {
            "total_feeds": total_feeds,
            "feeds_needing_poll": feeds_needing_poll_count,
            "polling_feeds": polling_feeds,
            "interval_distribution": interval_counts
        }
    except Exception as e:
        logger.error(f"❌ RSS TASK ERROR: Failed to get feeds summary: {e}")
        return {
            "total_feeds": 0,
            "feeds_needing_poll": 0,
            "polling_feeds": 0,
            "interval_distribution": {}
        }


@celery_app.task(bind=True, name="services.celery_tasks.rss_tasks.poll_rss_feeds_task", rate_limit="1/m")  # Limit to 1 poll per minute
def poll_rss_feeds_task(
    self,
    user_id: Optional[str] = None,
    feed_ids: Optional[List[str]] = None,
    force_poll: bool = False
) -> Dict[str, Any]:
    """
    Background task for RSS feed polling with concurrency control
    
    **BULLY!** This task handles scheduled RSS feed monitoring and article discovery!
    """
    try:
        logger.info(f"📡 RSS TASK: Starting RSS feed polling for user {user_id}")
        
        # First, cleanup any stuck polling feeds
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            cleaned_count = loop.run_until_complete(_cleanup_stuck_polling_feeds())
            if cleaned_count > 0:
                logger.info(f"🧹 RSS TASK: Cleaned up {cleaned_count} stuck polling feeds before polling")
        except Exception as e:
            logger.warning(f"⚠️ RSS TASK: Failed to cleanup stuck polling feeds: {e}")
        finally:
            loop.close()
        
        update_task_progress(self, 1, 4, "Initializing RSS background agent...")
        
        # Create new event loop for this task to avoid "Event loop is closed" errors
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(_async_poll_rss_feeds(
                self, user_id, feed_ids, force_poll
            ))
        finally:
            loop.close()
        
        logger.info(f"✅ RSS TASK: Completed RSS feed polling successfully")
        return result
        
    except SoftTimeLimitExceeded as e:
        logger.error(f"❌ RSS TASK ERROR: Soft time limit exceeded: {e}")
        self.update_state(
            state=TaskStatus.FAILURE,
            meta={
                "exc_type": "SoftTimeLimitExceeded",
                "error": "Soft time limit exceeded",
                "message": "RSS feed polling exceeded time limit",
                "timestamp": datetime.now().isoformat()
            }
        )
        return {
            "success": False,
            "error": "SoftTimeLimitExceeded",
            "message": "RSS feed polling exceeded time limit"
        }
    except Exception as e:
        logger.error(f"❌ RSS TASK ERROR: {e}")
        
        self.update_state(
            state=TaskStatus.FAILURE,
            meta={
                "exc_type": type(e).__name__,
                "error": str(e),
                "message": "RSS feed polling failed",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return {
            "success": False,
            "error": str(e),
            "message": "Background RSS feed polling failed"
        }


@celery_app.task(
    bind=True, 
    name="services.celery_tasks.rss_tasks.process_rss_article_task", 
    rate_limit="2/m", 
    priority=5,  # High priority
    autoretry_for=(Exception,),  # Retry on any exception
    retry_kwargs={'max_retries': 3, 'countdown': 60},  # Retry 3 times with 60s delay
    retry_backoff=True  # Exponential backoff
)
def process_rss_article_task(
    self,
    article_id: str,
    user_id: str,
    collection_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Background task for processing individual RSS articles
    
    **By George!** This task handles full article download, processing, and embedding!
    """
    try:
        logger.info(f"📰 RSS TASK: Starting article processing for article {article_id} (Task ID: {self.request.id})")
        
        update_task_progress(self, 1, 5, "Initializing article processing...")
        
        # Create new event loop for this task to avoid "Event loop is closed" errors
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(_async_process_rss_article(
                self, article_id, user_id, collection_name
            ))
        finally:
            loop.close()
        
        logger.info(f"✅ RSS TASK: Completed article processing successfully")
        return result
        
    except Exception as e:
        logger.error(f"❌ RSS TASK ERROR: {e}")
        
        self.update_state(
            state=TaskStatus.FAILURE,
            meta={
                "error": str(e),
                "message": "RSS article processing failed",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return {
            "success": False,
            "error": str(e),
            "message": "Background RSS article processing failed"
        }


@celery_app.task(bind=True, name="services.celery_tasks.rss_tasks.cleanup_stuck_rss_feeds_task")
def cleanup_stuck_rss_feeds_task(self) -> Dict[str, Any]:
    """
    Background task for cleaning up RSS feeds stuck in polling state

    **Trust busting for stuck RSS feeds!** This task cleans up feeds that have been
    stuck in polling state for too long, preventing them from being polled again!
    """
    try:
        logger.info(f"🧹 RSS TASK: Starting cleanup of stuck RSS feeds")

        update_task_progress(self, 1, 2, "Cleaning up stuck RSS feeds...")

        # Create new event loop for this task to avoid "Event loop is closed" errors
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            cleaned_count = loop.run_until_complete(_cleanup_stuck_polling_feeds())
        finally:
            loop.close()

        update_task_progress(self, 2, 2, "Cleanup completed")

        logger.info(f"✅ RSS TASK: Completed cleanup of stuck RSS feeds")
        return {
            "success": True,
            "task_id": self.request.id,
            "timestamp": datetime.now().isoformat(),
            "cleaned_feeds": cleaned_count,
            "message": f"Successfully cleaned up {cleaned_count} stuck RSS feeds"
        }

    except Exception as e:
        logger.error(f"❌ RSS TASK ERROR: {e}")

        self.update_state(
            state=TaskStatus.FAILURE,
            meta={
                "error": str(e),
                "message": "RSS cleanup failed",
                "timestamp": datetime.now().isoformat()
            }
        )

        return {
            "success": False,
            "error": str(e),
            "message": "Background RSS cleanup failed"
        }


@celery_app.task(bind=True, name="services.celery_tasks.rss_tasks.rss_health_check_task")
def rss_health_check_task(self) -> Dict[str, Any]:
    """
    Background task for RSS feed health monitoring
    
    **Trust busting for unhealthy RSS feeds!** This task monitors feed reliability!
    """
    try:
        logger.info(f"🏥 RSS TASK: Starting RSS feed health check")
        
        update_task_progress(self, 1, 3, "Checking RSS feed health...")
        
        # Create new event loop for this task to avoid "Event loop is closed" errors
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(_async_rss_health_check(self))
        finally:
            loop.close()
        
        logger.info(f"✅ RSS TASK: Completed RSS health check successfully")
        return result
        
    except Exception as e:
        logger.error(f"❌ RSS TASK ERROR: {e}")
        
        self.update_state(
            state=TaskStatus.FAILURE,
            meta={
                "error": str(e),
                "message": "RSS health check failed",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return {
            "success": False,
            "error": str(e),
            "message": "Background RSS health check failed"
        }


async def _async_poll_rss_feeds(
    task,
    user_id: Optional[str],
    feed_ids: Optional[List[str]],
    force_poll: bool
) -> Dict[str, Any]:
    """Async RSS feed polling processing"""
    try:
        update_task_progress(task, 2, 4, "Creating RSS background agent...")
        
        # Initialize RSS background agent - LAZY IMPORT
        from services.langgraph_agents.rss_background_agent import RSSBackgroundAgent
        rss_agent = RSSBackgroundAgent()
        
        # Prepare state for RSS processing
        state = {
            "user_id": user_id,
            "feeds_to_poll": feed_ids,
            "force_poll": force_poll,
            "task_id": task.request.id
        }
        
        update_task_progress(task, 3, 4, "Polling RSS feeds...")
        
        # Process RSS feeds (RSSBackgroundAgent exposes process(), not _process_request)
        result = await rss_agent.process(state)
        
        update_task_progress(task, 4, 4, "RSS feed polling completed")
        
        # Clean and return result
        cleaned_result = clean_result_for_storage(result)
        
        return {
            "success": True,
            "task_id": task.request.id,
            "timestamp": datetime.now().isoformat(),
            "result": cleaned_result
        }
        
    except Exception as e:
        logger.error(f"❌ ASYNC RSS ERROR: {e}")
        raise e


async def _async_process_rss_article(
    task,
    article_id: str,
    user_id: str,
    collection_name: Optional[str]
) -> Dict[str, Any]:
    """Async RSS article processing"""
    try:
        update_task_progress(task, 2, 5, "Retrieving article metadata...")
        
        # Retrieve article from database
        from tools_service.services.rss_service import get_rss_service
        rss_service = await get_rss_service()
        
        # Add retry logic for database connection issues
        max_retries = 5
        retry_delay = 2  # Start with 2 seconds, then increase
        
        for attempt in range(max_retries):
            try:
                article = await rss_service.get_article(article_id)
                break
            except Exception as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ["another operation is in progress", "connection was closed", "connection does not exist"]) and attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)  # Exponential backoff
                    logger.warning(f"⚠️ Database connection issue (attempt {attempt + 1}/{max_retries}), retrying in {wait_time} seconds: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"❌ Database error after {max_retries} attempts: {e}")
                    raise e
        
        if not article:
            raise Exception(f"Article {article_id} not found")
        
        update_task_progress(task, 3, 5, "Downloading full article content...")
        
        # Download full article content
        import aiohttp
        import re
        from bs4 import BeautifulSoup
        
        async with aiohttp.ClientSession() as session:
            async with session.get(article.link, timeout=30) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download article: HTTP {response.status}")
                
                content = await response.text()
        
        # Use universal content extractor for better results
        from services.universal_content_extractor import get_universal_content_extractor
        universal_extractor = await get_universal_content_extractor()
        
        clean_content, original_html, enhanced_images = await universal_extractor.extract_main_content(content, article.link)
        
        # If universal extractor didn't work well, fall back to our enhanced cleaning
        if not clean_content or len(clean_content.strip()) < 100:
            logger.warning(f"⚠️ Universal extractor produced insufficient content, falling back to enhanced cleaning")
            
            # Fallback to enhanced BeautifulSoup cleaning
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove navigation and website chrome elements
            for element in soup(["script", "style", "nav", "header", "footer", "aside", "menu", "sidebar"]):
                element.decompose()
            
            # Remove common navigation and menu classes/IDs
            for element in soup.find_all(class_=re.compile(r'(nav|menu|header|footer|sidebar|breadcrumb|pagination|social|share|ad|banner|logo|widget|sidebar|column|panel)', re.I)):
                element.decompose()
            
            for element in soup.find_all(id=re.compile(r'(nav|menu|header|footer|sidebar|breadcrumb|pagination|social|share|ad|banner|logo|widget|sidebar|column|panel)', re.I)):
                element.decompose()
            
            # Remove common navigation patterns
            for element in soup.find_all("div", class_=re.compile(r'(navigation|navbar|menubar|toolbar|banner|advertisement|sidebar|widget|column|panel|menu|nav)', re.I)):
                element.decompose()
            
            # Remove Hackaday-specific elements
            for element in soup.find_all("div", class_=re.compile(r'(sidebar|widget|column|panel|menu|nav|related|popular|trending|recommended)', re.I)):
                element.decompose()
            
            # Remove elements with common sidebar/column patterns
            for element in soup.find_all("div", class_=re.compile(r'(col-|column-|sidebar-|widget-|panel-)', re.I)):
                element.decompose()
            
            # Remove elements with specific Hackaday patterns
            for element in soup.find_all("div", class_=re.compile(r'(hackaday|hack|sidebar|widget)', re.I)):
                element.decompose()
            
            # Get text content
            text_content = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_content = ' '.join(chunk for chunk in chunks if chunk)
            
            # Remove common web artifacts and navigation text
            artifacts_to_remove = [
                r'Share this article', r'Follow us on', r'Subscribe to', r'BBC Homepage', 
                r'Skip to content', r'Accessibility Help', r'Cookie Policy', r'Privacy Policy',
                r'Terms of Service', r'Contact Us', r'About Us', r'Home', r'News', r'Sports',
                r'Entertainment', r'Business', r'Technology', r'Science', r'Health',
                r'Search', r'Login', r'Sign up', r'Subscribe', r'Newsletter',
                r'Follow', r'Share', r'Like', r'Comment', r'Related Articles',
                r'Recommended', r'Popular', r'Trending', r'Most Read', r'Latest',
                r'Previous', r'Next', r'Back to top', r'Return to top',
                r'Advertisement', r'Ad', r'Sponsored', r'Promoted',
                r'Menu', r'Navigation', r'Breadcrumb', r'Pagination',
                r'Footer', r'Header', r'Sidebar', r'Widget',
                # Hackaday-specific patterns
                r'Hackaday', r'Hack a Day', r'Hackaday\.com', r'Hackaday Blog',
                r'Submit a Tip', r'Submit Tip', r'Submit Your Tip',
                r'Recent Posts', r'Recent Articles', r'Latest Posts', r'Latest Articles',
                r'Popular Posts', r'Popular Articles', r'Featured Posts', r'Featured Articles',
                r'Related Posts', r'Related Articles', r'You might also like',
                r'Comments', r'Comment', r'Leave a comment', r'Post a comment',
                r'Tagged with', r'Tags', r'Categories', r'Category',
                r'Posted by', r'Author', r'Written by', r'By',
                r'Posted on', r'Published on', r'Date', r'Time',
                r'Read more', r'Continue reading', r'Full article',
                r'Subscribe to Hackaday', r'Follow Hackaday', r'Hackaday Newsletter',
                r'RSS Feed', r'RSS', r'Atom Feed', r'Atom',
                r'Twitter', r'Facebook', r'Reddit', r'YouTube', r'Instagram',
                r'Email', r'Contact', r'About', r'Privacy', r'Terms'
            ]
            
            for artifact in artifacts_to_remove:
                clean_content = re.sub(artifact, '', clean_content, flags=re.IGNORECASE)
            
            # Remove common website chrome patterns
            clean_content = re.sub(r'^\s*(Home|News|Sports|Entertainment|Business|Technology|Science|Health)\s*$', '', clean_content, flags=re.MULTILINE | re.IGNORECASE)
            clean_content = re.sub(r'^\s*(Search|Login|Sign up|Subscribe|Follow|Share)\s*$', '', clean_content, flags=re.MULTILINE | re.IGNORECASE)
            
            # Remove Hackaday-specific patterns
            clean_content = re.sub(r'^\s*(Hackaday|Hack a Day|Submit a Tip|Recent Posts|Popular Posts|Related Posts)\s*$', '', clean_content, flags=re.MULTILINE | re.IGNORECASE)
            clean_content = re.sub(r'^\s*(Comments|Comment|Leave a comment|Posted by|Posted on|Tagged with)\s*$', '', clean_content, flags=re.MULTILINE | re.IGNORECASE)
            
            # Remove common sidebar/menu patterns
            clean_content = re.sub(r'^\s*(Sidebar|Widget|Column|Panel|Menu|Navigation)\s*$', '', clean_content, flags=re.MULTILINE | re.IGNORECASE)
            
            # Clean up any remaining excessive whitespace
            clean_content = re.sub(r'\s+', ' ', clean_content)
            clean_content = clean_content.strip()
            
            # Limit content length
            if len(clean_content) > 50000:
                clean_content = clean_content[:50000] + "..."
        
        update_task_progress(task, 4, 5, "Processing and embedding article...")
        
        # Use FileManager service for centralized file placement
        from services.file_manager import get_file_manager
        from services.file_manager.models.file_placement_models import FilePlacementRequest, SourceType
        from services.service_container import get_service_container
        from services.embedding_service_wrapper import get_embedding_service
        from utils.document_processor import get_document_processor
        from models.api_models import DocumentType, DocumentCategory, ProcessingStatus
        
        # Get services from service container (has WebSocket manager access)
        service_container = await get_service_container()
        file_manager = await get_file_manager()
        document_service = service_container.document_service  # Use service container's document service
        
        embedding_manager = await get_embedding_service()
        document_processor = await get_document_processor()
        
        # Get the RSS feed to determine folder location
        # Add retry logic for feed retrieval
        feed = None
        for attempt in range(max_retries):
            try:
                feed = await rss_service.get_feed(article.feed_id)
                if feed:
                    break
                else:
                    logger.warning(f"⚠️ Feed {article.feed_id} not found (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)
                        await asyncio.sleep(wait_time)
                        continue
            except Exception as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ["another operation is in progress", "connection was closed", "connection does not exist"]) and attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    logger.warning(f"⚠️ Database connection issue during feed retrieval (attempt {attempt + 1}/{max_retries}), retrying in {wait_time} seconds: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"❌ Feed retrieval error after {max_retries} attempts: {e}")
                    raise e
        
        if not feed:
            raise Exception(f"Feed {article.feed_id} not found after {max_retries} attempts")
        
        # Determine folder location based on RSS feed's scope (Global vs My Documents)
        # Global feeds import to Global Documents -> Web Sources
        # User feeds import to My Documents -> Web Sources
        # A feed is global if user_id is None, user-specific if user_id is set
        is_global_feed = feed.user_id is None
        if is_global_feed:
            # For global feeds, we'll create a real folder structure
            collection_type = "global"  # Global collection
        else:
            # For user feeds, we'll create a real folder structure
            collection_type = "user"  # User collection
        
        # FileManager will handle folder creation automatically based on source type
        
        # Generate a unique document ID for the imported article
        import hashlib
        document_id = hashlib.sha256(f"{article_id}_{user_id}".encode()).hexdigest()[:32]
        
        # Prepare metadata for the document
        document_metadata = {
            "title": article.title,
            "description": article.description or f"Article from {feed.feed_name}",
            "author": feed.feed_name,
            "category": "rss_import",
            "tags": ["rss", "imported", feed.feed_name],
            "language": "en",
            "source_url": article.link,
            "rss_article_id": article_id,
            "rss_feed_id": article.feed_id,
            "rss_feed_name": feed.feed_name,
            "rss_feed_owner": feed.user_id,  # Track if feed was global or user-specific
            "rss_feed_scope": "global" if feed.user_id is None else "user",  # Track feed scope
            "published_date": article.published_date.isoformat() if article.published_date else None,
            "imported_from": "rss",
            "imported_by": user_id,
            "collection_type": collection_type  # Track which collection this belongs to
        }
        
        # Get user role information for admin context
        current_user_role = "user"  # Default
        admin_user_id = None
        
        # For global feeds, we need to get the admin context from the user who initiated the task
        if collection_type == "global" and user_id:
            try:
                from services.auth_service import auth_service
                user_info = await auth_service.get_user_by_id(user_id)
                if user_info and user_info.role == "admin":
                    current_user_role = "admin"
                    admin_user_id = user_id
                    logger.info(f"🔐 Confirmed admin user {user_id} for global RSS folder creation")
                else:
                    logger.warning(f"⚠️ User {user_id} is not admin but trying to import global RSS article - will attempt anyway")
                    # For global feeds, still allow creation but with admin context from system
                    current_user_role = "admin"
                    admin_user_id = user_id  # System admin context for global RSS operations
            except Exception as e:
                logger.warning(f"⚠️ Could not verify user admin status: {e} - proceeding with admin context for global RSS")
                current_user_role = "admin"
                admin_user_id = user_id
        
        # Use FileManager to place the RSS article with proper error handling
        try:
            logger.info(f"📁 Initializing FileManager for RSS article placement...")
            await file_manager.initialize()
            logger.info(f"📁 FileManager initialized successfully")
            
            placement_request = FilePlacementRequest(
                content=clean_content,
                title=article.title,
                source_type=SourceType.RSS,
                source_metadata={
                    "feed_name": feed.feed_name,
                    "feed_url": feed.feed_url,
                    "article_url": article.link,
                    "published_date": article.published_date.isoformat() if article.published_date else None,
                    "rss_article_id": article_id,
                    "rss_feed_id": article.feed_id,
                    "rss_feed_owner": feed.user_id,
                    "rss_feed_scope": "global" if feed.user_id is None else "user",
                    "images": enhanced_images  # **BULLY!** Pass through extracted images
                },
                doc_type=DocumentType.TXT,
                category=DocumentCategory.NEWS,
                description=article.description or f"Article from {feed.feed_name}",
                author=feed.feed_name,
                language="en",
                user_id=None if collection_type == "global" else user_id,
                collection_type=collection_type,
                current_user_role=current_user_role,
                admin_user_id=admin_user_id,
                process_immediately=True  # Process immediately to create chunks for vector search
            )
            
            # Place file using FileManager
            placement_response = await file_manager.place_file(placement_request)
            document_id = placement_response.document_id
            folder_id = placement_response.folder_id
            
            logger.info(f"📝 File placed via FileManager: {document_id} in folder {folder_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to place file via FileManager: {e}")
            logger.error(f"❌ Error type: {type(e).__name__}")
            logger.error(f"❌ Error details: {str(e)}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            raise Exception(f"FileManager placement failed: {e}")
        
        # Store document record in database
        try:
            logger.info(f"📄 Creating document record in database: {document_id}")
            logger.info(f"📄 Document metadata: title='{document_metadata['title']}', collection_type={document_metadata['collection_type']}")
            # FileManager already created the document, so we don't need to create it again
            logger.info(f"✅ Document already created by FileManager: {document_id}")
            logger.info(f"✅ Document record created successfully: {document_id}")
            
            # Add a small delay to ensure the transaction is committed
            await asyncio.sleep(0.1)
            
            # Verify the document was created successfully via FileManager
            try:
                # FileManager handles document creation, so we just log success
                logger.info(f"✅ Document created successfully via FileManager: {document_id}")
            except Exception as e:
                logger.warning(f"⚠️ Could not verify document creation: {e}")
            
        except Exception as e:
            logger.error(f"❌ Failed to process document via FileManager: {e}")
            raise Exception(f"FileManager document processing failed: {e}")
        
        # Update status to embedding using FileManager's WebSocket notifier
        try:
            # Emit WebSocket notification via FileManager
            await file_manager.websocket_notifier.notify_processing_status_update(
                document_id, "embedding", folder_id, 
                user_id=None if collection_type == "global" else user_id, 
                progress=0.5
            )
            logger.info(f"✅ Document status updated to EMBEDDING: {document_id}")
        except Exception as e:
            logger.error(f"❌ Failed to update document status: {e}")
            # Don't raise here, continue with processing
        
        # FileManager's document service already handled content processing, so we just need to update final status
        try:
            # Use the parallel document service's enhanced capabilities
            if hasattr(file_manager.document_service, 'store_text_document'):
                # The parallel document service handles all processing internally
                logger.info(f"✅ Document processing completed by ParallelDocumentService: {document_id}")
            else:
                # Fallback to basic processing
                logger.info(f"✅ Document processing completed by basic document service: {document_id}")
            
            # Emit WebSocket notification via FileManager
            await file_manager.websocket_notifier.notify_file_processed(
                document_id, folder_id, 
                user_id=None if collection_type == "global" else user_id,
                processing_info={"chunks_processed": "completed_by_parallel_document_service"}
            )
        except Exception as e:
            logger.error(f"❌ Failed to update final document status: {e}")
            # Don't raise here, the processing was successful
        
        # Mark article as processed
        await rss_service.mark_article_processed(article_id)
        
        update_task_progress(task, 5, 5, "Article processing completed")
        
        return {
            "success": True,
            "task_id": task.request.id,
            "article_id": article_id,
            "user_id": user_id,
            "collection_name": feed.feed_name,
            "document_id": document_id, # Return the document ID
            "content_length": len(clean_content),
            "timestamp": datetime.now().isoformat(),
            "message": f"Article imported to {feed.feed_name} folder via FileManager"
        }
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"❌ ASYNC ARTICLE PROCESSING ERROR: {e}")
        logger.error(f"❌ ERROR TRACEBACK: {error_traceback}")
        
        # Return a proper error response instead of raising
        return {
            "success": False,
            "task_id": task.request.id,
            "article_id": article_id,
            "user_id": user_id,
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat(),
            "message": f"Failed to import article: {str(e)}"
        }


async def _async_extract_full_content(task, user_id: str, article_ids: List[str]) -> Dict[str, Any]:
    """Async full content extraction using Crawl4AI"""
    try:
        update_task_progress(task, 2, 4, "Retrieving articles needing full content...")
        
        # Get RSS service
        from tools_service.services.rss_service import get_rss_service
        rss_service = await get_rss_service()
        
        # Get articles that need full content
        articles = []
        for article_id in article_ids:
            article = await rss_service.get_article(article_id)
            if article:
                articles.append(article)
        
        if not articles:
            return {
                "success": True,
                "articles_processed": 0,
                "message": "No articles found for full content extraction"
            }
        
        update_task_progress(task, 3, 4, f"Extracting full content for {len(articles)} articles...")
        
        # Initialize RSS background agent for content extraction - LAZY IMPORT
        from services.langgraph_agents.rss_background_agent import RSSBackgroundAgent
        rss_agent = RSSBackgroundAgent()
        
        articles_processed = 0
        articles_successful = 0
        
        for article in articles:
            try:
                # Extract full content (text + HTML) and images using Crawl4AI
                full_content, full_content_html, images = await rss_agent._extract_full_content_with_crawl4ai(article.link)
                
                if full_content:
                    # Update article with full content and images
                    success = await rss_service.update_article_full_content(article.article_id, full_content, images, full_content_html)
                    if success:
                        articles_successful += 1
                        logger.info(f"✅ Successfully extracted full content for article {article.article_id}")
                    else:
                        logger.warning(f"⚠️ Failed to update full content for article {article.article_id}")
                else:
                    logger.warning(f"⚠️ No full content extracted for article {article.article_id}")
                
                articles_processed += 1
                
            except Exception as e:
                logger.error(f"❌ Failed to extract full content for article {article.article_id}: {e}")
                articles_processed += 1
        
        update_task_progress(task, 4, 4, "Full content extraction completed")
        
        return {
            "success": True,
            "articles_processed": articles_processed,
            "articles_successful": articles_successful,
            "timestamp": datetime.now().isoformat(),
            "message": f"Processed {articles_processed} articles, successfully extracted content for {articles_successful} articles"
        }
        
    except Exception as e:
        logger.error(f"❌ ASYNC FULL CONTENT EXTRACTION ERROR: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "message": f"Full content extraction failed: {str(e)}"
        }


async def _async_rss_health_check(task) -> Dict[str, Any]:
    """Async RSS feed health check"""
    try:
        update_task_progress(task, 2, 3, "Analyzing feed health metrics...")
        
        # TODO: Implement RSS feed health analysis
        # health_results = await analyze_rss_feed_health()
        
        update_task_progress(task, 3, 3, "Health check completed")
        
        # For now, return placeholder result
        return {
            "success": True,
            "task_id": task.request.id,
            "timestamp": datetime.now().isoformat(),
            "message": "RSS health check completed (placeholder implementation)",
            "healthy_feeds": 0,
            "unhealthy_feeds": 0,
            "total_feeds": 0
        }
        
    except Exception as e:
        logger.error(f"❌ ASYNC HEALTH CHECK ERROR: {e}")
        raise e


# Scheduled task for automatic RSS feed polling
@celery_app.task(bind=True, name="services.celery_tasks.rss_tasks.scheduled_rss_poll_task")
def scheduled_rss_poll_task(self) -> Dict[str, Any]:
    """
    Scheduled task for automatic RSS feed polling

    **BULLY!** This is the scheduled cavalry charge that keeps RSS feeds fresh!
    This task is triggered by Celery Beat to poll all active feeds
    """
    try:
        logger.info(f"⏰ RSS TASK: Starting scheduled RSS feed polling - cavalry charge commencing!")

        update_task_progress(self, 1, 3, "Checking feeds needing poll...")

        # First, get a summary of feeds that need polling for better logging
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            feeds_summary = loop.run_until_complete(_get_feeds_summary())
        finally:
            loop.close()

        logger.info(f"📊 RSS TASK: Found {feeds_summary['total_feeds']} total feeds, {feeds_summary['feeds_needing_poll']} need polling")

        update_task_progress(self, 2, 3, f"Polling {feeds_summary['feeds_needing_poll']} RSS feeds...")

        # Poll all active feeds
        result = poll_rss_feeds_task.delay(
            user_id=None,  # Global polling
            feed_ids=None,  # All feeds
            force_poll=False
        )

        update_task_progress(self, 3, 3, "RSS polling task queued")

        logger.info(f"✅ RSS TASK: Scheduled RSS polling initiated - task {result.id} queued")

        return {
            "success": True,
            "scheduled_task_id": result.id,
            "timestamp": datetime.now().isoformat(),
            "feeds_summary": feeds_summary,
            "message": f"Scheduled RSS polling initiated - {feeds_summary['feeds_needing_poll']} feeds queued for polling"
        }

    except Exception as e:
        logger.error(f"❌ SCHEDULED RSS TASK ERROR: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "message": "Scheduled RSS polling failed"
        }


@celery_app.task(bind=True, name="services.celery_tasks.rss_tasks.purge_old_news_task")
def purge_old_news_task(self):
    """Purge synthesized news articles older than retention_days."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        def _run():
            async def _purge():
                try:
                    from services.settings_service import settings_service
                    from services.database_manager.database_helpers import execute
                    
                    if not getattr(settings_service, "_initialized", False):
                        await settings_service.initialize()
                    days = await settings_service.get_setting("news.retention_days", 14)
                    days = int(days or 14)
                    # First fetch file paths to delete from disk
                    from services.database_manager.database_helpers import fetch_all
                    rows = await fetch_all(
                        "SELECT file_path FROM news_articles WHERE updated_at < (NOW() - ($1 || ' days')::interval)",
                        days,
                    )
                    import os
                    deleted_files = 0
                    for row in rows or []:
                        p = row.get("file_path")
                        try:
                            if p and os.path.exists(p):
                                os.remove(p)
                                deleted_files += 1
                        except Exception:
                            pass
                    # Then delete SQL rows
                    await execute(
                        "DELETE FROM news_articles WHERE updated_at < (NOW() - ($1 || ' days')::interval)",
                        days,
                    )
                    return {"status": "ok", "purged": True, "retention_days": days, "files_deleted": deleted_files}
                except Exception as e:
                    return {"status": "error", "error": str(e)}
            return loop.run_until_complete(_purge())
        try:
            result = _run()
        finally:
            loop.close()
        return result
    except Exception as e:
        logger.error(f"❌ PURGE NEWS TASK ERROR: {e}")
        return {"status": "error", "error": str(e)}
