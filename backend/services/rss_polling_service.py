"""
RSS polling service for Celery: feed fetch, parse, persist, and optional full-content extraction.
No LLM inference; deterministic HTTP + feedparser + Crawl4AI.
"""

import logging
import asyncio
import hashlib
import re
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from urllib.parse import urljoin, urlparse
import aiohttp
import feedparser

from tools_service.models.rss_models import RSSFeed, RSSArticle, RSSFeedPollResult

logger = logging.getLogger(__name__)

# Substrings for stripping likely ad/tracking images (src URL, lowercase).
_RSS_AD_IMG_URL_MARKERS = (
    "doubleclick",
    "googlesyndication",
    "adnxs",
    "amazon-adsystem",
    "facebook.com/tr",
    "analytics",
    "/pixel",
    "pixel.",
    "beacon",
    "ad-server",
    "adserver",
    "adsrvr",
    "ruamupr",
    "adservice",
    "pagead",
    "adsafe",
    "adform",
    "criteo",
    "taboola",
    "outbrain",
)

# class/id bucket substrings for non-article chrome (matched with word-style caution).
_RSS_NON_ARTICLE_BUCKET_KEYS = (
    "adcovery",
    "ad-container",
    "advert",
    "advertisement",
    "sponsored",
    "paid",
    "sidebar",
    "social-share",
    "social_share",
    "share-buttons",
    "newsletter",
    "subscribe-box",
    "subscribe-form",
    "popup",
    "modal",
    "disqus",
    "outbrain",
    "taboola",
    "recirculation",
    "related-posts",
    "related_posts",
    "related-articles",
    "comment-section",
    "comments-area",
)


class RSSPollingService:
    """
    Background RSS feed polling and article ingestion.
    Used by Celery tasks; not an LLM agent.
    """

    def __init__(self):
        self._current_user_id: Optional[str] = None
    
    async def _get_rss_service(self):
        """
        Get RSS service from tools-service.
        
        RSS service has been migrated to tools-service container.
        """
        from tools_service.services.rss_service import get_rss_service
        return await get_rss_service()
    
    async def poll_feeds(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Poll RSS feeds from state: feeds_to_poll, user_id, force_poll.
        Returns a dict suitable for Celery result storage (task_status, metadata, feed_results).
        """
        try:
            logger.info("RSS poll: starting feed processing")
            
            # Extract RSS processing parameters from state
            feeds_to_poll = state.get("feeds_to_poll", [])
            user_id = state.get("user_id")
            force_poll = state.get("force_poll", False)
            
            logger.info(
                "RSS poll: state feeds_to_poll=%s user_id=%s force_poll=%s",
                feeds_to_poll,
                user_id,
                force_poll,
            )
            
            # Store user_id for feed filtering
            self._current_user_id = user_id
            
            if not feeds_to_poll:
                # Get all active feeds that need polling
                logger.info("RSS poll: no feeds in state, loading feeds needing poll")
                feeds_to_poll = await self._get_feeds_needing_poll()
                logger.info("RSS poll: %s feeds need polling", len(feeds_to_poll))
            else:
                # Normalize provided feed references (IDs/dicts) into RSSFeed objects
                logger.info("RSS poll: normalizing %s feed references", len(feeds_to_poll))
                normalized_feeds: List[RSSFeed] = []
                for feed_ref in feeds_to_poll:
                    logger.debug("RSS poll: resolving feed reference %s", feed_ref)
                    feed_obj = await self._resolve_feed_reference(feed_ref)
                    if feed_obj is not None:
                        normalized_feeds.append(feed_obj)
                        logger.info(
                            "RSS poll: resolved feed %s (%s)",
                            feed_obj.feed_id,
                            feed_obj.feed_name,
                        )
                    else:
                        logger.error("RSS poll: failed to resolve feed reference %s", feed_ref)
                feeds_to_poll = normalized_feeds
                logger.info("RSS poll: normalized %s feeds", len(feeds_to_poll))
            
            if not feeds_to_poll:
                return self._build_poll_result(
                    task_status="complete",
                    response="No RSS feeds require polling at this time",
                    metadata={"feeds_polled": 0, "articles_found": 0, "articles_added": 0}
                )
            
            # Process RSS feeds in parallel for maximum efficiency
            start_time = datetime.utcnow()
            logger.info("RSS poll: polling %s feeds in parallel", len(feeds_to_poll))
            
            polling_tasks = [
                self._poll_single_feed_with_timeout(feed, user_id, force_poll, timeout=300) 
                for feed in feeds_to_poll
            ]
            
            # Execute all feeds in parallel with error handling
            feed_results = await asyncio.gather(*polling_tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            processed_results = []
            total_articles_found = 0
            total_articles_added = 0
            total_duplicates_skipped = 0
            errors = []
            
            for i, result in enumerate(feed_results):
                feed = feeds_to_poll[i]
                
                if isinstance(result, Exception):
                    # Handle exception from parallel execution
                    error_msg = f"Feed {feed.feed_id}: {str(result)}"
                    logger.error("RSS poll error: %s", error_msg)
                    errors.append(error_msg)
                    
                    processed_results.append(RSSFeedPollResult(
                        feed_id=feed.feed_id,
                        status="error",
                        error_message=str(result)
                    ))
                else:
                    # Process successful result
                    processed_results.append(result)
                    
                    if result.status == "success":
                        total_articles_found += result.articles_found
                        total_articles_added += result.articles_added
                    elif result.status == "error":
                        errors.append(f"Feed {feed.feed_id}: {result.error_message}")
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Create structured response
            response = self._build_poll_result(
                task_status="complete",
                response=f"RSS feed polling completed. Found {total_articles_found} articles, added {total_articles_added} new articles.",
                metadata={
                    "feeds_polled": len(feeds_to_poll),
                    "articles_found": total_articles_found,
                    "articles_added": total_articles_added,
                    "duplicates_skipped": total_duplicates_skipped,
                    "errors": errors,
                    "processing_time": processing_time,
                    "parallel_execution": True
                },
                feed_results=processed_results
            )
            
            logger.info(
                "RSS poll: completed %s feeds in %.2fs",
                len(feeds_to_poll),
                processing_time,
            )
            return response
            
        except Exception as e:
            logger.error("RSS poll failed: %s", e)
            return self._build_poll_result(
                task_status="error",
                response=f"RSS processing failed: {str(e)}",
                metadata={"errors": [str(e)]}
            )
    
    async def _resolve_feed_reference(self, feed_ref: Any) -> Optional[RSSFeed]:
        """Resolve a feed reference (RSSFeed | str ID | dict) into an RSSFeed object."""
        try:
            logger.debug(f"🔍 RSS AGENT: Resolving feed reference type: {type(feed_ref)}")
            
            if isinstance(feed_ref, RSSFeed):
                return feed_ref
            
            if isinstance(feed_ref, str):
                rss_service = await self._get_rss_service()
                feed = await rss_service.get_feed(feed_ref)
                if not feed:
                    logger.error(f"🔍 RSS AGENT: Failed to resolve feed ID {feed_ref} - feed not found")
                return feed
            
            if isinstance(feed_ref, dict):
                # Best-effort parse from dict payload
                return RSSFeed(**feed_ref)
            
            logger.error(f"❌ RSS AGENT ERROR: Unsupported feed reference type: {type(feed_ref)}")
            return None
        except Exception as e:
            logger.error(f"❌ RSS AGENT ERROR: Failed to resolve feed reference {feed_ref}: {e}")
            return None
    
    async def _get_feeds_needing_poll(self) -> List[RSSFeed]:
        """Get RSS feeds that need polling based on check intervals"""
        try:
            rss_service = await self._get_rss_service()
            
            # Get user_id from state if available
            user_id = getattr(self, '_current_user_id', None)
            return await rss_service.get_feeds_needing_poll(user_id)
        except Exception as e:
            logger.error(f"❌ RSS AGENT ERROR: Failed to get feeds needing poll: {e}")
            return []
    
    def _validate_feed_url(self, url: str) -> bool:
        """
        Validate feed URL before fetching.
        
        Skip feeds with invalid or empty URLs
        """
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.scheme in ('http', 'https') and bool(parsed.netloc)
        except Exception:
            return False
    
    async def _poll_single_feed_with_timeout(self, feed: RSSFeed, user_id: Optional[str], force_poll: bool, timeout: int = 300) -> RSSFeedPollResult:
        """
        Poll a single feed with timeout protection.
        
        Isolate hung feeds so one feed cannot block the poll cycle
        Default timeout is 5 minutes per feed.
        """
        try:
            return await asyncio.wait_for(
                self._poll_single_feed(feed, user_id, force_poll),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"❌ RSS AGENT: Feed {feed.feed_id} polling timeout after {timeout}s")
            # Clean up polling status on timeout
            try:
                rss_service = await self._get_rss_service()
                await rss_service.mark_feed_polling(feed.feed_id, is_polling=False)
            except Exception as cleanup_error:
                logger.error(f"❌ RSS AGENT: Failed to cleanup after timeout: {cleanup_error}")
            
            return RSSFeedPollResult(
                feed_id=feed.feed_id,
                status="error",
                error_message=f"Polling timeout after {timeout} seconds"
            )
    
    async def _poll_single_feed(self, feed: RSSFeed, user_id: Optional[str], force_poll: bool) -> RSSFeedPollResult:
        """Poll a single RSS feed for new articles with concurrency control"""
        rss_service = await self._get_rss_service()
        
        try:
            logger.debug(f"📡 RSS AGENT: Polling feed {feed.feed_id}")
            
            if not self._validate_feed_url(feed.feed_url):
                logger.error(f"❌ RSS AGENT: Invalid feed URL for {feed.feed_id}: {feed.feed_url}")
                return RSSFeedPollResult(
                    feed_id=feed.feed_id,
                    status="error",
                    error_message="Invalid feed URL"
                )
            
            # Try to mark feed as polling
            polling_marked = await rss_service.mark_feed_polling(feed.feed_id, is_polling=True)
            if not polling_marked:
                logger.warning(f"⚠️ RSS AGENT: Feed {feed.feed_id} is already being polled by another process")
                return RSSFeedPollResult(
                    feed_id=feed.feed_id,
                    status="already_polling",
                    articles_found=0,
                    articles_added=0
                )
            
            # Check if feed needs polling (unless forced)
            if not force_poll and not self._feed_needs_polling(feed):
                return RSSFeedPollResult(
                    feed_id=feed.feed_id,
                    status="no_new_articles",
                    articles_found=0,
                    articles_added=0
                )
            
            # Parse RSS feed (set headers to avoid 403 from some hosts)
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; PlatoRSS/1.0; +https://example.local)",
                "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
            }
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(feed.feed_url, timeout=30) as response:
                    if response.status != 200:
                        raise Exception(f"HTTP {response.status}: {response.reason}")
                    
                    content = await response.text()
            
            # Parse RSS content
            parsed_feed = feedparser.parse(content)
            
            if parsed_feed.bozo:
                raise Exception(f"Invalid RSS feed: {parsed_feed.bozo_exception}")
            
            # Process articles
            articles_found = len(parsed_feed.entries)
            articles_added = 0
            
            logger.info(f"📡 RSS AGENT: Found {articles_found} articles in feed {feed.feed_id}")
            
            for entry in parsed_feed.entries:
                try:
                    # Create RSS article (summary/teaser from description or summary)
                    raw_description = getattr(entry, 'description', None) or getattr(entry, 'summary', None)
                    content_encoded = self._get_entry_content_encoded(entry)
                    article = RSSArticle(
                        article_id=self._generate_article_id(entry.link),
                        feed_id=feed.feed_id,
                        title=entry.title,
                        description=raw_description,
                        link=entry.link,
                        published_date=self._parse_published_date(entry),
                        user_id=user_id
                    )
                    
                    # Generate content hash for duplicate detection
                    article.content_hash = article.generate_content_hash()
                    
                    # Check for duplicates
                    is_duplicate = await self._is_duplicate_article(article)
                    if is_duplicate:
                        continue
                    
                    desc_plain_for_trunc = ""
                    if article.description:
                        raw_desc = article.description
                        try:
                            if '<' in raw_desc:
                                desc_plain_for_trunc = self._html_to_plain_text(raw_desc)
                                article.description = self._sanitize_article_html(raw_desc, article.link)
                            else:
                                article.description = raw_desc.strip()
                                desc_plain_for_trunc = article.description
                        except Exception:
                            article.description = (raw_desc or "").strip()
                            desc_plain_for_trunc = article.description

                    full_body_from_feed = False
                    if content_encoded:
                        article.full_content_html = self._sanitize_article_html(
                            content_encoded, article.link
                        )
                        article.full_content = self._html_to_plain_text(content_encoded)
                        full_body_from_feed = True
                        logger.debug(
                            "RSS ingest: using content:encoded for %s (len=%s)",
                            (article.title or "")[:80],
                            len(content_encoded),
                        )

                    # Crawl only when feed did not supply full HTML body and teaser looks truncated
                    if not full_body_from_feed and self._is_content_truncated(desc_plain_for_trunc):
                        logger.debug(
                            "RSS ingest: truncated teaser for %s, extracting full content",
                            article.title,
                        )
                        full_content, full_content_html, images = await self.extract_full_content(article.link)
                        if full_content:
                            # Plain text for search/embeddings; sanitized HTML keeps paragraphs and images
                            article.full_content = self._html_to_plain_text(full_content)
                            if full_content_html:
                                article.full_content_html = self._sanitize_article_html(
                                    full_content_html, article.link
                                )
                            if images:
                                article.images = images
                            logger.debug(
                                "RSS ingest: extracted full content for %s", article.title
                            )
                    
                    # Save article to database
                    save_success = await self._save_article(article)
                    if save_success:
                        articles_added += 1
                    else:
                        logger.error(f"📡 RSS AGENT: Failed to save article: {article.title}")
                    
                except Exception as e:
                    logger.error(f"❌ RSS AGENT ERROR: Failed to process article {entry.link}: {e}")
                    continue
            
            # Update feed last_check timestamp (this also sets is_polling=false)
            await self._update_feed_last_check(feed.feed_id)
            
            return RSSFeedPollResult(
                feed_id=feed.feed_id,
                status="success",
                articles_found=articles_found,
                articles_added=articles_added
            )
        
        except Exception as e:
            logger.error(f"❌ RSS AGENT ERROR: Failed to poll feed {feed.feed_id}: {e}")
            return RSSFeedPollResult(
                feed_id=feed.feed_id,
                status="error",
                error_message=str(e)
            )
        finally:
            # This was previously done in both except and finally, causing redundant calls
            try:
                await rss_service.mark_feed_polling(feed.feed_id, is_polling=False)
            except Exception as cleanup_error:
                logger.error(f"❌ RSS AGENT ERROR: Failed to cleanup polling status for {feed.feed_id}: {cleanup_error}")
    
    def _feed_needs_polling(self, feed: RSSFeed) -> bool:
        """Check if a feed needs polling based on its check interval"""
        if not feed.last_check:
            return True
        
        next_check_time = feed.last_check + timedelta(seconds=feed.check_interval)
        return datetime.utcnow() >= next_check_time
    
    def _generate_article_id(self, link: str) -> str:
        """Generate a unique article ID from the link"""
        return hashlib.sha256(link.encode()).hexdigest()[:32]
    
    def _parse_published_date(self, entry) -> Optional[datetime]:
        """Parse published date from RSS entry"""
        try:
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                return datetime(*entry.published_parsed[:6])
            elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                return datetime(*entry.updated_parsed[:6])
            return None
        except Exception:
            return None

    @staticmethod
    def _get_entry_content_encoded(entry) -> Optional[str]:
        """Prefer Atom/RSS full body (content:encoded) when present and substantial."""
        if not getattr(entry, "content", None):
            return None
        for block in entry.content:
            val = None
            if hasattr(block, "value"):
                val = block.value
            elif isinstance(block, dict):
                val = block.get("value")
            if val is None:
                continue
            text = str(val).strip()
            if len(text) > 200:
                return text
        return None
    
    async def _is_duplicate_article(self, article: RSSArticle) -> bool:
        """Check if article is a duplicate based on content hash"""
        try:
            rss_service = await self._get_rss_service()
            return await rss_service.is_duplicate_article(article)
        except Exception as e:
            logger.error(f"❌ RSS AGENT ERROR: Failed to check duplicate: {e}")
            return False
    
    async def _save_article(self, article: RSSArticle) -> bool:
        """Save RSS article to database"""
        try:
            rss_service = await self._get_rss_service()
            return await rss_service.save_article(article)
        except Exception as e:
            logger.error(f"❌ RSS AGENT ERROR: Failed to save article: {e}")
            return False
    
    async def _update_feed_last_check(self, feed_id: str) -> bool:
        """Update feed last_check timestamp"""
        try:
            rss_service = await self._get_rss_service()
            return await rss_service.update_feed_last_check(feed_id)
        except Exception as e:
            logger.error(f"❌ RSS AGENT ERROR: Failed to update feed last_check: {e}")
            return False
    
    def _is_content_truncated(self, description: Optional[str]) -> bool:
        """Detect if RSS teaser is truncated and needs full-page extraction."""
        if not description:
            return False

        truncation_patterns = [
            r"\.\.\.read more",
            r"\.\.\.continue reading",
            r"read more\s*</a>",
            r"continue reading\s*</a>",
            r"full story\s*</a>",
            r"full article\s*</a>",
            r"\[…\]",
            r"&hellip;\s*</a>",
        ]

        description_lower = description.lower()
        for pattern in truncation_patterns:
            if re.search(pattern, description_lower):
                return True

        stripped = description.strip()
        if len(stripped) < 300:
            sentence_marks = len(re.findall(r"[.!?](?:\s|$)", stripped))
            if sentence_marks < 2:
                return True

        return False
    
    async def extract_full_content(self, url: str) -> tuple[Optional[str], Optional[str], Optional[List[Dict[str, Any]]]]:
        """Extract full article content (clean text + HTML) and images using Crawl4AI."""
        try:
            logger.info("RSS extract: fetching full content from %s", url[:120])
            
            # Use module-level wrapper for singleton management
            from services.langgraph_tools.crawl4ai_web_tools import crawl_web_content
            
            # Extract content using Crawl4AI (markdown extraction - we don't configure Crawl4AI with LLM)
            result = await crawl_web_content(
                url=url,
                extraction_strategy="markdown",
                chunking_strategy="NlpSentenceChunking",
                word_count_threshold=10
            )
            
            if result and result.get("results") and len(result["results"]) > 0:
                # Get the first result (we only crawled one URL)
                crawl_result = result["results"][0]
                
                if crawl_result.get("success"):
                    # Crawl4AI puts markdown/fit text in full_content; real DOM (with <img>) is in html.
                    page_html = (crawl_result.get("html") or "").strip()
                    text_or_md = (crawl_result.get("full_content") or "").strip()
                    enhanced_images = self._crawl_images_to_dicts(
                        crawl_result.get("images"), limit=20
                    )

                    extracted_text = ""
                    content_blocks = crawl_result.get("content_blocks", [])
                    if content_blocks:
                        for block in content_blocks:
                            if isinstance(block, dict) and "content" in block:
                                extracted_text += block["content"] + "\n\n"
                            elif isinstance(block, str):
                                extracted_text += block + "\n\n"

                    cleaned_from_blocks = (
                        self._clean_extracted_content(extracted_text)
                        if extracted_text.strip()
                        else ""
                    )

                    cleaned_ue = ""
                    article_html = ""
                    ext_imgs: List[Dict[str, Any]] = []

                    if len(page_html) > 80:
                        from services.universal_content_extractor import (
                            get_universal_content_extractor,
                        )

                        universal_extractor = await get_universal_content_extractor()
                        cleaned_ue, article_html, ext_imgs = (
                            await universal_extractor.extract_main_content(page_html, url)
                        )
                        if ext_imgs:
                            enhanced_images = ext_imgs

                    if cleaned_from_blocks and len(cleaned_from_blocks.strip()) >= 50:
                        cleaned_content = cleaned_from_blocks
                    elif cleaned_ue and len(cleaned_ue.strip()) >= 50:
                        cleaned_content = cleaned_ue
                    elif text_or_md:
                        cleaned_content = self._clean_extracted_content(text_or_md)
                    else:
                        cleaned_content = cleaned_ue or cleaned_from_blocks

                    if not cleaned_content or len(cleaned_content.strip()) < 20:
                        logger.warning(
                            "RSS extract: insufficient text for %s (blocks=%s md=%s html=%s)",
                            url[:80],
                            bool(extracted_text.strip()),
                            len(text_or_md),
                            len(page_html),
                        )
                        return None, None, None

                    if not article_html or len(article_html.strip()) < 50:
                        article_html = ""

                    logger.info(
                        "RSS extract: ok %s text_len=%s html_len=%s images=%s",
                        url[:80],
                        len(cleaned_content),
                        len(article_html),
                        len(enhanced_images),
                    )
                    return cleaned_content, article_html, enhanced_images

                logger.warning(
                    "RSS extract: crawl failed for %s: %s",
                    url[:120],
                    crawl_result.get("error", "Unknown error"),
                )
            else:
                logger.warning(f"⚠️ RSS AGENT: No results from Crawl4AI for {url}")
            
            return None, None, None
            
        except Exception as e:
            logger.error(f"❌ RSS AGENT ERROR: Failed to extract full content from {url}: {e}")
            return None, None, None
    
    def _clean_extracted_content(self, content: str) -> str:
        """Clean and format extracted content - focus on main article content"""
        if not content:
            return ""
        
        # If content looks like HTML, try to extract text from it
        if "<html>" in content.lower() or "<body>" in content.lower():
            try:
                from bs4 import BeautifulSoup
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
                content = soup.get_text()
            except ImportError:
                # If BeautifulSoup is not available, do basic HTML tag removal
                content = re.sub(r'<[^>]+>', '', content)
        
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
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
            content = re.sub(artifact, '', content, flags=re.IGNORECASE)
        
        # Remove common website chrome patterns
        content = re.sub(r'^\s*(Home|News|Sports|Entertainment|Business|Technology|Science|Health)\s*$', '', content, flags=re.MULTILINE | re.IGNORECASE)
        content = re.sub(r'^\s*(Search|Login|Sign up|Subscribe|Follow|Share)\s*$', '', content, flags=re.MULTILINE | re.IGNORECASE)
        
        # Remove Hackaday-specific patterns
        content = re.sub(r'^\s*(Hackaday|Hack a Day|Submit a Tip|Recent Posts|Popular Posts|Related Posts)\s*$', '', content, flags=re.MULTILINE | re.IGNORECASE)
        content = re.sub(r'^\s*(Comments|Comment|Leave a comment|Posted by|Posted on|Tagged with)\s*$', '', content, flags=re.MULTILINE | re.IGNORECASE)
        
        # Remove common sidebar/menu patterns
        content = re.sub(r'^\s*(Sidebar|Widget|Column|Panel|Menu|Navigation)\s*$', '', content, flags=re.MULTILINE | re.IGNORECASE)
        
        # Clean up any remaining excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        return content
    
    def _build_poll_result(self, task_status: str, response: str, metadata: Dict[str, Any], feed_results: Optional[List[RSSFeedPollResult]] = None) -> Dict[str, Any]:
        """Build structured result dict for Celery / task storage."""
        return {
            "task_status": task_status,
            "response": response,
            "metadata": metadata,
            "feed_results": [result.dict() for result in feed_results] if feed_results else [],
            "timestamp": datetime.utcnow().isoformat(),
            "success": task_status == "complete",
        }

    def _safe_url_for_href(self, href: str, base_url: Optional[str]) -> Optional[str]:
        if not href or not str(href).strip():
            return None
        h = str(href).strip()
        low = h.lower()
        if low.startswith("javascript:") or low.startswith("vbscript:") or low.startswith("data:"):
            return None
        if base_url and not urlparse(h).netloc:
            h = urljoin(base_url, h)
        parsed = urlparse(h)
        if parsed.scheme and parsed.scheme.lower() not in ("http", "https", "mailto", ""):
            return None
        return h

    def _safe_url_for_img_src(self, src: str, base_url: Optional[str]) -> Optional[str]:
        if not src or not str(src).strip():
            return None
        s = str(src).strip()
        low = s.lower()
        if low.startswith("javascript:") or low.startswith("vbscript:"):
            return None
        if s.startswith("//") and base_url:
            s = urljoin(base_url, s)
        elif base_url and not urlparse(s).netloc:
            s = urljoin(base_url, s)
        parsed = urlparse(s)
        scheme = (parsed.scheme or "").lower()
        if scheme in ("http", "https", ""):
            return s
        if scheme == "data" and low.startswith("data:image/"):
            return s
        return None

    def _sanitize_srcset(self, srcset: str, base_url: Optional[str]) -> Optional[str]:
        if not srcset or not str(srcset).strip():
            return None
        parts_out = []
        for chunk in str(srcset).split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            url_part = chunk.split()[0] if chunk.split() else chunk
            fixed = self._safe_url_for_img_src(url_part, base_url)
            if not fixed:
                continue
            rest = chunk[len(url_part) :].strip()
            parts_out.append(f"{fixed} {rest}".strip() if rest else fixed)
        if not parts_out:
            return None
        return ", ".join(parts_out)

    def _crawl_images_to_dicts(
        self, raw_images: Optional[List[Any]], limit: int = 20
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not raw_images:
            return out
        for img in raw_images[:limit]:
            if not isinstance(img, dict):
                continue
            out.append(
                {
                    "src": img.get("src") or img.get("url"),
                    "alt": img.get("alt", ""),
                    "title": img.get("title", ""),
                    "width": img.get("width"),
                    "height": img.get("height"),
                    "caption": img.get("caption", ""),
                    "position": img.get("position", "inline"),
                    "type": img.get("type", "content"),
                }
            )
        return out

    @staticmethod
    def _rss_src_looks_like_ad_or_tracking(src: str) -> bool:
        if not src or not str(src).strip():
            return False
        sl = str(src).lower()
        return any(m in sl for m in _RSS_AD_IMG_URL_MARKERS)

    @staticmethod
    def _rss_img_dimensions_tracking_pixel(img) -> bool:
        """Drop images whose width/height attrs look like tracking pixels."""
        for dim_name in ("width", "height"):
            v = img.get(dim_name)
            if v is None:
                continue
            s = str(v).strip().lower().rstrip("px")
            try:
                n = int(float(s))
            except ValueError:
                continue
            if n <= 5:
                return True
        return False

    def _sanitize_article_html(self, html: str, base_url: Optional[str] = None) -> str:
        """Strip scripts/embedded media risks, resolve relative image URLs, keep article images and block structure."""
        if not html:
            return ""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")

            for tag in soup.find_all(["script", "style", "iframe", "object", "embed", "video", "svg"]):
                tag.decompose()
            for tag in list(soup.find_all("noscript")):
                for img in list(tag.find_all("img")):
                    tag.insert_before(img.extract())
                tag.decompose()

            for tag in list(soup.find_all(True)):
                classes = " ".join(tag.get("class") or [])
                elem_id = tag.get("id") or ""
                bucket = f"{classes} {elem_id}".lower()
                if any(k in bucket for k in _RSS_NON_ARTICLE_BUCKET_KEYS):
                    tag.decompose()

            for p in list(soup.find_all("p")):
                text = (p.get_text() or "").strip().lower()
                if (
                    "appeared first on" in text
                    or text.startswith("the post ")
                    or "advertisement" in text
                    or "paid content" in text
                ):
                    p.decompose()

            for tag in list(soup.find_all(True)):
                for attr in list(tag.attrs.keys()):
                    al = attr.lower()
                    val = tag.get(attr)
                    val_s = str(val) if val is not None else ""
                    if al.startswith("on") or al == "style":
                        del tag.attrs[attr]
                    elif "javascript:" in val_s.lower() or "vbscript:" in val_s.lower():
                        del tag.attrs[attr]

            for a in list(soup.find_all("a")):
                href = a.get("href")
                fixed = self._safe_url_for_href(href, base_url) if href else None
                if fixed:
                    a["href"] = fixed
                    a["target"] = "_blank"
                    a["rel"] = "noopener noreferrer"
                elif href:
                    a.unwrap()

            _lazy_src_attrs = (
                "data-src",
                "data-lazy-src",
                "data-original",
                "data-lazyload",
                "data-url",
                "data-image",
                "data-lazy",
            )
            img_count_before = len(soup.find_all("img"))
            for img in list(soup.find_all("img")):
                src0 = (img.get("src") or "").strip()
                src0_l = src0.lower()
                needs_real_src = (
                    not src0
                    or src0_l.startswith("data:")
                    or "about:blank" in src0_l
                )
                if needs_real_src:
                    for attr in _lazy_src_attrs:
                        altv = img.get(attr)
                        if not altv or not str(altv).strip():
                            continue
                        raw = str(altv).strip().split(",")[0].strip().split()[0]
                        fixed_ls = self._safe_url_for_img_src(raw, base_url)
                        if fixed_ls:
                            img["src"] = fixed_ls
                            break
                    dss = img.get("data-srcset")
                    if dss and not img.get("srcset"):
                        ss = self._sanitize_srcset(dss, base_url)
                        if ss:
                            img["srcset"] = ss

            for img in list(soup.find_all("img")):
                src = img.get("src")
                fixed = self._safe_url_for_img_src(src, base_url) if src else None
                if fixed:
                    if self._rss_src_looks_like_ad_or_tracking(fixed):
                        img.decompose()
                        continue
                    if self._rss_img_dimensions_tracking_pixel(img):
                        img.decompose()
                        continue
                    img["src"] = fixed
                    if img.get("srcset"):
                        ss = self._sanitize_srcset(img["srcset"], base_url)
                        if ss:
                            img["srcset"] = ss
                        else:
                            del img["srcset"]
                else:
                    img.decompose()

            img_count_after = len(soup.find_all("img"))
            if img_count_before > 0:
                logger.info(
                    "RSS sanitize: images %d -> %d for %s",
                    img_count_before,
                    img_count_after,
                    (base_url or "unknown")[:80],
                )

            for src_tag in list(soup.find_all("source")):
                if src_tag.get("srcset"):
                    ss = self._sanitize_srcset(src_tag["srcset"], base_url)
                    if ss:
                        src_tag["srcset"] = ss
                    else:
                        src_tag.decompose()

            body = soup.find("body")
            if body:
                inner = "".join(str(c) for c in body.children)
                return inner.strip() if inner.strip() else str(soup)
            return str(soup)
        except Exception:
            import re as _re

            cleaned = _re.sub(
                r"<\s*(script|style|iframe|object|embed|video|svg)[^>]*>.*?</\s*\1\s*>",
                " ",
                html,
                flags=_re.IGNORECASE | _re.DOTALL,
            )
            cleaned = _re.sub(
                r"<\s*(script|style|iframe|object|embed|video|svg)[^>]*/>",
                " ",
                cleaned,
                flags=_re.IGNORECASE,
            )
            cleaned = _re.sub(r"\bon\w+\s*=\s*[^>]+", "", cleaned, flags=_re.IGNORECASE)
            return cleaned

    def _html_to_plain_text(self, html: str) -> str:
        if not html:
            return ""
        if "<" not in html:
            return html.strip()
        try:
            from bs4 import BeautifulSoup, NavigableString

            soup = BeautifulSoup(html, "html.parser")
            for tag in soup.find_all(
                ["script", "style", "iframe", "object", "embed", "video", "svg"]
            ):
                tag.decompose()
            for tag in list(soup.find_all(True)):
                classes = " ".join(tag.get("class") or [])
                elem_id = tag.get("id") or ""
                bucket = f"{classes} {elem_id}".lower()
                if any(
                    k in bucket
                    for k in [
                        "adcovery",
                        "ad-container",
                        "advert",
                        "advertisement",
                        "sponsored",
                        "paid",
                    ]
                ):
                    tag.decompose()
            for p in list(soup.find_all("p")):
                text = (p.get_text() or "").strip()
                tl = text.lower()
                if (
                    "appeared first on" in tl
                    or tl.startswith("the post ")
                    or "advertisement" in tl
                    or "paid content" in tl
                ):
                    p.decompose()
            for img in list(soup.find_all("img")):
                alt = (img.get("alt") or "").strip()
                img.replace_with(
                    NavigableString(f"\n[Image: {alt}]\n" if alt else "\n[Image]\n")
                )
            for br in list(soup.find_all("br")):
                br.replace_with(NavigableString("\n"))
            for block in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote"]):
                block.append(NavigableString("\n\n"))
            for li in list(soup.find_all("li")):
                li.insert(0, NavigableString("\n"))
                li.append(NavigableString("\n"))
            text = soup.get_text()
        except Exception:
            import re as _re

            text = _re.sub(
                r"<\s*(script|style)[^>]*>.*?</\s*(script|style)\s*>",
                " ",
                html,
                flags=_re.IGNORECASE | _re.DOTALL,
            )
            text = _re.sub(r"<[^>]+>", " ", text)
            text = _re.sub(r"\battachment-post[^\s]*", " ", text, flags=_re.IGNORECASE)

        import re as _re

        text = _re.sub(r'\{[^}]*"client_callback_domain"[^}]*\}', " ", text, flags=_re.IGNORECASE)
        text = _re.sub(r'\{[^}]*"widget_type"[^}]*\}', " ", text, flags=_re.IGNORECASE)
        text = _re.sub(r'\{[^}]*"publisher_website_id"[^}]*\}', " ", text, flags=_re.IGNORECASE)
        text = _re.sub(r'\{[^}]*"target_selector"[^}]*\}', " ", text, flags=_re.IGNORECASE)
        text = _re.sub(r'\{[^}]*"widget_div_id"[^}]*\}', " ", text, flags=_re.IGNORECASE)
        text = _re.sub(r'\{[^}]*"adcovery"[^}]*\}', " ", text, flags=_re.IGNORECASE)
        text = _re.sub(r'\{[^}]*ruamupr\.com[^}]*\}', " ", text, flags=_re.IGNORECASE)
        text = _re.sub(r'\{[^}]*doubleclick[^}]*\}', " ", text, flags=_re.IGNORECASE)
        text = _re.sub(r'\{[^}]*googlesyndication[^}]*\}', " ", text, flags=_re.IGNORECASE)
        text = _re.sub(r'\{[^}]*adsystem[^}]*\}', " ", text, flags=_re.IGNORECASE)
        text = _re.sub(r"ruamupr\.com\S*", " ", text, flags=_re.IGNORECASE)
        text = _re.sub(r"adcovery[^\s]*", " ", text, flags=_re.IGNORECASE)
        text = _re.sub(r"\bADVERTISEMENT\b", " ", text, flags=_re.IGNORECASE)
        text = _re.sub(r"(?:Paid\s*Content\s*:*)+", " ", text, flags=_re.IGNORECASE)

        chunks = [p.strip() for p in _re.split(r"\n\s*\n", text) if p.strip()]
        if chunks:
            joined = "\n\n".join(_re.sub(r"[ \t\r]+", " ", c) for c in chunks)
        else:
            joined = _re.sub(r"[ \t\r]+", " ", text).strip()
        return joined.strip()
