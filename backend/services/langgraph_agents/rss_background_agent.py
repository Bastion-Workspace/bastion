"""
RSS Background Agent
Handles background RSS feed polling and article processing
"""

import logging
import asyncio
import hashlib
import re
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import aiohttp
import feedparser

from services.langgraph_agents.base_agent import BaseAgent
from models.rss_models import RSSFeed, RSSArticle, RSSFeedPollResult
from models.agent_response_models import RSSManagementResult

logger = logging.getLogger(__name__)


class RSSBackgroundAgent(BaseAgent):
    """
    RSS Background Agent for feed monitoring and article processing
    
    **BULLY!** This agent handles RSS feed polling, article discovery,
    and structured processing following LangGraph best practices!
    """
    
    def __init__(self):
        super().__init__("rss_background_agent")
        self.system_prompt = self._build_system_prompt()
        # **ROOSEVELT FIX**: Cache service container to avoid repeated async calls
        self._service_container = None
        self._current_user_id = None
    
    def _build_system_prompt(self) -> str:
        """
        Build the system prompt for RSS processing
        
        **NOTE**: This agent doesn't actually use LLM inference - it does direct RSS parsing.
        The system prompt is kept for consistency with BaseAgent but isn't actively used.
        This is deterministic feed parsing, not semantic reasoning.
        """
        return """
RSS Background Agent for automated feed monitoring.
This agent performs direct RSS parsing without LLM inference.
"""
    
    async def _get_rss_service(self):
        """
        Get RSS service with caching to avoid repeated service container lookups.
        
        **ROOSEVELT FIX**: This eliminates 10+ redundant service container calls throughout the agent.
        """
        if not self._service_container:
            from services.service_container import get_service_container
            self._service_container = await get_service_container()
        return self._service_container.rss_service
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process RSS feed polling and article processing
        
        **Trust busting for unstructured RSS processing!** We use structured outputs only!
        """
        try:
            logger.info(f"🤖 RSS AGENT: Starting RSS feed processing")
            
            # Extract RSS processing parameters from state
            feeds_to_poll = state.get("feeds_to_poll", [])
            user_id = state.get("user_id")
            force_poll = state.get("force_poll", False)
            
            logger.info(f"📡 RSS AGENT: Received state - feeds_to_poll: {feeds_to_poll}, user_id: {user_id}, force_poll: {force_poll}")
            
            # Store user_id for feed filtering
            self._current_user_id = user_id
            
            if not feeds_to_poll:
                # Get all active feeds that need polling
                logger.info(f"📡 RSS AGENT: No feeds provided, getting feeds needing poll")
                feeds_to_poll = await self._get_feeds_needing_poll()
                logger.info(f"📡 RSS AGENT: Found {len(feeds_to_poll)} feeds needing poll")
            else:
                # Normalize provided feed references (IDs/dicts) into RSSFeed objects
                logger.info(f"📡 RSS AGENT: Normalizing {len(feeds_to_poll)} feed references")
                normalized_feeds: List[RSSFeed] = []
                for feed_ref in feeds_to_poll:
                    logger.info(f"📡 RSS AGENT: Resolving feed reference: {feed_ref}")
                    feed_obj = await self._resolve_feed_reference(feed_ref)
                    if feed_obj is not None:
                        normalized_feeds.append(feed_obj)
                        logger.info(f"📡 RSS AGENT: Successfully resolved feed: {feed_obj.feed_id} - {feed_obj.feed_name}")
                    else:
                        logger.error(f"📡 RSS AGENT: Failed to resolve feed reference: {feed_ref}")
                feeds_to_poll = normalized_feeds
                logger.info(f"📡 RSS AGENT: Successfully normalized {len(feeds_to_poll)} feeds")
            
            if not feeds_to_poll:
                return self._create_response(
                    task_status="complete",
                    response="No RSS feeds require polling at this time",
                    metadata={"feeds_polled": 0, "articles_found": 0, "articles_added": 0}
                )
            
            # Process RSS feeds in parallel for maximum efficiency
            start_time = datetime.utcnow()
            logger.info(f"📡 RSS AGENT: Polling {len(feeds_to_poll)} feeds in parallel")
            
            # **ROOSEVELT FIX**: Create concurrent tasks with timeout protection (5 min per feed)
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
                    logger.error(f"❌ RSS AGENT ERROR: {error_msg}")
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
            response = self._create_response(
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
            
            logger.info(f"✅ RSS AGENT: Completed parallel processing of {len(feeds_to_poll)} feeds in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"❌ RSS AGENT ERROR: {e}")
            return self._create_response(
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
                # **ROOSEVELT FIX**: Use cached service instead of repeated lookups
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
            # **ROOSEVELT FIX**: Use cached service instead of repeated lookups
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
        
        **ROOSEVELT FIX**: Prevents attempting to fetch invalid URLs that would fail anyway.
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
        
        **ROOSEVELT FIX**: Prevents hung feeds from blocking the entire polling cycle.
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
        # **ROOSEVELT FIX**: Get cached RSS service once at the start
        rss_service = await self._get_rss_service()
        
        try:
            logger.debug(f"📡 RSS AGENT: Polling feed {feed.feed_id}")
            
            # **ROOSEVELT FIX**: Validate URL before attempting to fetch
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
                    # Create RSS article
                    article = RSSArticle(
                        article_id=self._generate_article_id(entry.link),
                        feed_id=feed.feed_id,
                        title=entry.title,
                        description=getattr(entry, 'description', None) or getattr(entry, 'summary', None),
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
                    
                    # Clean description HTML before storage
                    if article.description:
                        try:
                            article.description = self._html_to_plain_text(self._clean_html_remove_media_and_boilerplate(article.description))
                        except Exception:
                            pass

                    # Check if content is truncated and extract full content if needed
                    if self._is_content_truncated(article.description):
                        logger.debug(f"🕷️ RSS AGENT: Detected truncated content for {article.title}, extracting full content")
                        full_content, full_content_html, images = await self._extract_full_content_with_crawl4ai(article.link)
                        if full_content:
                            # Clean both full text and HTML before saving
                            article.full_content = self._html_to_plain_text(full_content)
                            if full_content_html:
                                article.full_content_html = self._clean_html_remove_media_and_boilerplate(full_content_html)
                            if images:
                                article.images = images
                            logger.debug(f"✅ RSS AGENT: Successfully extracted full content for {article.title}")
                    
                    # Save article to database
                    save_success = await self._save_article(article)
                    if save_success:
                        articles_added += 1
                        # Also upsert a simple News headline so it surfaces in News API
                        try:
                            await self._upsert_news_from_rss(article, feed)
                        except Exception as news_e:
                            logger.warning(f"⚠️ RSS AGENT: Failed to upsert news from RSS article {article.article_id}: {news_e}")
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
            # **ROOSEVELT FIX**: Don't cleanup here - let finally block handle it to avoid double cleanup
            return RSSFeedPollResult(
                feed_id=feed.feed_id,
                status="error",
                error_message=str(e)
            )
        finally:
            # **ROOSEVELT FIX**: Cleanup polling status in finally block (runs on success AND error)
            # This was previously done in both except and finally, causing redundant calls
            try:
                await rss_service.mark_feed_polling(feed.feed_id, is_polling=False)
            except Exception as cleanup_error:
                logger.error(f"❌ RSS AGENT ERROR: Failed to cleanup polling status for {feed.feed_id}: {cleanup_error}")

    async def _upsert_news_from_rss(self, article: "RSSArticle", feed: "RSSFeed") -> None:
        """Create a minimal News article from an RSS article and upsert into NewsService."""
        try:
            from services.service_container import get_service_container
            from models.news_models import NewsArticleSynth, NewsSourceRef

            service_container = await get_service_container()
            news_service = getattr(service_container, 'news_service', None)
            if not news_service:
                return

            # Build a conservative lede and body
            lede = (article.description or article.full_content or "").strip()
            if len(lede) > 280:
                lede = (lede[:277] + "...").strip()

            # Ensure we have robust full content: fetch if missing/too short
            try:
                need_full = not article.full_content or len(article.full_content.strip()) < 800
                if need_full:
                    fc_text, fc_html, fc_images = await self._extract_full_content_with_crawl4ai(article.link)
                    if fc_text or fc_html:
                        article.full_content = self._html_to_plain_text(fc_text or fc_html or "")
                        if fc_html:
                            article.full_content_html = self._clean_html_remove_media_and_boilerplate(fc_html)
                        if fc_images:
                            article.images = fc_images
            except Exception:
                pass

            # Enforce freshness before News upsert (skip very old items)
            try:
                from services.settings_service import settings_service
                if not getattr(settings_service, "_initialized", False):
                    await settings_service.initialize()
                recency_minutes = int(await settings_service.get_setting("news.recency_minutes", 60))
                if article.published_date:
                    age = (datetime.utcnow() - article.published_date).total_seconds() / 60.0
                    if age > recency_minutes:
                        logger.info(f"⏱️ Skipping News upsert for stale article {article.article_id} (> {recency_minutes} min)")
                        # Still save RSS article but skip News upsert
                        # Fall through without building body; balanced_body unused when skipping
                        pass
            except Exception:
                pass

            # Build a conservative body by stripping common footer boilerplate and images
            def _strip_boilerplate(text: str) -> str:
                if not text:
                    return ""
                import re as _re
                t = text
                # Remove 'The post ... appeared first on ...' and variants
                t = _re.sub(r"\bThe post\s+.*?\s+appeared first on\s+.*$", "", t, flags=_re.IGNORECASE)
                t = _re.sub(r"\s*appeared first on\s+.*$", "", t, flags=_re.IGNORECASE)
                # Remove leading/trailing inline image tag fragments
                t = _re.sub(r"^\s*<img[^>]*>\s*", "", t, flags=_re.IGNORECASE)
                t = _re.sub(r"^\s*<[^>]*attachment-post[^>]*>\s*", "", t, flags=_re.IGNORECASE)
                return t.strip()

            # Prefer full content if present after extraction
            body_source = article.full_content or article.description or ""
            balanced_body = _strip_boilerplate(body_source.strip()) or article.title

            citation = NewsSourceRef(
                name=feed.feed_name,
                url=article.link,
                published_at=article.published_date.isoformat() if article.published_date else None,
            )

            # Compute severity (single-source → likely normal)
            severity = news_service.compute_severity([citation], [article.published_date] if article.published_date else [])

            news = NewsArticleSynth(
                id=article.article_id,
                title=article.title,
                lede=lede or article.title,
                balanced_body=balanced_body,
                key_points=[],
                citations=[citation],
                diversity_score=0.0,
                severity=severity,
                images=article.images,  # **BULLY!** Pass through extracted images
            )

            await news_service.upsert_article(news)
        except Exception as e:
            # Log and continue; failure here should not impact RSS ingest
            logger.warning(f"⚠️ Failed to upsert News from RSS: {e}")
    
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
    
    async def _is_duplicate_article(self, article: RSSArticle) -> bool:
        """Check if article is a duplicate based on content hash"""
        try:
            # **ROOSEVELT FIX**: Use cached service instead of repeated lookups
            rss_service = await self._get_rss_service()
            return await rss_service.is_duplicate_article(article)
        except Exception as e:
            logger.error(f"❌ RSS AGENT ERROR: Failed to check duplicate: {e}")
            return False
    
    async def _save_article(self, article: RSSArticle) -> bool:
        """Save RSS article to database"""
        try:
            # **ROOSEVELT FIX**: Use cached service instead of repeated lookups
            rss_service = await self._get_rss_service()
            return await rss_service.save_article(article)
        except Exception as e:
            logger.error(f"❌ RSS AGENT ERROR: Failed to save article: {e}")
            return False
    
    async def _update_feed_last_check(self, feed_id: str) -> bool:
        """Update feed last_check timestamp"""
        try:
            # **ROOSEVELT FIX**: Use cached service instead of repeated lookups
            rss_service = await self._get_rss_service()
            return await rss_service.update_feed_last_check(feed_id)
        except Exception as e:
            logger.error(f"❌ RSS AGENT ERROR: Failed to update feed last_check: {e}")
            return False
    
    def _is_content_truncated(self, description: Optional[str]) -> bool:
        """Detect if RSS content is truncated and needs full extraction"""
        if not description:
            return False
        
        # Common truncation indicators
        truncation_patterns = [
            r'\.\.\.read more',
            r'\.\.\.read more</a>',
            r'\.\.\.continue reading',
            r'\.\.\.more',
            r'\.\.\.',
            r'read more',
            r'continue reading',
            r'full story',
            r'full article'
        ]
        
        description_lower = description.lower()
        
        # Check for truncation patterns
        for pattern in truncation_patterns:
            if re.search(pattern, description_lower):
                return True
        
        # Check if content is suspiciously short (likely truncated)
        if len(description.strip()) < 200:
            return True
        
        return False
    
    async def _extract_full_content_with_crawl4ai(self, url: str) -> tuple[Optional[str], Optional[str], Optional[List[Dict[str, Any]]]]:
        """Extract full article content (clean text + HTML) and images using Crawl4AI"""
        try:
            logger.info(f"🕷️ RSS AGENT: Extracting full content from {url}")
            
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
                    # Try to get clean extracted content from content_blocks first
                    content_blocks = crawl_result.get("content_blocks", [])
                    if content_blocks:
                        # Extract text from content blocks
                        extracted_text = ""
                        for block in content_blocks:
                            if isinstance(block, dict) and "content" in block:
                                extracted_text += block["content"] + "\n\n"
                            elif isinstance(block, str):
                                extracted_text += block + "\n\n"
                        
                        if extracted_text.strip():
                            # Clean up the content for text storage
                            cleaned_content = self._clean_extracted_content(extracted_text)
                            
                            # Get original HTML content for display
                            original_html = crawl_result.get("full_content", "")
                            
                            # Extract and enhance images from the crawl result
                            raw_images = crawl_result.get("images", [])
                            enhanced_images = []
                            
                            for img in raw_images:
                                enhanced_image = {
                                    "src": img.get("src") or img.get("url"),
                                    "alt": img.get("alt", ""),
                                    "title": img.get("title", ""),
                                    "width": img.get("width"),
                                    "height": img.get("height"),
                                    "caption": img.get("caption", ""),
                                    "position": img.get("position", "inline"),  # inline, header, sidebar, etc.
                                    "type": img.get("type", "content")  # content, advertisement, logo, etc.
                                }
                                enhanced_images.append(enhanced_image)
                            
                            logger.info(f"✅ RSS AGENT: Successfully extracted {len(cleaned_content)} characters, {len(original_html)} HTML chars, and {len(enhanced_images)} images from {url}")
                            return cleaned_content, original_html, enhanced_images
                    
                    # Use universal content extractor for better results
                    if crawl_result.get("full_content"):
                        content = crawl_result["full_content"]
                        
                        # Use universal content extractor
                        from services.universal_content_extractor import get_universal_content_extractor
                        universal_extractor = await get_universal_content_extractor()
                        
                        cleaned_content, original_html, enhanced_images = await universal_extractor.extract_main_content(content, url)
                        
                        # If universal extractor didn't work well, fall back to our enhanced cleaning
                        if not cleaned_content or len(cleaned_content.strip()) < 100:
                            logger.warning(f"⚠️ Universal extractor produced insufficient content, falling back to enhanced cleaning")
                            cleaned_content = self._clean_extracted_content(content)
                            original_html = crawl_result.get("full_content", "")
                        
                        # Only use crawl result images if universal extractor found none
                        if not enhanced_images:
                            raw_images = crawl_result.get("images", [])
                            if raw_images:
                                enhanced_images = []
                                for img in raw_images:
                                    enhanced_image = {
                                        "src": img.get("src") or img.get("url"),
                                        "alt": img.get("alt", ""),
                                        "title": img.get("title", ""),
                                        "width": img.get("width"),
                                        "height": img.get("height"),
                                        "caption": img.get("caption", ""),
                                        "position": img.get("position", "inline"),
                                        "type": img.get("type", "content")
                                    }
                                    enhanced_images.append(enhanced_image)
                                logger.info(f"📸 Using {len(enhanced_images)} images from crawl result as fallback")
                        
                        logger.info(f"✅ RSS AGENT: Successfully extracted {len(cleaned_content)} characters, {len(original_html)} HTML chars, and {len(enhanced_images)} images from {url} (universal extractor)")
                        return cleaned_content, original_html, enhanced_images
                    
                    logger.warning(f"⚠️ RSS AGENT: No content found in Crawl4AI result for {url}")
                else:
                    logger.warning(f"⚠️ RSS AGENT: Crawl4AI extraction failed for {url}: {crawl_result.get('error', 'Unknown error')}")
            else:
                logger.warning(f"⚠️ RSS AGENT: No results from Crawl4AI for {url}")
            
            # **ROOSEVELT FIX**: Must return 3 values (text, html, images) - was missing images
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
        
        # Limit content length to prevent database issues
        if len(content) > 50000:
            content = content[:50000] + "..."
        
        return content
    
    def _create_response(self, task_status: str, response: str, metadata: Dict[str, Any], feed_results: Optional[List[RSSFeedPollResult]] = None) -> Dict[str, Any]:
        """Create structured agent response"""
        return {
            "task_status": task_status,
            "response": response,
            "metadata": metadata,
            "feed_results": [result.dict() for result in feed_results] if feed_results else [],
            "timestamp": datetime.utcnow().isoformat()
        }

    def _html_to_plain_text(self, html: str) -> str:
        if not html:
            return ""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            # Remove media and decorative containers
            for tag in soup.find_all(['img', 'figure', 'figcaption', 'picture', 'source', 'svg', 'video', 'iframe']):
                tag.decompose()
            # Remove scripts/styles
            for tag in soup.find_all(['script', 'style']):
                tag.decompose()
            # Remove obvious ad/paid content containers by class/id
            for tag in soup.find_all(True):
                classes = ' '.join(tag.get('class') or [])
                elem_id = tag.get('id') or ''
                bucket = f"{classes} {elem_id}".lower()
                if any(k in bucket for k in ['adcovery', 'ad-container', 'advert', 'advertisement', 'sponsored', 'paid']):
                    tag.decompose()
            # Remove boilerplate paragraphs like "The post ... appeared first on ..."
            for p in soup.find_all('p'):
                text = (p.get_text() or '').strip()
                tl = text.lower()
                if 'appeared first on' in tl or tl.startswith('the post ') or 'advertisement' in tl or 'paid content' in tl:
                    p.decompose()
            text = soup.get_text(separator=' ')
        except Exception:
            import re as _re
            # Fallback regex stripping
            text = _re.sub(r"<(img|figure|figcaption|picture|source|svg|video|iframe)[^>]*>", " ", html, flags=_re.IGNORECASE)
            text = _re.sub(r"</(figure|figcaption|picture|svg|video|iframe)>", " ", text, flags=_re.IGNORECASE)
            text = _re.sub(r"<\s*(script|style)[^>]*>.*?</\s*(script|style)\s*>", " ", text, flags=_re.IGNORECASE|_re.DOTALL)
            text = _re.sub(r"<[^>]+>", " ", text)
            text = _re.sub(r"\battachment-post[^\s]*", " ", text, flags=_re.IGNORECASE)
        import re as _re
        # **BULLY!** Remove advertisement JSON widget configurations comprehensively
        text = _re.sub(r'\{[^}]*"client_callback_domain"[^}]*\}', ' ', text, flags=_re.IGNORECASE)
        text = _re.sub(r'\{[^}]*"widget_type"[^}]*\}', ' ', text, flags=_re.IGNORECASE)
        text = _re.sub(r'\{[^}]*"publisher_website_id"[^}]*\}', ' ', text, flags=_re.IGNORECASE)
        text = _re.sub(r'\{[^}]*"target_selector"[^}]*\}', ' ', text, flags=_re.IGNORECASE)
        text = _re.sub(r'\{[^}]*"widget_div_id"[^}]*\}', ' ', text, flags=_re.IGNORECASE)
        text = _re.sub(r'\{[^}]*"adcovery"[^}]*\}', ' ', text, flags=_re.IGNORECASE)
        
        # Remove any JSON objects containing common ad network domains
        text = _re.sub(r'\{[^}]*ruamupr\.com[^}]*\}', ' ', text, flags=_re.IGNORECASE)
        text = _re.sub(r'\{[^}]*doubleclick[^}]*\}', ' ', text, flags=_re.IGNORECASE)
        text = _re.sub(r'\{[^}]*googlesyndication[^}]*\}', ' ', text, flags=_re.IGNORECASE)
        text = _re.sub(r'\{[^}]*adsystem[^}]*\}', ' ', text, flags=_re.IGNORECASE)
        
        # Remove leftover ad domain references and general ad markers
        text = _re.sub(r"ruamupr\.com\S*", " ", text, flags=_re.IGNORECASE)
        text = _re.sub(r"adcovery[^\s]*", " ", text, flags=_re.IGNORECASE)
        text = _re.sub(r"\bADVERTISEMENT\b", " ", text, flags=_re.IGNORECASE)
        text = _re.sub(r"(?:Paid\s*Content\s*:*)+", " ", text, flags=_re.IGNORECASE)
        text = _re.sub(r"\s+", " ", text).strip()
        return text

    def _clean_html_remove_media_and_boilerplate(self, html: str) -> str:
        if not html:
            return ""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            # Remove media and decorative containers
            for tag in soup.find_all(['img', 'figure', 'figcaption', 'picture', 'source', 'svg', 'video', 'iframe']):
                tag.decompose()
            # Remove scripts/styles
            for tag in soup.find_all(['script', 'style']):
                tag.decompose()
            # Remove webfeedsFeaturedVisual or attachment-post blocks
            for tag in soup.find_all(attrs={"class": True}):
                cls = ' '.join(tag.get('class') or [])
                if 'webfeedsFeaturedVisual' in cls or 'attachment-post' in cls:
                    tag.decompose()
            # Remove common ad/sponsored containers by class/id keywords
            for tag in soup.find_all(True):
                classes = ' '.join(tag.get('class') or [])
                elem_id = tag.get('id') or ''
                bucket = f"{classes} {elem_id}".lower()
                if any(k in bucket for k in ['adcovery', 'ad-container', 'advert', 'advertisement', 'sponsored', 'paid']):
                    tag.decompose()
            # Remove boilerplate paragraphs like "The post ... appeared first on ..."
            for p in soup.find_all('p'):
                text = (p.get_text() or '').strip().lower()
                if 'appeared first on' in text or text.startswith('the post ') or 'advertisement' in text or 'paid content' in text:
                    p.decompose()
            return str(soup)
        except Exception:
            import re as _re
            cleaned = _re.sub(r"<(img|figure|figcaption|picture|source|svg|video|iframe)[^>]*>", " ", html, flags=_re.IGNORECASE)
            cleaned = _re.sub(r"</(figure|figcaption|picture|svg|video|iframe)>", " ", cleaned, flags=_re.IGNORECASE)
            cleaned = _re.sub(r"<[^>]*attachment-post[^>]*>", " ", cleaned, flags=_re.IGNORECASE)
            cleaned = _re.sub(r"<\s*(script|style)[^>]*>.*?</\s*(script|style)\s*>", " ", cleaned, flags=_re.IGNORECASE|_re.DOTALL)
            cleaned = _re.sub(r"\bThe post\s+.*?\s+appeared first on\s+.*$", "", cleaned, flags=_re.IGNORECASE)
            
            # **BULLY!** Clean JSON ad widgets from fallback HTML cleaning too
            cleaned = _re.sub(r'\{[^}]*"client_callback_domain"[^}]*\}', ' ', cleaned, flags=_re.IGNORECASE)
            cleaned = _re.sub(r'\{[^}]*"widget_type"[^}]*\}', ' ', cleaned, flags=_re.IGNORECASE)
            cleaned = _re.sub(r'\{[^}]*"publisher_website_id"[^}]*\}', ' ', cleaned, flags=_re.IGNORECASE)
            cleaned = _re.sub(r'\{[^}]*"target_selector"[^}]*\}', ' ', cleaned, flags=_re.IGNORECASE)
            cleaned = _re.sub(r'\{[^}]*"widget_div_id"[^}]*\}', ' ', cleaned, flags=_re.IGNORECASE)
            cleaned = _re.sub(r'\{[^}]*ruamupr\.com[^}]*\}', ' ', cleaned, flags=_re.IGNORECASE)
            cleaned = _re.sub(r"ruamupr\.com\S*", " ", cleaned, flags=_re.IGNORECASE)
            cleaned = _re.sub(r"adcovery[^\s]*", " ", cleaned, flags=_re.IGNORECASE)
            cleaned = _re.sub(r"\bADVERTISEMENT\b", " ", cleaned, flags=_re.IGNORECASE)
            cleaned = _re.sub(r"(?:Paid\s*Content\s*:*)+", " ", cleaned, flags=_re.IGNORECASE)
            return cleaned
