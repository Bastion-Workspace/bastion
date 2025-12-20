"""
Crawl4AI gRPC Service Implementation
"""

import logging
import time
import asyncio
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler

# Try to import AdaptiveCrawler - the proper API for adaptive crawling
try:
    from crawl4ai import AdaptiveCrawler, AdaptiveConfig
except ImportError:
    try:
        from crawl4ai.adaptive import AdaptiveCrawler, AdaptiveConfig
    except ImportError:
        AdaptiveCrawler = None
        AdaptiveConfig = None

from config.settings import settings
from service.crawl_strategies import get_extraction_strategy

logger = logging.getLogger(__name__)


class CrawlServiceImplementation:
    """gRPC service implementation for Crawl4AI"""
    
    def __init__(self):
        self.crawler: Optional[AsyncWebCrawler] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the crawler"""
        try:
            if not self._initialized:
                logger.info("Initializing Crawl4AI service...")
                self.crawler = AsyncWebCrawler(
                    headless=settings.HEADLESS
                )
                await self.crawler.__aenter__()
                self._initialized = True
                logger.info("Crawl4AI service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Crawl4AI: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup crawler resources"""
        if self.crawler and self._initialized:
            try:
                await self.crawler.__aexit__(None, None, None)
                self._initialized = False
                logger.info("Crawl4AI service cleaned up")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
    
    async def crawl(self, request) -> Any:
        """
        Single URL crawl
        
        Args:
            request: CrawlRequest protobuf message
            
        Returns:
            CrawlResponse protobuf message
        """
        from protos import crawl_service_pb2
        
        start_time = time.time()
        
        try:
            if not self._initialized:
                await self.initialize()
            
            logger.info(f"Crawling URL: {request.url} with strategy: {request.extraction_strategy}")
            
            # Use CrawlerRunConfig per new API: https://docs.crawl4ai.com/api/arun/
            from crawl4ai import CrawlerRunConfig, CacheMode
            
            # Prepare extraction strategy if provided
            extraction_strategy = None
            if request.extraction_strategy:
                extraction_strategy = get_extraction_strategy(
                    request.extraction_strategy,
                    request.chunking_strategy if request.chunking_strategy else "RegexChunking",
                    10,  # word_count_threshold
                    request.llm_question if request.llm_question else None
                )
            
            # Create CrawlerRunConfig with valid parameters only
            config = CrawlerRunConfig(
                cache_mode=CacheMode.ENABLED,
                css_selector=request.css_selector if request.css_selector else None,
                word_count_threshold=10,
                page_timeout=(request.timeout_seconds * 1000) if request.timeout_seconds and request.timeout_seconds > 0 else (settings.DEFAULT_TIMEOUT_SECONDS * 1000),
                extraction_strategy=extraction_strategy,
                scan_full_page=True  # Automatically handle infinite scroll pages
            )
            
            # Handle virtual scroll via page interaction strategy if requested
            if request.virtual_scroll:
                try:
                    from crawl4ai.page_interaction_strategy import VirtualScrollStrategy
                    scroll_delay = request.scroll_delay if request.scroll_delay > 0 else 1.0
                    config.page_interaction_strategy = VirtualScrollStrategy(delay=scroll_delay)
                except ImportError:
                    logger.warning("VirtualScrollStrategy not available, virtual scroll disabled")
                    # Fallback: try setting via js_code
                    try:
                        config.js_code = "window.scrollTo(0, document.body.scrollHeight); await new Promise(r => setTimeout(r, 1000));"
                    except:
                        logger.warning("Could not configure virtual scroll")
            
            # Handle fit_markdown if requested
            # Per docs: https://docs.crawl4ai.com/core/fit-markdown/
            # Fit markdown is configured via content_filter with DefaultMarkdownGenerator
            if request.use_fit_markdown:
                try:
                    from crawl4ai.content_filter_strategy import PruningContentFilter
                    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
                    
                    # Create pruning filter for fit markdown
                    prune_filter = PruningContentFilter(
                        threshold=0.48,  # Default threshold - removes low-value content
                        threshold_type="dynamic",  # Adjust based on tag type, text density
                        min_word_threshold=5  # Ignore nodes with <5 words
                    )
                    
                    # Create markdown generator with pruning filter
                    md_generator = DefaultMarkdownGenerator(content_filter=prune_filter)
                    config.markdown_generator = md_generator
                    
                    logger.debug("✅ Fit markdown configured with PruningContentFilter")
                except ImportError as e:
                    logger.warning(f"⚠️ Could not configure fit markdown: {e}, using default markdown")
                    # Don't set any markdown generator - let Crawl4AI use default
            
            # Execute crawl with new API: arun(url, config)
            result = await self.crawler.arun(url=request.url, config=config)
            
            if not result or not result.success:
                error_msg = result.error_message if result and hasattr(result, 'error_message') else "Crawl failed"
                logger.warning(f"Crawl failed for {request.url}: {error_msg}")
                return crawl_service_pb2.CrawlResponse(
                    success=False,
                    url=request.url,
                    error=error_msg
                )
            
            # Extract content based on strategy
            content = ""
            markdown_content = ""
            
            # Per docs: https://docs.crawl4ai.com/core/fit-markdown/
            # When content_filter is used, result.markdown has .fit_markdown and .raw_markdown
            if request.use_fit_markdown and hasattr(result, 'markdown') and hasattr(result.markdown, 'fit_markdown'):
                # Use fit_markdown (filtered, relevant content)
                markdown_content = result.markdown.fit_markdown or result.markdown.raw_markdown or ""
                logger.debug(f"Using fit_markdown: {len(markdown_content)} chars")
            elif hasattr(result, 'markdown'):
                # Use raw markdown or string markdown
                if hasattr(result.markdown, 'raw_markdown'):
                    markdown_content = result.markdown.raw_markdown or ""
                else:
                    markdown_content = result.markdown or ""
            else:
                markdown_content = ""
            
            if request.extraction_strategy == "markdown" or request.use_fit_markdown:
                content = markdown_content
            elif request.extraction_strategy == "text":
                content = result.text or ""
            elif request.extraction_strategy == "html":
                content = result.html or ""
            elif request.extraction_strategy == "llm_extraction" and hasattr(result, 'extracted_content'):
                content = result.extracted_content or markdown_content or ""
            else:
                content = markdown_content or result.text or ""
            
            # Truncate if needed
            max_length = request.max_content_length if request.max_content_length else settings.DEFAULT_MAX_CONTENT_LENGTH
            if len(content) > max_length:
                content = content[:max_length]
                logger.info(f"Content truncated to {max_length} characters")
            
            # Extract metadata
            metadata = {}
            if result.metadata:
                metadata = {
                    "title": str(result.metadata.get("title", "")),
                    "description": str(result.metadata.get("description", "")),
                    "author": str(result.metadata.get("author", "")),
                    "keywords": str(result.metadata.get("keywords", "")),
                    "language": str(result.metadata.get("language", "")),
                    "published_time": str(result.metadata.get("published_time", "")),
                    "modified_time": str(result.metadata.get("modified_time", ""))
                }
            
            # Extract links
            links = []
            if result.links and hasattr(result.links, '__iter__'):
                links = list(result.links)[:50]  # Limit to 50 links
            
            # Extract images
            images = []
            if result.media and isinstance(result.media, dict):
                img_list = result.media.get("images", [])
                if isinstance(img_list, (list, tuple)):
                    images = list(img_list)[:20]  # Limit to 20 images
            
            fetch_time = time.time() - start_time
            
            return crawl_service_pb2.CrawlResponse(
                success=True,
                url=request.url,
                title=metadata.get("title", "No title"),
                content=content,
                markdown=markdown_content,  # Use fit_markdown if available, otherwise raw_markdown
                html=result.html or "",
                metadata=metadata,
                links=links,
                images=images,
                content_length=len(content),
                fetch_time_seconds=fetch_time,
                status_code=result.status_code if hasattr(result, 'status_code') else 200,
                extracted_content=result.extracted_content if hasattr(result, 'extracted_content') else None
            )
            
        except Exception as e:
            logger.error(f"Error crawling {request.url}: {e}")
            return crawl_service_pb2.CrawlResponse(
                success=False,
                url=request.url,
                error=str(e)
            )
    
    async def crawl_many(self, request) -> Any:
        """
        Parallel multi-URL crawl using Crawl4AI's native arun_many()
        
        This uses Crawl4AI's optimized parallel crawling with built-in dispatcher
        management, which is more efficient than manual asyncio.gather()
        
        Args:
            request: CrawlManyRequest protobuf message
            
        Returns:
            CrawlManyResponse protobuf message
        """
        from protos import crawl_service_pb2
        
        start_time = time.time()
        max_concurrent = request.max_concurrent if request.max_concurrent > 0 else settings.MAX_CONCURRENT_CRAWLS
        
        try:
            if not self._initialized:
                await self.initialize()
            
            logger.info(f"Crawling {len(request.urls)} URLs using Crawl4AI arun_many() with max_concurrent={max_concurrent}")
            
            # Use Crawl4AI's native arun_many() with proper CrawlerRunConfig
            # According to docs: https://docs.crawl4ai.com/api/arun_many/
            # All parameters go in CrawlerRunConfig, not as kwargs
            try:
                from crawl4ai import CrawlerRunConfig, CacheMode
                
                # Prepare extraction strategy if provided
                extraction_strategy = None
                if request.extraction_strategy:
                    extraction_strategy = get_extraction_strategy(
                        request.extraction_strategy,
                        request.chunking_strategy or "RegexChunking",
                        10,  # word_count_threshold
                        request.llm_question if request.llm_question else None
                    )
                
                # Create CrawlerRunConfig with valid parameters only
                # Note: virtual_scroll and scroll_delay are not CrawlerRunConfig parameters
                # They may need to be set via page interaction strategies or kwargs
                config = CrawlerRunConfig(
                    # Cache settings
                    cache_mode=CacheMode.ENABLED,
                    
                    # Content selection
                    css_selector=request.css_selector if request.css_selector else None,
                    word_count_threshold=10,
                    
                    # Timeout (in milliseconds)
                    page_timeout=(request.timeout_seconds * 1000) if request.timeout_seconds and request.timeout_seconds > 0 else None,
                    
                    # Extraction strategy
                    extraction_strategy=extraction_strategy,
                    
                    # Automatic infinite scroll handling
                    scan_full_page=True  # Automatically handle infinite scroll pages
                )
                
                # Handle virtual scroll via page interaction if requested
                # Virtual scroll is typically handled via page interaction strategies
                if request.virtual_scroll:
                    try:
                        from crawl4ai.page_interaction_strategy import VirtualScrollStrategy
                        scroll_delay = request.scroll_delay if request.scroll_delay > 0 else 1.0
                        config.page_interaction_strategy = VirtualScrollStrategy(delay=scroll_delay)
                    except ImportError:
                        logger.warning("VirtualScrollStrategy not available, virtual scroll disabled")
                        # Fallback: try setting via js_code
                        try:
                            config.js_code = "window.scrollTo(0, document.body.scrollHeight); await new Promise(r => setTimeout(r, 1000));"
                        except:
                            logger.warning("Could not configure virtual scroll")
                
                # Handle fit_markdown if requested
                # Per docs: https://docs.crawl4ai.com/core/fit-markdown/
                # Fit markdown is configured via content_filter with DefaultMarkdownGenerator
                if request.use_fit_markdown:
                    try:
                        from crawl4ai.content_filter_strategy import PruningContentFilter
                        from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
                        
                        # Create pruning filter for fit markdown
                        prune_filter = PruningContentFilter(
                            threshold=0.48,  # Default threshold - removes low-value content
                            threshold_type="dynamic",  # Adjust based on tag type, text density
                            min_word_threshold=5  # Ignore nodes with <5 words
                        )
                        
                        # Create markdown generator with pruning filter
                        md_generator = DefaultMarkdownGenerator(content_filter=prune_filter)
                        config.markdown_generator = md_generator
                        
                        logger.debug("✅ Fit markdown configured with PruningContentFilter")
                    except ImportError as e:
                        logger.warning(f"⚠️ Could not configure fit markdown: {e}, using default markdown")
                        # Don't set any markdown generator - let Crawl4AI use default
                
                # Optional: Use a dispatcher for advanced concurrency control
                dispatcher = None
                try:
                    from crawl4ai.dispatcher import MemoryAdaptiveDispatcher
                    # Use memory-adaptive dispatcher for optimal resource management
                    dispatcher = MemoryAdaptiveDispatcher(
                        memory_threshold_percent=70.0,
                        max_session_permit=max_concurrent
                    )
                except ImportError:
                    # Fallback: dispatcher not available, arun_many will use default
                    logger.debug("MemoryAdaptiveDispatcher not available, using default dispatcher")
                
                # Call native arun_many() - this is the optimized parallel method
                # Per docs: arun_many(urls, config, dispatcher)
                if dispatcher:
                    results = await self.crawler.arun_many(
                        urls=request.urls,
                        config=config,
                        dispatcher=dispatcher
                    )
                else:
                    results = await self.crawler.arun_many(
                        urls=request.urls,
                        config=config
                    )
                
            except AttributeError:
                # Fallback if arun_many doesn't exist in this Crawl4AI version
                logger.warning("arun_many() not available, falling back to manual parallelization")
                results = []
                for url in request.urls:
                    kwargs = {**common_kwargs, "url": url}
                    result = await self.crawler.arun(**kwargs)
                    results.append(result)
            
            # Process results from arun_many()
            crawl_responses = []
            successful = 0
            failed = 0
            total_content_length = 0
            
            for i, result in enumerate(results):
                if not result or not result.success:
                    error_msg = result.error_message if result and hasattr(result, 'error_message') else "Crawl failed"
                    logger.warning(f"Crawl failed for {request.urls[i]}: {error_msg}")
                    crawl_responses.append(
                        crawl_service_pb2.CrawlResponse(
                            success=False,
                            url=request.urls[i],
                            error=error_msg
                        )
                    )
                    failed += 1
                    continue
                
                # Extract content - prefer fit_markdown if available
                # Per docs: https://docs.crawl4ai.com/core/fit-markdown/
                content = ""
                if request.use_fit_markdown and hasattr(result, 'markdown') and hasattr(result.markdown, 'fit_markdown'):
                    content = result.markdown.fit_markdown or result.markdown.raw_markdown or ""
                elif hasattr(result, 'markdown'):
                    if hasattr(result.markdown, 'raw_markdown'):
                        content = result.markdown.raw_markdown or ""
                    else:
                        content = result.markdown or ""
                else:
                    content = result.text or ""
                
                # Truncate if needed
                max_length = request.max_content_length if request.max_content_length else settings.DEFAULT_MAX_CONTENT_LENGTH
                if len(content) > max_length:
                    content = content[:max_length]
                
                # Extract metadata
                metadata = {}
                if result.metadata:
                    metadata = {
                        "title": str(result.metadata.get("title", "")),
                        "description": str(result.metadata.get("description", "")),
                        "author": str(result.metadata.get("author", "")),
                        "keywords": str(result.metadata.get("keywords", "")),
                        "language": str(result.metadata.get("language", "")),
                        "published_time": str(result.metadata.get("published_time", "")),
                        "modified_time": str(result.metadata.get("modified_time", ""))
                    }
                
                # Extract links
                links = []
                if result.links and hasattr(result.links, '__iter__'):
                    links = list(result.links)[:50]
                
                # Extract images
                images = []
                if result.media and isinstance(result.media, dict):
                    img_list = result.media.get("images", [])
                    if isinstance(img_list, (list, tuple)):
                        images = list(img_list)[:20]
                
                # Extract markdown (prefer fit_markdown if available)
                markdown_out = ""
                if hasattr(result, 'markdown'):
                    if hasattr(result.markdown, 'fit_markdown'):
                        markdown_out = str(result.markdown.fit_markdown or result.markdown.raw_markdown or "")
                    elif hasattr(result.markdown, 'raw_markdown'):
                        markdown_out = str(result.markdown.raw_markdown or "")
                    else:
                        markdown_out = str(result.markdown or "")
                
                # Ensure all values are proper types for protobuf
                html_content = str(result.html or "") if hasattr(result, 'html') else ""
                content_str = str(content) if content else ""
                
                # Ensure metadata values are strings
                metadata_dict = {}
                for k, v in metadata.items():
                    metadata_dict[str(k)] = str(v) if v is not None else ""
                
                # Ensure links and images are lists of strings
                links_list = [str(link) for link in links if link]
                images_list = [str(img) for img in images if img]
                
                crawl_responses.append(
                    crawl_service_pb2.CrawlResponse(
                        success=True,
                        url=str(request.urls[i]),
                        title=str(metadata.get("title", "No title")),
                        content=content_str,
                        markdown=str(markdown_out),
                        html=html_content,
                        metadata=metadata_dict,
                        links=links_list,
                        images=images_list,
                        content_length=len(content_str),
                        fetch_time_seconds=0.0,  # arun_many doesn't provide individual times
                        status_code=int(result.status_code) if hasattr(result, 'status_code') and result.status_code else 200,
                        extracted_content=str(result.extracted_content) if hasattr(result, 'extracted_content') and result.extracted_content else None
                    )
                )
                successful += 1
                total_content_length += len(content)
            
            total_time = time.time() - start_time
            
            logger.info(f"✅ arun_many() completed: {successful}/{len(request.urls)} successful in {total_time:.2f}s")
            
            return crawl_service_pb2.CrawlManyResponse(
                success=True,
                results=crawl_responses,
                urls_requested=len(request.urls),
                successful_crawls=successful,
                failed_crawls=failed,
                total_content_length=total_content_length,
                total_time_seconds=total_time
            )
            
        except Exception as e:
            logger.error(f"Error in crawl_many: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return crawl_service_pb2.CrawlManyResponse(
                success=False,
                urls_requested=len(request.urls),
                successful_crawls=0,
                failed_crawls=len(request.urls),
                error=str(e)
            )
    
    async def adaptive_crawl(self, request) -> Any:
        """
        Adaptive intelligent crawl using Crawl4AI's AdaptiveCrawler
        
        This uses Crawl4AI's three-layer scoring system (coverage, consistency, saturation)
        to intelligently crawl a website until sufficient information is gathered.
        
        Key difference from our research agent:
        - Research agent: Multi-source search (local docs, entities, web search)
        - AdaptiveCrawler: Deep single-domain exploration with intelligent stopping
        
        Best for: Deep-diving into a specific website/documentation for comprehensive info
        
        Args:
            request: AdaptiveCrawlRequest protobuf message
            
        Returns:
            CrawlResponse protobuf message with aggregated content
        """
        from protos import crawl_service_pb2
        
        start_time = time.time()
        
        try:
            if not self._initialized:
                await self.initialize()
            
            # Check if AdaptiveCrawler is available
            if AdaptiveCrawler is None or AdaptiveConfig is None:
                raise ValueError("AdaptiveCrawler not available in this Crawl4AI version. Requires crawl4ai>=0.7.0")
            
            max_pages = request.max_pages if request.max_pages > 0 else 50
            confidence_threshold = 0.7  # Default confidence threshold
            
            logger.info(f"Adaptive crawl: {request.seed_url} with query: '{request.query}' (max_pages={max_pages})")
            
            # Create adaptive config
            config = AdaptiveConfig(
                confidence_threshold=confidence_threshold,
                max_pages=max_pages,
                top_k_links=5,  # Follow top 5 most relevant links per page
                min_gain_threshold=0.05  # Minimum expected information gain to continue
            )
            
            # Create adaptive crawler wrapper
            adaptive = AdaptiveCrawler(self.crawler, config)
            
            # Execute adaptive crawl using digest() method
            result = await adaptive.digest(
                start_url=request.seed_url,
                query=request.query
            )
            
            if not result:
                raise Exception("AdaptiveCrawler returned no result")
            
            # Get the most relevant pages (aggregated content)
            relevant_pages = adaptive.get_relevant_content(top_k=10)
            
            if not relevant_pages:
                logger.warning(f"No relevant pages found for query: {request.query}")
                return crawl_service_pb2.CrawlResponse(
                    success=False,
                    url=request.seed_url,
                    error="No relevant content found for query"
                )
            
            # Aggregate content from all relevant pages
            aggregated_content = []
            aggregated_markdown = []
            all_links = set()
            all_images = []
            
            for page in relevant_pages:
                page_content = page.get('content', '')
                page_url = page.get('url', '')
                page_score = page.get('score', 0.0)
                
                # Add page header with relevance score
                header = f"\n## From: {page_url} (Relevance: {page_score:.1%})\n\n"
                aggregated_content.append(header + page_content)
                aggregated_markdown.append(header + page_content)
                all_links.add(page_url)
            
            # Combine all content
            combined_content = "\n\n".join(aggregated_content)
            combined_markdown = "\n\n".join(aggregated_markdown)
            
            # Get statistics
            stats = {}
            try:
                adaptive.print_stats(detailed=False)  # Logs stats
                # Note: print_stats doesn't return values, it just prints
                # We could parse the output or access internal state if needed
            except:
                pass
            
            # Extract metadata from first page or result
            metadata = {
                "query": request.query,
                "pages_crawled": len(relevant_pages),
                "confidence_achieved": getattr(result, 'confidence', 0.0) if hasattr(result, 'confidence') else 0.0,
                "strategy": "adaptive"
            }
            
            fetch_time = time.time() - start_time
            
            logger.info(f"✅ Adaptive crawl completed: {len(relevant_pages)} relevant pages, {len(combined_content)} chars in {fetch_time:.2f}s")
            
            return crawl_service_pb2.CrawlResponse(
                success=True,
                url=request.seed_url,
                title=f"Adaptive Crawl: {request.query[:50]}",
                content=combined_content,
                markdown=combined_markdown,
                html="",  # AdaptiveCrawler doesn't return HTML directly
                metadata=metadata,
                links=list(all_links),
                images=all_images,
                content_length=len(combined_content),
                fetch_time_seconds=fetch_time,
                status_code=200
            )
            
        except Exception as e:
            logger.error(f"Error in adaptive crawl: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return crawl_service_pb2.CrawlResponse(
                success=False,
                url=request.seed_url,
                error=str(e)
            )
    
    async def health_check(self, request) -> Any:
        """
        Health check
        
        Args:
            request: HealthCheckRequest protobuf message
            
        Returns:
            HealthCheckResponse protobuf message
        """
        from protos import crawl_service_pb2
        
        try:
            # Check if crawler is initialized
            crawl4ai_available = self._initialized and self.crawler is not None
            
            # Try a simple browser check
            browser_available = False
            if crawl4ai_available:
                try:
                    # Simple check - just verify crawler is ready
                    browser_available = True
                except:
                    browser_available = False
            
            status = "healthy" if (crawl4ai_available and browser_available) else "degraded"
            
            return crawl_service_pb2.HealthCheckResponse(
                status=status,
                crawl4ai_available=crawl4ai_available,
                browser_available=browser_available,
                service_version="0.7.2",
                details={
                    "max_concurrent": str(settings.MAX_CONCURRENT_CRAWLS),
                    "headless": str(settings.HEADLESS)
                }
            )
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return crawl_service_pb2.HealthCheckResponse(
                status="unhealthy",
                crawl4ai_available=False,
                browser_available=False,
                service_version="0.7.2",
                details={"error": str(e)}
            )

