"""
Crawl4AI Web Tools Module
Advanced web scraping and content extraction using Crawl4AI Service via gRPC for LangGraph agents
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from urllib.parse import urlparse, quote_plus

from clients.crawl_service_client import get_crawl_service_client

logger = logging.getLogger(__name__)


class Crawl4AIWebTools:
    """Advanced web content tools using Crawl4AI Service for superior content extraction"""
    
    def __init__(self):
        self._crawl_client = None
        self.rate_limit = 2.0  # seconds between requests
        self.last_request_time = 0
        logger.info("🕷️ Crawl4AI Web Tools initialized (using gRPC service)")
    
    async def _get_crawl_client(self):
        """Get Crawl4AI service client with lazy initialization"""
        if self._crawl_client is None:
            try:
                self._crawl_client = await get_crawl_service_client()
                logger.info("✅ Crawl4AI service client initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Crawl4AI service client: {e}")
                raise
                
        return self._crawl_client
    
    def get_tools(self) -> Dict[str, Any]:
        """Get all Crawl4AI web tools"""
        return {
            "crawl_web_content": self.crawl_web_content,
            "crawl_site": self.crawl_site,
        }
    
    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all Crawl4AI web tools"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "crawl_web_content",
                    "description": "Extract full content from web URLs using advanced Crawl4AI scraping with proper citations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "urls": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of URLs to crawl and extract content from"
                            },
                            "extraction_strategy": {
                                "type": "string",
                                "enum": ["markdown", "NoExtractionStrategy", "CosineStrategy"],
                                "default": "markdown",
                                "description": "Content extraction strategy (always markdown - we don't configure Crawl4AI with LLM)"
                            },
                            "chunking_strategy": {
                                "type": "string", 
                                "enum": ["RegexChunking", "NlpSentenceChunking", "FixedLengthWordChunking"],
                                "default": "NlpSentenceChunking",
                                "description": "How to chunk the extracted content"
                            },
                            "css_selector": {
                                "type": "string",
                                "description": "Optional CSS selector to target specific content"
                            },
                            "word_count_threshold": {
                                "type": "integer",
                                "default": 10,
                                "description": "Minimum word count for content blocks"
                            }
                        },
                        "required": ["urls"]
                    }
                }
            }
        ]
    
    async def crawl_web_content(
        self, 
        urls: List[str], 
        extraction_strategy: str = "markdown",  # Always use markdown - we don't configure Crawl4AI with LLM
        chunking_strategy: str = "NlpSentenceChunking",
        css_selector: Optional[str] = None,
        word_count_threshold: int = 10,
        user_id: str = None
    ) -> Dict[str, Any]:
        """Extract full content from web URLs using Crawl4AI Service via gRPC
        
        Note: Always uses markdown extraction. We don't configure Crawl4AI with an LLM
        since we have our own LLM infrastructure (OpenRouter, etc.) for processing content.
        """
        try:
            logger.info(f"🕷️ Crawling {len(urls)} URLs with Crawl4AI (markdown extraction)")
            
            crawl_client = await self._get_crawl_client()
            
            # Limit to 5 URLs to prevent abuse
            urls_to_crawl = urls[:5]
            
            # Always use markdown extraction - we don't configure Crawl4AI with LLM
            # Our LLM processing happens separately in the research agent
            mapped_strategy = "markdown"
            
            # Use parallel crawling via gRPC
            response = await crawl_client.crawl_many(
                urls=urls_to_crawl,
                extraction_strategy=mapped_strategy,
                chunking_strategy=chunking_strategy,
                max_concurrent=5,
                rate_limit_seconds=self.rate_limit,
                css_selector=css_selector,
                max_content_length=1000000,
                include_links=True,
                include_metadata=True,
                timeout_seconds=60,
                use_fit_markdown=True,  # Use fit markdown for LLM optimization
                user_id=user_id
            )
            
            if not response.get("success"):
                return {
                    "success": False,
                    "error": response.get("error", "Unknown error"),
                    "results": [],
                    "urls_crawled": 0,
                    "successful_crawls": 0
                }
            
            # Convert gRPC response to expected format
            results = []
            for result in response.get("results", []):
                if result.get("success"):
                    metadata = result.get("metadata", {})
                    content = result.get("content", "")
                    markdown = result.get("markdown", "")
                    
                    # Parse extracted content if available
                    content_blocks = []
                    extracted_content = result.get("extracted_content")
                    if extracted_content:
                        try:
                            import json
                            extracted_data = json.loads(extracted_content)
                            if isinstance(extracted_data, list):
                                content_blocks = extracted_data
                            elif isinstance(extracted_data, dict):
                                content_blocks = [extracted_data]
                        except:
                            content_blocks = [{"content": extracted_content, "type": "text"}]
                    
                    # Build metadata
                    full_metadata = {
                        "url": result.get("url", ""),
                        "title": metadata.get("title", "") or result.get("title", ""),
                        "description": metadata.get("description", ""),
                        "keywords": metadata.get("keywords", ""),
                        "author": metadata.get("author", ""),
                        "language": metadata.get("language", ""),
                        "published_time": metadata.get("published_time", ""),
                        "modified_time": metadata.get("modified_time", ""),
                        "crawl_timestamp": datetime.now().isoformat(),
                        "word_count": len(content.split()),
                        "extraction_strategy": extraction_strategy,
                        "domain": urlparse(result.get("url", "")).netloc
                    }
                    
                    results.append({
                        "url": result.get("url", ""),
                        "success": True,
                        "metadata": full_metadata,
                        "content_blocks": content_blocks,
                        "full_content": content[:50000],  # Limit content size
                        "html": result.get("html", ""),  # Include HTML content
                        "links": result.get("links", [])[:20],
                        "images": result.get("images", [])[:10],
                        "fetch_time": f"{result.get('fetch_time_seconds', 0):.2f}s",
                        "citations": self._generate_citations(full_metadata, content_blocks)
                    })
                else:
                    results.append({
                        "url": result.get("url", ""),
                        "success": False,
                        "error": result.get("error", "Unknown crawl error"),
                        "citations": []
                    })
            
            successful_crawls = [r for r in results if r["success"]]
            
            return {
                "success": True,
                "results": results,
                "urls_crawled": len(urls_to_crawl),
                "successful_crawls": len(successful_crawls),
                "total_content_length": sum(len(r.get("full_content", "")) for r in successful_crawls),
                "total_citations": sum(len(r.get("citations", [])) for r in results)
            }
            
        except Exception as e:
            logger.error(f"❌ Crawl4AI web content extraction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "urls_crawled": 0,
                "successful_crawls": 0
            }

    async def crawl_site(
        self,
        seed_url: str,
        query_criteria: str,
        max_pages: int = 50,
        max_depth: int = 2,
        allowed_path_prefix: Optional[str] = None,
        include_pdfs: bool = False,
        user_id: str = None
    ) -> Dict[str, Any]:
        """Bounded, domain-scoped crawl starting from seed_url and filtering by query_criteria.

        - Enforces same-host policy (scheme+host must match seed).
        - Optionally restricts to a path prefix under the host.
        - Performs polite, rate-limited crawling using Crawl4AI for each fetched page.
        - Filters pages by simple keyword heuristic first; advanced relevance to be applied by agents.
        """
        try:
            from collections import deque
            from urllib.parse import urljoin
            import re

            parsed_seed = urlparse(seed_url)
            seed_host = parsed_seed.netloc
            seed_scheme = parsed_seed.scheme
            if not seed_scheme or not seed_host:
                return {"success": False, "error": "Invalid seed_url"}

            # Frontier and visited sets
            frontier = deque([(seed_url, 0)])
            visited: set = set()

            def normalize(u: str) -> str:
                try:
                    p = urlparse(u)
                    # Drop fragment and normalize path
                    return f"{p.scheme}://{p.netloc}{p.path}?{p.query}".rstrip('?')
                except Exception:
                    return u

            crawl_client = await self._get_crawl_client()
            results: List[Dict[str, Any]] = []
            considered = 0

            # Simple keyword list from criteria for fast prefilter
            criteria_terms = [t.strip().lower() for t in re.split(r"[,;/]|\s+", query_criteria) if t.strip()]
            keyword_paths = ["news", "newsroom", "release", "press", "media", "statement"]

            def is_in_scope(url: str) -> bool:
                p = urlparse(url)
                if p.scheme not in ("http", "https"):
                    return False
                if p.netloc != seed_host:
                    return False
                # Exclude obviously irrelevant/site-internal endpoints
                pl = p.path.lower()
                if pl.startswith("/internal") or pl.startswith("/external"):
                    return False
                # Allow within prefix OR news-related sections on same host
                within_prefix = bool(allowed_path_prefix and p.path.startswith(allowed_path_prefix))
                news_like = any(k in pl for k in keyword_paths)
                if not (within_prefix or news_like or p.path == parsed_seed.path):
                    return False
                # Basic extension filter
                if any(p.path.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".zip", ".mp4", ".mp3"]):
                    return False
                if (p.path.lower().endswith(".pdf")) and not include_pdfs:
                    return False
                return True

            while frontier and len(results) < max_pages:
                url, depth = frontier.popleft()
                norm_url = normalize(url)
                if norm_url in visited:
                    continue
                visited.add(norm_url)
                considered += 1

                try:
                    await self._rate_limit()
                    
                    # Use gRPC client for individual page crawl
                    crawl_response = await crawl_client.crawl(
                        url=url,
                        extraction_strategy="llm_extraction",
                        chunking_strategy="RegexChunking",
                        include_links=True,
                        include_metadata=True,
                        timeout_seconds=60,
                        use_fit_markdown=True,
                        user_id=user_id
                    )
                    
                    if not crawl_response.get("success"):
                        logger.warning(f"⚠️ Crawl failed for {url}: {crawl_response.get('error', 'Unknown error')}")
                        continue
                    
                    # Convert gRPC response to expected format
                    page_success = True
                    page_links = crawl_response.get("links", [])
                    page_content = crawl_response.get("content", "")
                    page_markdown = crawl_response.get("markdown", "")
                    page_html = crawl_response.get("html", "")
                    
                    # Create a mock crawl_result object for compatibility
                    class MockCrawlResult:
                        def __init__(self, response):
                            self.success = response.get("success", False)
                            self.links = response.get("links", [])
                            self.markdown = response.get("markdown", "")
                            self.html = response.get("html", "")
                            self.text = response.get("content", "")
                            self.metadata = response.get("metadata", {})
                    
                    # Extract data from gRPC response
                    page_success = crawl_response.get("success", False)
                    page_links = list(crawl_response.get("links", []) or [])
                    page_html = crawl_response.get("html", "")
                    page_markdown = crawl_response.get("markdown", "")
                    page_metadata = crawl_response.get("metadata", {}) or {}
                    
                    # Fallback: extract anchors from HTML if links are sparse
                    try:
                        if page_html:
                            from bs4 import BeautifulSoup
                            soup = BeautifulSoup(page_html, "lxml")
                            # Prefer explicit rel=next if present
                            try:
                                next_a = soup.find("a", attrs={"rel": lambda v: v and "next" in str(v).lower()})
                                if next_a and isinstance(next_a.get("href"), str):
                                    page_links.append(next_a.get("href"))
                            except Exception:
                                pass
                            for a in soup.find_all("a"):
                                href = a.get("href")
                                if isinstance(href, str) and len(href) > 1:
                                    page_links.append(href)
                    except Exception:
                        pass
                    
                    # Prefer raw HTML then markdown for text extraction
                    html_raw = page_html or page_markdown or ""

                    if page_success:
                        title = page_metadata.get("title", "") if isinstance(page_metadata, dict) else ""

                        # Extract main text using robust fallback chain
                        main_text = ""
                        try:
                            import trafilatura
                            extracted = trafilatura.extract(html_raw, include_links=False, include_images=False)
                            if isinstance(extracted, str) and len(extracted) > 200:
                                main_text = extracted
                        except Exception:
                            pass
                        if not main_text:
                            try:
                                from bs4 import BeautifulSoup
                                soup = BeautifulSoup(html_raw, "lxml")
                                container = soup.find("main") or soup.find("article") or soup.find(id="main-content")
                                text_blocks = (container.get_text("\n", strip=True) if container else soup.get_text("\n", strip=True))
                                if isinstance(text_blocks, str) and len(text_blocks) > 200:
                                    main_text = text_blocks
                            except Exception:
                                pass
                        if not main_text:
                            main_text = html_raw

                        # Quick relevance: keyword match in main text + path cues
                        text_for_relevance = (main_text or "").lower()[:200000]
                        term_hits = sum(1 for t in criteria_terms if t and t in text_for_relevance)
                        path = urlparse(url).path.lower()
                        path_boost = 0.2 if "/news/releases" in path or "/news/" in path else 0.0
                        year_boost = 0.2 if "2025" in text_for_relevance or "2025" in (title or "").lower() else 0.0
                        domain_keywords = ["arrest", "arrests", "deport", "removal", "removed", "charged", "sentence", "sentenced"]
                        domain_hits = sum(1 for k in domain_keywords if k in text_for_relevance)
                        base = (term_hits + domain_hits) / max(4, (len(criteria_terms) or 1) + len(domain_keywords))
                        relevance_score = max(0.0, min(1.0, base + path_boost + year_boost))

                        # Record result
                        results.append({
                            "url": url,
                            "success": True,
                            "metadata": {
                                "url": url,
                                "title": title or "",
                                "domain": seed_host,
                                "crawl_timestamp": datetime.now().isoformat(),
                            },
                            "full_content": (main_text or "")[:50000],
                            "content_blocks": [],
                            "links": page_links[:20],
                            "relevance_score": relevance_score,
                        })

                        # Expand frontier if within depth
                        if depth < max_depth:
                            for link in page_links[:100]:
                                try:
                                    # Ignore on-page fragment links
                                    if isinstance(link, str) and link.startswith('#'):
                                        continue
                                    abs_url = urljoin(url, link)
                                    abs_norm = normalize(abs_url)
                                    if abs_norm not in visited and is_in_scope(abs_norm):
                                        frontier.append((abs_norm, depth + 1))
                                except Exception:
                                    continue
                    else:
                        results.append({
                            "url": url,
                            "success": False,
                            "error": crawl_response.get("error", "Unknown crawl error")
                        })

                except Exception as e:
                    results.append({"url": url, "success": False, "error": str(e)})

                if len(visited) >= max_pages:
                    break

            successful = [r for r in results if r.get("success")]
            return {
                "success": True,
                "results": results,
                "successful_crawls": len(successful),
                "urls_crawled": len(visited),
                "urls_considered": considered,
                "domain": seed_host,
                "seed_url": seed_url,
            }
        except Exception as e:
            logger.error(f"❌ crawl_site failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _get_extraction_strategy(self, strategy_name: str, chunking_strategy: str, word_count_threshold: int):
        """
        Get configured extraction strategy
        
        NOTE: This method is deprecated - extraction strategies are now handled
        by the Crawl4AI service via gRPC. This is kept for backward compatibility
        but should not be used for new code.
        """
        # No longer needed - extraction strategies are configured in the gRPC service
        # This method is kept for compatibility but returns None
        logger.debug(f"⚠️ _get_extraction_strategy called but extraction is handled by Crawl4AI service")
        return None
    
    def _generate_citations(self, metadata: Dict[str, Any], content_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate proper citations for crawled content"""
        citations = []
        
        # Main page citation
        citation = {
            "type": "webpage",
            "url": metadata.get("url", ""),
            "title": metadata.get("title", "Untitled"),
            "author": metadata.get("author", ""),
            "published_date": metadata.get("published_time", ""),
            "accessed_date": metadata.get("crawl_timestamp", ""),
            "domain": metadata.get("domain", ""),
            "description": metadata.get("description", "")
        }
        
        # Format citation text
        if citation["author"]:
            citation_text = f"{citation['author']}. \"{citation['title']}.\" {citation['domain']}"
        else:
            citation_text = f"\"{citation['title']}.\" {citation['domain']}"
            
        if citation["published_date"]:
            citation_text += f", {citation['published_date'][:10]}"  # Just date part
            
        citation_text += f". Web. {citation['accessed_date'][:10]}."
        
        citation["citation_text"] = citation_text
        citation["confidence"] = 0.9  # High confidence for direct page scraping
        
        citations.append(citation)
        
        return citations
    
    async def _search_searxng(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search using SearXNG"""
        try:
            import os
            import httpx
            
            searxng_url = os.getenv("SEARXNG_URL", "http://searxng:8080")
            search_url = f"{searxng_url}/search"
            
            params = {
                "q": query,
                "format": "json",
                "categories": "general",
                "engines": "bing,google,duckduckgo",
                "language": "en",
                "time_range": None,
                "safesearch": 1,
                "pageno": 1
            }
            
            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/json",
                "X-Forwarded-For": "172.18.0.1",  # Docker bridge gateway IP for bot detection
                "X-Real-IP": "172.18.0.1"  # Required by SearXNG bot detection
            }
            
            async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
                response = await client.get(search_url, params=params)
                response.raise_for_status()
                data = response.json()
            
            results = []
            search_results = data.get("results", [])
            
            for i, result in enumerate(search_results[:limit]):
                title = result.get("title", "").strip()
                url = result.get("url", "").strip()
                content = result.get("content", "").strip()
                
                if title and url and len(title) > 3:
                    results.append({
                        "title": title[:200],
                        "url": url,
                        "snippet": content[:500] if content else f"Search result {i+1}",
                        "source": urlparse(url).netloc,
                        "relevance_score": max(0.9 - (i * 0.05), 0.1),
                        "engine": result.get("engine", "unknown")
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ SearXNG search failed: {e}")
            return []
    
    async def _rate_limit(self):
        """Apply rate limiting to requests"""
        import time
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit:
            sleep_time = self.rate_limit - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    async def cleanup(self):
        """Cleanup Crawl4AI resources"""
        if self._crawler:
            try:
                await self._crawler.close()
                logger.info("✅ Crawl4AI crawler closed")
            except Exception as e:
                logger.warning(f"⚠️ Error closing Crawl4AI crawler: {e}")


# Module-level wrapper functions for tool registry
_crawl4ai_instance = None

async def get_crawl4ai_instance():
    """Get or create singleton Crawl4AI tools instance"""
    global _crawl4ai_instance
    if _crawl4ai_instance is None:
        _crawl4ai_instance = Crawl4AIWebTools()
    return _crawl4ai_instance

async def crawl_web_content(
    url: Optional[str] = None,
    urls: Optional[List[str]] = None,
    extraction_strategy: str = "markdown",  # Always markdown - we don't configure Crawl4AI with LLM
    chunking_strategy: str = "NlpSentenceChunking",
    css_selector: Optional[str] = None,
    word_count_threshold: int = 10,
    user_id: str = None
) -> Dict[str, Any]:
    """
    Module-level wrapper for crawl_web_content tool
    
    Accepts either single URL or multiple URLs for backward compatibility.
    
    Note: Always uses markdown extraction. We don't configure Crawl4AI with an LLM
    since we have our own LLM infrastructure for processing content.
    """
    instance = await get_crawl4ai_instance()
    
    # Handle both single URL and multiple URLs
    if url and not urls:
        urls_list = [url]
    elif urls and not url:
        urls_list = urls
    elif url and urls:
        urls_list = urls + [url]
    else:
        return {
            "success": False,
            "error": "Must provide either 'url' or 'urls' parameter",
            "results": [],
            "urls_crawled": 0,
            "successful_crawls": 0
        }
    
    return await instance.crawl_web_content(
        urls=urls_list,
        extraction_strategy=extraction_strategy,
        chunking_strategy=chunking_strategy,
        css_selector=css_selector,
        word_count_threshold=word_count_threshold,
        user_id=user_id
    )

async def crawl_site(
    seed_url: str,
    query_criteria: str,
    max_pages: int = 50,
    max_depth: int = 2,
    allowed_path_prefix: Optional[str] = None,
    include_pdfs: bool = False,
    user_id: str = None
) -> Dict[str, Any]:
    """Module-level wrapper for crawl_site tool"""
    instance = await get_crawl4ai_instance()
    return await instance.crawl_site(
        seed_url=seed_url,
        query_criteria=query_criteria,
        max_pages=max_pages,
        max_depth=max_depth,
        allowed_path_prefix=allowed_path_prefix,
        include_pdfs=include_pdfs,
        user_id=user_id
    )
