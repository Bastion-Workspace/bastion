"""
Crawl4AI Web Tools Module
Advanced web scraping and content extraction using Crawl4AI for LangGraph agents
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from urllib.parse import urlparse, quote_plus

logger = logging.getLogger(__name__)


class Crawl4AIWebTools:
    """Advanced web content tools using Crawl4AI for superior content extraction"""
    
    def __init__(self):
        self._crawler = None
        self.rate_limit = 2.0  # seconds between requests
        self.last_request_time = 0
        logger.info("🕷️ Crawl4AI Web Tools initialized")
    
    async def _get_crawler(self):
        """Get Crawl4AI crawler with lazy initialization"""
        if self._crawler is None:
            try:
                from crawl4ai import AsyncWebCrawler
                
                # Configure crawler for optimal content extraction
                self._crawler = AsyncWebCrawler(
                    headless=True,
                    browser_type="chromium",
                    verbose=False,
                    always_by_pass_cache=False,
                    base_directory="/tmp/crawl4ai",
                    # Anti-detection settings
                    user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    headers={
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.5",
                        "Accept-Encoding": "gzip, deflate",
                        "Connection": "keep-alive",
                        "Upgrade-Insecure-Requests": "1",
                    }
                )
                
                # Start the crawler
                await self._crawler.start()
                logger.info("✅ Crawl4AI crawler initialized and started")
                
            except ImportError:
                logger.error("❌ Crawl4AI not installed. Run: pip install crawl4ai")
                raise
            except Exception as e:
                logger.error(f"❌ Failed to initialize Crawl4AI crawler: {e}")
                raise
                
        return self._crawler
    
    def get_tools(self) -> Dict[str, Any]:
        """Get all Crawl4AI web tools"""
        return {
            "crawl_web_content": self.crawl_web_content,
            "search_and_crawl": self.search_and_crawl,
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
                                "enum": ["NoExtractionStrategy", "LLMExtractionStrategy", "CosineStrategy"],
                                "default": "LLMExtractionStrategy",
                                "description": "Content extraction strategy to use"
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
            },
            {
                "type": "function",
                "function": {
                    "name": "search_and_crawl",
                    "description": "Search the web using SearXNG and then crawl selected results with Crawl4AI for full content",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "max_results": {
                                "type": "integer",
                                "default": 5,
                                "description": "Maximum number of search results to crawl"
                            },
                            "crawl_top_results": {
                                "type": "integer",
                                "default": 3,
                                "description": "Number of top results to crawl with Crawl4AI"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
    
    async def crawl_web_content(
        self, 
        urls: List[str], 
        extraction_strategy: str = "LLMExtractionStrategy",
        chunking_strategy: str = "NlpSentenceChunking",
        css_selector: Optional[str] = None,
        word_count_threshold: int = 10,
        user_id: str = None
    ) -> Dict[str, Any]:
        """Extract full content from web URLs using Crawl4AI"""
        try:
            logger.info(f"🕷️ Crawling {len(urls)} URLs with Crawl4AI extraction strategy: {extraction_strategy}")
            
            crawler = await self._get_crawler()
            results = []
            
            for url in urls[:5]:  # Limit to 5 URLs to prevent abuse
                try:
                    await self._rate_limit()
                    
                    logger.info(f"🕷️ Crawling URL: {url}")
                    
                    # Configure extraction strategy
                    extraction_config = await self._get_extraction_strategy(
                        extraction_strategy, 
                        chunking_strategy,
                        word_count_threshold
                    )
                    
                    # Crawl the page
                    kwargs = {
                        "url": url,
                        "css_selector": css_selector,
                        "bypass_cache": False,
                        "js_code": None,  # Can add custom JS if needed
                        "wait_for": None,  # Can wait for specific elements
                    }
                    
                    # Only add extraction_strategy if it's not None
                    if extraction_config is not None:
                        kwargs["extraction_strategy"] = extraction_config
                    
                    crawl_result = await crawler.arun(**kwargs)
                    
                    if crawl_result.success:
                        # Extract metadata safely
                        metadata_dict = getattr(crawl_result, 'metadata', {}) or {}
                        if not isinstance(metadata_dict, dict):
                            metadata_dict = {}
                            
                        metadata = {
                            "url": url,
                            "title": metadata_dict.get("title", "") if isinstance(metadata_dict.get("title"), str) else "",
                            "description": metadata_dict.get("description", "") if isinstance(metadata_dict.get("description"), str) else "",
                            "keywords": metadata_dict.get("keywords", "") if isinstance(metadata_dict.get("keywords"), str) else "",
                            "author": metadata_dict.get("author", "") if isinstance(metadata_dict.get("author"), str) else "",
                            "language": metadata_dict.get("language", "") if isinstance(metadata_dict.get("language"), str) else "",
                            "published_time": metadata_dict.get("published_time", "") if isinstance(metadata_dict.get("published_time"), str) else "",
                            "modified_time": metadata_dict.get("modified_time", "") if isinstance(metadata_dict.get("modified_time"), str) else "",
                            "crawl_timestamp": datetime.now().isoformat(),
                            "word_count": len(str(crawl_result.cleaned_html).split()) if crawl_result.cleaned_html else 0,
                            "extraction_strategy": extraction_strategy,
                            "domain": urlparse(url).netloc
                        }
                        
                        # Extract structured content
                        content_blocks = []
                        if crawl_result.extracted_content:
                            # Parse extracted content (JSON format from LLM extraction)
                            try:
                                import json
                                extracted_data = json.loads(crawl_result.extracted_content)
                                if isinstance(extracted_data, list):
                                    content_blocks = extracted_data
                                elif isinstance(extracted_data, dict):
                                    content_blocks = [extracted_data]
                            except:
                                # Fallback to plain text
                                content_blocks = [{"content": crawl_result.extracted_content, "type": "text"}]
                        
                        # Include full cleaned HTML as fallback
                        full_content = crawl_result.cleaned_html or crawl_result.html or ""
                        
                        results.append({
                            "url": url,
                            "success": True,
                            "metadata": metadata,
                            "content_blocks": content_blocks,
                            "full_content": full_content[:50000],  # Limit content size
                            "links": list(crawl_result.links)[:20] if crawl_result.links and hasattr(crawl_result.links, '__iter__') else [],  # Include outbound links
                            "images": list(crawl_result.media.get("images", []))[:10] if crawl_result.media and isinstance(crawl_result.media.get("images", []), (list, tuple)) else [],
                            "fetch_time": getattr(crawl_result, "response_headers", {}).get("crawl-time", "unknown"),
                            "citations": self._generate_citations(metadata, content_blocks)
                        })
                        
                        logger.info(f"✅ Successfully crawled {url}: {len(full_content)} chars, {len(content_blocks)} blocks")
                        
                    else:
                        logger.warning(f"⚠️ Crawl failed for {url}: {crawl_result.error_message}")
                        results.append({
                            "url": url,
                            "success": False,
                            "error": crawl_result.error_message or "Unknown crawl error",
                            "citations": []
                        })
                        
                except Exception as e:
                    error_msg = str(e)
                    if "unhashable type" in error_msg:
                        logger.error(f"❌ Data structure error crawling {url}: {error_msg}")
                        logger.error(f"🔍 Crawl result type: {type(crawl_result)}")
                        if hasattr(crawl_result, 'metadata'):
                            logger.error(f"🔍 Metadata type: {type(getattr(crawl_result, 'metadata', None))}")
                    else:
                        logger.error(f"❌ Error crawling {url}: {error_msg}")
                    
                    results.append({
                        "url": url,
                        "success": False,
                        "error": error_msg,
                        "citations": []
                    })
            
            successful_crawls = [r for r in results if r["success"]]
            
            return {
                "success": True,
                "results": results,
                "urls_crawled": len(urls),
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

            crawler = await self._get_crawler()
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
                    extraction_config = await self._get_extraction_strategy(
                        "LLMExtractionStrategy",
                        "NlpSentenceChunking",
                        10
                    )

                    kwargs = {
                        "url": url,
                        "bypass_cache": False,
                        # Improve link availability on dynamic pages
                        "wait_for": "a",
                    }
                    if extraction_config is not None:
                        kwargs["extraction_strategy"] = extraction_config

                    crawl_result = await crawler.arun(**kwargs)

                    page_success = getattr(crawl_result, "success", False)
                    page_links = list(getattr(crawl_result, "links", []) or [])
                    # Fallback: extract anchors from HTML if crawler.links is sparse
                    try:
                        html_for_links = crawl_result.html or crawl_result.cleaned_html or ""
                        if html_for_links:
                            from bs4 import BeautifulSoup
                            soup = BeautifulSoup(html_for_links, "lxml")
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
                    # Prefer raw HTML then cleaned for text extraction
                    html_raw = crawl_result.html or crawl_result.cleaned_html or ""

                    if page_success:
                        meta = getattr(crawl_result, 'metadata', {}) or {}
                        title = meta.get("title") if isinstance(meta, dict) else None

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
                            "error": getattr(crawl_result, 'error_message', 'Unknown crawl error')
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
    
    async def search_and_crawl(
        self, 
        query: str, 
        max_results: int = 5, 
        crawl_top_results: int = 3,
        user_id: str = None
    ) -> Dict[str, Any]:
        """Search the web and crawl top results with Crawl4AI"""
        try:
            logger.info(f"🔍🕷️ Search and crawl: {query} (crawling top {crawl_top_results} of {max_results} results)")
            
            # First, perform web search using SearXNG
            search_results = await self._search_searxng(query, max_results)
            
            if not search_results:
                return {
                    "success": False,
                    "error": "No search results found",
                    "search_results": [],
                    "crawled_results": []
                }
            
            # Extract URLs from top search results
            top_urls = [result["url"] for result in search_results[:crawl_top_results]]
            
            # Crawl the top URLs with Crawl4AI
            crawl_response = await self.crawl_web_content(
                urls=top_urls,
                extraction_strategy="LLMExtractionStrategy",
                user_id=user_id
            )
            
            # Combine search and crawl results
            combined_results = []
            for i, search_result in enumerate(search_results):
                combined_result = {
                    "search_rank": i + 1,
                    "title": search_result["title"],
                    "url": search_result["url"],
                    "snippet": search_result["snippet"],
                    "source": search_result["source"],
                    "relevance_score": search_result["relevance_score"],
                    "crawled": False,
                    "full_content": None,
                    "content_blocks": [],
                    "citations": []
                }
                
                # Add crawled content if available
                if i < crawl_top_results and crawl_response["success"]:
                    crawled_result = next(
                        (r for r in crawl_response["results"] if r["url"] == search_result["url"]), 
                        None
                    )
                    if crawled_result and crawled_result["success"]:
                        combined_result.update({
                            "crawled": True,
                            "full_content": crawled_result["full_content"],
                            "content_blocks": crawled_result["content_blocks"],
                            "metadata": crawled_result["metadata"],
                            "citations": crawled_result["citations"],
                            "links": crawled_result.get("links", []),
                            "images": crawled_result.get("images", [])
                        })
                
                combined_results.append(combined_result)
            
            crawled_count = sum(1 for r in combined_results if r["crawled"])
            total_citations = sum(len(r.get("citations", [])) for r in combined_results)
            
            return {
                "success": True,
                "query": query,
                "search_results_count": len(search_results),
                "crawled_results_count": crawled_count,
                "total_citations": total_citations,
                "results": combined_results,
                "summary": f"Found {len(search_results)} search results, crawled {crawled_count} with full content extraction"
            }
            
        except Exception as e:
            logger.error(f"❌ Search and crawl failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "search_results": [],
                "crawled_results": []
            }
    
    async def _get_extraction_strategy(self, strategy_name: str, chunking_strategy: str, word_count_threshold: int):
        """Get configured extraction strategy"""
        try:
            from crawl4ai import LLMExtractionStrategy, CosineStrategy, RegexChunking, LLMConfig
            
            # Configure chunking strategy
            if chunking_strategy == "RegexChunking":
                chunking = RegexChunking()
            else:  # Default to RegexChunking for now (other strategies need different imports)
                chunking = RegexChunking()
            
            # Configure extraction strategy
            if strategy_name == "LLMExtractionStrategy":
                return LLMExtractionStrategy(
                    llm_config=LLMConfig(provider="openai"),  # Use new LLMConfig approach
                    instruction="Extract the main content, key points, and important information from this web page. Focus on factual content, quotes, data, and key insights. Format as structured JSON with content blocks.",
                    extraction_type="block",
                    apply_chunking=True,
                    chunking_strategy=chunking,
                    word_count_threshold=word_count_threshold
                )
            elif strategy_name == "CosineStrategy":
                return CosineStrategy(
                    semantic_filter="Extract main content and key information",
                    word_count_threshold=word_count_threshold,
                    apply_chunking=True,
                    chunking_strategy=chunking
                )
            else:  # Basic extraction - return None to use default
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to configure extraction strategy: {e}, using basic extraction")
            # Return a simple extraction config that doesn't require complex imports
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
