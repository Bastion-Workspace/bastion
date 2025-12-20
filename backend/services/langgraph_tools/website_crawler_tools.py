"""
Website Crawler Tools Module
Recursive website crawling using Crawl4AI Service via gRPC for complete site capture and vectorization
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from urllib.parse import urlparse, urljoin, urldefrag
from collections import deque
import hashlib
import time

from clients.crawl_service_client import get_crawl_service_client

logger = logging.getLogger(__name__)


class WebsiteCrawlerTools:
    """Advanced recursive website crawling tools using Crawl4AI Service via gRPC"""
    
    def __init__(self):
        self._crawl_client = None
        self.rate_limit = 2.0  # seconds between requests
        self.last_request_time = 0
        logger.info("🕷️ Website Crawler Tools initialized (using gRPC service)")
    
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
    
    async def _rate_limit(self):
        """Apply rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit:
            wait_time = self.rate_limit - time_since_last
            logger.debug(f"⏳ Rate limiting: waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def _normalize_url(self, url: str, base_url: str) -> str:
        """Normalize URL and resolve relative paths"""
        # Remove fragment identifiers
        url, _ = urldefrag(url)
        
        # Resolve relative URLs
        if not url.startswith(('http://', 'https://')):
            url = urljoin(base_url, url)
        
        # Remove trailing slashes for consistency
        url = url.rstrip('/')
        
        return url
    
    def _is_same_domain(self, url1: str, url2: str) -> bool:
        """Check if two URLs are from the same domain"""
        parsed1 = urlparse(url1)
        parsed2 = urlparse(url2)
        
        return parsed1.netloc == parsed2.netloc
    
    def _get_url_type(self, url: str) -> str:
        """Determine the type of URL (html, image, document, skip)"""
        parsed = urlparse(url)
        path_lower = parsed.path.lower()
        
        # Image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp', '.ico'}
        # Document extensions
        document_extensions = {'.pdf', '.doc', '.docx', '.txt'}
        # Skip these entirely
        skip_extensions = {
            '.zip', '.tar', '.gz', '.mp4', '.mp3', '.avi', '.mov',
            '.css', '.js', '.woff', '.woff2', '.ttf', '.eot'
        }
        
        # Skip patterns
        skip_patterns = [
            '/api/',
            'mailto:',
            'tel:',
            'javascript:',
            '#'
        ]
        
        # Check for skip patterns
        for pattern in skip_patterns:
            if pattern in url.lower():
                return 'skip'
        
        # Check extensions
        for ext in image_extensions:
            if path_lower.endswith(ext):
                return 'image'
        
        for ext in document_extensions:
            if path_lower.endswith(ext):
                return 'document'
        
        for ext in skip_extensions:
            if path_lower.endswith(ext):
                return 'skip'
        
        # Default to HTML
        return 'html'
    
    def _should_skip_url(self, url: str) -> bool:
        """Check if URL should be skipped entirely"""
        return self._get_url_type(url) == 'skip'
    
    def _extract_links_and_media_from_html(self, html_content: str, base_url: str) -> Dict[str, List[str]]:
        """Extract all internal links, images, and documents from HTML content"""
        html_links = []
        image_links = []
        document_links = []
        
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find all anchor tags
            for anchor in soup.find_all('a', href=True):
                href = anchor['href']
                full_url = self._normalize_url(href, base_url)
                
                if self._is_same_domain(full_url, base_url) and not self._should_skip_url(full_url):
                    url_type = self._get_url_type(full_url)
                    if url_type == 'html':
                        html_links.append(full_url)
                    elif url_type == 'document':
                        document_links.append(full_url)
            
            # Find all image tags
            for img in soup.find_all('img', src=True):
                src = img['src']
                full_url = self._normalize_url(src, base_url)
                
                if self._is_same_domain(full_url, base_url):
                    url_type = self._get_url_type(full_url)
                    if url_type == 'image':
                        image_links.append(full_url)
            
            logger.debug(f"🔗 Extracted from {base_url}: {len(html_links)} HTML links, {len(image_links)} images, {len(document_links)} documents")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to extract links from HTML: {e}")
        
        return {
            'html': list(set(html_links)),
            'images': list(set(image_links)),
            'documents': list(set(document_links))
        }
    
    async def crawl_page(self, url: str) -> Dict[str, Any]:
        """Crawl a single page and extract content and links"""
        try:
            await self._rate_limit()
            
            url_type = self._get_url_type(url)
            
            # Handle different content types
            if url_type == 'image':
                return await self._download_image(url)
            elif url_type == 'document':
                return await self._download_document(url)
            else:
                # Default to HTML crawling
                return await self._crawl_html_page(url)
            
        except Exception as e:
            logger.error(f"❌ Error crawling {url}: {e}")
            return {
                "success": False,
                "url": url,
                "content_type": "error",
                "error": str(e)
            }
    
    async def _crawl_html_page(self, url: str) -> Dict[str, Any]:
        """Crawl an HTML page using Crawl4AI Service via gRPC"""
        try:
            crawl_client = await self._get_crawl_client()
            
            logger.info(f"🌐 Crawling HTML page: {url}")
            
            # Perform crawl via gRPC service
            result = await crawl_client.crawl(
                url=url,
                extraction_strategy="markdown",
                chunking_strategy="RegexChunking",
                include_links=True,
                include_metadata=True,
                timeout_seconds=60,
                use_fit_markdown=True
            )
            
            if not result.get("success"):
                error_msg = result.get("error", "Unknown error")
                logger.error(f"❌ Failed to crawl {url}: {error_msg}")
                return {
                    "success": False,
                    "url": url,
                    "content_type": "html",
                    "error": error_msg
                }
            
            # Extract content from gRPC response
            markdown_content = result.get("markdown", "")
            html_content = result.get("html", "")
            
            # Extract metadata
            metadata_dict = result.get("metadata", {}) or {}
            metadata = {
                "title": metadata_dict.get("title", ""),
                "description": metadata_dict.get("description", ""),
                "keywords": metadata_dict.get("keywords", ""),
                "language": metadata_dict.get("language", "en"),
            }
            
            # Extract links, images, and documents
            extracted = self._extract_links_and_media_from_html(html_content, url)
            
            logger.info(f"✅ Successfully crawled {url}: {len(markdown_content)} chars, {len(extracted['html'])} links, {len(extracted['images'])} images, {len(extracted['documents'])} docs")
            
            return {
                "success": True,
                "url": url,
                "content_type": "html",
                "markdown_content": markdown_content,
                "html_content": html_content,
                "metadata": metadata,
                "internal_links": extracted['html'],
                "image_links": extracted['images'],
                "document_links": extracted['documents'],
                "crawl_time": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error crawling HTML page {url}: {e}")
            return {
                "success": False,
                "url": url,
                "content_type": "html",
                "error": str(e)
            }
    
    async def _download_image(self, url: str) -> Dict[str, Any]:
        """Download an image file"""
        try:
            import httpx
            
            logger.info(f"📸 Downloading image: {url}")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                # Get content type and size
                content_type = response.headers.get('content-type', 'image/unknown')
                content_size = len(response.content)
                
                # Extract filename
                parsed = urlparse(url)
                filename = parsed.path.split('/')[-1] or 'image'
                
                logger.info(f"✅ Downloaded image {url}: {content_size} bytes")
                
                return {
                    "success": True,
                    "url": url,
                    "content_type": "image",
                    "binary_content": response.content,
                    "filename": filename,
                    "mime_type": content_type,
                    "size_bytes": content_size,
                    "metadata": {
                        "content_type": content_type,
                        "filename": filename
                    },
                    "crawl_time": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to download image {url}: {e}")
            return {
                "success": False,
                "url": url,
                "content_type": "image",
                "error": str(e)
            }
    
    async def _download_document(self, url: str) -> Dict[str, Any]:
        """Download a document file (PDF, DOC, etc.)"""
        try:
            import httpx
            
            logger.info(f"📄 Downloading document: {url}")
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                # Get content type and size
                content_type = response.headers.get('content-type', 'application/octet-stream')
                content_size = len(response.content)
                
                # Extract filename
                parsed = urlparse(url)
                filename = parsed.path.split('/')[-1] or 'document'
                
                logger.info(f"✅ Downloaded document {url}: {content_size} bytes")
                
                return {
                    "success": True,
                    "url": url,
                    "content_type": "document",
                    "binary_content": response.content,
                    "filename": filename,
                    "mime_type": content_type,
                    "size_bytes": content_size,
                    "metadata": {
                        "content_type": content_type,
                        "filename": filename
                    },
                    "crawl_time": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to download document {url}: {e}")
            return {
                "success": False,
                "url": url,
                "content_type": "document",
                "error": str(e)
            }
    
    async def crawl_website_recursive(
        self,
        start_url: str,
        max_pages: int = 500,
        max_depth: int = 10,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Recursively crawl entire website starting from seed URL
        
        Args:
            start_url: Starting URL for the crawl
            max_pages: Maximum number of pages to crawl
            max_depth: Maximum depth to traverse
            user_id: User ID for document storage
            
        Returns:
            Dictionary with crawl results and statistics
        """
        try:
            logger.info(f"🕷️ Starting recursive website crawl: {start_url}")
            logger.info(f"📊 Parameters: max_pages={max_pages}, max_depth={max_depth}")
            
            # Parse base URL
            parsed_base = urlparse(start_url)
            base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"
            
            # Initialize crawl state
            crawl_session_id = hashlib.md5(f"{start_url}_{datetime.utcnow().isoformat()}".encode()).hexdigest()[:8]
            
            visited_urls: Set[str] = set()
            failed_urls: List[Dict[str, Any]] = []
            crawled_pages: List[Dict[str, Any]] = []
            
            # BFS queue: (url, depth, parent_url)
            queue = deque([(start_url, 0, None)])
            
            # Track statistics by content type
            stats = {
                "html_pages": 0,
                "images": 0,
                "documents": 0
            }
            start_time = time.time()
            
            while queue and len(crawled_pages) < max_pages:
                url, depth, parent_url = queue.popleft()
                
                # Skip if already visited
                if url in visited_urls:
                    continue
                
                # Skip if depth exceeded
                if depth > max_depth:
                    logger.debug(f"⚠️ Skipping {url}: depth {depth} exceeds max {max_depth}")
                    continue
                
                visited_urls.add(url)
                
                # Crawl the page/image/document
                page_result = await self.crawl_page(url)
                
                if page_result["success"]:
                    # Add crawl metadata
                    page_result["crawl_session_id"] = crawl_session_id
                    page_result["depth"] = depth
                    page_result["parent_url"] = parent_url
                    page_result["base_domain"] = base_domain
                    
                    crawled_pages.append(page_result)
                    
                    # Update statistics
                    content_type = page_result.get("content_type", "html")
                    if content_type == "html":
                        stats["html_pages"] += 1
                        # Add HTML links, images, and documents to queue
                        for link in page_result.get("internal_links", []):
                            if link not in visited_urls:
                                queue.append((link, depth + 1, url))
                        for img in page_result.get("image_links", []):
                            if img not in visited_urls:
                                queue.append((img, depth, url))  # Same depth for media
                        for doc in page_result.get("document_links", []):
                            if doc not in visited_urls:
                                queue.append((doc, depth, url))  # Same depth for documents
                    elif content_type == "image":
                        stats["images"] += 1
                    elif content_type == "document":
                        stats["documents"] += 1
                    
                    logger.info(f"📄 Crawled {len(crawled_pages)}/{max_pages} items ({stats['html_pages']} pages, {stats['images']} images, {stats['documents']} docs) - depth: {depth}, queue: {len(queue)}")
                    
                else:
                    failed_urls.append({
                        "url": url,
                        "error": page_result.get("error", "Unknown error"),
                        "depth": depth,
                        "content_type": page_result.get("content_type", "unknown")
                    })
            
            elapsed_time = time.time() - start_time
            
            # Generate crawl summary
            summary = {
                "crawl_session_id": crawl_session_id,
                "start_url": start_url,
                "base_domain": base_domain,
                "total_items_crawled": len(crawled_pages),
                "html_pages_crawled": stats["html_pages"],
                "images_downloaded": stats["images"],
                "documents_downloaded": stats["documents"],
                "total_items_failed": len(failed_urls),
                "max_depth_reached": max(p["depth"] for p in crawled_pages) if crawled_pages else 0,
                "elapsed_time_seconds": elapsed_time,
                "crawled_pages": crawled_pages,
                "failed_urls": failed_urls,
                "success": True
            }
            
            logger.info(f"✅ Website crawl completed: {len(crawled_pages)} pages in {elapsed_time:.2f}s")
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Website crawl failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "start_url": start_url
            }
    
    async def cleanup(self):
        """Cleanup crawler resources"""
        if self._crawl_client:
            try:
                await self._crawl_client.close()
                self._crawl_client = None
                logger.info("✅ Crawl4AI service client closed")
            except Exception as e:
                logger.error(f"❌ Failed to close crawl client: {e}")


# Module-level wrapper functions for tool registry
_crawler_instance = None

async def crawl_website_recursive(
    start_url: str,
    max_pages: int = 500,
    max_depth: int = 10,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Wrapper function for tool registry"""
    global _crawler_instance
    if _crawler_instance is None:
        _crawler_instance = WebsiteCrawlerTools()
    
    return await _crawler_instance.crawl_website_recursive(
        start_url=start_url,
        max_pages=max_pages,
        max_depth=max_depth,
        user_id=user_id
    )

