"""
Web Content Tool - MCP Tool for Fetching and Summarizing Web Content
Allows LLM to fetch web pages and extract their content for analysis
"""

import asyncio
import logging
import time
import random
from typing import List, Dict, Any, Optional
import httpx
from urllib.parse import urlparse, urljoin
import re

from mcp.schemas.tool_schemas import ToolResponse, ToolError
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WebContentInput(BaseModel):
    """Input for web content fetching"""
    url: str = Field(..., description="URL to fetch content from")
    extract_text: bool = Field(True, description="Whether to extract text content")
    extract_links: bool = Field(False, description="Whether to extract links")
    extract_metadata: bool = Field(True, description="Whether to extract metadata")
    max_content_length: int = Field(10000, ge=1000, le=50000, description="Maximum content length to extract")
    timeout: int = Field(30, ge=5, le=60, description="Request timeout in seconds")


class WebMetadata(BaseModel):
    """Web page metadata"""
    title: Optional[str] = Field(None, description="Page title")
    description: Optional[str] = Field(None, description="Page description")
    author: Optional[str] = Field(None, description="Page author")
    published_date: Optional[str] = Field(None, description="Publication date")
    keywords: List[str] = Field(default_factory=list, description="Page keywords")
    language: Optional[str] = Field(None, description="Page language")
    content_type: Optional[str] = Field(None, description="Content type")


class WebContentResult(BaseModel):
    """Result from web content fetching"""
    url: str = Field(..., description="Original URL")
    title: str = Field(..., description="Page title")
    content: str = Field(..., description="Extracted content")
    metadata: WebMetadata = Field(..., description="Page metadata")
    links: List[str] = Field(default_factory=list, description="Extracted links")
    content_length: int = Field(..., description="Content length in characters")
    fetch_time: float = Field(..., description="Time taken to fetch content")
    status_code: int = Field(..., description="HTTP status code")


class WebContentOutput(BaseModel):
    """Output from web content fetching"""
    url: str = Field(..., description="Original URL")
    result: WebContentResult = Field(..., description="Fetched content result")
    success: bool = Field(..., description="Whether fetch was successful")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    fetch_summary: str = Field(..., description="Summary of fetch operation")


class WebContentTool:
    """MCP tool for fetching and analyzing web content"""
    
    def __init__(self, config=None):
        """Initialize with configuration"""
        self.config = config or {}
        self.name = "fetch_web_content"
        self.description = "Fetch and extract content from web pages"
        
        # Multiple modern user agents for rotation
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/119.0"
        ]
        self.current_ua_index = 0
        
        # Rate limiting
        self.last_request_time = {}
        self.rate_limit = 2.0  # seconds between requests
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        
    def _get_next_user_agent(self) -> str:
        """Get next user agent in rotation"""
        user_agent = self.user_agents[self.current_ua_index]
        self.current_ua_index = (self.current_ua_index + 1) % len(self.user_agents)
        return user_agent

    async def initialize(self):
        """Initialize the web content tool"""
        logger.info("📄 WebContentTool initialized")
        
        # Test connectivity
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get("https://httpbin.org/get")
                if response.status_code == 200:
                    logger.info("✅ Web content connectivity verified")
                else:
                    logger.warning("⚠️ Web content connectivity test failed")
        except Exception as e:
            logger.warning(f"⚠️ Web content connectivity test failed: {e}")
    
    async def execute(self, input_data: WebContentInput) -> ToolResponse:
        """Execute web content fetching"""
        start_time = time.time()
        
        try:
            logger.info(f"📄 Fetching web content from: {input_data.url}")
            
            # Rate limiting
            await self._rate_limit()
            
            # Fetch the web page
            content_result = await self._fetch_web_page(input_data)
            
            if not content_result:
                return ToolResponse(
                    success=False,
                    error=ToolError(
                        error_code="FETCH_FAILED",
                        error_message="Failed to fetch web content",
                        details={"url": input_data.url}
                    ),
                    execution_time=time.time() - start_time
                )
            
            # Create summary
            fetch_summary = f"Successfully fetched {content_result.content_length} characters from {content_result.url}"
            if content_result.metadata.title:
                fetch_summary += f" - '{content_result.metadata.title}'"
            
            # Create output
            output = WebContentOutput(
                url=input_data.url,
                result=content_result,
                success=True,
                fetch_summary=fetch_summary
            )
            
            logger.info(f"✅ Web content fetch completed: {content_result.content_length} chars in {content_result.fetch_time:.2f}s")
            
            return ToolResponse(
                success=True,
                data=output,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"❌ Web content fetch failed: {e}")
            return ToolResponse(
                success=False,
                error=ToolError(
                    error_code="FETCH_FAILED",
                    error_message=str(e),
                    details={"url": input_data.url}
                ),
                execution_time=time.time() - start_time
            )
    
    async def _rate_limit(self):
        """Apply rate limiting"""
        current_time = time.time()
        if hasattr(self, 'last_request_time') and self.last_request_time:
            elapsed = current_time - self.last_request_time.get('global', 0)
            if elapsed < self.rate_limit:
                wait_time = self.rate_limit - elapsed
                await asyncio.sleep(wait_time)
        self.last_request_time['global'] = time.time()
    
    async def _fetch_web_page(self, input_data: WebContentInput) -> Optional[WebContentResult]:
        """Fetch web page content with retry logic and enhanced headers"""
        start_time = time.time()
        
        # Enhanced headers for better compatibility with modern websites and Mastodon
        current_user_agent = self._get_next_user_agent()
        headers = {
            "User-Agent": current_user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "cross-site",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Referer": "https://www.google.com/",
            "Origin": "https://www.google.com",
            "X-Requested-With": "XMLHttpRequest"
        }
        
        # Special handling for Mastodon instances
        parsed_url = urlparse(input_data.url)
        if "mastodon" in parsed_url.netloc.lower():
            headers.update({
                "Accept": "application/activity+json, application/ld+json, text/html",
                "X-Forwarded-For": f"192.168.1.{random.randint(1, 254)}",
                "X-Real-IP": f"10.0.0.{random.randint(1, 254)}",
                "Via": "1.1 proxy.example.com"
            })
        
        # Retry logic with exponential backoff
        for attempt in range(self.max_retries):
            try:
                # Add random delay between retries to avoid rate limiting
                if attempt > 0:
                    delay = self.retry_delay * (2 ** (attempt - 1)) + (random.random() * 0.5)
                    await asyncio.sleep(delay)
                    logger.info(f"🔄 Retry attempt {attempt + 1}/{self.max_retries} for {input_data.url}")
                
                async with httpx.AsyncClient(
                    timeout=input_data.timeout,
                    follow_redirects=True,
                    max_redirects=10
                ) as client:
                    response = await client.get(input_data.url, headers=headers)
                    response.raise_for_status()
                    
                    content_type = response.headers.get("content-type", "").lower()
                    
                    # Handle different content types
                    if "text/html" in content_type:
                        html_content = response.text
                        
                        # Check for common blocking indicators
                        if self._is_blocked(html_content):
                            logger.warning(f"⚠️ Content appears to be blocked for {input_data.url}")
                            # For Mastodon, try alternative extraction even if blocked
                            if "mastodon" in parsed_url.netloc.lower():
                                content = self._extract_mastodon_fallback(html_content, input_data.max_content_length)
                                if content and len(content) > 50:  # If we got some meaningful content
                                    logger.info(f"✅ Mastodon fallback extraction successful: {len(content)} chars")
                                    return WebContentResult(
                                        url=input_data.url,
                                        title=self._extract_title(html_content) or "Mastodon Content",
                                        content=content,
                                        metadata=self._extract_metadata(html_content),
                                        links=[],
                                        content_length=len(content),
                                        fetch_time=time.time() - start_time,
                                        status_code=response.status_code
                                    )
                            if attempt < self.max_retries - 1:
                                continue
                            else:
                                return self._create_blocked_result(input_data.url, start_time)
                        
                        # Extract content and metadata
                        title = self._extract_title(html_content)
                        content = self._extract_text_content(html_content, input_data.max_content_length)
                        metadata = self._extract_metadata(html_content)
                        links = self._extract_links(html_content, input_data.url) if input_data.extract_links else []
                        
                        return WebContentResult(
                            url=input_data.url,
                            title=title or "No title",
                            content=content,
                            metadata=metadata,
                            links=links,
                            content_length=len(content),
                            fetch_time=time.time() - start_time,
                            status_code=response.status_code
                        )
                    
                    elif "application/json" in content_type:
                        # Handle JSON content (common in modern APIs)
                        try:
                            json_data = response.json()
                            content = str(json_data)[:input_data.max_content_length]
                            return WebContentResult(
                                url=input_data.url,
                                title="JSON Content",
                                content=content,
                                metadata=WebMetadata(),
                                links=[],
                                content_length=len(content),
                                fetch_time=time.time() - start_time,
                                status_code=response.status_code
                            )
                        except:
                            pass
                    
                    elif "text/plain" in content_type:
                        # Handle plain text content
                        content = response.text[:input_data.max_content_length]
                        return WebContentResult(
                            url=input_data.url,
                            title="Text Content",
                            content=content,
                            metadata=WebMetadata(),
                            links=[],
                            content_length=len(content),
                            fetch_time=time.time() - start_time,
                            status_code=response.status_code
                        )
                    
                    else:
                        logger.warning(f"⚠️ Unsupported content type: {content_type}")
                        return None
                        
            except httpx.HTTPStatusError as e:
                logger.error(f"❌ HTTP error {e.response.status_code}: {e}")
                if e.response.status_code in [403, 429, 503] and attempt < self.max_retries - 1:
                    continue  # Retry on rate limiting and server errors
                return None
            except httpx.RequestError as e:
                logger.error(f"❌ Request error: {e}")
                if attempt < self.max_retries - 1:
                    continue
                return None
            except Exception as e:
                logger.error(f"❌ Fetch error: {e}")
                if attempt < self.max_retries - 1:
                    continue
                return None
        
        return None
    
    def _extract_mastodon_fallback(self, html_content: str, max_length: int) -> str:
        """Fallback extraction method specifically for Mastodon content"""
        try:
            # Look for JSON-LD structured data first
            json_ld_pattern = r'<script type="application/ld\+json"[^>]*>(.*?)</script>'
            json_matches = re.findall(json_ld_pattern, html_content, re.DOTALL | re.IGNORECASE)
            
            extracted_content = []
            
            for json_str in json_matches:
                try:
                    import json
                    data = json.loads(json_str)
                    if isinstance(data, dict):
                        # Extract relevant text content
                        if 'name' in data:
                            extracted_content.append(f"Title: {data['name']}")
                        if 'description' in data:
                            extracted_content.append(f"Description: {data['description']}")
                        if 'content' in data:
                            extracted_content.append(f"Content: {data['content']}")
                        if 'text' in data:
                            extracted_content.append(f"Text: {data['text']}")
                except json.JSONDecodeError:
                    continue
            
            # Also try to extract from specific Mastodon/ActivityPub patterns
            # Look for post content in data attributes
            post_patterns = [
                r'data-content="([^"]*)"',
                r'property="og:description" content="([^"]*)"',
                r'name="description" content="([^"]*)"',
                r'class="[^"]*content[^"]*"[^>]*>([^<]+)</[^>]*>',
                r'class="[^"]*toot[^"]*"[^>]*>([^<]+)</[^>]*>',
                r'class="[^"]*status[^"]*"[^>]*>([^<]+)</[^>]*>'
            ]
            
            for pattern in post_patterns:
                matches = re.findall(pattern, html_content, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    if match and len(match.strip()) > 10:  # Only meaningful content
                        extracted_content.append(match.strip())
            
            # Combine and clean up the content
            if extracted_content:
                combined = '\n'.join(extracted_content)
                # Clean up HTML entities and extra whitespace
                combined = re.sub(r'&[a-z]+;', ' ', combined)
                combined = re.sub(r'\s+', ' ', combined)
                return combined[:max_length]
            
            return ""
            
        except Exception as e:
            logger.debug(f"Mastodon fallback extraction failed: {e}")
            return ""
    
    def _is_blocked(self, html_content: str) -> bool:
        """Check if content appears to be blocked by anti-bot measures"""
        blocked_indicators = [
            "access denied",
            "blocked",
            "captcha",
            "cloudflare",
            "ddos protection",
            "forbidden",
            "rate limit",
            "too many requests",
            "please wait",
            "checking your browser",
            "security check",
            "bot detected",
            "automated access",
            "javascript required",
            "enable javascript"
        ]
        
        content_lower = html_content.lower()
        for indicator in blocked_indicators:
            if indicator in content_lower:
                return True
        return False
    
    def _create_blocked_result(self, url: str, start_time: float) -> WebContentResult:
        """Create a result indicating the content was blocked"""
        return WebContentResult(
            url=url,
            title="Content Blocked",
            content="This content appears to be blocked by anti-bot measures. The website may require JavaScript, CAPTCHA verification, or has rate limiting in place.",
            metadata=WebMetadata(),
            links=[],
            content_length=0,
            fetch_time=time.time() - start_time,
            status_code=403
        )
    
    def _extract_title(self, html_content: str) -> Optional[str]:
        """Extract page title from HTML"""
        try:
            # Look for title tag
            title_match = re.search(r'<title[^>]*>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
            if title_match:
                title = title_match.group(1).strip()
                # Clean up the title
                title = re.sub(r'\s+', ' ', title)  # Replace multiple spaces
                title = re.sub(r'[\n\r\t]', ' ', title)  # Replace newlines and tabs
                return title[:200]  # Limit length
            return None
        except Exception as e:
            logger.warning(f"⚠️ Failed to extract title: {e}")
            return None
    
    def _extract_text_content(self, html_content: str, max_length: int) -> str:
        """Extract text content from HTML with improved handling for modern websites"""
        try:
            # Remove script and style tags
            content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.IGNORECASE | re.DOTALL)
            content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.IGNORECASE | re.DOTALL)
            
            # Remove common modern website elements that don't add value
            content = re.sub(r'<nav[^>]*>.*?</nav>', '', content, flags=re.IGNORECASE | re.DOTALL)
            content = re.sub(r'<header[^>]*>.*?</header>', '', content, flags=re.IGNORECASE | re.DOTALL)
            content = re.sub(r'<footer[^>]*>.*?</footer>', '', content, flags=re.IGNORECASE | re.DOTALL)
            content = re.sub(r'<aside[^>]*>.*?</aside>', '', content, flags=re.IGNORECASE | re.DOTALL)
            
            # Remove social media widgets and tracking elements
            content = re.sub(r'<div[^>]*class="[^"]*share[^"]*"[^>]*>.*?</div>', '', content, flags=re.IGNORECASE | re.DOTALL)
            content = re.sub(r'<div[^>]*class="[^"]*social[^"]*"[^>]*>.*?</div>', '', content, flags=re.IGNORECASE | re.DOTALL)
            content = re.sub(r'<div[^>]*class="[^"]*cookie[^"]*"[^>]*>.*?</div>', '', content, flags=re.IGNORECASE | re.DOTALL)
            content = re.sub(r'<div[^>]*class="[^"]*popup[^"]*"[^>]*>.*?</div>', '', content, flags=re.IGNORECASE | re.DOTALL)
            
            # Remove HTML tags but preserve some structure
            content = re.sub(r'<br[^>]*>', '\n', content, flags=re.IGNORECASE)
            content = re.sub(r'<p[^>]*>', '\n', content, flags=re.IGNORECASE)
            content = re.sub(r'</p>', '\n', content, flags=re.IGNORECASE)
            content = re.sub(r'<div[^>]*>', '\n', content, flags=re.IGNORECASE)
            content = re.sub(r'</div>', '\n', content, flags=re.IGNORECASE)
            
            # Remove remaining HTML tags
            content = re.sub(r'<[^>]+>', ' ', content)
            
            # Clean up whitespace and normalize
            content = re.sub(r'\n\s*\n', '\n\n', content)  # Multiple newlines to double newlines
            content = re.sub(r'[ \t]+', ' ', content)  # Multiple spaces/tabs to single space
            content = re.sub(r'\n +', '\n', content)  # Remove leading spaces after newlines
            content = re.sub(r' +\n', '\n', content)  # Remove trailing spaces before newlines
            content = content.strip()
            
            # Limit length while preserving word boundaries
            if len(content) > max_length:
                # Try to cut at a word boundary
                cut_point = max_length - 100
                while cut_point > max_length // 2:
                    if content[cut_point] == ' ' or content[cut_point] == '\n':
                        break
                    cut_point -= 1
                content = content[:cut_point] + "..."
            
            return content
        except Exception as e:
            logger.warning(f"⚠️ Failed to extract text content: {e}")
            return "Content extraction failed"
    
    def _extract_metadata(self, html_content: str) -> WebMetadata:
        """Extract metadata from HTML"""
        try:
            metadata = WebMetadata()
            
            # Extract meta tags
            meta_tags = re.findall(r'<meta[^>]+>', html_content, re.IGNORECASE)
            
            for tag in meta_tags:
                # Extract name and content attributes
                name_match = re.search(r'name=["\']([^"\']+)["\']', tag, re.IGNORECASE)
                content_match = re.search(r'content=["\']([^"\']+)["\']', tag, re.IGNORECASE)
                
                if name_match and content_match:
                    name = name_match.group(1).lower()
                    content = content_match.group(1)
                    
                    if name == "description":
                        metadata.description = content[:500]  # Limit length
                    elif name == "author":
                        metadata.author = content
                    elif name == "keywords":
                        metadata.keywords = [kw.strip() for kw in content.split(',') if kw.strip()]
                    elif name == "language":
                        metadata.language = content
            
            # Extract language from html tag
            lang_match = re.search(r'<html[^>]*lang=["\']([^"\']+)["\']', html_content, re.IGNORECASE)
            if lang_match and not metadata.language:
                metadata.language = lang_match.group(1)
            
            return metadata
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to extract metadata: {e}")
            return WebMetadata()
    
    def _extract_links(self, html_content: str, base_url: str) -> List[str]:
        """Extract links from HTML"""
        try:
            links = []
            link_pattern = r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>'
            
            for match in re.finditer(link_pattern, html_content, re.IGNORECASE):
                href = match.group(1)
                
                # Skip empty links and anchors
                if not href or href.startswith('#'):
                    continue
                
                # Convert relative URLs to absolute
                if not href.startswith(('http://', 'https://')):
                    href = urljoin(base_url, href)
                
                # Only include HTTP/HTTPS links
                if href.startswith(('http://', 'https://')):
                    links.append(href)
            
            return list(set(links))[:50]  # Remove duplicates and limit
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to extract links: {e}")
            return []
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get MCP tool definition for registration"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": WebContentInput.schema(),
            "outputSchema": WebContentOutput.schema()
        } 