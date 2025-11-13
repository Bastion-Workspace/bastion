"""
Web Search Tool - MCP Tool for Internet Search
Allows LLM to search the web for current information and external sources using SearXNG
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
import httpx
import json
from urllib.parse import quote_plus

from mcp.schemas.tool_schemas import ToolResponse, ToolError
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WebSearchInput(BaseModel):
    """Input for web search"""
    query: str = Field(..., description="Search query")
    num_results: int = Field(20, ge=1, le=20, description="Number of results to return")
    search_type: str = Field("web", description="Search type: 'web', 'news', 'images'")
    language: str = Field("en", description="Search language (ISO code)")
    region: str = Field("us", description="Search region")


class WebSearchResult(BaseModel):
    """Result from web search"""
    title: str = Field(..., description="Result title")
    url: str = Field(..., description="Result URL")
    snippet: str = Field(..., description="Result snippet/description")
    source: str = Field(..., description="Source domain")
    relevance_score: Optional[float] = Field(None, description="Relevance score")
    published_date: Optional[str] = Field(None, description="Publication date if available")


class WebSearchOutput(BaseModel):
    """Output from web search"""
    query: str = Field(..., description="Original search query")
    results: List[WebSearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results found")
    search_summary: str = Field(..., description="Summary of search results")
    search_time: float = Field(..., description="Time taken for search")


class WebSearchTool:
    """MCP tool for web search capabilities"""
    
    def __init__(self, config=None):
        """Initialize with configuration"""
        self.config = config or {}
        self.name = "web_search"
        self.description = "Search the web for current information and external sources"
        
        # Get SearXNG URL from config
        if hasattr(config, 'SEARXNG_URL'):
            self.searxng_url = config.SEARXNG_URL
        elif isinstance(config, dict):
            self.searxng_url = config.get("SEARXNG_URL", "http://searxng:8080")
        else:
            self.searxng_url = "http://searxng:8080"
        
        # Rate limiting
        self.last_request_time = 0
        self.rate_limit = 1.0  # seconds between requests
        
    async def initialize(self):
        """Initialize the web search tool"""
        logger.info("🌐 WebSearchTool initialized with SearXNG")
        
        # Test SearXNG connectivity
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.searxng_url}/config")
                if response.status_code == 200:
                    logger.info("✅ SearXNG connectivity verified")
                else:
                    logger.warning("⚠️ SearXNG connectivity test failed")
        except Exception as e:
            logger.warning(f"⚠️ SearXNG connectivity test failed: {e}")
    
    async def execute(self, input_data: WebSearchInput) -> ToolResponse:
        """Execute web search using SearXNG"""
        start_time = time.time()
        
        try:
            logger.info(f"🌐 Executing SearXNG search: '{input_data.query}'")
            
            # Rate limiting
            await self._rate_limit()
            
            # Perform search using SearXNG
            results = await self._search_searxng(input_data)
            
            # Create search summary
            search_summary = f"Found {len(results)} results for '{input_data.query}' via SearXNG"
            if results:
                domains = list(set(result.source for result in results))
                search_summary += f" from {len(domains)} sources: {', '.join(domains[:3])}"
            
            # Create output
            output = WebSearchOutput(
                query=input_data.query,
                results=results,
                total_results=len(results),
                search_summary=search_summary,
                search_time=time.time() - start_time
            )
            
            logger.info(f"✅ Web search completed: {len(results)} results in {output.search_time:.2f}s")
            
            return ToolResponse(
                success=True,
                data=output,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"❌ Web search failed: {e}")
            return ToolResponse(
                success=False,
                error=ToolError(
                    error_code="WEB_SEARCH_FAILED",
                    error_message=str(e),
                    details={"query": input_data.query, "provider": "searxng"}
                ),
                execution_time=time.time() - start_time
            )
    
    async def _rate_limit(self):
        """Apply rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            wait_time = self.rate_limit - elapsed
            await asyncio.sleep(wait_time)
        self.last_request_time = time.time()
    
    async def _search_searxng(self, input_data: WebSearchInput) -> List[WebSearchResult]:
        """Search using SearXNG JSON API"""
        try:
            # SearXNG search endpoint
            search_url = f"{self.searxng_url}/search"
            params = {
                "q": input_data.query,
                "format": "json",
                "categories": "general",
                "engines": "google,duckduckgo",  # Use Google and DuckDuckGo only
                "lang": input_data.language,
                "pageno": 1
            }
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "X-Forwarded-For": "172.18.0.1",  # Docker bridge gateway IP for bot detection
                "X-Real-IP": "172.18.0.1"  # Required by SearXNG bot detection
            }
            
            async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
                response = await client.get(search_url, params=params)
                response.raise_for_status()
                data = response.json()
            
            results = []
            search_results = data.get("results", [])
            
            for i, result in enumerate(search_results[:input_data.num_results]):
                # Extract data from SearXNG result
                title = result.get("title", "").strip()
                url = result.get("url", "").strip()
                content = result.get("content", "").strip()
                engine = result.get("engine", "unknown")
                
                # Skip invalid results
                if not title or not url or len(title) < 3:
                    continue
                
                # Create WebSearchResult
                results.append(WebSearchResult(
                    title=title[:200],  # Limit title length
                    url=url,
                    snippet=content[:500] if content else f"Search result from {engine}",
                    source=self._extract_domain(url),
                    relevance_score=max(0.9 - (i * 0.05), 0.1),  # Decreasing relevance
                    published_date=result.get("publishedDate", None)
                ))
            
            logger.info(f"✅ SearXNG returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"❌ SearXNG search failed: {e}")
            # Return a fallback result
            return [WebSearchResult(
                title=f"Web search for: {input_data.query}",
                url=f"{self.searxng_url}/search?q={quote_plus(input_data.query)}",
                snippet=f"SearXNG search for '{input_data.query}'. This is a fallback result - check SearXNG service status.",
                source="searxng",
                relevance_score=0.5
            )]
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            if not url:
                return "unknown"
            
            # Simple domain extraction
            if url.startswith("http"):
                # Remove protocol
                domain = url.split("//")[1] if "//" in url else url
            else:
                domain = url
            
            # Remove path and query parameters
            domain = domain.split("/")[0]
            domain = domain.split("?")[0]
            
            return domain
        except:
            return "unknown"
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get MCP tool definition for registration"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": WebSearchInput.schema(),
            "outputSchema": WebSearchOutput.schema()
        } 