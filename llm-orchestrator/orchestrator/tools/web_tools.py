"""
Web Tools - Web search and crawling via backend gRPC
"""

import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action
from orchestrator.utils.tool_type_models import WebResult

logger = logging.getLogger(__name__)


# ── I/O models for search_web_tool ───────────────────────────────────────

class SearchWebInputs(BaseModel):
    """Required inputs for web search."""
    query: str = Field(description="Search query")


class SearchWebParams(BaseModel):
    """Optional configuration."""
    max_results: int = Field(default=15, description="Maximum results to return")


class SearchWebOutputs(BaseModel):
    """Typed outputs for search_web_tool."""
    results: List[WebResult] = Field(description="Web search results")
    count: int = Field(description="Number of results")
    query_used: str = Field(description="The query that was executed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def search_web_tool(
    query: str,
    max_results: int = 15
) -> Dict[str, Any]:
    """
    Search the web for information.
    Returns structured dict with results, count, query_used, and formatted.
    """
    try:
        logger.info(f"Web search: {query[:100]}")

        client = await get_backend_tool_client()
        raw = await client.search_web(query=query, max_results=max_results)

        if not raw:
            return {
                "results": [],
                "count": 0,
                "query_used": query,
                "formatted": "No web results found.",
            }

        results = []
        for r in raw:
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("snippet", ""),
            })

        formatted_parts = [f"Found {len(results)} web results:\n"]
        for i, result in enumerate(results, 1):
            formatted_parts.append(f"\n{i}. **{result['title']}**")
            formatted_parts.append(f"   URL: {result['url']}")
            if result.get("snippet"):
                formatted_parts.append(f"   {result['snippet']}")

        return {
            "results": results,
            "count": len(results),
            "query_used": query,
            "formatted": "\n".join(formatted_parts),
        }

    except Exception as e:
        logger.error(f"Web search tool error: {e}")
        return {
            "results": [],
            "count": 0,
            "query_used": query,
            "formatted": f"Error searching web: {str(e)}",
        }


class CrawlWebContentInputs(BaseModel):
    """Inputs for crawl_web_content_tool (url or urls)."""
    url: Optional[str] = Field(default=None, description="Single URL to crawl")
    urls: Optional[List[str]] = Field(default=None, description="Multiple URLs to crawl")


class CrawlWebContentParams(BaseModel):
    """Optional configuration for crawl_web_content_tool."""
    paginate: bool = Field(default=False, description="Follow pagination across multiple pages")
    max_pages: int = Field(default=10, description="Max pages to crawl when paginating")
    pagination_param: Optional[str] = Field(default=None, description="URL query param for page number, e.g. 'page'")
    start_page: int = Field(default=0, description="Starting page number")
    next_page_css_selector: Optional[str] = Field(default=None, description="CSS selector for next-page link")
    css_selector: Optional[str] = Field(default=None, description="CSS selector for content extraction")


class CrawledPage(BaseModel):
    """Single crawled page result."""
    content: str = Field(description="Extracted content")
    title: str = Field(description="Page title")
    url: str = Field(description="URL")
    links: List[str] = Field(default_factory=list, description="Links found on page")
    images: List[str] = Field(default_factory=list, description="Image URLs found on page")


class CrawlWebContentOutputs(BaseModel):
    """Outputs for crawl_web_content_tool."""
    results: List[Dict[str, Any]] = Field(description="List of {content, title, url, links, images} per page")
    count: int = Field(description="Number of URLs crawled")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


_UNCRAWLABLE_EXTENSIONS = frozenset({
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".tar", ".gz", ".rar", ".7z",
})


def _is_uncrawlable_url(u: str) -> bool:
    path = urlparse(u).path.lower().split("?")[0]
    return any(path.endswith(ext) for ext in _UNCRAWLABLE_EXTENSIONS)


async def crawl_web_content_tool(
    url: str = None,
    urls: List[str] = None,
    paginate: bool = False,
    max_pages: int = 10,
    pagination_param: Optional[str] = None,
    start_page: int = 0,
    next_page_css_selector: Optional[str] = None,
    css_selector: Optional[str] = None
) -> Dict[str, Any]:
    """
    Crawl and extract content from web URLs, with optional pagination.
    Returns structured dict with results, count, and formatted.
    """
    try:
        url_list = urls if urls else ([url] if url else [])

        skipped = [u for u in url_list if _is_uncrawlable_url(u)]
        crawlable = [u for u in url_list if not _is_uncrawlable_url(u)]

        if skipped:
            logger.info(
                "Skipping %d uncrawlable URL(s) (PDF/binary): %s",
                len(skipped),
                ", ".join(skipped[:5]),
            )

        if not crawlable:
            skip_lines = [f"- {u}" for u in skipped]
            return {
                "results": [],
                "count": 0,
                "formatted": (
                    "Cannot crawl these URLs (binary/PDF files are not supported by the web crawler):\n"
                    + "\n".join(skip_lines)
                    + "\nTry an HTML documentation page instead."
                ),
            }

        crawl_url = crawlable[0] if len(crawlable) == 1 else None
        crawl_urls = crawlable if len(crawlable) > 1 else None

        logger.info(f"Crawling {len(crawlable)} URLs (paginate={paginate}, max_pages={max_pages})")

        client = await get_backend_tool_client()
        results = await client.crawl_web_content(
            url=crawl_url,
            urls=crawl_urls,
            paginate=paginate,
            max_pages=max_pages,
            pagination_param=pagination_param,
            start_page=start_page,
            next_page_css_selector=next_page_css_selector,
            css_selector=css_selector
        )

        if not results:
            return {"results": [], "count": 0, "formatted": "No content crawled."}

        formatted_parts = [f"Crawled {len(results)} URLs:\n"]
        for i, result in enumerate(results, 1):
            content = result.get("content", "")
            content_length = len(content)
            logger.info(f"Crawled URL {i}/{len(results)}: {result['url']} - Content length: {content_length} chars")
            formatted_parts.append(f"\n{i}. **{result.get('title', '')}**")
            formatted_parts.append(f"   URL: {result.get('url', '')}")
            if content_length == 0:
                formatted_parts.append("   Content: [EMPTY - crawl may have been blocked]")
            elif content_length < 100:
                formatted_parts.append(f"   Content: {content} [WARNING: Very short content - {content_length} chars]")
            else:
                formatted_parts.append(f"   Content: {content}")
            images = result.get("images", [])
            if images:
                formatted_parts.append(f"   Images: {len(images)} URL(s)")

        if skipped:
            formatted_parts.append(
                f"\n(Skipped {len(skipped)} binary/PDF URL(s) that cannot be crawled: "
                + ", ".join(skipped[:3])
                + ("..." if len(skipped) > 3 else "")
                + ")"
            )

        formatted = "\n".join(formatted_parts)
        return {"results": results, "count": len(results), "formatted": formatted}

    except Exception as e:
        logger.error(f"Crawl tool error: {e}")
        err = str(e)
        return {"results": [], "count": 0, "formatted": f"Error crawling content: {err}"}


register_action(
    name="crawl_web_content",
    category="search",
    description="Crawl and extract content from web URLs, with optional pagination",
    inputs_model=CrawlWebContentInputs,
    params_model=CrawlWebContentParams,
    outputs_model=CrawlWebContentOutputs,
    tool_function=crawl_web_content_tool,
)


register_action(
    name="search_web",
    category="search",
    description="Search the web for information",
    inputs_model=SearchWebInputs,
    params_model=SearchWebParams,
    outputs_model=SearchWebOutputs,
    tool_function=search_web_tool,
)


# Tool registry
WEB_TOOLS = {
    'search_web': search_web_tool,
    'crawl_web_content': crawl_web_content_tool,
}

