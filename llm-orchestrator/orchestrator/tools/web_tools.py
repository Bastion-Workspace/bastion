"""
Web Tools - Web search and crawling via backend gRPC
"""

import logging
from typing import List, Dict, Any

from orchestrator.backend_tool_client import get_backend_tool_client

logger = logging.getLogger(__name__)


async def search_web_tool(
    query: str,
    max_results: int = 15
) -> str:
    """
    Search the web for information

    Args:
        query: Search query
        max_results: Maximum number of results to return

    Returns:
        Formatted search results
    """
    try:
        logger.info(f"Web search: {query[:100]}")
        
        client = await get_backend_tool_client()
        results = await client.search_web(query=query, max_results=max_results)
        
        if not results:
            return "No web results found."
        
        # Format results
        formatted_parts = [f"Found {len(results)} web results:\n"]
        
        for i, result in enumerate(results, 1):
            formatted_parts.append(f"\n{i}. **{result['title']}**")
            formatted_parts.append(f"   URL: {result['url']}")
            if result.get('snippet'):
                formatted_parts.append(f"   {result['snippet']}")
        
        return '\n'.join(formatted_parts)
        
    except Exception as e:
        logger.error(f"Web search tool error: {e}")
        return f"Error searching web: {str(e)}"


async def crawl_web_content_tool(
    url: str = None,
    urls: List[str] = None
) -> str:
    """
    Crawl and extract content from web URLs
    
    Args:
        url: Single URL to crawl
        urls: Multiple URLs to crawl
        
    Returns:
        Formatted crawled content
    """
    try:
        url_list = urls if urls else ([url] if url else [])
        logger.info(f"Crawling {len(url_list)} URLs")
        
        client = await get_backend_tool_client()
        results = await client.crawl_web_content(url=url, urls=urls)
        
        if not results:
            return "No content crawled."
        
        # Format results
        formatted_parts = [f"Crawled {len(results)} URLs:\n"]
        
        for i, result in enumerate(results, 1):
            formatted_parts.append(f"\n{i}. **{result['title']}**")
            formatted_parts.append(f"   URL: {result['url']}")
            formatted_parts.append(f"   Content: {result['content'][:500]}...")
        
        return '\n'.join(formatted_parts)
        
    except Exception as e:
        logger.error(f"Crawl tool error: {e}")
        return f"Error crawling content: {str(e)}"


async def search_and_crawl_tool(
    query: str,
    max_results: int = 10
) -> str:
    """
    Combined search and crawl - searches web and crawls top results

    Args:
        query: Search query
        max_results: Maximum number of results to crawl

    Returns:
        Formatted search + crawl results
    """
    try:
        logger.info(f"Search and crawl: {query[:100]}")
        
        client = await get_backend_tool_client()
        result = await client.search_and_crawl(query=query, max_results=max_results)
        
        search_results = result.get('search_results', [])
        crawled_content = result.get('crawled_content', [])
        
        formatted_parts = [
            f"Search and Crawl Results:",
            f"Search: {len(search_results)} results",
            f"Crawled: {len(crawled_content)} pages\n"
        ]
        
        # Format search results
        if search_results:
            formatted_parts.append("\n=== Search Results ===")
            for i, sr in enumerate(search_results[:5], 1):
                formatted_parts.append(f"\n{i}. {sr['title']}")
                formatted_parts.append(f"   {sr['url']}")
                formatted_parts.append(f"   {sr['snippet']}")
        
        # Format crawled content
        if crawled_content:
            formatted_parts.append("\n\n=== Crawled Content ===")
            for i, cc in enumerate(crawled_content, 1):
                formatted_parts.append(f"\n{i}. {cc['title']}")
                formatted_parts.append(f"   {cc['url']}")
                formatted_parts.append(f"   {cc['content'][:300]}...")
        
        return '\n'.join(formatted_parts)
        
    except Exception as e:
        logger.error(f"Search and crawl tool error: {e}")
        return f"Error in search and crawl: {str(e)}"


async def search_web_structured(
    query: str,
    max_results: int = 15
) -> List[Dict[str, Any]]:
    """
    Search the web for information - returns structured data

    Args:
        query: Search query
        max_results: Maximum number of results to return

    Returns:
        List of result dictionaries with 'title', 'url', 'snippet', etc.
    """
    try:
        logger.info(f"Web search (structured): {query[:100]}")
        
        client = await get_backend_tool_client()
        results = await client.search_web(query=query, max_results=max_results)
        
        if not results:
            logger.warning(f"⚠️ Web search returned empty results for query: {query[:100]}")
            return []
        
        logger.info(f"✅ Web search completed: {len(results)} results")
        return results
        
    except Exception as e:
        logger.error(f"❌ Web search tool error: {e}")
        import traceback
        logger.error(f"❌ Web search traceback: {traceback.format_exc()}")
        return []


async def crawl_site_tool(
    seed_url: str,
    query_criteria: str,
    max_pages: int = 50,
    max_depth: int = 2,
    allowed_path_prefix: str = None,
    include_pdfs: bool = False
) -> str:
    """
    Domain-scoped crawl starting from seed URL, filtering pages by query criteria.
    
    This is useful when you have a specific website URL and want to crawl
    multiple pages from that domain that match your query criteria.
    
    Args:
        seed_url: Starting URL for the crawl (e.g., "https://example.com/news")
        query_criteria: Criteria to identify relevant pages (e.g., "2025 announcements")
        max_pages: Maximum number of pages to crawl (default: 50)
        max_depth: Maximum depth to traverse from seed (default: 2)
        allowed_path_prefix: Optional path prefix to restrict crawl (e.g., "/newsroom")
        include_pdfs: Whether to include PDFs in crawl scope (default: False)
        
    Returns:
        Formatted crawl results with relevant pages
    """
    try:
        logger.info(f"Domain crawl: {seed_url}, query={query_criteria[:100]}")
        
        client = await get_backend_tool_client()
        result = await client.crawl_site(
            seed_url=seed_url,
            query_criteria=query_criteria,
            max_pages=max_pages,
            max_depth=max_depth,
            allowed_path_prefix=allowed_path_prefix,
            include_pdfs=include_pdfs,
            user_id="system"  # Tool calls don't have user context
        )
        
        if not result.get("success"):
            error = result.get("error", "Unknown error")
            return f"Domain crawl failed: {error}"
        
        # Format results
        domain = result.get("domain", "")
        successful = result.get("successful_crawls", 0)
        considered = result.get("urls_considered", 0)
        results = result.get("results", [])
        
        formatted_parts = [
            f"Domain crawl for {domain}",
            f"Crawled {successful} of {considered} considered URLs\n"
        ]
        
        for i, item in enumerate(results[:20], 1):
            title = item.get("title", "No title")
            url = item.get("url", "")
            score = item.get("relevance_score", 0.0)
            formatted_parts.append(f"\n{i}. **{title}** (relevance: {score:.2f})")
            formatted_parts.append(f"   URL: {url}")
            if item.get("full_content"):
                snippet = item["full_content"][:200]
                formatted_parts.append(f"   {snippet}...")
        
        return '\n'.join(formatted_parts)
        
    except Exception as e:
        logger.error(f"Crawl site tool error: {e}")
        return f"Error crawling site: {str(e)}"


# Tool registry
WEB_TOOLS = {
    'search_web': search_web_tool,
    'crawl_web_content': crawl_web_content_tool,
    'search_and_crawl': search_and_crawl_tool,
    'search_web_structured': search_web_structured,
    'crawl_site': crawl_site_tool
}

