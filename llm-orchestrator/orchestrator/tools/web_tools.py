"""
Web Tools - Web search and crawling via backend gRPC
"""

import logging
from typing import List, Dict, Any

from orchestrator.backend_tool_client import get_backend_tool_client

logger = logging.getLogger(__name__)


async def search_web_tool(
    query: str,
    num_results: int = 15
) -> str:
    """
    Search the web for information
    
    Args:
        query: Search query
        num_results: Number of results to return
        
    Returns:
        Formatted search results
    """
    try:
        logger.info(f"Web search: {query[:100]}")
        
        client = await get_backend_tool_client()
        results = await client.search_web(query=query, num_results=num_results)
        
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
    num_results: int = 10
) -> str:
    """
    Combined search and crawl - searches web and crawls top results
    
    Args:
        query: Search query
        num_results: Number of results to crawl
        
    Returns:
        Formatted search + crawl results
    """
    try:
        logger.info(f"Search and crawl: {query[:100]}")
        
        client = await get_backend_tool_client()
        result = await client.search_and_crawl(query=query, num_results=num_results)
        
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


# Tool registry
WEB_TOOLS = {
    'search_web': search_web_tool,
    'crawl_web_content': crawl_web_content_tool,
    'search_and_crawl': search_and_crawl_tool
}

