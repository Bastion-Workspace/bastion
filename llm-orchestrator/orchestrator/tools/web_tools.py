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
        
        # Format results - NO TRUNCATION since we're using markdown extraction
        formatted_parts = [f"Crawled {len(results)} URLs:\n"]
        
        for i, result in enumerate(results, 1):
            content = result.get('content', '')
            content_length = len(content)
            
            # Log content length for debugging
            logger.info(f"Crawled URL {i}/{len(results)}: {result['url']} - Content length: {content_length} chars")
            
            formatted_parts.append(f"\n{i}. **{result['title']}**")
            formatted_parts.append(f"   URL: {result['url']}")
            
            if content_length == 0:
                formatted_parts.append(f"   Content: [EMPTY - crawl may have been blocked]")
            elif content_length < 100:
                formatted_parts.append(f"   Content: {content} [WARNING: Very short content - {content_length} chars]")
            else:
                # Include full markdown content - no truncation
                formatted_parts.append(f"   Content: {content}")
        
        return '\n'.join(formatted_parts)
        
    except Exception as e:
        logger.error(f"Crawl tool error: {e}")
        return f"Error crawling content: {str(e)}"


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


# Tool registry
WEB_TOOLS = {
    'search_web': search_web_tool,
    'crawl_web_content': crawl_web_content_tool,
    'search_web_structured': search_web_structured
}

