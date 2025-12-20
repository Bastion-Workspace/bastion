"""
Web Research Subgraph

Reusable subgraph for web search + URL extraction + crawling.
Handles the complete web research workflow: search, extract URLs, crawl top results.

Can be used by:
- Full Research Agent (Web Round 1 & 2)
- Entertainment Agent (movie/TV info)
- Electronics Agent (product research)
- Site Crawl Agent (targeted searches)

Inputs:
- query: Main search query
- queries: Optional list of queries for parallel search
- max_results: Max search results per query (default: 10)
- crawl_top_n: Number of top URLs to crawl (default: 3)

Outputs:
- web_results: Combined search + crawl results
- search_results: Raw search results
- crawled_content: Crawled content
- sources_found: Structured sources for citations
- urls_crawled: List of URLs that were crawled
"""

import logging
import re
import asyncio
from typing import Dict, Any, List, Optional

from langgraph.graph import StateGraph, END

from orchestrator.tools import search_web_tool, crawl_web_content_tool, search_web_structured

logger = logging.getLogger(__name__)


# Use Dict[str, Any] for compatibility with any agent state
WebResearchSubgraphState = Dict[str, Any]


async def web_search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Perform web search with single or multiple queries - returns structured results"""
    try:
        query = state.get("query", "")
        queries = state.get("queries", None)  # Optional: list of queries for parallel search
        max_results = state.get("max_results", 10)
        shared_memory = state.get("shared_memory", {})
        
        # Track tool usage
        previous_tools = shared_memory.get("previous_tools_used", [])
        if "search_web_structured" not in previous_tools:
            previous_tools.append("search_web_structured")
            shared_memory["previous_tools_used"] = previous_tools
            state["shared_memory"] = shared_memory
        
        # If multiple queries provided, search in parallel
        if queries and len(queries) > 1:
            logger.info(f"Web search: Parallel search with {len(queries)} queries")
            search_tasks = [
                search_web_structured(query=q, max_results=max_results) 
                for q in queries[:3]  # Limit to top 3 queries
            ]
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Combine structured results
            combined_structured = []
            for i, result in enumerate(search_results):
                if isinstance(result, Exception):
                    logger.warning(f"Search query {i+1} failed: {result}")
                    continue
                if isinstance(result, list):
                    combined_structured.extend(result)
            
            # Deduplicate by URL
            seen_urls = set()
            unique_structured = []
            for res in combined_structured:
                url = res.get("url")
                if url and url not in seen_urls:
                    unique_structured.append(res)
                    seen_urls.add(url)
            
            structured_results = unique_structured
        else:
            # Single query search
            search_query = queries[0] if queries and len(queries) == 1 else query
            logger.info(f"Web search: {search_query[:100]}")
            structured_results = await search_web_structured(query=search_query, max_results=max_results)
        
        logger.info("Tool used: search_web_structured (web search)")
        
        # Format for display (backward compatibility)
        formatted_results = []
        for i, res in enumerate(structured_results, 1):
            formatted_results.append(f"\n{i}. **{res.get('title', 'No Title')}**")
            formatted_results.append(f"   URL: {res.get('url', 'No URL')}")
            if res.get('snippet'):
                formatted_results.append(f"   {res['snippet']}")
        search_result = '\n'.join(formatted_results) if formatted_results else ""
        
        return {
            "search_results": search_result,
            "structured_search_results": structured_results,
            "query_used": query
        }
        
    except Exception as e:
        logger.error(f"Web search node error: {e}")
        return {
            "search_results": "",
            "structured_search_results": [],
            "error": str(e)
        }


async def url_extraction_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and prioritize URLs from structured search results based on relevance"""
    try:
        structured_results = state.get("structured_search_results", [])
        crawl_top_n = state.get("crawl_top_n", 3)
        search_results = state.get("search_results", "")
        
        if not structured_results:
            # Fallback to regex extraction from formatted string
            logger.warning("No structured search results, falling back to regex extraction")
            urls = re.findall(r'URL: (https?://[^\s]+)', search_results) if search_results else []
            seen = set()
            unique_urls = []
            for url in urls:
                if url not in seen:
                    seen.add(url)
                    unique_urls.append(url)
            top_urls = unique_urls[:crawl_top_n] if unique_urls else []
            logger.info(f"Extracted {len(top_urls)} URLs from formatted results (fallback)")
            return {
                "urls_to_crawl": top_urls,
                "search_results": search_results
            }
        
        # Prioritize URLs based on relevance scores
        from urllib.parse import urlparse
        
        # Keywords for official/preferred sites (can be made configurable)
        official_keywords = ["allin.com", "allinpodcast.co", "youtube.com/@allin", "podcasts.apple.com"]
        
        priority_urls = []
        seen_domains = set()
        
        # Sort by relevance score (descending)
        sorted_results = sorted(
            structured_results, 
            key=lambda x: x.get('relevance_score', 0.0), 
            reverse=True
        )
        
        for result in sorted_results:
            url = result.get('url')
            if not url:
                continue
            
            domain = urlparse(url).netloc
            relevance = result.get('relevance_score', 0.0)
            is_official = any(kw in url for kw in official_keywords)
            is_high_relevance = relevance >= 0.7
            
            # Prioritize official sites and high-relevance results
            if is_official and url not in priority_urls:
                priority_urls.insert(0, url)  # Add to front
            elif is_high_relevance and domain not in seen_domains and url not in priority_urls:
                priority_urls.append(url)
                seen_domains.add(domain)
            elif url not in priority_urls and domain not in seen_domains:
                priority_urls.append(url)
                seen_domains.add(domain)
        
        # Smart crawl limit: crawl more if there are many high-relevance results
        high_relevance_count = sum(1 for r in structured_results if r.get('relevance_score', 0.0) >= 0.7)
        
        # If many high-relevance results, crawl more (up to 5-7)
        if high_relevance_count >= 5:
            effective_limit = min(7, len(priority_urls))
            logger.info(f"Many high-relevance results ({high_relevance_count}), crawling {effective_limit} URLs")
        elif high_relevance_count >= 3:
            effective_limit = min(5, len(priority_urls))
            logger.info(f"Several high-relevance results ({high_relevance_count}), crawling {effective_limit} URLs")
        else:
            effective_limit = min(crawl_top_n, len(priority_urls))
        
        top_urls = priority_urls[:effective_limit] if priority_urls else []
        
        logger.info(f"Extracted {len(top_urls)} URLs from {len(structured_results)} search results (prioritized by relevance)")
        
        return {
            "urls_to_crawl": top_urls,
            "search_results": search_results
        }
        
    except Exception as e:
        logger.error(f"URL extraction node error: {e}")
        return {
            "urls_to_crawl": [],
            "search_results": state.get("search_results", ""),
            "error": str(e)
        }


async def crawl_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Crawl extracted URLs in parallel"""
    try:
        urls_to_crawl = state.get("urls_to_crawl", [])
        shared_memory = state.get("shared_memory", {})
        
        if not urls_to_crawl:
            logger.info("No URLs to crawl")
            return {
                "crawled_content": "",
                "crawl_results": [],
                "urls_crawled": []
            }
        
        # Track tool usage
        previous_tools = shared_memory.get("previous_tools_used", [])
        if "crawl_web_content_tool" not in previous_tools:
            previous_tools.append("crawl_web_content_tool")
            shared_memory["previous_tools_used"] = previous_tools
            state["shared_memory"] = shared_memory
        
        logger.info(f"Crawling {len(urls_to_crawl)} URLs")
        
        # Crawl URLs (backend handles parallel crawling)
        crawl_result = await crawl_web_content_tool(urls=urls_to_crawl)
        logger.info("Tool used: crawl_web_content_tool (crawled top results)")
        
        # Get structured crawl results from backend for sources
        from orchestrator.backend_tool_client import get_backend_tool_client
        structured_results = []
        try:
            client = await get_backend_tool_client()
            user_id = shared_memory.get("user_id", "system")
            structured_results = await client.crawl_web_content(urls=urls_to_crawl, user_id=user_id)
        except Exception as e:
            logger.warning(f"Failed to get structured crawl results: {e}")
            structured_results = []
        
        # Format crawled content
        if crawl_result and not crawl_result.startswith("Error"):
            crawled_content = f"\n\n=== Crawled Content ===\n{crawl_result}"
        else:
            crawled_content = ""
        
        return {
            "crawled_content": crawled_content,
            "crawl_results": structured_results if isinstance(structured_results, list) else [],
            "urls_crawled": urls_to_crawl
        }
        
    except Exception as e:
        logger.error(f"Crawl node error: {e}")
        return {
            "crawled_content": "",
            "crawl_results": [],
            "urls_crawled": [],
            "error": str(e)
        }


async def format_results_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Combine search + crawl results into structured output"""
    try:
        search_results = state.get("search_results", "")
        crawled_content = state.get("crawled_content", "")
        crawl_results = state.get("crawl_results", [])
        urls_crawled = state.get("urls_crawled", [])
        
        # Combine search and crawl results
        web_result = f"{search_results}{crawled_content}".strip()
        
        # Build structured sources for citations
        sources_found = []
        if isinstance(crawl_results, list):
            for result in crawl_results:
                if isinstance(result, dict):
                    sources_found.append({
                        "type": "web",
                        "url": result.get("url", ""),
                        "title": result.get("title", ""),
                        "source": "web"
                    })
        
        # Build web_results dict
        web_results = {
            "content": web_result,
            "query_used": state.get("query_used", state.get("query", "")),
            "search_results": search_results,
            "crawled_content": crawled_content,
            "urls_crawled": urls_crawled
        }
        
        logger.info(f"Formatted web research results: {len(sources_found)} sources, {len(urls_crawled)} URLs crawled")
        
        return {
            "web_results": web_results,
            "sources_found": sources_found,
            "urls_crawled": urls_crawled
        }
        
    except Exception as e:
        logger.error(f"Format results node error: {e}")
        return {
            "web_results": {
                "content": state.get("search_results", ""),
                "error": str(e)
            },
            "sources_found": [],
            "urls_crawled": [],
            "error": str(e)
        }


def build_web_research_subgraph(checkpointer) -> StateGraph:
    """
    Build web research subgraph for web search + crawling
    
    This subgraph handles the complete web research workflow:
    1. Search web (supports single or multiple queries)
    2. Extract URLs from search results
    3. Crawl top N URLs in parallel
    4. Format results with structured sources
    
    Expected state inputs:
    - query: str - Main search query
    - queries: List[str] (optional) - Multiple queries for parallel search
    - max_results: int (optional, default: 10) - Max search results per query
    - crawl_top_n: int (optional, default: 3) - Number of top URLs to crawl
    - shared_memory: Dict[str, Any] - Shared memory for tool tracking
    - messages: List (optional) - Conversation history
    - metadata: Dict[str, Any] (optional) - Metadata for checkpointing
    
    Returns state with:
    - web_results: Dict[str, Any] - Combined search + crawl results
    - search_results: str - Raw search results
    - crawled_content: str - Crawled content
    - sources_found: List[Dict[str, Any]] - Structured sources for citations
    - urls_crawled: List[str] - URLs that were crawled
    """
    subgraph = StateGraph(Dict[str, Any])
    
    # Add nodes
    subgraph.add_node("web_search", web_search_node)
    subgraph.add_node("url_extraction", url_extraction_node)
    subgraph.add_node("crawl", crawl_node)
    subgraph.add_node("format_results", format_results_node)
    
    # Set entry point
    subgraph.set_entry_point("web_search")
    
    # Linear flow: search -> extract -> crawl -> format
    subgraph.add_edge("web_search", "url_extraction")
    subgraph.add_edge("url_extraction", "crawl")
    subgraph.add_edge("crawl", "format_results")
    subgraph.add_edge("format_results", END)
    
    return subgraph.compile(checkpointer=checkpointer)

