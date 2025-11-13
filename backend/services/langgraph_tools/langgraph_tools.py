"""
LangGraph Tools - Roosevelt's "Official LangGraph Integration"
Using LangGraph's built-in tool system and decorators
"""

import logging
from typing import Dict, Any, List, Optional
from langchain.tools import tool
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


from pydantic import BaseModel, Field
from typing import Optional

class SearchLocalInput(BaseModel):
    query: str = Field(description="Search query")
    limit: int = Field(default=200, description="Maximum number of results")
    search_types: Optional[List[str]] = Field(default=["vector", "entities"], description="Types of search to perform")

@tool(args_schema=SearchLocalInput)
async def search_local(query: str, limit: int = 200, search_types: List[str] = None) -> str:
    """Search local documents and entities"""
    try:
        from services.langgraph_tools.unified_search_tools import _get_unified_search
        
        if search_types is None:
            search_types = ["vector", "entities"]
        
        # Get the raw structured data with citations
        search_instance = await _get_unified_search()
        result = await search_instance.search_local(query, search_types, limit, None)
        
        if not result.get("success"):
            return f"❌ Search failed: {result.get('error', 'Unknown error')}"
        
        results = result.get("results", [])
        count = result.get("count", 0)
        search_summary = result.get("search_summary", [])
        
        if count == 0:
            return f"🔍 No results found for '{query}'. Search summary: {', '.join(search_summary)}"
        
        # Format results with FULL citation information
        formatted_results = [f"🔍 **Found {count} relevant results for '{query}':**\n"]
        
        for i, item in enumerate(results[:20], 1):  # Limit to top 20 for readability
            doc_id = item.get("document_id", "unknown")
            score = item.get("score", 0.0)
            content = item.get("content", "")
            source_collection = item.get("source_collection", "unknown")
            metadata = item.get("metadata", {})
            
            # Get document title/filename from metadata
            title = metadata.get("title") or metadata.get("filename") or f"Document {doc_id[:8]}"
            
            # Include FULL citation information
            citation_info = f"📄 **Source:** {title}"
            if metadata.get("author"):
                citation_info += f" by {metadata['author']}"
            if metadata.get("date"):
                citation_info += f" ({metadata['date']})"
            if metadata.get("url"):
                citation_info += f" - {metadata['url']}"
            
            # Truncate content for readability
            content_preview = content[:300] + "..." if len(content) > 300 else content
            
            formatted_results.append(
                f"\n**{i}. {title}** (Score: {score:.3f}, Collection: {source_collection})\n"
                f"{citation_info}\n"
                f"Content: {content_preview}\n"
            )
        
        if count > 20:
            formatted_results.append(f"\n... and {count - 20} more results")
        
        # Add search summary
        formatted_results.append(f"\n📊 **Search Summary:** {', '.join(search_summary)}")
        
        return {
            "success": True,
            "content": "".join(formatted_results),
            "results_count": count,
            "search_summary": search_summary
        }
        
    except Exception as e:
        logger.error(f"❌ search_local failed: {e}")
        return {
            "success": False,
            "content": f"Search failed: {str(e)}",
            "error": str(e)
        }


class GetDocumentInput(BaseModel):
    document_id: str = Field(description="Document ID to retrieve")

@tool(args_schema=GetDocumentInput)
async def get_document(document_id: str) -> str:
    """Retrieve full document content by ID"""
    try:
        from services.langgraph_tools.unified_search_tools import get_document_content
        
        result = await get_document_content(document_id=document_id)
        
        content = result.get("content", "Document not found")
        return {"success": bool(content and content != "Document not found"), "content": content}
        
    except Exception as e:
        logger.error(f"❌ get_document failed: {e}")
        return {"success": False, "content": f"Document retrieval failed: {str(e)}", "error": str(e)}


class SearchWebInput(BaseModel):
    query: str = Field(description="Web search query")
    num_results: Optional[int] = Field(default=None, description="Number of results to return (alias for limit)")
    limit: Optional[int] = Field(default=None, description="Number of results to return")

@tool(args_schema=SearchWebInput)
async def search_web(query: str, num_results: Optional[int] = None, limit: Optional[int] = None, user_id: Optional[str] = None) -> str:
    """Search the web for information"""
    try:
        from services.langgraph_tools.web_content_tools import search_web as web_search
        
        # Handle both num_results and limit parameters
        search_limit = num_results or limit or 10
        result = await web_search(query=query, limit=search_limit)
        
        if result.get("success"):
            results = result.get("results", [])
            if results:
                formatted_results = [f"🌐 **Found {len(results)} web results for '{query}':**\n"]
                for i, item in enumerate(results[:10], 1):
                    title = item.get("title", "No title")
                    url = item.get("url", "")
                    snippet = item.get("snippet", "")
                    formatted_results.append(f"\n**{i}. {title}**\nURL: {url}\n{snippet}\n")
                return {"success": True, "content": "".join(formatted_results), "results_count": len(formatted_results)}
            else:
                return f"🌐 No web results found for '{query}'"
        else:
            return f"❌ Web search failed: {result.get('error', 'Unknown error')}"
        
    except Exception as e:
        logger.error(f"❌ search_web failed: {e}")
        return {"success": False, "content": f"Web search failed: {str(e)}", "error": str(e)}


class CrawlWebContentInput(BaseModel):
    url: Optional[str] = Field(default=None, description="Single URL to crawl (primary usage)")
    urls: Optional[List[str]] = Field(default=None, description="Multiple URLs to crawl (alternative)")

@tool(args_schema=CrawlWebContentInput)
async def crawl_web_content(url: Optional[str] = None, urls: Optional[List[str]] = None, user_id: Optional[str] = None) -> str:
    """Crawl web content from URL(s) - accepts either single URL or list of URLs"""
    try:
        from services.langgraph_tools.crawl4ai_web_tools import Crawl4AIWebTools
        
        # Handle both single URL and multiple URLs
        if url and not urls:
            target_urls = [url]  # Convert single URL to list
        elif urls and not url:
            target_urls = urls   # Use provided list
        elif url and urls:
            target_urls = [url] + urls  # Combine both
        else:
                            return {"success": False, "content": "❌ Error: Either 'url' or 'urls' parameter must be provided", "error": "Missing required parameters"}
        
        logger.info(f"🕷️ Roosevelt's Web Crawler: Processing {len(target_urls)} URL(s)")
        logger.info(f"🕷️ ROOSEVELT'S DEBUG: Calling crawler.crawl_web_content(urls={target_urls})")
        
        crawler = Crawl4AIWebTools()
        result = await crawler.crawl_web_content(urls=target_urls)
        
        if result.get("success"):
            crawled_results = result.get("results", [])
            formatted_results = [f"🕷️ **Crawled {len(crawled_results)} URLs:**\n"]
            for item in crawled_results:
                if item.get("success"):
                    title = item.get("metadata", {}).get("title", "No title")
                    url = item.get("url", "")
                    content_length = len(item.get("full_content", ""))
                    formatted_results.append(f"\n**{title}**\nURL: {url}\nContent: {content_length} characters\n")
            return "".join(formatted_results)
        else:
            return f"❌ Web crawling failed: {result.get('error', 'Unknown error')}"
        
    except Exception as e:
        logger.error(f"❌ crawl_web_content failed: {e}")
        return {"success": False, "content": f"Web crawling failed: {str(e)}", "error": str(e)}


class SearchAndCrawlInput(BaseModel):
    query: str = Field(description="Search query")
    max_results: Optional[int] = Field(default=None, description="Max results to crawl")
    num_results: Optional[int] = Field(default=None, description="Number of results to crawl (alias for max_results)")

@tool(args_schema=SearchAndCrawlInput)
async def search_and_crawl(query: str, max_results: Optional[int] = None, num_results: Optional[int] = None, user_id: Optional[str] = None) -> str:
    """Search the web and crawl top results"""
    try:
        # Handle both max_results and num_results parameters
        crawl_limit = max_results or num_results or 15
        from services.langgraph_tools.crawl4ai_web_tools import Crawl4AIWebTools
        
        crawler = Crawl4AIWebTools()
        result = await crawler.search_and_crawl(query, max_results=crawl_limit)
        
        if result.get("success"):
            results = result.get("results", [])
            formatted_results = [f"🔍🕷️ **Search and crawl results for '{query}':**\n"]
            for i, item in enumerate(results[:crawl_limit], 1):
                title = item.get("title", "No title")
                url = item.get("url", "")
                crawled = item.get("crawled", False)
                if crawled:
                    content_length = len(item.get("full_content", ""))
                    formatted_results.append(f"\n**{i}. {title}** ✅ Crawled ({content_length} chars)\nURL: {url}\n")
                else:
                    formatted_results.append(f"\n**{i}. {title}** ⏭️ Search only\nURL: {url}\n")
            return "".join(formatted_results)
        else:
            return f"❌ Search and crawl failed: {result.get('error', 'Unknown error')}"
        
    except Exception as e:
        logger.error(f"❌ search_and_crawl failed: {e}")
        return {"success": False, "content": f"Search and crawl failed: {str(e)}", "error": str(e)}


class CrawlSiteInput(BaseModel):
    seed_url: str = Field(description="Seed URL whose domain will be crawled")
    query_criteria: str = Field(description="Criteria to identify relevant pages on the site")
    max_pages: Optional[int] = Field(default=50, description="Maximum pages to crawl")
    max_depth: Optional[int] = Field(default=2, description="Maximum depth from seed")
    allowed_path_prefix: Optional[str] = Field(default=None, description="Restrict crawl to path prefix under the domain")
    include_pdfs: Optional[bool] = Field(default=False, description="Whether to include PDFs in crawl scope")

@tool(args_schema=CrawlSiteInput)
async def crawl_site(seed_url: str, query_criteria: str, max_pages: Optional[int] = 50, max_depth: Optional[int] = 2, allowed_path_prefix: Optional[str] = None, include_pdfs: Optional[bool] = False, user_id: Optional[str] = None) -> str:
    """Domain-scoped crawl starting from a seed URL, filtering pages by query criteria."""
    try:
        from services.langgraph_tools.crawl4ai_web_tools import Crawl4AIWebTools
        crawler = Crawl4AIWebTools()
        result = await crawler.crawl_site(
            seed_url=seed_url,
            query_criteria=query_criteria,
            max_pages=max_pages or 50,
            max_depth=max_depth or 2,
            allowed_path_prefix=allowed_path_prefix,
            include_pdfs=include_pdfs,
            user_id=user_id,
        )
        if result.get("success"):
            domain = result.get("domain", "")
            crawled = result.get("successful_crawls", 0)
            considered = result.get("urls_considered", 0)
            lines = [f"🕷️ Domain crawl for {domain} — crawled {crawled} of {considered} considered URLs\n"]
            for item in result.get("results", [])[:20]:
                if item.get("success"):
                    title = ((item.get("metadata") or {}).get("title") or "No title").strip() or "No title"
                    url = item.get("url", "")
                    score = item.get("relevance_score", 0.0)
                    lines.append(f"\n**{title}** (relevance: {score:.2f})\nURL: {url}\n")
            return "".join(lines)
        else:
            return {"success": False, "content": f"Site crawl failed: {result.get('error', 'Unknown error')}", "error": result.get('error', 'Unknown error')}
    except Exception as e:
        logger.error(f"❌ crawl_site failed: {e}")
        return {"success": False, "content": f"Site crawl failed: {str(e)}", "error": str(e)}

class AnalyzeAndIngestInput(BaseModel):
    urls: List[str] = Field(description="URLs to analyze and ingest")

@tool(args_schema=AnalyzeAndIngestInput)
async def analyze_and_ingest(urls: List[str]) -> str:
    """Analyze and ingest URLs as documents"""
    try:
        from services.langgraph_tools.web_content_tools import analyze_and_ingest_url
        
        result = await analyze_and_ingest_url(urls)
        
        if result.get("success"):
            analyzed = result.get("urls_analyzed", 0)
            ingested = result.get("urls_ingested", 0)
            return f"✅ Analyzed {analyzed} URLs, successfully ingested {ingested} as documents"
        else:
            return f"❌ Analysis and ingestion failed: {result.get('error', 'Unknown error')}"
        
    except Exception as e:
        logger.error(f"❌ analyze_and_ingest failed: {e}")
        return {"success": False, "content": f"Analysis and ingestion failed: {str(e)}", "error": str(e)}


class SummarizeContentInput(BaseModel):
    content: str = Field(description="Content to summarize")
    max_length: int = Field(default=500, description="Maximum length of summary")

@tool(args_schema=SummarizeContentInput)
async def summarize_content(content: str, max_length: int = 500) -> str:
    """Summarize content for analysis"""
    try:
        from services.langgraph_tools.content_analysis_tools import summarize_content as summarize
        
        result = await summarize(content=content, max_length=max_length)
        
        summary = result.get("summary", "Summarization failed")
        return {"success": bool(summary and summary != "Summarization failed"), "content": summary}
        
    except Exception as e:
        logger.error(f"❌ summarize_content failed: {e}")
        return {"success": False, "content": f"Summarization failed: {str(e)}", "error": str(e)}


class AnalyzeDocumentsInput(BaseModel):
    documents: List[Dict[str, Any]] = Field(description="List of documents to analyze")

@tool(args_schema=AnalyzeDocumentsInput)
async def analyze_documents(documents: List[Dict[str, Any]]) -> str:
    """Analyze document content and structure"""
    try:
        from services.langgraph_tools.content_analysis_tools import analyze_documents as analyze
        
        result = await analyze(documents=documents)
        
        return result.get("analysis", "Analysis failed")
        
    except Exception as e:
        logger.error(f"❌ analyze_documents failed: {e}")
        return f"Analysis failed: {str(e)}"


class SearchLocalStructuredInput(BaseModel):
    query: str = Field(description="Search query")
    limit: int = Field(default=50, description="Maximum number of results")
    search_types: Optional[List[str]] = Field(default=["vector", "entities"], description="Types of search to perform")

@tool(args_schema=SearchLocalStructuredInput)
async def search_local_structured(query: str, limit: int = 50, search_types: List[str] = None) -> str:
    """Search local documents and return structured results with full citation data"""
    try:
        from services.langgraph_tools.unified_search_tools import _get_unified_search
        import json
        
        if search_types is None:
            search_types = ["vector", "entities"]
        
        # Get the raw structured data with citations
        search_instance = await _get_unified_search()
        result = await search_instance.search_local(query, search_types, limit, None)
        
        if not result.get("success"):
            return f"❌ Search failed: {result.get('error', 'Unknown error')}"
        
        # Return structured JSON with full citation data
        structured_result = {
            "success": True,
            "query": query,
            "count": result.get("count", 0),
            "search_summary": result.get("search_summary", []),
            "results": []
        }
        
        for item in result.get("results", [])[:limit]:
            structured_item = {
                "document_id": item.get("document_id"),
                "score": item.get("score"),
                "content": item.get("content"),
                "source_collection": item.get("source_collection"),
                "metadata": item.get("metadata", {}),
                "citation": {
                    "title": item.get("metadata", {}).get("title") or item.get("metadata", {}).get("filename"),
                    "author": item.get("metadata", {}).get("author"),
                    "date": item.get("metadata", {}).get("date"),
                    "url": item.get("metadata", {}).get("url"),
                    "source": item.get("source_collection")
                }
            }
            structured_result["results"].append(structured_item)
        
        return json.dumps(structured_result, indent=2)
        
    except Exception as e:
        logger.error(f"❌ search_local_structured failed: {e}")
        return f"Search failed: {str(e)}"


# Tool collections for different agent types
def get_research_tools() -> List[BaseTool]:
    """Get tools for research agents"""
    return [
        search_local,
        search_local_structured,
        get_document,
        search_web,
        summarize_content,
        analyze_documents
    ]


def get_local_research_tools() -> List[BaseTool]:
    """Get local-only tools for research"""
    return [
        search_local,
        search_local_structured,
        get_document,
        summarize_content,
        analyze_documents
    ]


def get_web_research_tools() -> List[BaseTool]:
    """Get web-only tools for research"""
    return [
        search_web,
        crawl_web_content,
        search_and_crawl,
        analyze_and_ingest,
        crawl_site
    ]


def get_chat_tools() -> List[BaseTool]:
    """Get tools for chat agents"""
    return [
        search_local,
        get_document
    ]


def get_direct_tools() -> List[BaseTool]:
    """Get tools for direct agents"""
    return [
        search_local,
        get_document
    ]
