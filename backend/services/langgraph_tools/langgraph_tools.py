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
        from services.langgraph_tools.crawl4ai_web_tools import crawl_web_content as crawl_func
        
        logger.info(f"🕷️ Roosevelt's Web Crawler: Processing URL(s)")
        
        result = await crawl_func(url=url, urls=urls, user_id=user_id)
        
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
        crawl_web_content
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
