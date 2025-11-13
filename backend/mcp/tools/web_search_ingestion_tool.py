"""
Web Search Ingestion Tool - MCP Tool for Searching and Ingesting Web Content
Allows LLM to search the web and automatically ingest results into the knowledge base
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import httpx
import hashlib
from datetime import datetime
from urllib.parse import urlparse

from mcp.schemas.tool_schemas import ToolResponse, ToolError
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from mcp.tools.web_search_tool import WebSearchResult as WebSearchResultType

logger = logging.getLogger(__name__)


class WebSearchIngestionInput(BaseModel):
    """Input for web search with ingestion"""
    query: str = Field(..., description="Search query")
    num_results: int = Field(20, ge=1, le=20, description="Number of results to return and ingest")
    search_type: str = Field("web", description="Search type: 'web', 'news', 'images'")
    language: str = Field("en", description="Search language (ISO code)")
    region: str = Field("us", description="Search region")
    ingest_content: bool = Field(True, description="Whether to ingest fetched content into knowledge base")
    max_content_length: int = Field(10000, ge=1000, le=50000, description="Maximum content length to extract")
    category: str = Field("other", description="Category for ingested documents")
    tags: List[str] = Field(default_factory=list, description="Tags for ingested documents")


class IngestedWebResult(BaseModel):
    """Result from web search with ingestion"""
    title: str = Field(..., description="Result title")
    url: str = Field(..., description="Result URL")
    snippet: str = Field(..., description="Result snippet/description")
    source: str = Field(..., description="Source domain")
    document_id: Optional[str] = Field(None, description="Document ID if ingested")
    ingestion_status: str = Field(..., description="Ingestion status: 'success', 'failed', 'skipped'")
    content_length: int = Field(..., description="Content length in characters")
    fetch_time: float = Field(..., description="Time taken to fetch content")


class WebSearchIngestionOutput(BaseModel):
    """Output from web search with ingestion"""
    query: str = Field(..., description="Original search query")
    results: List[IngestedWebResult] = Field(..., description="Search and ingestion results")
    total_results: int = Field(..., description="Total number of results found")
    total_ingested: int = Field(..., description="Total number of results ingested")
    ingestion_summary: str = Field(..., description="Summary of search and ingestion results")
    search_time: float = Field(..., description="Time taken for search and ingestion")


class WebSearchIngestionTool:
    """MCP tool for web search with automatic ingestion"""
    
    def __init__(self, web_search_tool=None, web_content_tool=None, document_service=None, embedding_manager=None, user_document_service=None):
        """Initialize with required services"""
        self.web_search_tool = web_search_tool
        self.web_content_tool = web_content_tool
        self.document_service = document_service
        self.embedding_manager = embedding_manager
        self.user_document_service = user_document_service
        self.current_user_id = None  # Will be set by MCP server
        self.name = "web_search_and_ingest"
        self.description = "Search the web and automatically ingest results into the knowledge base"
        
    async def initialize(self):
        """Initialize the web search ingestion tool"""
        if not self.web_search_tool:
            raise ValueError("WebSearchTool is required")
        if not self.web_content_tool:
            raise ValueError("WebContentTool is required")
        if not self.document_service:
            raise ValueError("DocumentService is required")
        if not self.embedding_manager:
            raise ValueError("EmbeddingManager is required")
        
        logger.info("📥 WebSearchIngestionTool initialized")
    
    def set_current_user(self, user_id: str):
        """Set the current user for user-specific operations"""
        self.current_user_id = user_id
        logger.debug(f"🔐 WebSearchIngestionTool user set to: {user_id}")
    
    async def execute(self, input_data: WebSearchIngestionInput) -> ToolResponse:
        """Execute web search with ingestion"""
        start_time = time.time()
        
        try:
            logger.info(f"📥 Executing web search with ingestion: '{input_data.query}'")
            
            # Step 1: Perform web search
            search_results = await self._perform_web_search(input_data)
            
            if not search_results:
                return ToolResponse(
                    success=False,
                    error=ToolError(
                        error_code="SEARCH_FAILED",
                        error_message="No search results found",
                        details={"query": input_data.query}
                    ),
                    execution_time=time.time() - start_time
                )
            
            # Step 2: Fetch and ingest content for each result
            ingested_results = []
            total_ingested = 0
            
            for result in search_results[:input_data.num_results]:
                ingested_result = await self._process_and_ingest_result(
                    result, input_data
                )
                ingested_results.append(ingested_result)
                
                if ingested_result.ingestion_status == "success":
                    total_ingested += 1
            
            # Create ingestion summary
            ingestion_summary = f"Found {len(search_results)} results, ingested {total_ingested} documents"
            if total_ingested > 0:
                domains = list(set(r.source for r in ingested_results if r.ingestion_status == "success"))
                ingestion_summary += f" from {len(domains)} sources: {', '.join(domains[:3])}"
            
            # Create output
            output = WebSearchIngestionOutput(
                query=input_data.query,
                results=ingested_results,
                total_results=len(search_results),
                total_ingested=total_ingested,
                ingestion_summary=ingestion_summary,
                search_time=time.time() - start_time
            )
            
            logger.info(f"✅ Web search and ingestion completed: {total_ingested}/{len(ingested_results)} ingested in {output.search_time:.2f}s")
            
            return ToolResponse(
                success=True,
                data=output,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"❌ Web search and ingestion failed: {e}")
            return ToolResponse(
                success=False,
                error=ToolError(
                    error_code="INGESTION_FAILED",
                    error_message=str(e),
                    details={"query": input_data.query}
                ),
                execution_time=time.time() - start_time
            )
    
    async def _perform_web_search(self, input_data: WebSearchIngestionInput) -> List[Dict[str, Any]]:
        """Perform web search using the web search tool"""
        try:
            # Create input for web search tool
            from mcp.tools.web_search_tool import WebSearchInput
            
            search_input = WebSearchInput(
                query=input_data.query,
                num_results=input_data.num_results,
                search_type=input_data.search_type,
                language=input_data.language,
                region=input_data.region
            )
            
            # Execute web search
            search_response = await self.web_search_tool.execute(search_input)
            
            if not search_response.success:
                logger.error(f"❌ Web search failed: {search_response.error}")
                return []
            
            # Extract results
            search_data = search_response.data
            return search_data.results
            
        except Exception as e:
            logger.error(f"❌ Web search execution failed: {e}")
            return []
    
    async def _process_and_ingest_result(self, search_result: "WebSearchResultType", input_data: WebSearchIngestionInput) -> IngestedWebResult:
        """Process a search result and ingest it if possible"""
        try:
            url = search_result.url
            title = search_result.title
            snippet = search_result.snippet
            source = search_result.source
            
            # Check if URL is valid
            if not url or not self._is_valid_url(url):
                return IngestedWebResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    source=source,
                    ingestion_status="skipped",
                    content_length=0,
                    fetch_time=0.0
                )
            
            # Check if already ingested (avoid duplicates)
            existing_doc_id = await self._check_existing_document(url)
            if existing_doc_id:
                logger.info(f"📄 Document already exists: {url} -> {existing_doc_id}")
                return IngestedWebResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    source=source,
                    document_id=existing_doc_id,
                    ingestion_status="skipped",
                    content_length=0,
                    fetch_time=0.0
                )
            
            # Fetch content if ingestion is enabled
            if input_data.ingest_content:
                fetch_start = time.time()
                
                # Create input for web content tool
                from mcp.tools.web_content_tool import WebContentInput
                
                content_input = WebContentInput(
                    url=url,
                    extract_text=True,
                    extract_links=False,
                    extract_metadata=True,
                    max_content_length=input_data.max_content_length,
                    timeout=30
                )
                
                # Fetch content
                content_response = await self.web_content_tool.execute(content_input)
                
                if content_response.success and content_response.data.result.content:
                    # Ingest the content
                    document_id = await self._ingest_content(
                        url, title, content_response.data.result, input_data
                    )
                    
                    fetch_time = time.time() - fetch_start
                    
                    return IngestedWebResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source=source,
                        document_id=document_id,
                        ingestion_status="success",
                        content_length=len(content_response.data.result.content),
                        fetch_time=fetch_time
                    )
                else:
                    logger.warning(f"⚠️ Failed to fetch content from: {url}")
                    return IngestedWebResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source=source,
                        ingestion_status="failed",
                        content_length=0,
                        fetch_time=time.time() - fetch_start
                    )
            else:
                # Ingestion disabled, just return search result
                return IngestedWebResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    source=source,
                    ingestion_status="skipped",
                    content_length=0,
                    fetch_time=0.0
                )
                
        except Exception as e:
            logger.error(f"❌ Failed to process result {search_result.url}: {e}")
            return IngestedWebResult(
                title=search_result.title,
                url=search_result.url,
                snippet=search_result.snippet,
                source=search_result.source,
                ingestion_status="failed",
                content_length=0,
                fetch_time=0.0
            )
    
    async def _check_existing_document(self, url: str) -> Optional[str]:
        """Check if a document with this URL already exists"""
        try:
            # Generate a consistent document ID from URL
            url_hash = hashlib.md5(url.encode()).hexdigest()
            doc_id = f"web_{url_hash}"
            
            # Check if document exists using document service
            exists = await self.document_service.check_document_exists(doc_id)
            return doc_id if exists else None
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to check existing document: {e}")
            return None
    
    async def _ingest_content(self, url: str, title: str, content_result: Any, input_data: WebSearchIngestionInput) -> str:
        """Ingest web content into the knowledge base"""
        try:
            # Generate document ID
            url_hash = hashlib.md5(url.encode()).hexdigest()
            doc_id = f"web_{url_hash}"
            
            # Prepare metadata
            metadata = {
                "url": url,
                "source": content_result.metadata.source if hasattr(content_result.metadata, 'source') else "",
                "title": title,
                "author": content_result.metadata.author if hasattr(content_result.metadata, 'author') else "",
                "description": content_result.metadata.description if hasattr(content_result.metadata, 'description') else "",
                "keywords": content_result.metadata.keywords if hasattr(content_result.metadata, 'keywords') else [],
                "language": content_result.metadata.language if hasattr(content_result.metadata, 'language') else "en",
                "search_query": input_data.query,
                "ingestion_date": datetime.utcnow().isoformat(),
                "category": input_data.category,
                "tags": input_data.tags + ["web_search", "ingested"],
                "content_type": "web_page",
                "original_url": url
            }
            
            # Create document content
            document_content = f"""
Title: {title}
URL: {url}
Source: {metadata['source']}

{content_result.content}

---
Ingested from web search query: "{input_data.query}"
Ingestion date: {metadata['ingestion_date']}
"""
            
            # Store document using appropriate service (user-specific or global)
            if self.current_user_id and self.user_document_service:
                # Store in user's private collection
                logger.info(f"📥 Storing web document in user collection for user: {self.current_user_id}")
                success = await self.user_document_service.store_text_document_for_user(
                    doc_id=doc_id,
                    content=document_content,
                    metadata=metadata,
                    user_id=self.current_user_id,
                    filename=f"{title[:50]}.txt"
                )
            else:
                # Store in global collection (fallback)
                logger.info("📥 Storing web document in global collection")
                success = await self.document_service.store_text_document(
                    doc_id=doc_id,
                    content=document_content,
                    metadata=metadata,
                    filename=f"{title[:50]}.txt"
                )
            
            if success:
                logger.info(f"📥 Successfully ingested document: {doc_id} from {url}")
                return doc_id
            else:
                logger.error(f"❌ Failed to store document: {doc_id}")
                raise Exception("Document storage failed")
            
        except Exception as e:
            logger.error(f"❌ Failed to ingest content from {url}: {e}")
            raise
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and accessible"""
        try:
            parsed = urlparse(url)
            return bool(parsed.scheme and parsed.netloc)
        except:
            return False
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get MCP tool definition for registration"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": WebSearchIngestionInput.schema(),
            "outputSchema": WebSearchIngestionOutput.schema()
        } 