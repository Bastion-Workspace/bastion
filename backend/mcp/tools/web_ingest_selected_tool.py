"""
Web Ingest Selected Results Tool - MCP Tool for Ingesting Selected Web Content
Allows LLM to ingest only the selected results from web search analysis
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
import hashlib
from datetime import datetime

from mcp.schemas.tool_schemas import ToolResponse, ToolError
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SelectedResult(BaseModel):
    """A selected result for ingestion"""
    url: str = Field(..., description="Result URL")
    title: str = Field(..., description="Result title")
    source: str = Field(..., description="Source domain")
    selection_reason: str = Field(..., description="Reason for selecting this result")
    priority: int = Field(1, ge=1, le=5, description="Priority level (1=low, 5=high)")


class WebIngestSelectedInput(BaseModel):
    """Input for ingesting selected web results"""
    selected_results: List[SelectedResult] = Field(..., description="Results selected for ingestion")
    original_query: str = Field(..., description="Original search query that produced these results")
    category: str = Field("other", description="Category for ingested documents")
    tags: List[str] = Field(default_factory=list, description="Tags for ingested documents")
    max_content_length: int = Field(10000, ge=1000, le=50000, description="Maximum content length to extract")
    skip_existing: bool = Field(True, description="Skip results that already exist in knowledge base")


class IngestedResult(BaseModel):
    """Result from ingestion attempt"""
    url: str = Field(..., description="Result URL")
    title: str = Field(..., description="Result title")
    source: str = Field(..., description="Source domain")
    document_id: Optional[str] = Field(None, description="Document ID if ingested")
    ingestion_status: str = Field(..., description="Ingestion status: 'success', 'failed', 'skipped'")
    content_length: int = Field(..., description="Content length in characters")
    fetch_time: float = Field(..., description="Time taken to fetch content")
    selection_reason: str = Field(..., description="Reason for selection")
    priority: int = Field(..., description="Priority level")


class WebIngestSelectedOutput(BaseModel):
    """Output from ingesting selected results"""
    original_query: str = Field(..., description="Original search query")
    selected_count: int = Field(..., description="Number of results selected")
    ingested_count: int = Field(..., description="Number of results successfully ingested")
    skipped_count: int = Field(..., description="Number of results skipped (duplicates)")
    failed_count: int = Field(..., description="Number of results that failed ingestion")
    results: List[IngestedResult] = Field(..., description="Ingestion results")
    ingestion_summary: str = Field(..., description="Summary of ingestion results")
    total_time: float = Field(..., description="Total time taken for ingestion")


class WebIngestSelectedTool:
    """MCP tool for ingesting selected web search results"""
    
    def __init__(self, web_content_tool=None, document_service=None, embedding_manager=None, user_document_service=None):
        """Initialize with required services"""
        self.web_content_tool = web_content_tool
        self.document_service = document_service
        self.user_document_service = user_document_service
        self.embedding_manager = embedding_manager
        self.current_user_id = None  # Will be set by MCP chat service
        self.name = "web_ingest_selected_results"
        self.description = "Ingest selected web search results into the knowledge base"
        
    async def initialize(self):
        """Initialize the web ingest selected tool"""
        if not self.web_content_tool:
            raise ValueError("WebContentTool is required")
        if not self.document_service:
            raise ValueError("DocumentService is required")
        if not self.embedding_manager:
            raise ValueError("EmbeddingManager is required")
        
        # Validate user document service if provided
        if self.user_document_service:
            if not hasattr(self.user_document_service, 'store_text_document_for_user'):
                logger.error(f"❌ UserDocumentService missing method 'store_text_document_for_user'. Available methods: {[m for m in dir(self.user_document_service) if not m.startswith('_')]}")
                raise ValueError("UserDocumentService missing required method")
            logger.info("✅ UserDocumentService validated with required methods")
        
        logger.info("📥 WebIngestSelectedTool initialized")
    
    def set_current_user(self, user_id: str):
        """Set the current user ID for document storage"""
        self.current_user_id = user_id
        logger.debug(f"🔐 WebIngestSelectedTool user set to: {user_id}")
    
    async def execute(self, input_data: WebIngestSelectedInput) -> ToolResponse:
        """Execute ingestion of selected results"""
        start_time = time.time()
        
        try:
            logger.info(f"📥 Ingesting {len(input_data.selected_results)} selected results for query: '{input_data.original_query}'")
            
            ingested_results = []
            ingested_count = 0
            skipped_count = 0
            failed_count = 0
            
            for selected_result in input_data.selected_results:
                ingested_result = await self._process_selected_result(
                    selected_result, input_data
                )
                ingested_results.append(ingested_result)
                
                if ingested_result.ingestion_status == "success":
                    ingested_count += 1
                elif ingested_result.ingestion_status == "skipped":
                    skipped_count += 1
                else:
                    failed_count += 1
            
            # Create ingestion summary
            ingestion_summary = f"Selected {len(input_data.selected_results)} results, ingested {ingested_count} documents"
            if ingested_count > 0:
                sources = list(set(r.source for r in ingested_results if r.ingestion_status == "success"))
                ingestion_summary += f" from {len(sources)} sources: {', '.join(sources[:3])}"
            
            if skipped_count > 0:
                ingestion_summary += f", skipped {skipped_count} duplicates"
            if failed_count > 0:
                ingestion_summary += f", {failed_count} failed"
            
            # Create output
            output = WebIngestSelectedOutput(
                original_query=input_data.original_query,
                selected_count=len(input_data.selected_results),
                ingested_count=ingested_count,
                skipped_count=skipped_count,
                failed_count=failed_count,
                results=ingested_results,
                ingestion_summary=ingestion_summary,
                total_time=time.time() - start_time
            )
            
            logger.info(f"✅ Selected results ingestion completed: {ingested_count}/{len(input_data.selected_results)} ingested in {output.total_time:.2f}s")
            
            return ToolResponse(
                success=True,
                data=output,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"❌ Selected results ingestion failed: {e}")
            return ToolResponse(
                success=False,
                error=ToolError(
                    error_code="INGESTION_FAILED",
                    error_message=str(e),
                    details={"original_query": input_data.original_query}
                ),
                execution_time=time.time() - start_time
            )
    
    async def _process_selected_result(self, selected_result: SelectedResult, input_data: WebIngestSelectedInput) -> IngestedResult:
        """Process a selected result for ingestion"""
        try:
            url = selected_result.url
            title = selected_result.title
            source = selected_result.source
            
            # Check if URL is valid
            if not url or not self._is_valid_url(url):
                return IngestedResult(
                    url=url,
                    title=title,
                    source=source,
                    ingestion_status="failed",
                    content_length=0,
                    fetch_time=0.0,
                    selection_reason=selected_result.selection_reason,
                    priority=selected_result.priority
                )
            
            # Check if already exists (if skip_existing is enabled)
            if input_data.skip_existing:
                existing_doc_id = await self._check_existing_document(url)
                if existing_doc_id:
                    logger.info(f"📄 Document already exists: {url} -> {existing_doc_id}")
                    return IngestedResult(
                        url=url,
                        title=title,
                        source=source,
                        document_id=existing_doc_id,
                        ingestion_status="skipped",
                        content_length=0,
                        fetch_time=0.0,
                        selection_reason=selected_result.selection_reason,
                        priority=selected_result.priority
                    )
            
            # Fetch content using web_content_tool
            fetch_start = time.time()
            logger.info(f"🌐 Fetching content for: {url}")
            
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
                    url, title, content_response.data.result, selected_result, input_data
                )
                
                fetch_time = time.time() - fetch_start
                
                return IngestedResult(
                    url=url,
                    title=title,
                    source=source,
                    document_id=document_id,
                    ingestion_status="success",
                    content_length=len(content_response.data.result.content),
                    fetch_time=fetch_time,
                    selection_reason=selected_result.selection_reason,
                    priority=selected_result.priority
                )
            else:
                logger.warning(f"⚠️ web_content_tool failed for: {url}")
                
                # Create fallback content from search result metadata
                fallback_content = self._create_fallback_content(selected_result, input_data.original_query)
                
                # Create a mock result object for fallback content
                class FallbackResult:
                    def __init__(self, content, title):
                        self.content = content
                        self.title = title
                
                fallback_result = FallbackResult(fallback_content, title)
                
                # Ingest the fallback content
                document_id = await self._ingest_content(
                    url, title, fallback_result, selected_result, input_data
                )
                
                fetch_time = time.time() - fetch_start
                
                return IngestedResult(
                    url=url,
                    title=title,
                    source=source,
                    document_id=document_id,
                    ingestion_status="success",
                    content_length=len(fallback_content),
                    fetch_time=fetch_time,
                    selection_reason=f"{selected_result.selection_reason} (fallback content)",
                    priority=selected_result.priority
                )
                
        except Exception as e:
            logger.error(f"❌ Failed to process selected result {selected_result.url}: {e}")
            return IngestedResult(
                url=selected_result.url,
                title=selected_result.title,
                source=selected_result.source,
                ingestion_status="failed",
                content_length=0,
                fetch_time=0.0,
                selection_reason=selected_result.selection_reason,
                priority=selected_result.priority
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
    
    async def _ingest_content(self, url: str, title: str, content_result: Any, selected_result: SelectedResult, input_data: WebIngestSelectedInput) -> str:
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
                "original_query": input_data.original_query,
                "ingestion_date": datetime.utcnow().isoformat(),
                "category": input_data.category,
                "tags": input_data.tags + ["web_search", "ingested", "selected"],
                "content_type": "web_page",
                "original_url": url,
                "selection_reason": selected_result.selection_reason,
                "priority": selected_result.priority,
                "selection_criteria": "llm_analyzed"
            }
            
            # Create document content
            document_content = f"""
Title: {title}
URL: {url}
Source: {metadata['source']}
Selection Reason: {selected_result.selection_reason}
Priority: {selected_result.priority}

{content_result.content}

---
Ingested from web search query: "{input_data.original_query}"
Selection reason: {selected_result.selection_reason}
Priority: {selected_result.priority}
Ingestion date: {metadata['ingestion_date']}
"""
            
            # Store document using appropriate service (user-specific or global)
            if self.current_user_id and self.user_document_service:
                # Store in user's private collection
                logger.info(f"📥 Storing web document in user collection for user: {self.current_user_id}")
                logger.debug(f"🔍 UserDocumentService type: {type(self.user_document_service)}")
                logger.debug(f"🔍 UserDocumentService has method: {hasattr(self.user_document_service, 'store_text_document_for_user')}")
                
                if not hasattr(self.user_document_service, 'store_text_document_for_user'):
                    logger.error(f"❌ Method not found! Available methods: {[m for m in dir(self.user_document_service) if 'store' in m.lower()]}")
                    raise AttributeError("store_text_document_for_user method not found")
                
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
                logger.info(f"📥 Successfully ingested selected document: {doc_id} from {url}")
                return doc_id
            else:
                logger.error(f"❌ Failed to store selected document: {doc_id}")
                raise Exception("Document storage failed")
            
        except Exception as e:
            logger.error(f"❌ Failed to ingest selected content from {url}: {e}")
            raise
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and accessible"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return bool(parsed.scheme and parsed.netloc)
        except:
            return False
    
    def _create_fallback_content(self, selected_result: SelectedResult, original_query: str) -> str:
        """Create fallback content when web fetching fails"""
        try:
            # Create a structured fallback content using available metadata
            fallback_content = f"""
# {selected_result.title}

**Source:** {selected_result.source}
**URL:** {selected_result.url}
**Original Query:** {original_query}
**Selection Reason:** {selected_result.selection_reason}
**Priority:** {selected_result.priority}/5

## Summary
This content was selected from search results for the query: "{original_query}"

**Reason for Selection:** {selected_result.selection_reason}

**Source Information:**
- Domain: {selected_result.source}
- URL: {selected_result.url}
- Title: {selected_result.title}

## Note
This is fallback content created from search result metadata because the original web page could not be accessed. The actual content may be more comprehensive than what is shown here.

**Search Context:**
- Query: {original_query}
- Selection Priority: {selected_result.priority}/5
- Selection Criteria: {selected_result.selection_reason}

This fallback content preserves the search context and selection reasoning for future reference and analysis.
"""
            
            return fallback_content.strip()
            
        except Exception as e:
            logger.error(f"❌ Failed to create fallback content for {selected_result.url}: {e}")
            return ""
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get MCP tool definition for registration"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": WebIngestSelectedInput.schema(),
            "outputSchema": WebIngestSelectedOutput.schema()
        } 