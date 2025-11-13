"""
Crawl4AI Tool - MCP Tool for Advanced Web Crawling and Content Extraction
Integrates Crawl4AI for LLM-optimized web content extraction with advanced anti-detection
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

from mcp.schemas.tool_schemas import ToolResponse, ToolError
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Crawl4AIInput(BaseModel):
    """Input for Crawl4AI web crawling"""
    url: str = Field(..., description="URL to crawl")
    extraction_strategy: str = Field("markdown", description="Extraction strategy: 'markdown', 'text', 'html', 'llm_extraction'")
    llm_question: Optional[str] = Field(None, description="Question for LLM extraction (if using llm_extraction strategy)")
    max_content_length: int = Field(50000, ge=1000, le=200000, description="Maximum content length to extract")
    include_links: bool = Field(True, description="Whether to include extracted links")
    include_metadata: bool = Field(True, description="Whether to include page metadata")
    browser_options: Dict[str, Any] = Field(default_factory=dict, description="Browser configuration options")
    timeout: int = Field(60, ge=10, le=300, description="Request timeout in seconds")


class Crawl4AIMetadata(BaseModel):
    """Metadata from Crawl4AI extraction"""
    title: Optional[str] = Field(None, description="Page title")
    description: Optional[str] = Field(None, description="Page description")
    author: Optional[str] = Field(None, description="Page author")
    published_date: Optional[str] = Field(None, description="Publication date")
    keywords: List[str] = Field(default_factory=list, description="Page keywords")
    language: Optional[str] = Field(None, description="Page language")
    content_type: Optional[str] = Field(None, description="Content type")
    word_count: Optional[int] = Field(None, description="Word count")
    reading_time: Optional[int] = Field(None, description="Estimated reading time in minutes")


class Crawl4AIResult(BaseModel):
    """Result from Crawl4AI crawling"""
    url: str = Field(..., description="Original URL")
    title: str = Field(..., description="Page title")
    content: str = Field(..., description="Extracted content")
    metadata: Crawl4AIMetadata = Field(..., description="Page metadata")
    links: List[str] = Field(default_factory=list, description="Extracted links")
    content_length: int = Field(..., description="Content length in characters")
    fetch_time: float = Field(..., description="Time taken to fetch content")
    status_code: int = Field(..., description="HTTP status code")
    extraction_strategy: str = Field(..., description="Strategy used for extraction")
    llm_response: Optional[str] = Field(None, description="LLM extraction response if used")


class Crawl4AIOutput(BaseModel):
    """Output from Crawl4AI crawling"""
    url: str = Field(..., description="Original URL")
    result: Crawl4AIResult = Field(..., description="Crawled content result")
    success: bool = Field(..., description="Whether crawl was successful")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    crawl_summary: str = Field(..., description="Summary of crawl operation")


class Crawl4AITool:
    """MCP tool for advanced web crawling using Crawl4AI"""
    
    def __init__(self, config=None):
        """Initialize with configuration"""
        self.config = config or {}
        self.name = "crawl4ai_web_crawler"
        self.description = "Advanced web crawling and content extraction using Crawl4AI"
        self.current_user_id = None  # Will be set by MCP chat service
        
        # Crawl4AI configuration
        self.browser_options = {
            "headless": True,
            "proxy": None,  # Can be configured via config
            "user_agent": None,  # Will use Crawl4AI's default
            "viewport": {"width": 1920, "height": 1080},
            "timeout": 30000
        }
        
        # Update with config if provided
        if config and "browser_options" in config:
            self.browser_options.update(config["browser_options"])
        
        # Initialize Crawl4AI components
        self.crawler = None
        self._initialized = False
    
    def set_current_user(self, user_id: str):
        """Set the current user ID for the tool"""
        self.current_user_id = user_id
        logger.debug(f"🔐 Crawl4AITool user set to: {user_id}")
    
    async def initialize(self):
        """Initialize the Crawl4AI tool"""
        try:
            # Import Crawl4AI components
            from crawl4ai import AsyncWebCrawler
            
            logger.info("📄 Crawl4AITool initializing...")
            
            # Test connectivity with a simple crawl
            async with AsyncWebCrawler() as test_crawler:
                test_result = await test_crawler.arun(
                    url="https://httpbin.org/get",
                    extraction_strategy="markdown"
                )
                if test_result and test_result.markdown:
                    logger.info("✅ Crawl4AI connectivity verified")
                else:
                    logger.warning("⚠️ Crawl4AI connectivity test failed")
            
            self._initialized = True
            logger.info("✅ Crawl4AITool initialized successfully")
            
        except ImportError as e:
            logger.error(f"❌ Crawl4AI not installed: {e}")
            raise ValueError("Crawl4AI is required but not installed. Run: pip install crawl4ai")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Crawl4AITool: {e}")
            raise
    
    async def execute(self, input_data: Crawl4AIInput) -> ToolResponse:
        """Execute web crawling with Crawl4AI"""
        start_time = time.time()
        
        try:
            if not self._initialized:
                await self.initialize()
            
            # Import Crawl4AI components
            from crawl4ai import AsyncWebCrawler
            
            # Prepare crawl parameters
            crawl_params = {
                "url": input_data.url,
                "extraction_strategy": input_data.extraction_strategy,
                "max_content_length": input_data.max_content_length,
                "include_links": input_data.include_links,
                "include_metadata": input_data.include_metadata,
                "timeout": input_data.timeout * 1000  # Convert to milliseconds
            }
            
            # Add LLM question if using LLM extraction
            if input_data.extraction_strategy == "llm_extraction" and input_data.llm_question:
                crawl_params["llm_question"] = input_data.llm_question
            
            # Add browser options
            if input_data.browser_options:
                crawl_params.update(input_data.browser_options)
            
            logger.info(f"🌐 Crawl4AI crawling: {input_data.url} with strategy: {input_data.extraction_strategy}")
            
            # Execute crawl
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(**crawl_params)
                
                if not result:
                    raise Exception("Crawl4AI returned no result")
                
                # Extract content based on strategy
                if input_data.extraction_strategy == "markdown":
                    content = result.markdown or ""
                elif input_data.extraction_strategy == "text":
                    content = result.text or ""
                elif input_data.extraction_strategy == "html":
                    content = result.html or ""
                elif input_data.extraction_strategy == "llm_extraction":
                    content = result.llm_response or result.markdown or ""
                else:
                    content = result.markdown or ""
                
                # Truncate content if needed
                if len(content) > input_data.max_content_length:
                    content = content[:input_data.max_content_length]
                    logger.info(f"📏 Content truncated to {input_data.max_content_length} characters")
                
                # Extract metadata
                metadata = Crawl4AIMetadata(
                    title=result.metadata.get("title") if result.metadata else None,
                    description=result.metadata.get("description") if result.metadata else None,
                    author=result.metadata.get("author") if result.metadata else None,
                    published_date=result.metadata.get("published_date") if result.metadata else None,
                    keywords=result.metadata.get("keywords", []) if result.metadata else [],
                    language=result.metadata.get("language") if result.metadata else None,
                    content_type=result.metadata.get("content_type") if result.metadata else None,
                    word_count=result.metadata.get("word_count") if result.metadata else None,
                    reading_time=result.metadata.get("reading_time") if result.metadata else None
                )
                
                # Extract links
                links = result.links if hasattr(result, 'links') and result.links else []
                
                # Create result
                crawl_result = Crawl4AIResult(
                    url=input_data.url,
                    title=metadata.title or "No title",
                    content=content,
                    metadata=metadata,
                    links=links,
                    content_length=len(content),
                    fetch_time=time.time() - start_time,
                    status_code=result.status_code if hasattr(result, 'status_code') else 200,
                    extraction_strategy=input_data.extraction_strategy,
                    llm_response=result.llm_response if hasattr(result, 'llm_response') else None
                )
                
                # Create output
                output = Crawl4AIOutput(
                    url=input_data.url,
                    result=crawl_result,
                    success=True,
                    error_message=None,
                    crawl_summary=f"Successfully crawled {input_data.url} using {input_data.extraction_strategy} strategy. Extracted {len(content)} characters in {crawl_result.fetch_time:.2f}s"
                )
                
                logger.info(f"✅ Crawl4AI crawl completed: {len(content)} chars in {crawl_result.fetch_time:.2f}s")
                
                return ToolResponse(
                    success=True,
                    data=output,
                    execution_time=time.time() - start_time
                )
                
        except Exception as e:
            logger.error(f"❌ Crawl4AI crawl failed for {input_data.url}: {e}")
            return ToolResponse(
                success=False,
                error=ToolError(
                    error_code="CRAWL_FAILED",
                    error_message=str(e),
                    details={"url": input_data.url, "strategy": input_data.extraction_strategy}
                ),
                execution_time=time.time() - start_time
            )
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get MCP tool definition for registration"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": Crawl4AIInput.model_json_schema(),
            "outputSchema": Crawl4AIOutput.model_json_schema()
        } 