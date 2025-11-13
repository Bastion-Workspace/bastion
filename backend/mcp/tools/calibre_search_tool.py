"""
Calibre Search Tool - MCP Tool for Calibre Library Search
Allows LLM to search the connected Calibre ebook library
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional

from mcp.schemas.tool_schemas import (
    CalibreSearchInput,
    CalibreSearchOutput,
    CalibreBookResult,
    ToolResponse,
    ToolError
)
from services.calibre_search_service import CalibreSearchService

logger = logging.getLogger(__name__)


class CalibreSearchTool:
    """MCP tool for Calibre library search"""
    
    def __init__(self, calibre_service: CalibreSearchService = None):
        """Initialize with Calibre search service"""
        self.calibre_service = calibre_service or CalibreSearchService()
        self.name = "search_calibre_library"
        self.description = "Search LOCAL Calibre ebook library (3000+ books) with full-text search capability for books, authors, and series"
        
    async def initialize(self):
        """Initialize the Calibre search tool"""
        try:
            if not self.calibre_service._initialized:
                await self.calibre_service.initialize()
            
            logger.info("📚 CalibreSearchTool initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize CalibreSearchTool: {e}")
            raise
    
    async def execute(self, input_data: CalibreSearchInput) -> ToolResponse:
        """Execute Calibre search with the given parameters"""
        start_time = time.time()
        
        try:
            logger.info(f"📚 Executing Calibre search: '{input_data.query}' (limit: {input_data.limit})")
            
            # Check if Calibre is available
            if not self.calibre_service.is_available():
                logger.warning("📚 Calibre library not available")
                return ToolResponse(
                    success=True,
                    data=CalibreSearchOutput(
                        results=[],
                        total_found=0,
                        query_used=input_data.query,
                        search_time=0.0,
                        library_available=False
                    ),
                    execution_time=time.time() - start_time
                )
            
            # Perform search
            search_results = await self._perform_calibre_search(input_data)
            
            # Create output
            output = CalibreSearchOutput(
                results=search_results,
                total_found=len(search_results),
                query_used=input_data.query,
                search_time=time.time() - start_time,
                library_available=True
            )
            
            logger.info(f"✅ Calibre search completed: {len(search_results)} results in {output.search_time:.2f}s")
            
            return ToolResponse(
                success=True,
                data=output,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"❌ Calibre search tool execution failed: {e}")
            return ToolResponse(
                success=False,
                error=ToolError(
                    error_code="CALIBRE_SEARCH_FAILED",
                    error_message=str(e),
                    details={"query": input_data.query, "limit": input_data.limit}
                ),
                execution_time=time.time() - start_time
            )
    
    async def _perform_calibre_search(self, input_data: CalibreSearchInput) -> List[Dict[str, Any]]:
        """Perform the actual Calibre search"""
        try:
            # Search Calibre library
            results = await self.calibre_service.search_calibre_only(
                query=input_data.query,
                limit=input_data.limit
            )
            
            # Format results using schema models
            formatted_results = []
            for result in results:
                book_result = CalibreBookResult(
                    book_id=result["document_id"],
                    title=result["title"],
                    authors=result["authors"],
                    content=result["content"],
                    score=result["score"],
                    formats=result["formats"],
                    source="calibre",
                    metadata={
                        "series": result.get("series_info"),
                        "publisher": result["metadata"].get("publisher"),
                        "publication_date": result["metadata"].get("pubdate"),
                        "isbn": result["metadata"].get("isbn"),
                        "tags": result["metadata"].get("tags", []),
                        "rating": result["metadata"].get("rating"),
                        "path": result["metadata"].get("path"),
                        "timestamp": result["metadata"].get("timestamp")
                    } if input_data.include_metadata else None
                )
                formatted_results.append(book_result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Calibre search execution failed: {e}")
            return []
    
    async def get_library_status(self) -> Dict[str, Any]:
        """Get current Calibre library status and information"""
        try:
            return await self.calibre_service.get_library_info()
        except Exception as e:
            logger.error(f"❌ Failed to get library status: {e}")
            return {"available": False, "error": str(e)}
    
    def get_tool_schema(self) -> Dict[str, Any]:
        """Get the MCP tool schema for this tool"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for books, authors, or series"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 20)",
                        "default": 20,
                        "minimum": 1,
                        "maximum": 100
                    },
                    "include_metadata": {
                        "type": "boolean",
                        "description": "Include detailed book metadata in results",
                        "default": True
                    },
                    "search_type": {
                        "type": "string",
                        "description": "Type of search to perform",
                        "enum": ["books", "authors", "series"],
                        "default": "books"
                    }
                },
                "required": ["query"]
            }
        }
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get the tool definition (alias for get_tool_schema for compatibility)"""
        return self.get_tool_schema()


# Helper function for testing and direct usage
async def search_calibre_books(
    query: str, 
    limit: int = 20, 
    include_metadata: bool = True
) -> Dict[str, Any]:
    """
    Direct function to search Calibre library
    Useful for testing and integration without MCP layer
    """
    tool = CalibreSearchTool()
    await tool.initialize()
    
    input_data = CalibreSearchInput(
        query=query,
        limit=limit,
        include_metadata=include_metadata
    )
    
    response = await tool.execute(input_data)
    return response.dict() if response.success else {"error": response.error}
