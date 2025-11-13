"""
Filename Search Tool - MCP Tool for Exact Filename Matching
Allows LLM to search for documents by exact filename when semantic search fails
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional

from mcp.schemas.tool_schemas import ToolResponse, ToolError
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FilenameSearchInput(BaseModel):
    """Input for filename search"""
    filename: str = Field(..., description="Filename to search for (with or without extension)")
    include_content: bool = Field(True, description="Whether to include document content in results")
    limit: int = Field(10, ge=1, le=50, description="Maximum number of results")


class FilenameSearchResult(BaseModel):
    """Result from filename search"""
    document_id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Exact filename")
    title: Optional[str] = Field(None, description="Document title")
    content: Optional[str] = Field(None, description="Document content preview")
    doc_type: str = Field(..., description="Document type")
    category: Optional[str] = Field(None, description="Document category")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    author: Optional[str] = Field(None, description="Document author")
    upload_date: str = Field(..., description="Upload date")
    file_size: int = Field(..., description="File size in bytes")
    match_score: float = Field(..., description="Match score (1.0 for exact match)")


class FilenameSearchOutput(BaseModel):
    """Output from filename search"""
    results: List[FilenameSearchResult] = Field(..., description="Search results")
    total_found: int = Field(..., description="Total number of results")
    search_query: str = Field(..., description="Filename that was searched")
    search_time: float = Field(..., description="Search execution time")


class FilenameSearchTool:
    """MCP tool for exact filename matching"""
    
    def __init__(self, embedding_manager=None, document_repository=None):
        """Initialize with required services"""
        self.embedding_manager = embedding_manager
        self.document_repository = document_repository
        self.name = "search_by_filename"
        self.description = "Search for documents by exact filename match"
        
    async def initialize(self):
        """Initialize the filename search tool"""
        if not self.document_repository:
            raise ValueError("DocumentRepository is required")
        if not self.embedding_manager:
            raise ValueError("EmbeddingManager is required")
        
        logger.info("📁 FilenameSearchTool initialized")
    
    async def execute(self, input_data: FilenameSearchInput) -> ToolResponse:
        """Execute filename search with the given parameters"""
        start_time = time.time()
        
        try:
            logger.info(f"📁 Searching for filename: '{input_data.filename}'")
            
            # Clean the filename
            clean_filename = input_data.filename.strip()
            
            # Search for documents with matching filename
            query = """
                SELECT document_id, filename, title, doc_type, category, tags, author,
                       upload_date, file_size, metadata_json
                FROM document_metadata 
                WHERE LOWER(filename) = LOWER($1) 
                   OR LOWER(filename) LIKE LOWER($2)
                   OR LOWER(filename) LIKE LOWER($3)
                ORDER BY upload_date DESC
                LIMIT $4
            """
            
            # Search patterns: exact match, with extension, without extension
            search_patterns = [
                clean_filename,
                f"{clean_filename}.%",
                f"%{clean_filename}%"
            ]
            
            results = await self.document_repository.execute_query(
                query, 
                *search_patterns,
                input_data.limit
            )
            
            # Format results
            formatted_results = []
            for row in results:
                # Get content if requested
                content = None
                if input_data.include_content:
                    chunks = await self.embedding_manager.get_all_document_chunks(row['document_id'])
                    if chunks:
                        # Use first chunk as preview
                        content = chunks[0]['content'][:500] + "..." if len(chunks[0]['content']) > 500 else chunks[0]['content']
                
                # Calculate match score
                match_score = 1.0 if row['filename'].lower() == clean_filename.lower() else 0.8
                
                formatted_results.append(FilenameSearchResult(
                    document_id=row['document_id'],
                    filename=row['filename'],
                    title=row['title'],
                    content=content,
                    doc_type=row['doc_type'],
                    category=row['category'],
                    tags=row['tags'] or [],
                    author=row['author'],
                    upload_date=row['upload_date'].isoformat() if row['upload_date'] else "",
                    file_size=row['file_size'],
                    match_score=match_score
                ))
            
            # Create output
            output = FilenameSearchOutput(
                results=formatted_results,
                total_found=len(formatted_results),
                search_query=input_data.filename,
                search_time=time.time() - start_time
            )
            
            logger.info(f"✅ Filename search completed: {len(formatted_results)} results in {output.search_time:.2f}s")
            
            return ToolResponse(
                success=True,
                data=output,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"❌ Filename search failed: {e}")
            return ToolResponse(
                success=False,
                error=ToolError(
                    error_code="FILENAME_SEARCH_FAILED",
                    error_message=str(e),
                    details={"filename": input_data.filename}
                ),
                execution_time=time.time() - start_time
            )
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get MCP tool definition for registration"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": FilenameSearchInput.schema(),
            "outputSchema": FilenameSearchOutput.schema()
        } 