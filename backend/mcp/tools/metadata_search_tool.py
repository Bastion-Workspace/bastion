"""
Metadata Search Tool - MCP Tool for Intelligent Metadata-Based Searches
Allows LLM to perform sophisticated metadata queries like "find other books by the same author"
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, date

from mcp.schemas.tool_schemas import ToolResponse, ToolError
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MetadataSearchInput(BaseModel):
    """Input for metadata-based search"""
    search_type: str = Field(..., description="Type of metadata search: 'by_author', 'by_category', 'by_tags', 'by_publication_date', 'similar_metadata'")
    reference_document_id: Optional[str] = Field(None, description="Document ID to use as reference for metadata matching")
    author: Optional[str] = Field(None, description="Author name to search for")
    category: Optional[str] = Field(None, description="Category to search for")
    tags: Optional[List[str]] = Field(None, description="Tags to search for")
    publication_date_from: Optional[date] = Field(None, description="Publication date range start")
    publication_date_to: Optional[date] = Field(None, description="Publication date range end")
    exclude_reference: bool = Field(True, description="Exclude the reference document from results")
    limit: int = Field(10, ge=1, le=50, description="Maximum number of results")


class MetadataSearchResult(BaseModel):
    """Result from metadata search"""
    document_id: str = Field(..., description="Document ID")
    title: str = Field(..., description="Document title")
    filename: str = Field(..., description="Original filename")
    author: Optional[str] = Field(None, description="Document author")
    category: Optional[str] = Field(None, description="Document category")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    publication_date: Optional[date] = Field(None, description="Publication date")
    upload_date: datetime = Field(..., description="Upload date")
    file_size: int = Field(..., description="File size in bytes")
    doc_type: str = Field(..., description="Document type")
    similarity_reason: str = Field(..., description="Why this document matches the search")


class MetadataSearchOutput(BaseModel):
    """Output from metadata search"""
    results: List[MetadataSearchResult] = Field(..., description="Search results")
    total_found: int = Field(..., description="Total number of results")
    search_summary: str = Field(..., description="Summary of what was searched")
    search_time: float = Field(..., description="Search execution time")


class MetadataSearchTool:
    """MCP tool for intelligent metadata-based searches"""
    
    def __init__(self, document_repository=None):
        """Initialize with required services"""
        self.document_repository = document_repository
        self.name = "search_by_metadata"
        self.description = "Search documents by metadata relationships. REQUIRED: search_type must be one of: 'by_author', 'by_category', 'by_tags', 'by_publication_date', 'similar_metadata'"
        
    async def initialize(self):
        """Initialize the metadata search tool"""
        if not self.document_repository:
            raise ValueError("DocumentRepository is required")
        
        logger.info("🔍 MetadataSearchTool initialized")
    
    async def execute(self, input_data: MetadataSearchInput) -> ToolResponse:
        """Execute metadata-based search"""
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Executing metadata search: {input_data.search_type}")
            
            # Get reference document metadata if provided
            reference_metadata = None
            if input_data.reference_document_id:
                reference_metadata = await self._get_document_metadata(input_data.reference_document_id)
                if not reference_metadata:
                    return ToolResponse(
                        success=False,
                        error=ToolError(
                            error_code="REFERENCE_NOT_FOUND",
                            error_message=f"Reference document {input_data.reference_document_id} not found"
                        ),
                        execution_time=time.time() - start_time
                    )
            
            # Perform the specific type of metadata search
            results = await self._perform_metadata_search(input_data, reference_metadata)
            
            # Format results
            formatted_results = await self._format_results(results, input_data, reference_metadata)
            
            # Create search summary
            search_summary = self._create_search_summary(input_data, reference_metadata, len(formatted_results))
            
            output = MetadataSearchOutput(
                results=formatted_results,
                total_found=len(formatted_results),
                search_summary=search_summary,
                search_time=time.time() - start_time
            )
            
            logger.info(f"✅ Metadata search completed: {len(formatted_results)} results")
            
            return ToolResponse(
                success=True,
                data=output,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"❌ Metadata search failed: {e}")
            return ToolResponse(
                success=False,
                error=ToolError(
                    error_code="SEARCH_FAILED",
                    error_message=str(e)
                ),
                execution_time=time.time() - start_time
            )
    
    async def _get_document_metadata(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific document"""
        try:
            doc_info = await self.document_repository.get_by_id(document_id)
            if doc_info:
                return {
                    "document_id": doc_info.document_id,
                    "title": doc_info.title,
                    "filename": doc_info.filename,
                    "author": doc_info.author,
                    "category": doc_info.category.value if doc_info.category else None,
                    "tags": doc_info.tags,
                    "publication_date": doc_info.publication_date,
                    "upload_date": doc_info.upload_date,
                    "file_size": doc_info.file_size,
                    "doc_type": doc_info.doc_type.value
                }
            return None
        except Exception as e:
            logger.error(f"❌ Failed to get document metadata: {e}")
            return None
    
    async def _perform_metadata_search(self, input_data: MetadataSearchInput, reference_metadata: Optional[Dict]) -> List[Dict[str, Any]]:
        """Perform the actual metadata search based on type"""
        
        if input_data.search_type == "by_author":
            return await self._search_by_author(input_data, reference_metadata)
        elif input_data.search_type == "by_category":
            return await self._search_by_category(input_data, reference_metadata)
        elif input_data.search_type == "by_tags":
            return await self._search_by_tags(input_data, reference_metadata)
        elif input_data.search_type == "by_publication_date":
            return await self._search_by_publication_date(input_data, reference_metadata)
        elif input_data.search_type == "similar_metadata":
            return await self._search_similar_metadata(input_data, reference_metadata)
        else:
            raise ValueError(f"Unknown search type: {input_data.search_type}")
    
    async def _search_by_author(self, input_data: MetadataSearchInput, reference_metadata: Optional[Dict]) -> List[Dict[str, Any]]:
        """Search for documents by the same author"""
        try:
            # Determine author to search for
            author = input_data.author
            if not author and reference_metadata:
                author = reference_metadata.get("author")
            
            if not author:
                return []
            
            # Build query
            query = "SELECT * FROM document_metadata WHERE author ILIKE $1"
            params = [f"%{author}%"]
            
            # Exclude reference document if requested
            if input_data.exclude_reference and reference_metadata:
                query += " AND document_id != $2"
                params.append(reference_metadata["document_id"])
            
            query += " ORDER BY publication_date DESC, upload_date DESC LIMIT $" + str(len(params) + 1)
            params.append(input_data.limit)
            
            results = await self.document_repository.execute_query(query, *params)
            return results
            
        except Exception as e:
            logger.error(f"❌ Author search failed: {e}")
            return []
    
    async def _search_by_category(self, input_data: MetadataSearchInput, reference_metadata: Optional[Dict]) -> List[Dict[str, Any]]:
        """Search for documents in the same category"""
        try:
            # Determine category to search for
            category = input_data.category
            if not category and reference_metadata:
                category = reference_metadata.get("category")
            
            if not category:
                return []
            
            # Build query
            query = "SELECT * FROM document_metadata WHERE category = $1"
            params = [category]
            
            # Exclude reference document if requested
            if input_data.exclude_reference and reference_metadata:
                query += " AND document_id != $2"
                params.append(reference_metadata["document_id"])
            
            query += " ORDER BY publication_date DESC, upload_date DESC LIMIT $" + str(len(params) + 1)
            params.append(input_data.limit)
            
            results = await self.document_repository.execute_query(query, *params)
            return results
            
        except Exception as e:
            logger.error(f"❌ Category search failed: {e}")
            return []
    
    async def _search_by_tags(self, input_data: MetadataSearchInput, reference_metadata: Optional[Dict]) -> List[Dict[str, Any]]:
        """Search for documents with overlapping tags"""
        try:
            # Determine tags to search for
            tags = input_data.tags
            if not tags and reference_metadata:
                tags = reference_metadata.get("tags", [])
            
            if not tags:
                return []
            
            # Build query for tag overlap
            query = "SELECT * FROM document_metadata WHERE tags && $1"
            params = [tags]
            
            # Exclude reference document if requested
            if input_data.exclude_reference and reference_metadata:
                query += " AND document_id != $2"
                params.append(reference_metadata["document_id"])
            
            query += " ORDER BY publication_date DESC, upload_date DESC LIMIT $" + str(len(params) + 1)
            params.append(input_data.limit)
            
            results = await self.document_repository.execute_query(query, *params)
            return results
            
        except Exception as e:
            logger.error(f"❌ Tags search failed: {e}")
            return []
    
    async def _search_by_publication_date(self, input_data: MetadataSearchInput, reference_metadata: Optional[Dict]) -> List[Dict[str, Any]]:
        """Search for documents published in a date range"""
        try:
            # Determine date range
            date_from = input_data.publication_date_from
            date_to = input_data.publication_date_to
            
            if not date_from and not date_to and reference_metadata:
                # Search for documents published in the same year as reference
                ref_date = reference_metadata.get("publication_date")
                if ref_date:
                    if isinstance(ref_date, str):
                        ref_date = datetime.fromisoformat(ref_date).date()
                    date_from = date(ref_date.year, 1, 1)
                    date_to = date(ref_date.year, 12, 31)
            
            if not date_from and not date_to:
                return []
            
            # Build query
            where_conditions = []
            params = []
            
            if date_from:
                params.append(date_from)
                where_conditions.append(f"publication_date >= ${len(params)}")
            
            if date_to:
                params.append(date_to)
                where_conditions.append(f"publication_date <= ${len(params)}")
            
            query = f"SELECT * FROM document_metadata WHERE {' AND '.join(where_conditions)}"
            
            # Exclude reference document if requested
            if input_data.exclude_reference and reference_metadata:
                params.append(reference_metadata["document_id"])
                query += f" AND document_id != ${len(params)}"
            
            query += f" ORDER BY publication_date DESC, upload_date DESC LIMIT ${len(params) + 1}"
            params.append(input_data.limit)
            
            results = await self.document_repository.execute_query(query, *params)
            return results
            
        except Exception as e:
            logger.error(f"❌ Publication date search failed: {e}")
            return []
    
    async def _search_similar_metadata(self, input_data: MetadataSearchInput, reference_metadata: Optional[Dict]) -> List[Dict[str, Any]]:
        """Search for documents with similar metadata profile"""
        try:
            if not reference_metadata:
                return []
            
            # Score documents based on metadata similarity
            query = """
            SELECT *,
                CASE 
                    WHEN author ILIKE $1 THEN 3
                    ELSE 0
                END +
                CASE 
                    WHEN category = $2 THEN 2
                    ELSE 0
                END +
                CASE 
                    WHEN tags && $3 THEN array_length(tags & $3, 1)
                    ELSE 0
                END as similarity_score
            FROM document_metadata
            WHERE (
                author ILIKE $1 OR 
                category = $2 OR 
                tags && $3
            )
            """
            
            params = [
                f"%{reference_metadata.get('author', '')}%",
                reference_metadata.get('category'),
                reference_metadata.get('tags', [])
            ]
            
            # Exclude reference document if requested
            if input_data.exclude_reference:
                query += " AND document_id != $4"
                params.append(reference_metadata["document_id"])
            
            query += " ORDER BY similarity_score DESC, publication_date DESC LIMIT $" + str(len(params) + 1)
            params.append(input_data.limit)
            
            results = await self.document_repository.execute_query(query, *params)
            return results
            
        except Exception as e:
            logger.error(f"❌ Similar metadata search failed: {e}")
            return []
    
    async def _format_results(self, results: List[Dict[str, Any]], input_data: MetadataSearchInput, reference_metadata: Optional[Dict]) -> List[MetadataSearchResult]:
        """Format raw database results into structured output"""
        formatted_results = []
        
        for result in results:
            # Determine similarity reason
            similarity_reason = self._generate_similarity_reason(result, input_data, reference_metadata)
            
            formatted_result = MetadataSearchResult(
                document_id=result["document_id"],
                title=result["title"] or result["filename"],
                filename=result["filename"],
                author=result["author"],
                category=result["category"],
                tags=result["tags"] or [],
                publication_date=result["publication_date"],
                upload_date=result["upload_date"],
                file_size=result["file_size"],
                doc_type=result["doc_type"],
                similarity_reason=similarity_reason
            )
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    def _generate_similarity_reason(self, result: Dict[str, Any], input_data: MetadataSearchInput, reference_metadata: Optional[Dict]) -> str:
        """Generate explanation for why this document matches"""
        
        if input_data.search_type == "by_author":
            author = input_data.author or (reference_metadata.get("author") if reference_metadata else "")
            return f"Same author: {author}"
        
        elif input_data.search_type == "by_category":
            category = input_data.category or (reference_metadata.get("category") if reference_metadata else "")
            return f"Same category: {category}"
        
        elif input_data.search_type == "by_tags":
            tags = input_data.tags or (reference_metadata.get("tags", []) if reference_metadata else [])
            matching_tags = set(result.get("tags", [])) & set(tags)
            return f"Shared tags: {', '.join(matching_tags)}"
        
        elif input_data.search_type == "by_publication_date":
            return f"Published: {result.get('publication_date', 'Unknown date')}"
        
        elif input_data.search_type == "similar_metadata":
            reasons = []
            if reference_metadata:
                if result.get("author") and reference_metadata.get("author"):
                    if reference_metadata["author"].lower() in result["author"].lower():
                        reasons.append("same author")
                if result.get("category") == reference_metadata.get("category"):
                    reasons.append("same category")
                if result.get("tags") and reference_metadata.get("tags"):
                    matching_tags = set(result["tags"]) & set(reference_metadata["tags"])
                    if matching_tags:
                        reasons.append(f"shared tags: {', '.join(matching_tags)}")
            return f"Similar metadata: {', '.join(reasons) if reasons else 'multiple factors'}"
        
        return "Metadata match"
    
    def _create_search_summary(self, input_data: MetadataSearchInput, reference_metadata: Optional[Dict], result_count: int) -> str:
        """Create a summary of what was searched"""
        
        if input_data.search_type == "by_author":
            author = input_data.author or (reference_metadata.get("author") if reference_metadata else "")
            return f"Found {result_count} documents by author '{author}'"
        
        elif input_data.search_type == "by_category":
            category = input_data.category or (reference_metadata.get("category") if reference_metadata else "")
            return f"Found {result_count} documents in category '{category}'"
        
        elif input_data.search_type == "by_tags":
            tags = input_data.tags or (reference_metadata.get("tags", []) if reference_metadata else [])
            return f"Found {result_count} documents with tags: {', '.join(tags)}"
        
        elif input_data.search_type == "by_publication_date":
            date_range = ""
            if input_data.publication_date_from and input_data.publication_date_to:
                date_range = f"from {input_data.publication_date_from} to {input_data.publication_date_to}"
            elif input_data.publication_date_from:
                date_range = f"from {input_data.publication_date_from}"
            elif input_data.publication_date_to:
                date_range = f"until {input_data.publication_date_to}"
            elif reference_metadata and reference_metadata.get("publication_date"):
                ref_date = reference_metadata["publication_date"]
                if isinstance(ref_date, str):
                    ref_date = datetime.fromisoformat(ref_date).date()
                date_range = f"in {ref_date.year}"
            return f"Found {result_count} documents published {date_range}"
        
        elif input_data.search_type == "similar_metadata":
            ref_title = reference_metadata.get("title", "reference document") if reference_metadata else "reference"
            return f"Found {result_count} documents with similar metadata to '{ref_title}'"
        
        return f"Found {result_count} documents"
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get MCP tool definition for registration"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": MetadataSearchInput.schema(),
            "outputSchema": MetadataSearchOutput.schema()
        } 