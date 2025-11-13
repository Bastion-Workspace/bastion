"""
MCP Tool Schemas
Pydantic models defining inputs and outputs for all MCP tools
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime, date
from pydantic import BaseModel, Field, validator
from enum import Enum


class DocumentType(str, Enum):
    """Supported document types"""
    PDF = "pdf"
    EPUB = "epub"
    TXT = "txt"
    DOCX = "docx"
    HTML = "html"
    URL = "url"
    EML = "eml"
    ZIP = "zip"


class AnalysisType(str, Enum):
    """Types of collection analysis"""
    SUMMARY = "summary"
    COMPREHENSIVE = "comprehensive"
    TEMPORAL = "temporal"
    THEMATIC = "thematic"


class SearchFilters(BaseModel):
    """Filters for document searches"""
    categories: Optional[List[str]] = Field(None, description="Filter by document categories")
    tags: Optional[List[str]] = Field(None, description="Filter by document tags")
    document_types: Optional[List[DocumentType]] = Field(None, description="Filter by document types")
    date_from: Optional[datetime] = Field(None, description="Filter documents from this date")
    date_to: Optional[datetime] = Field(None, description="Filter documents until this date")
    publication_date_from: Optional[date] = Field(None, description="Filter documents published from this date")
    publication_date_to: Optional[date] = Field(None, description="Filter documents published until this date")
    author: Optional[str] = Field(None, description="Filter by document author")
    file_size_min: Optional[int] = Field(None, description="Minimum file size in bytes")
    file_size_max: Optional[int] = Field(None, description="Maximum file size in bytes")


# ========== SEARCH TOOL SCHEMAS ==========

class SearchDocumentsInput(BaseModel):
    """Input for vector search tool"""
    query: str = Field(..., description="Search query text")
    limit: int = Field(100, ge=1, le=300, description="Maximum number of results (up to 300 for comprehensive coverage)")
    similarity_threshold: float = Field(0.4, ge=0.0, le=1.0, description="Minimum similarity score")
    filters: Optional[SearchFilters] = Field(None, description="Additional search filters")
    use_expansion: bool = Field(True, description="Whether to use query expansion")
    
    @validator('query', pre=True)
    def validate_query(cls, v):
        """Validate search query"""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        if len(v.strip()) > 1000:
            raise ValueError("Query too long (max 1000 characters)")
        return v.strip()


class SearchResult(BaseModel):
    """Individual search result"""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Document this chunk belongs to")
    document_title: str = Field(..., description="Document title or filename")
    content: str = Field(..., description="Chunk text content")
    similarity_score: float = Field(..., description="Similarity score to query")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SearchDocumentsOutput(BaseModel):
    """Output from vector search tool"""
    results: List[SearchResult] = Field(..., description="Search results")
    total_found: int = Field(..., description="Total number of results found")
    query_used: str = Field(..., description="Actual query used for search")
    search_time: float = Field(..., description="Search execution time in seconds")


# ========== DOCUMENT TOOL SCHEMAS ==========

class DocumentInfo(BaseModel):
    """Complete document information"""
    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    title: Optional[str] = Field(None, description="Document title")
    content: str = Field(..., description="Full document content")
    doc_type: DocumentType = Field(..., description="Document type")
    category: Optional[str] = Field(None, description="Document category")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    author: Optional[str] = Field(None, description="Document author")
    publication_date: Optional[date] = Field(None, description="Original publication date")
    upload_date: datetime = Field(..., description="Upload timestamp")
    file_size: int = Field(..., description="File size in bytes")
    page_count: Optional[int] = Field(None, description="Number of pages")
    chunk_count: int = Field(..., description="Number of text chunks")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class GetDocumentInput(BaseModel):
    """Input for document retrieval tool"""
    document_id: str = Field(..., description="Document ID to retrieve")
    include_content: bool = Field(True, description="Include full document content")
    include_metadata: bool = Field(True, description="Include document metadata")


class GetDocumentOutput(BaseModel):
    """Output from document retrieval tool"""
    document: DocumentInfo = Field(..., description="Document information and content")
    retrieval_time: float = Field(..., description="Retrieval time in seconds")


# ========== METADATA SEARCH SCHEMAS ==========

class SearchByMetadataInput(BaseModel):
    """Input for metadata-based search"""
    filters: SearchFilters = Field(..., description="Metadata filters to apply")
    limit: int = Field(50, ge=1, le=200, description="Maximum number of results")
    sort_by: str = Field("upload_date", description="Field to sort by")
    sort_desc: bool = Field(True, description="Sort in descending order")


class MetadataSearchResult(BaseModel):
    """Metadata search result"""
    document_id: str = Field(..., description="Document identifier")
    filename: str = Field(..., description="Document filename")
    title: Optional[str] = Field(None, description="Document title")
    doc_type: DocumentType = Field(..., description="Document type")
    category: Optional[str] = Field(None, description="Document category")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    upload_date: datetime = Field(..., description="Upload date")
    file_size: int = Field(..., description="File size in bytes")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SearchByMetadataOutput(BaseModel):
    """Output from metadata search"""
    results: List[MetadataSearchResult] = Field(..., description="Search results")
    total_found: int = Field(..., description="Total documents matching criteria")
    filters_applied: SearchFilters = Field(..., description="Filters that were applied")
    search_time: float = Field(..., description="Search execution time")


# ========== ENTITY EXPLORATION SCHEMAS ==========

class ExploreEntitiesInput(BaseModel):
    """Input for entity exploration tool"""
    entity_name: str = Field(..., description="Entity name to explore")
    max_documents: int = Field(20, ge=1, le=100, description="Maximum documents to return")
    include_related: bool = Field(True, description="Include related entities")
    relationship_depth: int = Field(1, ge=1, le=3, description="Depth of entity relationships")


class EntityRelationship(BaseModel):
    """Entity relationship information"""
    related_entity: str = Field(..., description="Related entity name")
    relationship_type: str = Field(..., description="Type of relationship")
    strength: float = Field(..., ge=0.0, le=1.0, description="Relationship strength")


class EntityInfo(BaseModel):
    """Entity information"""
    entity_name: str = Field(..., description="Entity name")
    entity_type: str = Field(..., description="Entity type (person, organization, etc.)")
    document_count: int = Field(..., description="Number of documents mentioning entity")
    relationships: List[EntityRelationship] = Field(default_factory=list, description="Related entities")


class ExploreEntitiesOutput(BaseModel):
    """Output from entity exploration"""
    entity_info: EntityInfo = Field(..., description="Information about the entity")
    relevant_documents: List[MetadataSearchResult] = Field(..., description="Documents mentioning entity")
    exploration_time: float = Field(..., description="Exploration time in seconds")


# ========== COLLECTION ANALYSIS SCHEMAS ==========

class AnalyzeCollectionInput(BaseModel):
    """Input for collection analysis tool"""
    document_ids: List[str] = Field(..., min_items=1, max_items=1000, description="Document IDs to analyze")
    analysis_type: AnalysisType = Field(AnalysisType.SUMMARY, description="Type of analysis to perform")
    focus_query: Optional[str] = Field(None, description="Specific focus for the analysis")
    include_citations: bool = Field(True, description="Include citations in analysis")


class AnalysisInsight(BaseModel):
    """Individual analysis insight"""
    insight_type: str = Field(..., description="Type of insight (theme, trend, etc.)")
    title: str = Field(..., description="Insight title")
    description: str = Field(..., description="Detailed description")
    supporting_documents: List[str] = Field(..., description="Document IDs supporting this insight")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class AnalyzeCollectionOutput(BaseModel):
    """Output from collection analysis"""
    analysis_summary: str = Field(..., description="Overall analysis summary")
    key_insights: List[AnalysisInsight] = Field(..., description="Key insights discovered")
    document_count: int = Field(..., description="Number of documents analyzed")
    analysis_type: AnalysisType = Field(..., description="Type of analysis performed")
    processing_time: float = Field(..., description="Analysis processing time")
    citations: List[str] = Field(default_factory=list, description="Citation references")


# ========== GENERAL TOOL SCHEMAS ==========

class ToolError(BaseModel):
    """Tool execution error"""
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class ToolResponse(BaseModel):
    """Generic tool response wrapper"""
    success: bool = Field(..., description="Whether tool execution succeeded")
    data: Optional[Any] = Field(None, description="Tool output data - can be any valid output model")
    error: Optional[ToolError] = Field(None, description="Error information if failed")
    execution_time: float = Field(..., description="Total execution time in seconds")


# ========== VALIDATORS ==========
# Validators are now embedded in their respective model classes above 

# ========== WEB SEARCH INGESTION SCHEMAS ==========

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


# ========== CALIBRE TOOL SCHEMAS ==========

class CalibreSearchInput(BaseModel):
    """Input for Calibre ebook library search"""
    query: str = Field(..., description="Search query for books, authors, or series")
    limit: int = Field(20, ge=1, le=100, description="Maximum number of results to return")
    include_metadata: bool = Field(True, description="Include detailed book metadata in results")
    search_type: str = Field("books", description="Type of search: 'books', 'authors', 'series'")
    
    @validator('query', pre=True)
    def validate_query(cls, v):
        """Validate search query"""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        if len(v.strip()) > 500:
            raise ValueError("Query too long (max 500 characters)")
        return v.strip()


class CalibreBookResult(BaseModel):
    """Individual Calibre book result"""
    book_id: str = Field(..., description="Unique book identifier")
    title: str = Field(..., description="Book title")
    authors: str = Field(..., description="Book authors")
    content: str = Field(..., description="Book description or excerpt")
    score: float = Field(..., description="Relevance score")
    formats: List[str] = Field(default_factory=list, description="Available formats (PDF, EPUB, etc.)")
    source: str = Field("calibre", description="Source identifier")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional book metadata")


class CalibreSearchOutput(BaseModel):
    """Output from Calibre search tool"""
    results: List[CalibreBookResult] = Field(..., description="Book search results")
    total_found: int = Field(..., description="Total number of results found")
    query_used: str = Field(..., description="Query that was executed")
    search_time: float = Field(..., description="Time taken for search in seconds")
    library_available: bool = Field(..., description="Whether Calibre library is available")


class CalibreBookAnalysisInput(BaseModel):
    """Input for Calibre book analysis tool"""
    book_id: int = Field(..., description="Calibre book ID to analyze")
    analysis_type: str = Field("metadata", description="Type of analysis: 'metadata', 'content', 'llm_ready'")
    preferred_format: str = Field("epub", description="Preferred book format for analysis")
    max_content_length: int = Field(2000, ge=100, le=10000, description="Maximum content length to extract")
    segment_size: int = Field(2000, ge=500, le=5000, description="Size of content segments for LLM processing")
    overlap_size: int = Field(200, ge=0, le=1000, description="Overlap between segments")
    break_at_sentences: bool = Field(True, description="Break segments at sentence boundaries")
    include_metadata: bool = Field(True, description="Include book metadata in analysis")


class CalibreBookAnalysisOutput(BaseModel):
    """Output from Calibre book analysis tool"""
    book_id: int = Field(..., description="Calibre book ID")
    title: str = Field(..., description="Book title")
    authors: str = Field(..., description="Book authors")
    analysis_type: str = Field(..., description="Type of analysis performed")
    content_preview: str = Field(..., description="Preview of book content")
    file_path: str = Field(..., description="Path to book file")
    file_size: int = Field(..., description="File size in bytes")
    available_formats: List[str] = Field(default_factory=list, description="Available book formats")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Book metadata")
    analysis_summary: str = Field(..., description="Summary of analysis results")
    extraction_successful: bool = Field(..., description="Whether content extraction was successful")
    content_segments: List[Dict[str, Any]] = Field(default_factory=list, description="Content segments for LLM processing")
    total_segments: int = Field(..., description="Total number of content segments")
    total_content_length: int = Field(..., description="Total content length in characters")


# ========== KNOWLEDGE GRAPH ANALYTICS SCHEMAS ========== 