"""
PDF Segmentation Models
Pydantic models for manual PDF segmentation functionality
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field


class SegmentType(str, Enum):
    """Types of content segments in a PDF"""
    ARTICLE = "article"
    ADVERTISEMENT = "advertisement"
    IMAGE = "image"
    CAPTION = "caption"
    HEADLINE = "headline"
    BYLINE = "byline"
    DATELINE = "dateline"
    MASTHEAD = "masthead"
    CLASSIFIED = "classified"
    EDITORIAL = "editorial"
    LETTER = "letter"
    CARTOON = "cartoon"
    CHART = "chart"
    TABLE = "table"
    FOOTER = "footer"
    HEADER = "header"
    OTHER = "other"


class RelationshipType(str, Enum):
    """Types of relationships between segments"""
    CAPTION_OF = "caption_of"
    CONTINUATION_OF = "continuation_of"
    RELATED_TO = "related_to"
    PART_OF = "part_of"
    BYLINE_OF = "byline_of"


class ProcessingStatus(str, Enum):
    """Processing status for PDF pages"""
    PENDING = "pending"
    SEGMENTED = "segmented"
    COMPLETED = "completed"


# === REQUEST MODELS ===

class SegmentBounds(BaseModel):
    """Bounding box coordinates for a segment in PDF coordinates"""
    x: float = Field(..., description="Top-left X coordinate in PDF points")
    y: float = Field(..., description="Top-left Y coordinate in PDF points")
    width: float = Field(..., description="Segment width in PDF points")
    height: float = Field(..., description="Segment height in PDF points")


class CreateSegmentRequest(BaseModel):
    """Request to create a new segment"""
    page_id: int = Field(..., description="PDF page ID")
    segment_type: SegmentType = Field(..., description="Type of content segment")
    bounds: SegmentBounds = Field(..., description="Segment bounding box")
    manual_text: Optional[str] = Field(None, description="Manually typed text content")
    tags: Optional[List[str]] = Field(default_factory=list, description="Additional tags")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class UpdateSegmentRequest(BaseModel):
    """Request to update an existing segment"""
    segment_type: Optional[SegmentType] = Field(None, description="Type of content segment")
    bounds: Optional[SegmentBounds] = Field(None, description="Segment bounding box")
    manual_text: Optional[str] = Field(None, description="Manually typed text content")
    tags: Optional[List[str]] = Field(None, description="Additional tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class CreateRelationshipRequest(BaseModel):
    """Request to create a relationship between segments"""
    parent_segment_id: str = Field(..., description="Parent segment ID")
    child_segment_id: str = Field(..., description="Child segment ID")
    relationship_type: RelationshipType = Field(..., description="Type of relationship")


class BulkSegmentRequest(BaseModel):
    """Request to create multiple segments at once"""
    page_id: int = Field(..., description="PDF page ID")
    segments: List[CreateSegmentRequest] = Field(..., description="List of segments to create")


# === RESPONSE MODELS ===

class PDFPageInfo(BaseModel):
    """Information about a PDF page"""
    id: int = Field(..., description="Page ID")
    document_id: str = Field(..., description="Parent document ID")
    page_number: int = Field(..., description="Page number within document")
    page_image_path: str = Field(..., description="Path to page image file")
    page_width: int = Field(..., description="Page width in pixels")
    page_height: int = Field(..., description="Page height in pixels")
    processing_status: ProcessingStatus = Field(..., description="Processing status")
    segment_count: Optional[int] = Field(None, description="Number of segments on this page")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class SegmentInfo(BaseModel):
    """Information about a PDF segment"""
    id: int = Field(..., description="Segment database ID")
    segment_id: str = Field(..., description="Unique segment identifier")
    page_id: int = Field(..., description="Parent page ID")
    segment_type: SegmentType = Field(..., description="Type of content segment")
    bounds: SegmentBounds = Field(..., description="Segment bounding box")
    manual_text: Optional[str] = Field(None, description="Manually typed text content")
    confidence_score: float = Field(..., description="Confidence score (always 1.0 for manual)")
    tags: List[str] = Field(default_factory=list, description="Additional tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class SegmentRelationship(BaseModel):
    """Relationship between two segments"""
    id: int = Field(..., description="Relationship ID")
    parent_segment_id: str = Field(..., description="Parent segment ID")
    child_segment_id: str = Field(..., description="Child segment ID")
    relationship_type: RelationshipType = Field(..., description="Type of relationship")
    created_at: datetime = Field(..., description="Creation timestamp")


class PageSegmentationResponse(BaseModel):
    """Response containing page and its segments"""
    page: PDFPageInfo = Field(..., description="Page information")
    segments: List[SegmentInfo] = Field(..., description="Segments on this page")
    relationships: List[SegmentRelationship] = Field(..., description="Segment relationships")


class DocumentSegmentationResponse(BaseModel):
    """Response containing all pages and segments for a document"""
    document_id: str = Field(..., description="Document ID")
    pages: List[PageSegmentationResponse] = Field(..., description="Pages with segments")
    total_segments: int = Field(..., description="Total number of segments")
    segment_types: Dict[str, int] = Field(..., description="Count by segment type")


class SegmentSearchRequest(BaseModel):
    """Request to search segments"""
    document_id: Optional[str] = Field(None, description="Filter by document")
    page_id: Optional[int] = Field(None, description="Filter by page")
    segment_types: Optional[List[SegmentType]] = Field(None, description="Filter by segment types")
    text_query: Optional[str] = Field(None, description="Search in manual text")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    skip: Optional[int] = Field(0, description="Number of results to skip")
    limit: Optional[int] = Field(50, description="Maximum number of results")


class SegmentSearchResponse(BaseModel):
    """Response for segment search"""
    segments: List[SegmentInfo] = Field(..., description="Found segments")
    total: int = Field(..., description="Total number of matching segments")
    segment_types: Dict[str, int] = Field(..., description="Count by segment type")


# === PROCESSING MODELS ===

class PDFExtractionRequest(BaseModel):
    """Request to extract pages from a PDF"""
    document_id: str = Field(..., description="Document ID to process")
    extract_images: bool = Field(default=True, description="Whether to extract page images")
    image_dpi: int = Field(default=300, description="DPI for extracted images")
    image_format: str = Field(default="PNG", description="Image format (PNG, JPEG)")


class PDFExtractionResponse(BaseModel):
    """Response for PDF page extraction"""
    document_id: str = Field(..., description="Document ID")
    pages_extracted: int = Field(..., description="Number of pages extracted")
    pages: List[PDFPageInfo] = Field(..., description="Extracted page information")
    extraction_time: float = Field(..., description="Processing time in seconds")


class SegmentationStats(BaseModel):
    """Statistics about segmentation progress"""
    total_documents: int = Field(..., description="Total documents with segmentation")
    total_pages: int = Field(..., description="Total pages processed")
    total_segments: int = Field(..., description="Total segments created")
    segments_by_type: Dict[str, int] = Field(..., description="Segment count by type")
    pages_by_status: Dict[str, int] = Field(..., description="Page count by status")
    avg_segments_per_page: float = Field(..., description="Average segments per page")


# === VECTORIZATION MODELS ===

class SegmentVectorizationRequest(BaseModel):
    """Request to vectorize segment text"""
    segment_ids: List[str] = Field(..., description="Segment IDs to vectorize")
    force_reprocess: bool = Field(default=False, description="Force reprocessing of existing vectors")


class SegmentVectorizationResponse(BaseModel):
    """Response for segment vectorization"""
    processed_segments: int = Field(..., description="Number of segments processed")
    successful_embeddings: int = Field(..., description="Number of successful embeddings")
    failed_embeddings: int = Field(..., description="Number of failed embeddings")
    processing_time: float = Field(..., description="Processing time in seconds")


# === NEW PDF-BASED MODELS ===

class PDFTextExtractionRequest(BaseModel):
    """Request to extract text from a PDF region"""
    document_id: str = Field(..., description="Document ID")
    page_number: int = Field(..., description="Page number (1-based)")
    bounds: SegmentBounds = Field(..., description="Region bounds in PDF coordinates")
    use_ocr_fallback: bool = Field(default=True, description="Use OCR if no selectable text")


class PDFTextExtractionResponse(BaseModel):
    """Response for PDF text extraction"""
    extracted_text: str = Field(..., description="Extracted text content")
    text_confidence: float = Field(..., description="Confidence score (0-1)")
    extraction_method: str = Field(..., description="Method used: 'direct', 'ocr', or 'hybrid'")
    word_count: int = Field(..., description="Number of words extracted")
    has_selectable_text: bool = Field(..., description="Whether PDF has selectable text")


class PDFSegmentCropRequest(BaseModel):
    """Request to crop a PDF segment to a new document"""
    segment_id: str = Field(..., description="Segment ID to crop")
    output_filename: Optional[str] = Field(None, description="Custom output filename")
    include_metadata: bool = Field(default=True, description="Include segment metadata")


class PDFSegmentCropResponse(BaseModel):
    """Response for PDF segment cropping"""
    cropped_pdf_path: str = Field(..., description="Path to cropped PDF file")
    file_size: int = Field(..., description="File size in bytes")
    page_dimensions: Tuple[float, float] = Field(..., description="Width, height in points")


class MultipleSelectionRequest(BaseModel):
    """Request to create multiple segments on a page"""
    page_id: int = Field(..., description="PDF page ID")
    selections: List[Dict[str, Any]] = Field(..., description="List of selection data")
    auto_extract_text: bool = Field(default=True, description="Automatically extract text")


class MultipleSelectionResponse(BaseModel):
    """Response for multiple selections"""
    created_segments: List[SegmentInfo] = Field(..., description="Created segments")
    failed_selections: List[Dict[str, Any]] = Field(..., description="Failed selections with errors")
    total_created: int = Field(..., description="Number of successfully created segments")


class TextEditRequest(BaseModel):
    """Request to edit extracted text"""
    segment_id: str = Field(..., description="Segment ID")
    edited_text: str = Field(..., description="User-edited text content")
    preserve_formatting: bool = Field(default=True, description="Attempt to preserve formatting")


class TextEditResponse(BaseModel):
    """Response for text editing"""
    updated_segment: SegmentInfo = Field(..., description="Updated segment info")
    text_changes: Dict[str, Any] = Field(..., description="Summary of changes made")


class PDFRegionInfo(BaseModel):
    """Information about a PDF region"""
    bounds: SegmentBounds = Field(..., description="Region bounds")
    text_content: str = Field(..., description="Extracted text")
    has_images: bool = Field(..., description="Whether region contains images")
    has_tables: bool = Field(..., description="Whether region contains tables")
    font_info: List[Dict[str, Any]] = Field(..., description="Font information")
    text_blocks: List[Dict[str, Any]] = Field(..., description="Text block structure")


# === EXPORT MODELS ===

class SegmentExportRequest(BaseModel):
    """Request to export segment data"""
    document_id: Optional[str] = Field(None, description="Export specific document")
    segment_types: Optional[List[SegmentType]] = Field(None, description="Export specific types")
    format: str = Field(default="json", description="Export format (json, csv, xml)")
    include_images: bool = Field(default=False, description="Include segment images")
    include_cropped_pdfs: bool = Field(default=False, description="Include cropped PDF files")


class SegmentExportResponse(BaseModel):
    """Response for segment export"""
    export_file_path: str = Field(..., description="Path to exported file")
    segments_exported: int = Field(..., description="Number of segments exported")
    file_size: int = Field(..., description="Export file size in bytes")
    export_time: float = Field(..., description="Export processing time")
    included_files: List[str] = Field(..., description="List of included file paths")
