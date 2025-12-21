"""
API Models for Plato Knowledge Base
Pydantic models for request/response validation
"""

from datetime import datetime, date
from enum import Enum
from typing import List, Optional, Dict, Any, ForwardRef
from pydantic import BaseModel, Field


class ProcessingStatus(str, Enum):
    """Document processing status"""
    UPLOADING = "uploading"
    PROCESSING = "processing"
    EMBEDDING = "embedding"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentType(str, Enum):
    """Supported document types"""
    PDF = "pdf"
    EPUB = "epub"
    TXT = "txt"
    MD = "md"
    DOCX = "docx"
    HTML = "html"
    URL = "url"
    EML = "eml"
    ZIP = "zip"
    SRT = "srt"
    ORG = "org"
    MP4 = "mp4"
    MKV = "mkv"
    AVI = "avi"
    MOV = "mov"
    WEBM = "webm"
    IMAGE = "image"  # For JPG, PNG, GIF, etc. - stored but not vectorized
    MP3 = "mp3"
    AAC = "aac"
    WAV = "wav"
    FLAC = "flac"
    OGG = "ogg"
    M4A = "m4a"
    WMA = "wma"
    OPUS = "opus"


class DocumentCategory(str, Enum):
    """Document categories for organization"""
    TECHNICAL = "technical"
    ACADEMIC = "academic"
    BUSINESS = "business"
    LEGAL = "legal"
    MEDICAL = "medical"
    LITERATURE = "literature"
    MANUAL = "manual"
    REFERENCE = "reference"
    RESEARCH = "research"
    PERSONAL = "personal"
    NEWS = "news"
    EDUCATION = "education"
    ENTERTAINMENT = "entertainment"
    OTHER = "other"


# === REQUEST MODELS ===

class URLImportRequest(BaseModel):
    """Request to import content from URL"""
    url: str = Field(..., description="URL to import content from")
    content_type: str = Field(default="auto", description="Content type hint")


class QueryRequest(BaseModel):
    """Natural language query request"""
    query: str = Field(..., description="Natural language query")
    session_id: Optional[str] = Field(None, description="Chat session ID")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for persistent conversations")
    max_results: Optional[int] = Field(default=10, description="Maximum results to return")
    execution_mode: Optional[str] = Field(default=None, description="Execution mode: 'plan', 'execute', 'chat', or 'direct' (auto-determined if not specified)")


class ModelConfigRequest(BaseModel):
    """Request to configure LLM model"""
    model_config = {"protected_namespaces": ()}
    
    model_name: str = Field(..., description="OpenRouter model name")


class DocumentFilterRequest(BaseModel):
    """Request to filter and search documents"""
    search_query: Optional[str] = Field(None, description="Text search in filename, title, description")
    category: Optional[DocumentCategory] = Field(None, description="Filter by category")
    tags: Optional[List[str]] = Field(None, description="Filter by tags (AND logic)")
    doc_type: Optional[DocumentType] = Field(None, description="Filter by document type")
    status: Optional[ProcessingStatus] = Field(None, description="Filter by processing status")
    author: Optional[str] = Field(None, description="Filter by author")
    language: Optional[str] = Field(None, description="Filter by language")
    date_from: Optional[datetime] = Field(None, description="Filter documents uploaded after this date")
    date_to: Optional[datetime] = Field(None, description="Filter documents uploaded before this date")
    publication_date_from: Optional[date] = Field(None, description="Filter documents published after this date")
    publication_date_to: Optional[date] = Field(None, description="Filter documents published before this date")
    min_quality: Optional[float] = Field(None, description="Minimum quality score")
    sort_by: Optional[str] = Field("upload_date", description="Sort field: upload_date, publication_date, filename, title, quality, size")
    sort_order: Optional[str] = Field("desc", description="Sort order: asc or desc")
    skip: Optional[int] = Field(0, description="Number of documents to skip")
    limit: Optional[int] = Field(50, description="Maximum number of documents to return")


class DocumentUpdateRequest(BaseModel):
    """Request to update document metadata"""
    title: Optional[str] = Field(None, description="Document title")
    category: Optional[DocumentCategory] = Field(None, description="Document category")
    tags: Optional[List[str]] = Field(None, description="Document tags")
    description: Optional[str] = Field(None, description="Document description")
    author: Optional[str] = Field(None, description="Document author")
    publication_date: Optional[date] = Field(None, description="Original publication date of the document")


class BulkCategorizeRequest(BaseModel):
    """Request to bulk categorize documents"""
    document_ids: List[str] = Field(..., description="List of document IDs")
    category: DocumentCategory = Field(..., description="Category to assign")
    tags: Optional[List[str]] = Field(None, description="Tags to add")




# Alias for backward compatibility with main.py

class CreateNoteRequest(BaseModel):
    """Request to create a new free-form note"""
    title: str = Field(..., min_length=1, max_length=500, description="Note title")
    content: str = Field(..., min_length=1, description="Note content")
    note_date: Optional[date] = Field(None, description="Date associated with the note")
    tags: Optional[List[str]] = Field(None, description="Optional tags for categorization")
    category: Optional[str] = Field(None, max_length=100, description="Optional category")


class UpdateNoteRequest(BaseModel):
    """Request to update an existing free-form note"""
    title: Optional[str] = Field(None, min_length=1, max_length=500, description="Note title")
    content: Optional[str] = Field(None, min_length=1, description="Note content")
    note_date: Optional[date] = Field(None, description="Date associated with the note")
    tags: Optional[List[str]] = Field(None, description="Optional tags for categorization")
    category: Optional[str] = Field(None, max_length=100, description="Optional category")


class SearchNotesRequest(BaseModel):
    """Request to search notes using vector similarity"""
    query: str = Field(..., min_length=1, description="Search query")
    max_results: Optional[int] = Field(default=10, description="Maximum number of results to return")
    similarity_threshold: Optional[float] = Field(default=0.7, description="Minimum similarity score")


class NotesCategoriesResponse(BaseModel):
    """Response containing note categories and tags"""
    categories: List[str] = Field(..., description="List of available categories")
    tags: List[str] = Field(..., description="List of available tags")


class NotesStatisticsResponse(BaseModel):
    """Response containing notes statistics"""
    total_notes: int = Field(..., description="Total number of notes")
    notes_by_category: Dict[str, int] = Field(..., description="Count of notes per category")
    notes_by_tag: Dict[str, int] = Field(..., description="Count of notes per tag")
    notes_this_month: int = Field(..., description="Notes created this month")
    notes_this_week: int = Field(..., description="Notes created this week")


# === RESPONSE MODELS ===

class QualityMetrics(BaseModel):
    """Document quality assessment metrics"""
    ocr_confidence: float = Field(..., description="OCR confidence score")
    language_confidence: float = Field(..., description="Language detection confidence")
    vocabulary_score: float = Field(..., description="Vocabulary coherence score")
    pattern_score: float = Field(..., description="Text pattern analysis score")
    overall_score: float = Field(..., description="Overall quality score")


class DocumentInfo(BaseModel):
    """Document information"""
    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    title: Optional[str] = Field(None, description="Document title")
    doc_type: DocumentType = Field(..., description="Document type")
    category: Optional[DocumentCategory] = Field(None, description="Document category")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    description: Optional[str] = Field(None, description="Document description")
    author: Optional[str] = Field(None, description="Document author")
    language: Optional[str] = Field(None, description="Document language")
    publication_date: Optional[date] = Field(None, description="Original publication date of the document")
    upload_date: datetime = Field(..., description="Upload timestamp")
    file_size: int = Field(..., description="File size in bytes")
    file_hash: Optional[str] = Field(None, description="SHA-256 hash of file content")
    status: ProcessingStatus = Field(..., description="Processing status")
    quality_metrics: Optional[QualityMetrics] = Field(None, description="Quality assessment")
    chunk_count: Optional[int] = Field(None, description="Number of text chunks")
    entity_count: Optional[int] = Field(None, description="Number of extracted entities")
    
    # Document ownership
    user_id: Optional[str] = Field(None, description="ID of user who uploaded this document (NULL for admin/global documents)")
    folder_id: Optional[str] = Field(None, description="ID of folder containing this document (NULL for root documents)")
    team_id: Optional[str] = Field(None, description="ID of team this document belongs to (NULL for user/global documents)")
    
    # Global submission workflow fields (deprecated - kept for database compatibility)
    submission_status: Optional[str] = Field(default="not_submitted", description="Global submission status (deprecated)")
    submitted_by: Optional[str] = Field(None, description="User ID who submitted document for global approval")
    submitted_at: Optional[datetime] = Field(None, description="Submission timestamp")
    submission_reason: Optional[str] = Field(None, description="Reason for submitting to global collection")
    reviewed_by: Optional[str] = Field(None, description="Admin user ID who reviewed the submission")
    reviewed_at: Optional[datetime] = Field(None, description="Review timestamp")
    review_comment: Optional[str] = Field(None, description="Admin review comment")
    collection_type: str = Field(default="user", description="Collection type: 'user' or 'global'")
    exempt_from_vectorization: Optional[bool] = Field(default=None, description="Three-state exemption: TRUE=exempt, FALSE=not exempt (override), NULL=inherit from folder")


class DocumentFolder(BaseModel):
    """Document folder information"""
    folder_id: str = Field(..., description="Unique folder identifier")
    name: str = Field(..., description="Folder name")
    parent_folder_id: Optional[str] = Field(None, description="Parent folder ID")
    user_id: Optional[str] = Field(None, description="Owner user ID")
    team_id: Optional[str] = Field(None, description="Team ID if this is a team folder")
    created_by: Optional[str] = Field(None, description="User ID who created this folder (for ownership tracking)")
    collection_type: str = Field(default="user", description="Collection type")
    category: Optional[DocumentCategory] = Field(None, description="Folder category (inherited by documents)")
    tags: List[str] = Field(default_factory=list, description="Folder tags (inherited by documents)")
    inherit_tags: bool = Field(True, description="Whether documents inherit folder tags on upload")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    document_count: Optional[int] = Field(0, description="Number of documents in folder")
    subfolder_count: Optional[int] = Field(0, description="Number of subfolders")
    children: Optional[List['DocumentFolder']] = Field(default_factory=list, description="Child folders")
    is_virtual_source: Optional[bool] = Field(False, description="Whether this is a virtual source (RSS Feeds, Web Sources)")
    exempt_from_vectorization: Optional[bool] = Field(default=None, description="Three-state exemption: TRUE=exempt, FALSE=not exempt (override parent), NULL=inherit from parent folder")


class FolderMetadataUpdateRequest(BaseModel):
    """Request to update folder metadata"""
    category: Optional[DocumentCategory] = Field(None, description="Folder category")
    tags: Optional[List[str]] = Field(None, description="Folder tags")
    inherit_tags: Optional[bool] = Field(None, description="Whether to inherit tags")


# Forward reference for DocumentFolder
DocumentFolder.model_rebuild()

class FolderCreateRequest(BaseModel):
    """Request to create a new folder"""
    name: str = Field(..., description="Folder name")
    parent_folder_id: Optional[str] = Field(None, description="Parent folder ID")
    collection_type: Optional[str] = Field("user", description="Collection type: 'user' or 'global'")

class FolderUpdateRequest(BaseModel):
    """Request to update folder information"""
    name: Optional[str] = Field(None, description="New folder name")
    parent_folder_id: Optional[str] = Field(None, description="New parent folder ID")

class DocumentCreateRequest(BaseModel):
    """Request to create a new text document"""
    filename: str = Field(..., description="Document filename")
    content: str = Field(..., description="Initial content")
    folder_id: Optional[str] = Field(None, description="Target folder ID")
    doc_type: str = Field(..., description="Document type (md/org)")

class FolderTreeResponse(BaseModel):
    """Response containing folder tree structure"""
    folders: List[DocumentFolder] = Field(..., description="List of folders")
    total_folders: int = Field(..., description="Total number of folders")

class FolderContentsResponse(BaseModel):
    """Response containing folder contents"""
    folder: DocumentFolder = Field(..., description="Folder information")
    documents: List[DocumentInfo] = Field(..., description="Documents in folder")
    subfolders: List[DocumentFolder] = Field(..., description="Subfolders")
    total_documents: int = Field(..., description="Total documents in folder")
    total_subfolders: int = Field(..., description="Total subfolders")

class DocumentUploadResponse(BaseModel):
    """Response for document upload"""
    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Uploaded filename")
    status: ProcessingStatus = Field(..., description="Initial processing status")
    message: str = Field(..., description="Status message")


class BulkUploadResponse(BaseModel):
    """Response for bulk document upload"""
    total_files: int = Field(..., description="Total number of files uploaded")
    successful_uploads: int = Field(..., description="Number of successful uploads")
    failed_uploads: int = Field(..., description="Number of failed uploads")
    upload_results: List[DocumentUploadResponse] = Field(..., description="Individual upload results")
    processing_time: float = Field(..., description="Total processing time in seconds")
    message: str = Field(..., description="Overall operation message")


class DocumentStatus(BaseModel):
    """Document processing status response"""
    document_id: str = Field(..., description="Document identifier")
    status: ProcessingStatus = Field(..., description="Current processing status")
    progress: float = Field(..., description="Processing progress (0-100)")
    message: str = Field(..., description="Status message")
    quality_metrics: Optional[QualityMetrics] = Field(None, description="Quality metrics")
    chunks_processed: int = Field(default=0, description="Number of chunks processed")
    entities_extracted: int = Field(default=0, description="Number of entities extracted")


class DocumentListResponse(BaseModel):
    """Response for document listing"""
    documents: List[DocumentInfo] = Field(..., description="List of documents")
    total: int = Field(..., description="Total number of documents")
    categories: Optional[Dict[str, int]] = Field(None, description="Category counts")
    tags: Optional[Dict[str, int]] = Field(None, description="Tag counts")
    filters_applied: Optional[Dict[str, Any]] = Field(None, description="Applied filters")


class CategorySummary(BaseModel):
    """Summary of documents by category"""
    category: DocumentCategory = Field(..., description="Document category")
    count: int = Field(..., description="Number of documents in category")
    total_size: int = Field(..., description="Total size of documents in bytes")
    avg_quality: Optional[float] = Field(None, description="Average quality score")


class TagSummary(BaseModel):
    """Summary of documents by tag"""
    tag: str = Field(..., description="Tag name")
    count: int = Field(..., description="Number of documents with this tag")
    categories: List[str] = Field(..., description="Categories that use this tag")


class DocumentCategoriesResponse(BaseModel):
    """Response for document categories overview"""
    categories: List[CategorySummary] = Field(..., description="Category summaries")
    tags: List[TagSummary] = Field(..., description="Tag summaries")
    total_documents: int = Field(..., description="Total number of documents")
    uncategorized_count: int = Field(..., description="Number of uncategorized documents")


class BulkOperationResponse(BaseModel):
    """Response for bulk operations"""
    success_count: int = Field(..., description="Number of successful operations")
    failed_count: int = Field(..., description="Number of failed operations")
    failed_documents: List[str] = Field(default_factory=list, description="IDs of failed documents")
    message: str = Field(..., description="Operation summary message")







class VisualCitation(BaseModel):
    """Visual citation information for document overlays"""
    page_number: Optional[int] = Field(None, description="Page number in document")
    page_id: Optional[str] = Field(None, description="Page ID for image retrieval")
    segment_id: Optional[str] = Field(None, description="Segment ID if from manual segmentation")
    segment_type: Optional[str] = Field(None, description="Type of segment (article, ad, image, etc.)")
    bounds: Optional[Dict[str, Any]] = Field(None, description="Bounding box coordinates {x, y, width, height}")
    page_image_url: Optional[str] = Field(None, description="URL to page image for overlay")
    highlight_text: Optional[str] = Field(None, description="Text to highlight in the overlay")


class Citation(BaseModel):
    """Source citation information"""
    document_id: str = Field(..., description="Source document ID")
    document_title: str = Field(..., description="Source document title")
    chunk_id: str = Field(..., description="Source chunk ID")
    relevance_score: float = Field(..., description="Relevance score")
    snippet: str = Field(..., description="Relevant text snippet")
    segment_id: Optional[str] = Field(None, description="PDF segment ID if from manual segmentation")
    page_number: Optional[int] = Field(None, description="Page number if from PDF")
    segment_type: Optional[str] = Field(None, description="Type of segment (article, ad, etc.)")
    segment_bounds: Optional[Dict[str, Any]] = Field(None, description="Segment coordinates for highlighting")
    visual_citation: Optional[VisualCitation] = Field(None, description="Visual citation data for document overlay")


class QueryResponse(BaseModel):
    """Response to natural language query"""
    answer: str = Field(..., description="Generated answer")
    citations: List[Citation] = Field(..., description="Source citations")
    session_id: str = Field(..., description="Chat session ID")
    query_time: float = Field(..., description="Query processing time in seconds")
    retrieval_count: int = Field(..., description="Number of retrieved chunks")
    research_plan: Optional[str] = Field(None, description="Generated research plan (for plan mode)")
    plan_approved: bool = Field(default=False, description="Whether research plan was approved")
    execution_mode: str = Field(default="direct", description="Execution mode used")
    iterations: int = Field(default=0, description="Number of LLM iterations used")


class QueryHistoryItem(BaseModel):
    """Single query history item"""
    timestamp: datetime = Field(..., description="Query timestamp")
    query: str = Field(..., description="User query")
    answer: str = Field(..., description="System response")
    citations: List[Citation] = Field(..., description="Source citations")


class QueryHistoryResponse(BaseModel):
    """Query history response"""
    history: List[QueryHistoryItem] = Field(..., description="Query history")


class ModelInfo(BaseModel):
    """Information about available LLM model - OpenRouter API enriched"""
    id: str = Field(..., description="Model ID (e.g., 'anthropic/claude-sonnet-4.5')")
    canonical_slug: Optional[str] = Field(None, description="Permanent slug that never changes")
    name: str = Field(..., description="Model display name")
    provider: str = Field(..., description="Model provider")
    context_length: int = Field(..., description="Maximum context window in tokens")
    input_cost: Optional[float] = Field(None, description="Input cost per token (USD)")
    output_cost: Optional[float] = Field(None, description="Output cost per token (USD)")
    request_cost: Optional[float] = Field(None, description="Fixed cost per API request (USD)")
    image_cost: Optional[float] = Field(None, description="Cost per image input (USD)")
    description: Optional[str] = Field(None, description="Model description")
    supported_parameters: Optional[List[str]] = Field(None, description="Supported API parameters")
    architecture: Optional[Dict[str, Any]] = Field(None, description="Model architecture details")
    top_provider: Optional[Dict[str, Any]] = Field(None, description="Top provider configuration")
    per_request_limits: Optional[Dict[str, Any]] = Field(None, description="Rate limiting information")
    created: Optional[int] = Field(None, description="Unix timestamp when model was added to OpenRouter")


class AvailableModelsResponse(BaseModel):
    """Available models response"""
    models: List[ModelInfo] = Field(..., description="Available models")


 


# === INTERNAL MODELS ===

class Chunk(BaseModel):
    """Text chunk with metadata"""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk text content")
    chunk_index: int = Field(..., description="Index within document")
    quality_score: float = Field(..., description="Quality assessment score")
    method: str = Field(..., description="Chunking method used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class Entity(BaseModel):
    """Extracted entity"""
    name: str = Field(..., description="Entity name")
    entity_type: str = Field(..., description="Entity type")
    confidence: float = Field(..., description="Extraction confidence")
    source_chunk: str = Field(..., description="Source chunk ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ProcessingResult(BaseModel):
    """Document processing result"""
    document_id: str = Field(..., description="Document identifier")
    chunks: List[Chunk] = Field(..., description="Extracted chunks")
    entities: List[Entity] = Field(..., description="Extracted entities")
    quality_metrics: QualityMetrics = Field(..., description="Quality assessment")
    processing_time: float = Field(..., description="Processing time in seconds")


# === WEBSOCKET MODELS ===

class WebSocketMessage(BaseModel):
    """WebSocket message format"""
    type: str = Field(..., description="Message type")
    data: Any = Field(..., description="Message data")
    session_id: Optional[str] = Field(None, description="Session identifier")


class ProcessingUpdate(BaseModel):
    """Processing status update"""
    document_id: str = Field(..., description="Document being processed")
    status: ProcessingStatus = Field(..., description="Current status")
    progress: float = Field(..., description="Progress percentage")
    message: str = Field(..., description="Status message")


class ChatStreamChunk(BaseModel):
    """Streaming chat response chunk"""
    type: str = Field(..., description="Chunk type: 'content' or 'citations'")
    data: str = Field(..., description="Chunk content")
    session_id: str = Field(..., description="Chat session ID")
    is_final: bool = Field(default=False, description="Whether this is the final chunk")


# === SETTINGS MODELS ===

class SettingValue(BaseModel):
    """Individual setting value with metadata"""
    value: Any = Field(..., description="Setting value")
    type: str = Field(..., description="Value type: string, integer, float, boolean, json")
    description: Optional[str] = Field(None, description="Setting description")
    is_secret: bool = Field(default=False, description="Whether this is a secret value")


class SettingsResponse(BaseModel):
    """Response containing all settings grouped by category"""
    settings: Dict[str, Dict[str, SettingValue]] = Field(..., description="Settings grouped by category")


class SettingUpdateRequest(BaseModel):
    """Request to update a single setting"""
    key: str = Field(..., description="Setting key")
    value: Any = Field(..., description="Setting value")
    description: Optional[str] = Field(None, description="Setting description")
    category: Optional[str] = Field("general", description="Setting category")


class BulkSettingsUpdateRequest(BaseModel):
    """Request to update multiple settings"""
    settings: Dict[str, Any] = Field(..., description="Settings to update")


class SettingUpdateResponse(BaseModel):
    """Response for setting update"""
    success: bool = Field(..., description="Whether the update was successful")
    message: str = Field(..., description="Update result message")
    updated_settings: Optional[Dict[str, bool]] = Field(None, description="Results for bulk updates")


# === OCR MODELS ===

class OCRProcessingRequest(BaseModel):
    """Request to process document with OCR"""
    document_id: str = Field(..., description="Document ID to process")
    force_ocr: bool = Field(default=False, description="Force OCR even if text exists")
    preserve_hocr: bool = Field(default=True, description="Save hOCR data for editing")


class HOCRWordInfo(BaseModel):
    """Information about a word in hOCR data"""
    word_id: str = Field(..., description="Unique word identifier")
    text: str = Field(..., description="Word text")
    bbox: List[int] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    confidence: int = Field(..., description="OCR confidence score (0-100)")


class HOCRLineInfo(BaseModel):
    """Information about a line in hOCR data"""
    line_id: str = Field(..., description="Unique line identifier")
    bbox: List[int] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    words: List[HOCRWordInfo] = Field(..., description="Words in this line")
    word_count: int = Field(..., description="Number of words in line")
    text: str = Field(..., description="Complete line text")


class HOCRPageInfo(BaseModel):
    """Information about a page in hOCR data"""
    page_number: int = Field(..., description="Page number")
    page_width: int = Field(..., description="Page width in pixels")
    page_height: int = Field(..., description="Page height in pixels")
    lines: List[HOCRLineInfo] = Field(..., description="Lines on this page")
    line_count: int = Field(..., description="Number of lines on page")


class HOCRData(BaseModel):
    """Complete hOCR document data"""
    document_type: str = Field(..., description="Document type (hocr)")
    page_count: int = Field(..., description="Number of pages")
    pages: List[HOCRPageInfo] = Field(..., description="Page data")
    total_words: int = Field(..., description="Total number of words")
    creation_info: Dict[str, Any] = Field(..., description="Creation metadata")


class OCRProcessingResponse(BaseModel):
    """Response from OCR processing"""
    status: str = Field(..., description="Processing status")
    document_id: str = Field(..., description="Document ID")
    hocr_available: bool = Field(..., description="Whether hOCR data is available")
    hocr_path: Optional[str] = Field(None, description="Path to hOCR file")
    hocr_data: Optional[HOCRData] = Field(None, description="Parsed hOCR data")
    pages_processed: int = Field(..., description="Number of pages processed")
    reason: Optional[str] = Field(None, description="Reason if skipped")
    error: Optional[str] = Field(None, description="Error message if failed")


class HOCRUpdateRequest(BaseModel):
    """Request to update text in hOCR file"""
    document_id: str = Field(..., description="Document ID")
    page_number: int = Field(..., description="Page number")
    word_id: str = Field(..., description="Word ID to update")
    new_text: str = Field(..., description="New text for the word")


class HOCRBatchUpdateRequest(BaseModel):
    """Request to batch update multiple words in hOCR file"""
    document_id: str = Field(..., description="Document ID")
    updates: List[Dict[str, str]] = Field(..., description="List of updates with word_id and new_text")


class HOCRBatchUpdateResponse(BaseModel):
    """Response from batch hOCR update"""
    success: bool = Field(..., description="Whether the operation was successful")
    successful_updates: int = Field(..., description="Number of successful updates")
    failed_updates: List[str] = Field(..., description="List of failed word IDs")
    total_updates: int = Field(..., description="Total number of updates attempted")
    error: Optional[str] = Field(None, description="Error message if failed")


class OCRConfidenceStats(BaseModel):
    """OCR confidence statistics for a document"""
    overall_avg_confidence: float = Field(..., description="Overall average confidence")
    overall_min_confidence: float = Field(..., description="Overall minimum confidence")
    overall_max_confidence: float = Field(..., description="Overall maximum confidence")
    total_words: int = Field(..., description="Total number of words")
    page_stats: List[Dict[str, Any]] = Field(..., description="Per-page statistics")
    low_confidence_words: int = Field(..., description="Number of low confidence words")
    high_confidence_words: int = Field(..., description="Number of high confidence words")


class LowConfidenceWord(BaseModel):
    """Information about a low confidence word for review"""
    page_number: int = Field(..., description="Page number")
    word_id: str = Field(..., description="Word ID")
    text: str = Field(..., description="Current text")
    confidence: int = Field(..., description="Confidence score")
    bbox: List[int] = Field(..., description="Bounding box coordinates")
    line_id: str = Field(..., description="Parent line ID")


class LowConfidenceWordsResponse(BaseModel):
    """Response containing low confidence words for review"""
    words: List[LowConfidenceWord] = Field(..., description="Low confidence words")
    total_count: int = Field(..., description="Total number of low confidence words")
    threshold: int = Field(..., description="Confidence threshold used")


class CorrectedTextResponse(BaseModel):
    """Response containing corrected text from hOCR"""
    document_id: str = Field(..., description="Document ID")
    corrected_text: str = Field(..., description="Corrected text content")
    page_count: int = Field(..., description="Number of pages processed")
    word_count: int = Field(..., description="Total number of words")


# === DIRECT SEARCH MODELS ===

class DirectSearchRequest(BaseModel):
    """Request for direct semantic search without LLM processing"""
    query: str = Field(..., min_length=1, description="Search query text")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum number of results")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    document_types: Optional[List[str]] = Field(None, description="Filter by document types")
    categories: Optional[List[str]] = Field(None, description="Filter by categories")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    date_from: Optional[datetime] = Field(None, description="Filter documents from this date")
    date_to: Optional[datetime] = Field(None, description="Filter documents to this date")
    include_metadata: bool = Field(default=True, description="Include document metadata in results")


class DirectSearchResult(BaseModel):
    """Single direct search result"""
    chunk_id: str = Field(..., description="Chunk identifier")
    similarity_score: float = Field(..., description="Similarity score (0.0 to 1.0)")
    text: str = Field(..., description="Chunk text content")
    highlighted_text: str = Field(..., description="Text with query terms highlighted")
    context: Dict[str, Any] = Field(..., description="Context around the match")
    chunk_metadata: Dict[str, Any] = Field(..., description="Chunk-specific metadata")
    document: Optional[Dict[str, Any]] = Field(None, description="Document metadata if requested")


class DirectSearchResponse(BaseModel):
    """Response from direct semantic search"""
    success: bool = Field(..., description="Whether the search was successful")
    query: str = Field(..., description="Original search query")
    results: List[DirectSearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results returned")
    similarity_threshold: float = Field(..., description="Similarity threshold used")
    filters_applied: Dict[str, Any] = Field(..., description="Filters that were applied")
    search_metadata: Dict[str, Any] = Field(..., description="Search execution metadata")
    error: Optional[str] = Field(None, description="Error message if search failed")


# Authentication Models
class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]
    expires_in: int

class UserCreateRequest(BaseModel):
    username: str
    email: str
    password: str
    display_name: Optional[str] = None
    role: str = "user"  # user, admin

class UserUpdateRequest(BaseModel):
    email: Optional[str] = None
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None

class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str

class UserResponse(BaseModel):
    user_id: str
    username: str
    email: str
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None
    role: str
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None

class UsersListResponse(BaseModel):
    users: List[UserResponse]
    total: int

class AuthenticatedUserResponse(BaseModel):
    user_id: str
    username: str
    email: str
    display_name: Optional[str] = None
    role: str
    preferences: Dict[str, Any] = Field(default_factory=dict)


# === EXPORT MODELS ===

class EpubExportRequest(BaseModel):
    """Request to export Markdown to EPUB"""
    content: str = Field(..., description="Markdown content to export")
    include_toc: bool = Field(default=True, description="Include EPUB navigation (TOC)")
    include_cover: bool = Field(default=True, description="Include cover page if resolvable")
    split_on_headings: bool = Field(default=True, description="Split chapters on headings")
    split_on_heading_levels: List[int] = Field(default_factory=lambda: [1, 2], description="Heading levels to split on (1-6)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata such as title, author, language, cover")
    heading_alignments: Dict[int, str] = Field(default_factory=dict, description="Per-heading alignment: {level: left|center|right|justify}")
