"""
File Placement Models for FileManager Service
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum
from pydantic import BaseModel, Field

from models.api_models import DocumentType, DocumentCategory, ProcessingStatus


class SourceType(str, Enum):
    """Types of file sources"""
    RSS = "rss"
    CHAT = "chat"
    CODING = "coding"
    UPLOAD = "upload"
    WEB_SCRAPING = "web_scraping"
    API_IMPORT = "api_import"
    MANUAL = "manual"
    AGENT_CREATED = "agent_created"


class FilePlacementRequest(BaseModel):
    """Request for file placement"""
    content: str = Field(..., description="File content")
    title: str = Field(..., description="File title")
    filename: Optional[str] = Field(None, description="Filename (auto-generated if not provided)")
    source_type: SourceType = Field(..., description="Source type")
    source_metadata: Dict[str, Any] = Field(default_factory=dict, description="Source-specific metadata")
    
    # Document properties
    doc_type: DocumentType = Field(default=DocumentType.TXT, description="Document type")
    category: DocumentCategory = Field(default=DocumentCategory.OTHER, description="Document category")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    description: Optional[str] = Field(None, description="Document description")
    author: Optional[str] = Field(None, description="Document author")
    language: str = Field(default="en", description="Document language")
    
    # Placement properties
    user_id: Optional[str] = Field(None, description="User ID (None for global documents)")
    collection_type: str = Field(default="user", description="Collection type (user/global)")
    folder_path: Optional[List[str]] = Field(None, description="Custom folder path")
    target_folder_id: Optional[str] = Field(None, description="Explicit target folder ID (overrides folder_path)")
    
    # Admin context for global folder creation
    current_user_role: str = Field(default="user", description="Role of the user placing the file")
    admin_user_id: Optional[str] = Field(None, description="Admin user ID (for permission validation)")
    
    # Processing options
    process_immediately: bool = Field(default=True, description="Process immediately or queue")
    priority: int = Field(default=5, description="Processing priority (1-10)")


class FilePlacementResponse(BaseModel):
    """Response from file placement"""
    document_id: str = Field(..., description="Generated document ID")
    folder_id: str = Field(..., description="Folder ID where file was placed")
    filename: str = Field(..., description="Final filename")
    processing_status: ProcessingStatus = Field(..., description="Current processing status")
    placement_timestamp: datetime = Field(..., description="When file was placed")
    websocket_notification_sent: bool = Field(..., description="Whether WebSocket notification was sent")
    
    # Optional processing info
    processing_task_id: Optional[str] = Field(None, description="Celery task ID for processing")
    estimated_processing_time: Optional[int] = Field(None, description="Estimated processing time in seconds")


class FileMoveRequest(BaseModel):
    """Request for moving files"""
    document_id: str = Field(..., description="Document ID to move")
    new_folder_id: str = Field(..., description="New folder ID")
    user_id: Optional[str] = Field(None, description="User ID for permission check")
    current_user_role: str = Field(default="user", description="Role of the user moving the file")


class FileMoveResponse(BaseModel):
    """Response from file move operation"""
    document_id: str = Field(..., description="Document ID that was moved")
    old_folder_id: str = Field(..., description="Previous folder ID")
    new_folder_id: str = Field(..., description="New folder ID")
    move_timestamp: datetime = Field(..., description="When file was moved")
    websocket_notification_sent: bool = Field(..., description="Whether WebSocket notification was sent")


class FileDeleteRequest(BaseModel):
    """Request for deleting files"""
    document_id: str = Field(..., description="Document ID to delete")
    user_id: Optional[str] = Field(None, description="User ID for permission check")
    recursive: bool = Field(default=False, description="Delete folder recursively if document is a folder")


class FileDeleteResponse(BaseModel):
    """Response from file delete operation"""
    document_id: str = Field(..., description="Document ID that was deleted")
    folder_id: Optional[str] = Field(None, description="Folder ID where file was located")
    delete_timestamp: datetime = Field(..., description="When file was deleted")
    websocket_notification_sent: bool = Field(..., description="Whether WebSocket notification was sent")
    items_deleted: int = Field(..., description="Number of items deleted")


class FileRenameRequest(BaseModel):
    """Request for renaming files"""
    document_id: str = Field(..., description="Document ID to rename")
    new_filename: str = Field(..., description="New filename including extension")
    user_id: Optional[str] = Field(None, description="User ID for permission check")


class FileRenameResponse(BaseModel):
    """Response from file rename operation"""
    document_id: str = Field(..., description="Document ID that was renamed")
    old_filename: str = Field(..., description="Old filename")
    new_filename: str = Field(..., description="New filename")
    folder_id: Optional[str] = Field(None, description="Folder ID where file is located")
    rename_timestamp: datetime = Field(..., description="When file was renamed")
    websocket_notification_sent: bool = Field(..., description="Whether WebSocket notification was sent")


class FolderStructureRequest(BaseModel):
    """Request for creating folder structure"""
    folder_path: List[str] = Field(..., description="Folder path to create")
    parent_folder_id: Optional[str] = Field(None, description="Parent folder ID (None for root level)")
    user_id: Optional[str] = Field(None, description="User ID (None for global folders)")
    collection_type: str = Field(default="user", description="Collection type (user/global)")
    description: Optional[str] = Field(None, description="Folder description")
    current_user_role: str = Field(default="user", description="Role of the user creating the folder")
    admin_user_id: Optional[str] = Field(None, description="Admin user ID (if current user is admin)")


class FolderStructureResponse(BaseModel):
    """Response from folder structure creation"""
    folder_id: str = Field(..., description="Created folder ID")
    folder_path: List[str] = Field(..., description="Folder path that was created")
    parent_folder_id: Optional[str] = Field(None, description="Parent folder ID")
    creation_timestamp: datetime = Field(..., description="When folder was created")
    websocket_notification_sent: bool = Field(..., description="Whether WebSocket notification was sent")
