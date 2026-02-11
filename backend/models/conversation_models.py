r"""
Conversation History Models for Bastion Workspace
Handles data models for persistent conversation storage
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from enum import Enum


class MessageType(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class PermissionLevel(str, Enum):
    READ = "read"
    COMMENT = "comment"
    EDIT = "edit"


class AttachmentMetadata(BaseModel):
    """Metadata for a message attachment"""
    attachment_id: str  # UUID
    filename: str
    content_type: str  # MIME type
    file_size: int  # bytes
    file_path: str  # Relative path in storage
    uploaded_at: datetime

    # Image-specific metadata
    is_image: bool = False
    image_width: Optional[int] = None
    image_height: Optional[int] = None

    # Analysis results (populated after processing)
    face_detection_results: Optional[Dict[str, Any]] = None
    vision_description: Optional[str] = None
    detected_identities: Optional[List[str]] = None


# Base models for conversation system
class ConversationMessage(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    message_id: str
    conversation_id: str
    message_type: MessageType
    content: str
    content_hash: Optional[str] = None
    model_used: Optional[str] = None
    query_time: Optional[float] = None
    token_count: Optional[int] = None
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    metadata_json: Dict[str, Any] = Field(default_factory=dict)
    parent_message_id: Optional[str] = None
    is_edited: bool = False
    edit_history: List[Dict[str, Any]] = Field(default_factory=list)
    attachments: List[AttachmentMetadata] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class ConversationSummary(BaseModel):
    conversation_id: str
    user_id: str
    title: str
    description: Optional[str] = None
    is_pinned: bool = False
    is_archived: bool = False
    tags: List[str] = Field(default_factory=list)
    metadata_json: Dict[str, Any] = Field(default_factory=dict)
    message_count: int = 0
    last_message_at: Optional[datetime] = None
    manual_order: Optional[int] = None
    order_locked: bool = False
    created_at: datetime
    updated_at: datetime


class ConversationDetail(ConversationSummary):
    messages: List[ConversationMessage] = Field(default_factory=list)


class ConversationFolder(BaseModel):
    folder_id: str
    user_id: str
    name: str
    description: Optional[str] = None
    color: Optional[str] = None
    parent_folder_id: Optional[str] = None
    sort_order: int = 0
    created_at: datetime
    updated_at: datetime


class ConversationShare(BaseModel):
    share_id: str
    conversation_id: str
    shared_by_user_id: str
    shared_with_user_id: Optional[str] = None
    permission_level: PermissionLevel = PermissionLevel.READ
    is_public: bool = False
    expires_at: Optional[datetime] = None
    access_count: int = 0
    created_at: datetime


class User(BaseModel):
    user_id: str
    username: Optional[str] = None
    email: Optional[str] = None
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None


# Request models
class CreateConversationRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    folder_id: Optional[str] = None
    initial_message: Optional[str] = None  # For title generation


class UpdateConversationRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    is_pinned: Optional[bool] = None
    is_archived: Optional[bool] = None
    tags: Optional[List[str]] = None
    manual_order: Optional[int] = None
    order_locked: Optional[bool] = None
    initial_message: Optional[str] = None  # For title generation


class CreateMessageRequest(BaseModel):
    content: str
    message_type: MessageType = MessageType.USER
    parent_message_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UpdateMessageRequest(BaseModel):
    content: str


class CreateFolderRequest(BaseModel):
    name: str
    description: Optional[str] = None
    color: Optional[str] = None
    parent_folder_id: Optional[str] = None


class UpdateFolderRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    color: Optional[str] = None
    parent_folder_id: Optional[str] = None
    sort_order: Optional[int] = None


class ShareConversationRequest(BaseModel):
    shared_with_user_id: Optional[str] = None  # None for public share
    permission_level: PermissionLevel = PermissionLevel.READ
    share_type: Optional[str] = None  # Alias for permission_level (read/comment/edit)
    is_public: bool = False
    expires_at: Optional[datetime] = None


class UpdateShareRequest(BaseModel):
    share_type: str  # read, comment, or edit
    expires_at: Optional[datetime] = None


class ConversationShareDetail(BaseModel):
    share_id: str
    conversation_id: str
    shared_by_user_id: str
    shared_with_user_id: Optional[str] = None
    share_type: str
    is_public: bool
    expires_at: Optional[datetime] = None
    created_at: datetime
    username: Optional[str] = None
    email: Optional[str] = None


class ConversationParticipant(BaseModel):
    user_id: str
    username: Optional[str] = None
    email: Optional[str] = None
    display_name: Optional[str] = None
    share_type: str
    is_owner: bool


class ShareConversationResponse(BaseModel):
    success: bool
    share_id: str
    message: str


class ConversationSharesResponse(BaseModel):
    shares: List[ConversationShareDetail]


class ConversationParticipantsResponse(BaseModel):
    participants: List[ConversationParticipant]


class SharedConversationsResponse(BaseModel):
    conversations: List[ConversationSummary]
    total_count: int


class ReorderConversationsRequest(BaseModel):
    conversation_ids: List[str]  # Ordered list of conversation IDs
    order_locked: bool = False  # Whether to lock the order


class ConversationSearchRequest(BaseModel):
    query: Optional[str] = None
    tags: Optional[List[str]] = None
    folder_id: Optional[str] = None
    is_pinned: Optional[bool] = None
    is_archived: Optional[bool] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    sort_by: str = Field(default="last_message_at")  # last_message_at, created_at, title, manual_order
    sort_order: str = Field(default="desc")  # asc, desc


class MessageSearchRequest(BaseModel):
    conversation_id: str
    query: Optional[str] = None
    message_type: Optional[MessageType] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)


# Response models
class ConversationListResponse(BaseModel):
    conversations: List[ConversationSummary]
    total_count: int
    has_more: bool
    folders: List[ConversationFolder] = Field(default_factory=list)


class ConversationResponse(BaseModel):
    conversation: ConversationDetail
    folder: Optional[ConversationFolder] = None


class MessageListResponse(BaseModel):
    messages: List[ConversationMessage]
    total_count: int
    has_more: bool


class MessageResponse(BaseModel):
    message: ConversationMessage


class ReactionRequest(BaseModel):
    emoji: str = Field(..., description="Emoji reaction (👍, 👎, 😂, ❤️, 😢)")


class ReactionResponse(BaseModel):
    success: bool
    message_id: str
    reactions: Dict[str, List[str]] = Field(default_factory=dict, description="Map of emoji to list of user_ids who reacted")


class FolderListResponse(BaseModel):
    folders: List[ConversationFolder]


class ShareResponse(BaseModel):
    share: ConversationShare
    share_url: Optional[str] = None


class ConversationStatsResponse(BaseModel):
    total_conversations: int
    total_messages: int
    pinned_conversations: int
    archived_conversations: int
    conversations_this_week: int
    messages_this_week: int
    most_active_day: Optional[str] = None
    average_messages_per_conversation: float = 0.0
    top_tags: List[Dict[str, Any]] = Field(default_factory=list)


# Export request for data portability
class ExportConversationsRequest(BaseModel):
    conversation_ids: Optional[List[str]] = None  # None for all conversations
    include_archived: bool = False
    format: str = Field(default="json")  # json, csv, markdown


class ExportResponse(BaseModel):
    export_id: str
    download_url: str
    expires_at: datetime
    file_size: Optional[int] = None
    format: str


# Import request for data migration
class ImportConversationsRequest(BaseModel):
    source_format: str  # json, csv, chatgpt_export, etc.
    merge_strategy: str = Field(default="create_new")  # create_new, merge_by_title, skip_duplicates
    default_folder_id: Optional[str] = None


class ImportResponse(BaseModel):
    import_id: str
    status: str  # pending, processing, completed, failed
    imported_conversations: int = 0
    imported_messages: int = 0
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class ConversationMode(str, Enum):
    """Available conversation modes"""
    CHAT = "chat"
    RESEARCH = "research"
    CODING = "coding"
    ANALYSIS = "analysis"
    HYBRID = "hybrid"

class ModeTransition(str, Enum):
    """Mode transition types"""
    STAYING = "staying"
    ENTERING = "entering"
    EXITING = "exiting"

class ModeResponse(BaseModel):
    """LLM mode decision response"""
    mode: ConversationMode = Field(..., description="Selected conversation mode")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in mode selection")
    reasoning: str = Field(..., description="Brief explanation of why this mode was chosen")
    mode_transition: ModeTransition = Field(ModeTransition.STAYING, description="Type of mode transition")
    next_mode: Optional[ConversationMode] = Field(None, description="Mode to transition to after completion")
    requires_approval: bool = Field(False, description="Whether this mode requires user approval")
    
    class Config:
        use_enum_values = True

class ConversationContext(BaseModel):
    """Conversation context including mode and research state"""
    conversation_id: str
    current_mode: ConversationMode = ConversationMode.CHAT
    research_plan: Optional[str] = None
    research_progress: Dict[str, Any] = Field(default_factory=dict)
    mode_history: List[Dict[str, Any]] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        use_enum_values = True
