"""Pydantic models for document and folder sharing and edit locks."""

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class CreateShareRequest(BaseModel):
    shared_with_user_id: str = Field(..., description="User ID to share with")
    share_type: Literal["read", "write"] = Field(..., description="read or write")
    expires_at: Optional[datetime] = None


class UpdateShareRequest(BaseModel):
    share_type: Literal["read", "write"] = Field(..., description="read or write")


class ShareInfoResponse(BaseModel):
    share_id: str
    document_id: Optional[str] = None
    folder_id: Optional[str] = None
    shared_by_user_id: str
    shared_with_user_id: str
    shared_with_username: Optional[str] = None
    share_type: str
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None


class ShareListResponse(BaseModel):
    shares: List[ShareInfoResponse]


class SharedItemResponse(BaseModel):
    item_type: str = Field(..., description="document or folder")
    document_id: Optional[str] = None
    folder_id: Optional[str] = None
    title: Optional[str] = None
    filename: Optional[str] = None
    name: Optional[str] = None
    parent_folder_id: Optional[str] = None
    share_type: str
    share_id: str


class SharerGroupResponse(BaseModel):
    sharer_user_id: str
    sharer_username: str
    items: List[SharedItemResponse]


class SharedWithMeResponse(BaseModel):
    groups: List[SharerGroupResponse]


class ShareableUserResponse(BaseModel):
    user_id: str
    username: str
    avatar_url: Optional[str] = None


class ShareableUsersListResponse(BaseModel):
    users: List[ShareableUserResponse]


class DocumentLockResponse(BaseModel):
    document_id: str
    locked_by_user_id: Optional[str] = None
    locked_by_username: Optional[str] = None
    acquired_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    active: bool = False


class AcquireLockResponse(BaseModel):
    success: bool
    message: str = ""
    lock: Optional[DocumentLockResponse] = None


class DocumentSharingContextResponse(BaseModel):
    """Effective share relationship for the current user viewing a document."""

    document_id: str
    is_owner: bool
    share_type: Optional[str] = Field(
        None, description="read or write when viewer is a share recipient; null if owner/admin path"
    )
    can_write: bool
    can_delete: bool
    collab_eligible: bool = Field(
        False,
        description="True when real-time Yjs collaboration should be used (team doc or multi-writer shares).",
    )
