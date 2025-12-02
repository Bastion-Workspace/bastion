"""
Team Models - Pydantic models for team operations
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class TeamRole(str, Enum):
    """Team member roles"""
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class PostType(str, Enum):
    """Team post types"""
    TEXT = "text"
    IMAGE = "image"
    FILE = "file"


# === REQUEST MODELS ===

class CreateTeamRequest(BaseModel):
    """Request to create a new team"""
    team_name: str = Field(..., min_length=1, max_length=255, description="Team name")
    description: Optional[str] = Field(None, max_length=1000, description="Team description")
    avatar_url: Optional[str] = Field(None, max_length=500, description="Team avatar URL")


class UpdateTeamRequest(BaseModel):
    """Request to update team details"""
    team_name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    avatar_url: Optional[str] = Field(None, max_length=500)
    settings: Optional[Dict[str, Any]] = Field(None, description="Team settings JSON")


class AddMemberRequest(BaseModel):
    """Request to add a team member"""
    user_id: str = Field(..., description="User ID to add")
    role: TeamRole = Field(TeamRole.MEMBER, description="Member role")


class UpdateMemberRoleRequest(BaseModel):
    """Request to update member role"""
    role: TeamRole = Field(..., description="New role")


class CreatePostRequest(BaseModel):
    """Request to create a team post"""
    content: str = Field(..., min_length=1, description="Post content")
    post_type: PostType = Field(PostType.TEXT, description="Post type")
    attachments: Optional[List[Dict[str, Any]]] = Field(default=[], description="File attachments metadata")


class AddReactionRequest(BaseModel):
    """Request to add a reaction to a post"""
    reaction_type: str = Field(..., max_length=10, description="Emoji reaction")


class CreateCommentRequest(BaseModel):
    """Request to create a comment on a post"""
    content: str = Field(..., min_length=1, description="Comment content")


# === RESPONSE MODELS ===

class TeamMemberResponse(BaseModel):
    """Team member information"""
    user_id: str
    username: str
    display_name: Optional[str]
    avatar_url: Optional[str]
    role: TeamRole
    joined_at: datetime
    invited_by: Optional[str]
    is_online: bool = False
    last_seen: Optional[datetime] = None


class TeamResponse(BaseModel):
    """Team information"""
    team_id: str
    team_name: str
    description: Optional[str]
    created_by: str
    created_at: datetime
    updated_at: datetime
    avatar_url: Optional[str]
    settings: Dict[str, Any] = {}
    member_count: int = 0
    user_role: Optional[TeamRole] = None


class TeamInvitationResponse(BaseModel):
    """Team invitation information"""
    invitation_id: str
    team_id: str
    team_name: str
    invited_user_id: str
    invited_by: str
    inviter_name: str
    status: str
    message_id: Optional[str]
    created_at: datetime
    expires_at: datetime
    responded_at: Optional[datetime]


class PostAttachmentResponse(BaseModel):
    """Post attachment information"""
    filename: str
    file_path: str
    mime_type: str
    file_size: int
    width: Optional[int] = None
    height: Optional[int] = None


class PostReactionResponse(BaseModel):
    """Post reaction information"""
    reaction_type: str
    count: int
    users: List[str]  # User IDs who reacted


class PostCommentResponse(BaseModel):
    """Post comment information"""
    comment_id: str
    post_id: str
    author_id: str
    author_name: str
    author_avatar: Optional[str]
    content: str
    created_at: datetime
    updated_at: datetime


class TeamPostResponse(BaseModel):
    """Team post information"""
    post_id: str
    team_id: str
    author_id: str
    author_name: str
    author_avatar: Optional[str]
    content: str
    post_type: PostType
    attachments: List[PostAttachmentResponse] = []
    reactions: List[PostReactionResponse] = []
    comment_count: int = 0
    created_at: datetime
    updated_at: datetime


class TeamsListResponse(BaseModel):
    """List of teams"""
    teams: List[TeamResponse]
    total: int


class TeamMembersListResponse(BaseModel):
    """List of team members"""
    members: List[TeamMemberResponse]
    total: int


class TeamPostsListResponse(BaseModel):
    """List of team posts"""
    posts: List[TeamPostResponse]
    total: int
    has_more: bool = False


class PostCommentsListResponse(BaseModel):
    """List of post comments"""
    comments: List[PostCommentResponse]
    total: int

