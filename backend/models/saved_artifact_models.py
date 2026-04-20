"""
Pydantic models for user-saved chat artifacts (dashboard embeds, share, export).
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

ALLOWED_ARTIFACT_TYPES = frozenset({"html", "mermaid", "chart", "svg", "react"})
MAX_SAVED_ARTIFACT_CODE_BYTES = 512_000


class SavedArtifactCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    artifact_type: Literal["html", "mermaid", "chart", "svg", "react"]
    code: str = Field(..., min_length=1)
    language: Optional[str] = Field(None, max_length=20)
    source_conversation_id: Optional[str] = Field(None, max_length=255)
    source_message_id: Optional[str] = Field(None, max_length=255)


class SavedArtifactUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    is_public: Optional[bool] = None


class SavedArtifactSummary(BaseModel):
    id: str
    title: str
    artifact_type: str
    is_public: bool
    created_at: datetime


class SavedArtifactListResponse(BaseModel):
    artifacts: List[SavedArtifactSummary]


class SavedArtifactResponse(BaseModel):
    id: str
    user_id: str
    title: str
    artifact_type: str
    code: str
    language: Optional[str] = None
    share_token: Optional[str] = None
    is_public: bool = False
    source_conversation_id: Optional[str] = None
    source_message_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class SavedArtifactShareResponse(BaseModel):
    share_token: str
    public_url: str
    embed_url: str
    api_url: str


class PublicArtifactResponse(BaseModel):
    title: str
    artifact_type: str
    code: str
    language: Optional[str] = None
