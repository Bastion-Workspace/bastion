"""API models for user document pins (Home dashboard pinned documents widget)."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class DocumentPinCreateRequest(BaseModel):
    document_id: str = Field(..., min_length=1, max_length=255)
    label: Optional[str] = Field(None, max_length=500)


class DocumentPinReorderRequest(BaseModel):
    pin_ids: List[str] = Field(..., min_length=1, description="Ordered pin UUIDs")


class DocumentPinItem(BaseModel):
    pin_id: str
    document_id: str
    label: Optional[str] = None
    sort_order: int
    title: Optional[str] = None
    filename: Optional[str] = None
    content_preview: Optional[str] = None


class DocumentPinsListResponse(BaseModel):
    pins: List[DocumentPinItem]
