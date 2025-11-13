"""
News Models - Roosevelt's Balanced Synthesis Structures

Defines structured outputs and API models for synthesized news.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class Severity(str):
    BREAKING = "breaking"
    URGENT = "urgent"
    NORMAL = "normal"


class NewsSourceRef(BaseModel):
    name: Optional[str] = Field(default=None, description="Source outlet name")
    url: Optional[str] = Field(default=None, description="Article URL")
    published_at: Optional[str] = Field(default=None, description="Original publication time (ISO)")


class NewsHeadline(BaseModel):
    id: str = Field(..., description="Synthesized article ID")
    title: str = Field(..., description="Balanced headline")
    summary: str = Field(..., description="1-2 sentence summary (lede)")
    key_points: List[str] = Field(default_factory=list, description="Key factual bullet points")
    sources_count: int = Field(default=0, description="Number of sources used in synthesis")
    diversity_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Diversity measure of source mix")
    severity: Literal["breaking", "urgent", "normal"] = Field(default="normal")
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Last update time (ISO)")


class NewsArticleSynth(BaseModel):
    id: str = Field(..., description="Synthesized article ID")
    title: str = Field(..., description="Balanced headline")
    lede: str = Field(..., description="Short opening paragraph capturing essence")
    balanced_body: str = Field(..., description="Full balanced narrative with neutral tone")
    key_points: List[str] = Field(default_factory=list)
    citations: List[NewsSourceRef] = Field(default_factory=list)
    diversity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    severity: Literal["breaking", "urgent", "normal"] = Field(default="normal")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    images: Optional[List[Dict[str, Any]]] = Field(default=None, description="Images extracted from source content")


class NewsHeadlinesResponse(BaseModel):
    headlines: List[NewsHeadline] = Field(default_factory=list)
    last_updated: str = Field(default_factory=lambda: datetime.utcnow().isoformat())



