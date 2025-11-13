"""
Site Crawl Models - Roosevelt's Focused Domain Reconnaissance
Pydantic structures for domain-scoped crawling and synthesis.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class CrawlFinding(BaseModel):
    """Single relevant page found during domain crawl."""
    url: str = Field(description="Canonical URL of the page")
    title: Optional[str] = Field(default=None, description="Page title if available")
    relevance_score: float = Field(ge=0.0, le=1.0, description="Relevance score 0-1")
    reasons: List[str] = Field(default_factory=list, description="Short reasons the page matches criteria")
    snippet: Optional[str] = Field(default=None, description="Representative snippet showcasing relevance")
    tokens: Optional[int] = Field(default=None, description="Approximate token count of extracted text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional extracted metadata")


class SiteCrawlResult(BaseModel):
    """Aggregated result of a focused domain crawl."""
    findings: List[CrawlFinding] = Field(default_factory=list, description="Relevant pages that matched criteria")
    total_crawled: int = Field(ge=0, description="Pages successfully crawled")
    total_considered: int = Field(ge=0, description="Pages discovered/considered during crawl")
    filters_applied: Dict[str, Any] = Field(default_factory=dict, description="Filter parameters used during crawl")
    summary: Optional[str] = Field(default=None, description="Brief summary of what was found")





































