"""
Article Analysis Models - Structured critique for non-fiction/articles/op-eds
"""

from __future__ import annotations

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


class ArticleVerdict(str):
    SOLID = "solid"
    NEEDS_MORE = "needs_more"


class ArticleAnalysisResult(BaseModel):
    task_status: str = Field(description="complete|incomplete|error")
    summary: str = Field(description="Overall assessment in 2-4 sentences")
    thesis_clarity: str = Field(description="Assessment of thesis/stance clarity")
    structure_coherence: str = Field(description="Assessment of organization and flow")
    evidence_quality: str = Field(description="Assessment of evidence, citations, examples")
    counterarguments: str = Field(description="Coverage of counterarguments and fairness")
    tone_audience_fit: str = Field(description="Tone and audience alignment")
    strengths: List[str] = Field(default_factory=list, description="Key strengths")
    weaknesses: List[str] = Field(default_factory=list, description="Key weaknesses")
    recommendations: List[str] = Field(default_factory=list, description="Concrete improvements to make")
    missing_elements: List[str] = Field(default_factory=list, description="Missing sections or arguments to add")
    outline_suggestions: Optional[List[str]] = Field(default_factory=list, description="Suggested outline bullets to strengthen piece")
    macro_improvements: List[str] = Field(default_factory=list, description="Specific macro-level improvements with examples")
    reframing_examples: List[str] = Field(default_factory=list, description="Before/after reframing examples using existing text")
    persuasion_enhancements: List[str] = Field(default_factory=list, description="Specific persuasion techniques with examples")
    verdict: str = Field(description="solid|needs_more")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence in assessment")
    metadata: Dict[str, Any] = Field(default_factory=dict)


def get_article_analysis_structured_output() -> ArticleAnalysisResult:
    return ArticleAnalysisResult


