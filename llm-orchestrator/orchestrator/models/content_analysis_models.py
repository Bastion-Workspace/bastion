"""
Content Analysis Models - Structured outputs for content analysis agent
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class ArticleAnalysisResult(BaseModel):
    """Structured output for single document analysis"""
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
    outline_suggestions: Optional[List[str]] = Field(default_factory=list, description="Suggested outline bullets")
    verdict: str = Field(description="solid|needs_more")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence in assessment")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentSummary(BaseModel):
    """Individual document summary for multi-document comparison"""
    title: str = Field(description="Document title or filename")
    summary: str = Field(description="Comprehensive summary (300-500 words)")
    main_topics: List[str] = Field(default_factory=list, description="Primary topics or themes")
    key_arguments: List[str] = Field(default_factory=list, description="Main arguments or findings")
    key_data_points: List[str] = Field(default_factory=list, description="Important data, dates, facts")
    author_perspective: Optional[str] = Field(default=None, description="Author's perspective or conclusions")
    unique_insights: List[str] = Field(default_factory=list, description="Unique or notable aspects")
    document_id: Optional[str] = Field(default=None, description="Source document ID")
    original_length: int = Field(default=0, ge=0, description="Original document length in characters")
    confidence: float = Field(default=0.9, ge=0.0, le=1.0, description="Confidence in summary quality")


class ComparisonAnalysisResult(BaseModel):
    """Structured output for multi-document comparison analysis"""
    task_status: str = Field(description="Status of the comparison task")
    key_similarities: List[str] = Field(default_factory=list, description="Themes/ideas/patterns across documents")
    key_differences: List[str] = Field(default_factory=list, description="How documents differ")
    conflicts_contradictions: List[str] = Field(default_factory=list, description="Contradictory claims or conflicts")
    unique_contributions: Dict[str, List[str]] = Field(default_factory=dict, description="Unique insights by document title")
    overall_assessment: str = Field(description="Synthesized collective picture from all documents")
    dominant_themes: List[str] = Field(default_factory=list, description="Most prominent themes")
    gaps_missing_perspectives: List[str] = Field(default_factory=list, description="Important gaps or missing perspectives")
    reading_recommendations: List[Dict[str, str]] = Field(default_factory=list, description="Which documents to read first")
    follow_up_questions: List[str] = Field(default_factory=list, description="Suggested follow-up questions")
    documents_compared: int = Field(default=0, ge=0, description="Number of documents analyzed")
    document_titles: List[str] = Field(default_factory=list, description="Titles of compared documents")
    comparison_strategy: Literal["direct", "summarize_then_compare"] = Field(description="Strategy used for comparison")
    confidence_level: float = Field(default=0.85, ge=0.0, le=1.0, description="Overall confidence in comparison")
    document_summaries: Optional[List[DocumentSummary]] = Field(default=None, description="Individual document summaries if summarization strategy was used")

