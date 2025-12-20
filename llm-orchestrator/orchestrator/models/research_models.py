"""
Pydantic models for structured research agent outputs
Type-safe models for assessment and gap analysis
"""

from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class ResearchAssessmentResult(BaseModel):
    """Structured output for research quality assessment - evaluating if results are sufficient"""
    sufficient: bool = Field(
        description="Whether the search results are sufficient to answer the query comprehensively"
    )
    has_relevant_info: bool = Field(
        description="Whether the results contain relevant information"
    )
    missing_info: List[str] = Field(
        default_factory=list,
        description="List of specific information gaps that need to be filled"
    )
    confidence: float = Field(
        description="Confidence in the assessment (0.0-1.0). Note: Anthropic API doesn't support min/max constraints on number types."
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of the assessment decision"
    )


class ResearchGapAnalysis(BaseModel):
    """Structured output for research gap analysis - identifying what information is missing"""
    missing_entities: List[str] = Field(
        default_factory=list,
        description="Specific entities, people, facts, or concepts that are missing from results"
    )
    suggested_queries: List[str] = Field(
        default_factory=list,
        description="Targeted search queries that could fill the identified gaps"
    )
    needs_web_search: bool = Field(
        default=False,
        description="Whether web search would likely help fill the gaps"
    )
    needs_local_search: Optional[bool] = Field(
        default=None,
        description="Whether local/document search would likely help fill the gaps"
    )
    gap_severity: Literal["minor", "moderate", "severe"] = Field(
        default="moderate",
        description="How significant the gaps are for answering the query"
    )
    reasoning: str = Field(
        default="",
        description="Explanation of why these gaps exist and how to fill them"
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence in the gap analysis (0.0-1.0)"
    )


class QuickAnswerAssessment(BaseModel):
    """Structured output for quick answer evaluation - determining if query can be answered from general knowledge"""
    can_answer_quickly: bool = Field(
        description="Whether this query can be answered accurately from general knowledge without searching"
    )
    confidence: float = Field(
        description="Confidence in the quick answer (0.0-1.0). Note: Anthropic API doesn't support min/max constraints on number types."
    )
    quick_answer: Optional[str] = Field(
        default=None,
        description="The quick answer text if can_answer_quickly=true, otherwise None"
    )
    reasoning: str = Field(
        default="",
        description="Explanation of why this can or cannot be answered quickly"
    )


class QueryTypeDetection(BaseModel):
    """Structured output for query type detection - determining if query should synthesize or present options"""
    query_type: Literal["objective", "subjective", "mixed"] = Field(
        description="Type of query: 'objective' for factual queries that benefit from synthesis, 'subjective' for preference-based queries that benefit from multiple options, 'mixed' for queries that need both"
    )
    confidence: float = Field(
        description="Confidence in the query type classification (0.0-1.0)"
    )
    reasoning: str = Field(
        default="",
        description="Explanation of why this query type was chosen"
    )
    should_present_options: bool = Field(
        description="Whether to present 2-3 distinct options/approaches instead of synthesizing a single answer"
    )
    num_options: Optional[int] = Field(
        default=3,
        description="Number of options to present if should_present_options=true (typically 2-3)"
    )





