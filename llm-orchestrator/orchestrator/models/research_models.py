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
    gap_severity: Literal["minor", "moderate", "severe"] = Field(
        default="moderate",
        description="How significant the gaps are for answering the query"
    )
    reasoning: str = Field(
        default="",
        description="Explanation of why these gaps exist and how to fill them"
    )





