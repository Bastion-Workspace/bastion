"""
Substack Models - Roosevelt's Long-Form Article Writing Structures

Structured request/response for the Substack agent producing
long-form markdown articles suitable for blog posts and Substack publications.
"""

from pydantic import BaseModel, Field
from typing import Dict, Optional, Literal, List


class SubstackArticleRequest(BaseModel):
    """User-facing request options for article generation."""
    topic: str = Field(description="Primary topic or article theme")
    target_length_words: int = Field(
        default=2500, 
        ge=500, 
        le=5000, 
        description="Target article length in words"
    )
    tone: Literal["professional", "conversational", "analytical", "opinionated", "storytelling"] = Field(
        default="conversational", 
        description="Overall writing tone"
    )
    style: Literal["journalistic", "essay", "commentary", "analysis", "narrative"] = Field(
        default="commentary",
        description="Article style and structure"
    )
    include_citations: bool = Field(
        default=True, 
        description="Include inline citations and references"
    )
    include_conclusion: bool = Field(
        default=True,
        description="Include a conclusion section"
    )


class SubstackArticleResponse(BaseModel):
    """Structured output for the Substack agent."""
    task_status: Literal["complete", "incomplete", "error"] = Field(
        description="Overall task status"
    )
    article_text: str = Field(
        description="Full article in markdown format"
    )
    metadata: Dict[str, int | float | str] = Field(
        default_factory=dict,
        description="Auxiliary data: word_count, reading_time_minutes, section_count, etc."
    )


class ResearchQuestion(BaseModel):
    """Individual research question for article enhancement."""
    question: str = Field(description="The research question to investigate")
    priority: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Importance of this question for the article"
    )
    search_type: Literal["local", "web", "both"] = Field(
        default="local",
        description="Where to search for answers"
    )
    rationale: str = Field(
        description="Why this question matters for the article"
    )


class ArticleResearchPlan(BaseModel):
    """Plan for researching additional context for article."""
    needs_research: bool = Field(
        description="Whether any research is needed"
    )
    research_questions: List[ResearchQuestion] = Field(
        default_factory=list,
        description="Questions to investigate"
    )
    estimated_depth: Literal["none", "light", "moderate", "deep"] = Field(
        default="none",
        description="How much research is needed"
    )
    web_search_needed: bool = Field(
        default=False,
        description="Whether web search will be required"
    )
    rationale: str = Field(
        default="",
        description="Overall explanation of research strategy"
    )


class ResearchFinding(BaseModel):
    """Result from a single research query."""
    question: str = Field(description="The question that was researched")
    sources_found: int = Field(description="Number of relevant sources")
    summary: str = Field(description="Key findings from research")
    citations: List[str] = Field(
        default_factory=list,
        description="Source citations for findings"
    )
    search_type_used: Literal["local", "web"] = Field(
        description="Type of search performed"
    )


def get_substack_structured_output() -> SubstackArticleResponse:
    """Convenience accessor for LangGraph structured output registration."""
    return SubstackArticleResponse

