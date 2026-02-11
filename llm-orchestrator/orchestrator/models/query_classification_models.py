"""
Pydantic models for LLM-powered query classification.
Structured output for routing research queries into collection_search, factual_query, or exploratory_search paths.
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal


class QueryPlan(BaseModel):
    """Structured query classification result for research workflow routing."""

    query_type: Literal["collection_search", "factual_query", "exploratory_search"] = Field(
        description="Which execution path to use"
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of the classification decision"
    )

    # Collection search fields (extracted when query_type is collection_search)
    has_collection_filters: bool = Field(
        default=False,
        description="Whether user specified series/author/date/type filters"
    )
    series: Optional[str] = Field(
        default=None,
        description="Series name if mentioned (e.g., Dilbert, Calvin and Hobbes)"
    )
    author: Optional[str] = Field(
        default=None,
        description="Author/creator if mentioned"
    )
    content_type: Optional[str] = Field(
        default=None,
        description="Content type: comic, photo, book, medical, screenshot, etc."
    )
    date_range: Optional[str] = Field(
        default=None,
        description="Date or date range if mentioned (e.g., 1989, December 2020)"
    )
    concept: Optional[str] = Field(
        default=None,
        description="Semantic concept for vector search (e.g., brains, dogs, office politics)"
    )
    requested_count: Optional[int] = Field(
        default=None,
        description="User-requested number of results (e.g. 5 from 'give me 5 more'); use as limit when set"
    )

    # Execution hints
    use_hybrid_search: bool = Field(
        default=False,
        description="Use vector (concept) + SQL (filters) hybrid search"
    )
    quick_local_check: bool = Field(
        default=False,
        description="Do a fast metadata-only local check before web"
    )
    needs_web_search: bool = Field(
        default=False,
        description="Likely needs web search for factual information"
    )
    skip_permission: bool = Field(
        default=False,
        description="Skip web search permission (obviously local-only query)"
    )

    estimated_time: str = Field(
        default="",
        description="Estimated response time (e.g., 1-2 seconds, 3-5 seconds)"
    )
