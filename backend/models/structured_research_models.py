"""
Structured Output Models for Research Agent - Roosevelt's "Best-of-Breed" Approach
LangGraph with_structured_output compatible models for pure structured responses
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List, Literal, Dict, Any, Union
from enum import Enum
from datetime import datetime


# ===== ENUMS AND TYPES =====

class TaskStatus(str, Enum):
    """Research task completion status"""
    COMPLETE = "complete"
    INCOMPLETE = "incomplete" 
    PERMISSION_REQUIRED = "permission_required"
    ERROR = "error"


class ConfidenceLevel(str, Enum):
    """Research confidence levels"""
    VERY_LOW = "very_low"      # 0.0-0.2
    LOW = "low"                # 0.2-0.4
    MEDIUM = "medium"          # 0.4-0.6
    HIGH = "high"              # 0.6-0.8
    VERY_HIGH = "very_high"    # 0.8-1.0


class SourceType(str, Enum):
    """Types of research sources"""
    DOCUMENT = "document"
    WEB_PAGE = "web_page"
    ENTITY = "entity"
    SEARCH_RESULT = "search_result"
    CITATION = "citation"


class PermissionType(str, Enum):
    """Types of permissions that can be requested"""
    WEB_SEARCH = "web_search"
    CRAWL_WEB_CONTENT = "crawl_web_content"


# ===== CORE DATA MODELS =====

class ResearchSource(BaseModel):
    """Individual research source with metadata"""
    title: str = Field(description="Title or name of the source")
    source_type: SourceType = Field(description="Type of source")
    url: Optional[str] = Field(default=None, description="URL if applicable")
    snippet: Optional[str] = Field(default=None, description="Brief excerpt or summary")
    relevance_score: Optional[float] = Field(
        default=None, 
        ge=0.0, 
        le=1.0, 
        description="Relevance score (0.0-1.0)"
    )
    document_id: Optional[str] = Field(default=None, description="Document ID for internal references")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional source metadata")


class ResearchCitation(BaseModel):
    """Citation for frontend display - Roosevelt's "Visual Citation" standard"""
    source_title: str = Field(description="Title of the cited source")
    source_url: Optional[str] = Field(default=None, description="URL of the source")
    quote_text: str = Field(description="The quoted/cited text")
    relevance_context: str = Field(description="Why this citation is relevant")
    page_reference: Optional[str] = Field(default=None, description="Page or section reference")
    
    # Enhanced citation fields for comprehensive source tracking
    citation_type: Optional[str] = Field(default="reference", description="Type of citation: webpage, document, book, quote, tool_usage")
    published_date: Optional[str] = Field(default=None, description="Publication or creation date")
    accessed_date: Optional[str] = Field(default=None, description="Date when source was accessed")
    author: Optional[str] = Field(default=None, description="Author or creator of the source")
    domain: Optional[str] = Field(default=None, description="Domain or publisher of web sources")
    document_id: Optional[str] = Field(default=None, description="Internal document ID for local sources")
    confidence: Optional[float] = Field(default=0.8, ge=0.0, le=1.0, description="Confidence in citation accuracy")


class ToolUsageRecord(BaseModel):
    """Record of tool usage during research"""
    tool_name: str = Field(description="Name of the tool used")
    success: bool = Field(description="Whether tool execution was successful")
    result_count: int = Field(default=0, description="Number of results returned")
    execution_time: Optional[float] = Field(default=None, description="Execution time in seconds")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")


class PermissionRequest(BaseModel):
    """Structured permission request for web tools"""
    permission_type: PermissionType = Field(description="Type of permission requested")
    justification: str = Field(description="Why this permission is needed for the research")
    scope: str = Field(description="Specific scope of what will be searched/accessed")
    query_context: str = Field(description="The research query that triggered this request")
    urgency: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Urgency level of the permission request"
    )
    alternative_available: bool = Field(
        default=False,
        description="Whether alternative research methods are available"
    )


# ===== MAIN STRUCTURED OUTPUTS =====

class LocalResearchResult(BaseModel):
    """Results from local-only research (no web access)"""
    task_status: TaskStatus = Field(description="Status of the research task")
    research_summary: str = Field(description="Natural language summary of research findings")
    sources_found: List[ResearchSource] = Field(
        default_factory=list,
        description="List of sources found during research"
    )
    citations: List[ResearchCitation] = Field(
        default_factory=list,
        description="Citations for frontend display"
    )
    tools_used: List[ToolUsageRecord] = Field(
        default_factory=list,
        description="Record of tools used during research"
    )
    confidence_level: ConfidenceLevel = Field(
        default=ConfidenceLevel.MEDIUM,
        description="Confidence in research completeness"
    )
    confidence_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Numeric confidence score (0.0-1.0)"
    )
    next_steps: Optional[str] = Field(
        default=None,
        description="Suggested next steps if research is incomplete"
    )
    permission_request: Optional[PermissionRequest] = Field(
        default=None,
        description="Permission request if web access needed"
    )
    processing_time: float = Field(description="Total processing time in seconds")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Timestamp of research completion"
    )


class WebResearchResult(BaseModel):
    """Results from web-enabled research"""
    task_status: TaskStatus = Field(description="Status of the research task")
    research_summary: str = Field(description="Natural language summary of research findings")
    local_sources: List[ResearchSource] = Field(
        default_factory=list,
        description="Sources found in local search"
    )
    web_sources: List[ResearchSource] = Field(
        default_factory=list,
        description="Sources found via web search"
    )
    combined_citations: List[ResearchCitation] = Field(
        default_factory=list,
        description="Citations from both local and web sources"
    )
    tools_used: List[ToolUsageRecord] = Field(
        default_factory=list,
        description="Record of all tools used during research"
    )
    confidence_level: ConfidenceLevel = Field(
        default=ConfidenceLevel.HIGH,
        description="Confidence in research completeness"
    )
    confidence_score: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Numeric confidence score (0.0-1.0)"
    )
    web_permission_used: bool = Field(
        default=True,
        description="Whether web permissions were utilized"
    )
    total_sources_found: int = Field(description="Total number of sources across all searches")
    processing_time: float = Field(description="Total processing time in seconds")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Timestamp of research completion"
    )


class PermissionRequestResponse(BaseModel):
    """Structured response when requesting permission"""
    task_status: Literal[TaskStatus.PERMISSION_REQUIRED] = Field(
        default=TaskStatus.PERMISSION_REQUIRED,
        description="Always permission_required for this response type"
    )
    permission_message: str = Field(
        description="Natural language message requesting permission"
    )
    permission_request: PermissionRequest = Field(
        description="Structured permission request details"
    )
    local_findings_summary: str = Field(
        description="Summary of what was found locally before requesting permission"
    )
    sources_found_locally: List[ResearchSource] = Field(
        default_factory=list,
        description="Sources found during local search"
    )
    tools_used: List[ToolUsageRecord] = Field(
        default_factory=list,
        description="Tools used before requesting permission"
    )
    confidence_without_web: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence level based on local results only"
    )
    estimated_improvement_with_web: float = Field(
        ge=0.0,
        le=1.0,
        description="Estimated confidence improvement with web access"
    )
    processing_time: float = Field(description="Processing time before permission request")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Timestamp of permission request"
    )


# ===== UNION TYPE FOR ALL RESEARCH OUTPUTS =====

ResearchAgentOutput = Union[
    LocalResearchResult,
    WebResearchResult, 
    PermissionRequestResponse
]


# ===== UTILITY FUNCTIONS =====

def confidence_score_to_level(score: float) -> ConfidenceLevel:
    """Convert numeric confidence score to confidence level enum"""
    if score < 0.2:
        return ConfidenceLevel.VERY_LOW
    elif score < 0.4:
        return ConfidenceLevel.LOW
    elif score < 0.6:
        return ConfidenceLevel.MEDIUM
    elif score < 0.8:
        return ConfidenceLevel.HIGH
    else:
        return ConfidenceLevel.VERY_HIGH


def confidence_level_to_score(level: ConfidenceLevel) -> float:
    """Convert confidence level enum to numeric score"""
    mapping = {
        ConfidenceLevel.VERY_LOW: 0.1,
        ConfidenceLevel.LOW: 0.3,
        ConfidenceLevel.MEDIUM: 0.5,
        ConfidenceLevel.HIGH: 0.7,
        ConfidenceLevel.VERY_HIGH: 0.9
    }
    return mapping.get(level, 0.5)
