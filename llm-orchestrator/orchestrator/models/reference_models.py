"""
Pydantic models for structured reference agent outputs
Type-safe models for query complexity analysis, pattern detection, and insights
"""

from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any


class QueryComplexityAnalysis(BaseModel):
    """LLM output for query complexity detection and routing"""
    complexity_level: Literal["simple_qa", "pattern_analysis", "insights"] = Field(
        description="Level of analysis needed: simple_qa (factual questions), pattern_analysis (trends/frequencies), insights (deep analysis with recommendations)"
    )
    needs_external_info: bool = Field(
        description="Whether the query requires external information (e.g., nutritional data, definitions, research)"
    )
    research_query: Optional[str] = Field(
        default=None,
        description="Formulated research query if external info is needed (e.g., 'nutritional content of pizza')"
    )
    needs_visualization: bool = Field(
        default=False,
        description="Whether the query would benefit from a chart or graph visualization"
    )
    visualization_type: Optional[str] = Field(
        default=None,
        description="Type of chart if visualization is needed (bar, line, pie, scatter, area, heatmap, box_plot, histogram)"
    )
    reasoning: str = Field(
        description="Brief explanation of complexity assessment and research need"
    )


class PatternItem(BaseModel):
    """Individual pattern found in reference document"""
    pattern_type: Literal["frequency", "temporal", "correlation", "anomaly", "trend"] = Field(
        description="Type of pattern detected"
    )
    description: str = Field(
        description="Human-readable description of the pattern"
    )
    occurrences: int = Field(
        description="Number of times this pattern appears"
    )
    examples: List[str] = Field(
        default_factory=list,
        description="Specific examples from the document (dates, entries, etc.)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional pattern metadata (dates, values, etc.)"
    )


class PatternAnalysisResult(BaseModel):
    """Structured pattern analysis results"""
    patterns: List[PatternItem] = Field(
        default_factory=list,
        description="List of patterns found in the reference document"
    )
    temporal_trends: List[str] = Field(
        default_factory=list,
        description="Temporal trends identified (e.g., 'increasing frequency in December', 'weekly pattern')"
    )
    frequencies: Dict[str, int] = Field(
        default_factory=dict,
        description="Frequency counts for recurring items (e.g., {'pizza': 5, 'salad': 12})"
    )
    correlations: List[str] = Field(
        default_factory=list,
        description="Correlations found between different data points"
    )
    summary: str = Field(
        description="Summary of all patterns found"
    )


class InsightItem(BaseModel):
    """Individual insight or observation"""
    insight_type: Literal["trend", "anomaly", "recommendation", "observation", "correlation"] = Field(
        description="Type of insight"
    )
    title: str = Field(
        description="Short title for the insight"
    )
    description: str = Field(
        description="Detailed description of the insight"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence level (0.0 to 1.0) in this insight"
    )
    supporting_evidence: List[str] = Field(
        default_factory=list,
        description="Specific evidence from the document supporting this insight"
    )


class InsightResult(BaseModel):
    """Structured insights and recommendations"""
    insights: List[InsightItem] = Field(
        default_factory=list,
        description="List of insights found"
    )
    key_trends: List[str] = Field(
        default_factory=list,
        description="Key trends identified"
    )
    anomalies: List[str] = Field(
        default_factory=list,
        description="Anomalies or outliers detected"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Actionable recommendations (if appropriate)"
    )
    summary: str = Field(
        description="Overall summary of insights"
    )


class ReferenceResponse(BaseModel):
    """Final structured response from reference agent"""
    task_status: Literal["complete", "incomplete", "error"] = Field(
        description="Task completion status"
    )
    response: str = Field(
        description="Natural language response to the user's query"
    )
    complexity_level: str = Field(
        description="Complexity level used: simple_qa, pattern_analysis, or insights"
    )
    patterns_found: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Patterns found (if pattern analysis was performed)"
    )
    insights: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Insights provided (if insights analysis was performed)"
    )
    research_used: bool = Field(
        default=False,
        description="Whether external research was used"
    )
    research_citations: Optional[List[str]] = Field(
        default=None,
        description="Citations from research (if research was used)"
    )
    visualization_used: bool = Field(
        default=False,
        description="Whether a visualization was generated"
    )
    visualization_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Visualization chart data (HTML or base64 image) if visualization was used"
    )
    static_visualization_data: Optional[str] = Field(
        default=None,
        description="Static visualization data (e.g. SVG string) for project library import"
    )
    static_format: Optional[str] = Field(
        default=None,
        description="Format of the static visualization (e.g. 'svg')"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        default=0.8,
        description="Confidence in the response (0.0 to 1.0)"
    )
    sources: List[str] = Field(
        default_factory=list,
        description="Sources used (document names, referenced files)"
    )

