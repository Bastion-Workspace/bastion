"""
Visualization Models - Structured outputs for chart generation

Pydantic models for visualization analysis and results
"""

from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any


class VisualizationAnalysis(BaseModel):
    """Structured output for LLM analysis of visualization needs"""
    chart_type: Literal["bar", "line", "pie", "scatter", "area", "heatmap", "box_plot", "histogram"] = Field(
        description="Type of chart that best represents the data"
    )
    title: str = Field(
        description="Chart title that summarizes what the visualization shows"
    )
    x_label: Optional[str] = Field(
        default=None,
        description="X-axis label (for charts with axes)"
    )
    y_label: Optional[str] = Field(
        default=None,
        description="Y-axis label (for charts with axes)"
    )
    data: Dict[str, Any] = Field(
        description="Chart data structure matching the chart type requirements"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the visualization analysis (0.0-1.0)"
    )
    reasoning: str = Field(
        description="Explanation of why this chart type and data structure were chosen"
    )


class VisualizationResult(BaseModel):
    """Final visualization result with chart metadata and rendering data"""
    success: bool = Field(
        description="Whether chart generation was successful"
    )
    chart_type: str = Field(
        description="Type of chart that was generated"
    )
    title: str = Field(
        description="Chart title"
    )
    output_format: Optional[str] = Field(
        default=None,
        description="Output format (html, png, svg, etc.)"
    )
    chart_data: Optional[str] = Field(
        default=None,
        description="Chart data (HTML string, base64 image, etc.)"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if generation failed"
    )

