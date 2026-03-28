"""
Visualization Tools - Chart and graph generation via backend gRPC service
Thin client wrapper that calls the Tools Service for chart generation
"""

import logging
from typing import Dict, Any, Optional

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)


# ── I/O models for create_chart_tool ────────────────────────────────────────

class CreateChartInputs(BaseModel):
    """Required inputs for create_chart_tool."""
    chart_type: str = Field(description="Type: bar, line, pie, scatter, area, heatmap, box_plot, histogram")
    data: Dict[str, Any] = Field(description="Chart data (format depends on chart type)")


class CreateChartParams(BaseModel):
    """Optional parameters."""
    title: str = Field(default="", description="Chart title")
    x_label: str = Field(default="", description="X-axis label")
    y_label: str = Field(default="", description="Y-axis label")
    interactive: bool = Field(default=True, description="Generate interactive chart")
    color_scheme: str = Field(default="plotly", description="Color scheme")
    width: int = Field(default=800, description="Chart width in pixels")
    height: int = Field(default=600, description="Chart height in pixels")
    include_static: bool = Field(default=False, description="Also generate static SVG")


class CreateChartOutputs(BaseModel):
    """Typed outputs for create_chart_tool."""
    success: bool = Field(description="Whether the chart was created")
    output_format: Optional[str] = Field(default=None, description="Output format e.g. html, svg")
    chart_data: Optional[str] = Field(default=None, description="Chart data or URL")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")

# Supported chart types (for validation/documentation)
SUPPORTED_CHART_TYPES = [
    "bar",
    "line",
    "pie",
    "scatter",
    "area",
    "heatmap",
    "box_plot",
    "histogram"
]


async def create_chart_tool(
    chart_type: str,
    data: Dict[str, Any],
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    interactive: bool = True,
    color_scheme: str = "plotly",
    width: int = 800,
    height: int = 600,
    include_static: bool = False
) -> Dict[str, Any]:
    """
    Create a chart or graph from structured data
    
    This is a thin client that calls the backend Tools Service via gRPC.
    All chart generation logic (Plotly, Kaleido) lives in the Tools Service
    to keep the orchestrator lean.
    
    Args:
        chart_type: Type of chart (bar, line, pie, scatter, area, heatmap, box_plot, histogram)
        data: Chart data (format depends on chart type)
        title: Chart title (optional)
        x_label: X-axis label (optional)
        y_label: Y-axis label (optional)
        interactive: Generate interactive chart (default: True)
        color_scheme: Color scheme to use (default: "plotly")
        width: Chart width in pixels (default: 800)
        height: Chart height in pixels (default: 600)
        include_static: Also generate a static SVG version (default: False)
        
    Returns:
        Dict with success status, output format, and chart_data
    """
    try:
        logger.info(f"Creating {chart_type} chart via Tools Service: {title} (static: {include_static})")
        
        # Validate chart type locally (quick check before gRPC call)
        if chart_type not in SUPPORTED_CHART_TYPES:
            err = f"Unsupported chart type: {chart_type}. Supported types: {', '.join(SUPPORTED_CHART_TYPES)}"
            return {
                "success": False,
                "error": err,
                "formatted": err
            }
        
        # Get backend tool client
        client = await get_backend_tool_client()
        
        # Call backend service via gRPC
        result = await client.create_chart(
            chart_type=chart_type,
            data=data,
            title=title,
            x_label=x_label,
            y_label=y_label,
            interactive=interactive,
            color_scheme=color_scheme,
            width=width,
            height=height,
            include_static=include_static
        )
        
        if result.get("success"):
            logger.info(f"Chart created successfully: {chart_type}, format: {result.get('output_format')}")
            formatted = f"Chart created successfully: {chart_type} ({result.get('output_format', '')})"
        else:
            logger.error(f"Chart creation failed: {result.get('error')}")
            formatted = f"Chart creation failed: {result.get('error', 'Unknown error')}"
        return {**result, "formatted": formatted}
        
    except Exception as e:
        logger.error(f"Error creating chart: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "formatted": f"Error creating chart: {str(e)}"
        }


register_action(
    name="create_chart",
    category="visualization",
    description="Create a chart or graph from structured data",
    inputs_model=CreateChartInputs,
    params_model=CreateChartParams,
    outputs_model=CreateChartOutputs,
    tool_function=create_chart_tool,
)


# Tool registry
VISUALIZATION_TOOLS = {
    'create_chart': create_chart_tool
}
