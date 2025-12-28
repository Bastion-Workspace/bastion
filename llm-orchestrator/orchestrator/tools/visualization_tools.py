"""
Visualization Tools - Chart and graph generation via backend gRPC service
Thin client wrapper that calls the Tools Service for chart generation
"""

import logging
from typing import Dict, Any

from orchestrator.backend_tool_client import get_backend_tool_client

logger = logging.getLogger(__name__)

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
            return {
                "success": False,
                "error": f"Unsupported chart type: {chart_type}. Supported types: {', '.join(SUPPORTED_CHART_TYPES)}"
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
        else:
            logger.error(f"Chart creation failed: {result.get('error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error creating chart: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }


# Tool registry
VISUALIZATION_TOOLS = {
    'create_chart': create_chart_tool
}
