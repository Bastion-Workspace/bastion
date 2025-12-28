"""
Visualization Service - Chart and graph generation using Plotly
Handles all chart generation logic for the Tools Service
"""

import logging
import base64
import json
from typing import Dict, Any, Optional, Tuple
import plotly.graph_objects as go
from plotly.io import to_image, to_html

logger = logging.getLogger(__name__)

# Supported chart types
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


def _validate_chart_data(chart_type: str, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate chart data structure matches chart type requirements
    
    Args:
        chart_type: Type of chart to create
        data: Data dictionary to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if chart_type not in SUPPORTED_CHART_TYPES:
        return False, f"Unsupported chart type: {chart_type}. Supported types: {', '.join(SUPPORTED_CHART_TYPES)}"
    
    if chart_type == "bar":
        if "labels" not in data or "values" not in data:
            return False, "Bar chart requires 'labels' and 'values' keys"
        if len(data["labels"]) != len(data["values"]):
            return False, "Bar chart 'labels' and 'values' must have same length"
    
    elif chart_type == "line":
        if "series" in data:
            # Multiple series format
            if not isinstance(data["series"], list) or len(data["series"]) == 0:
                return False, "Line chart 'series' must be a non-empty list"
            for series in data["series"]:
                if "x" not in series or "y" not in series:
                    return False, "Each series must have 'x' and 'y' keys"
        elif "x" in data and "y" in data:
            # Single series format
            if len(data["x"]) != len(data["y"]):
                return False, "Line chart 'x' and 'y' must have same length"
        else:
            return False, "Line chart requires either 'x'/'y' for single series or 'series' list for multiple series"
    
    elif chart_type == "pie":
        if "labels" not in data or "values" not in data:
            return False, "Pie chart requires 'labels' and 'values' keys"
        if len(data["labels"]) != len(data["values"]):
            return False, "Pie chart 'labels' and 'values' must have same length"
    
    elif chart_type == "scatter":
        if "x" not in data or "y" not in data:
            return False, "Scatter plot requires 'x' and 'y' keys"
        if len(data["x"]) != len(data["y"]):
            return False, "Scatter plot 'x' and 'y' must have same length"
    
    elif chart_type == "area":
        if "x" not in data or "y" not in data:
            return False, "Area chart requires 'x' and 'y' keys"
        if len(data["x"]) != len(data["y"]):
            return False, "Area chart 'x' and 'y' must have same length"
    
    elif chart_type == "heatmap":
        if "z" not in data:
            return False, "Heatmap requires 'z' key with 2D array"
        if not isinstance(data["z"], list) or len(data["z"]) == 0:
            return False, "Heatmap 'z' must be a non-empty 2D array"
        if not isinstance(data["z"][0], list):
            return False, "Heatmap 'z' must be a 2D array (list of lists)"
    
    elif chart_type == "box_plot":
        if "values" not in data:
            return False, "Box plot requires 'values' key"
        if not isinstance(data["values"], list) or len(data["values"]) == 0:
            return False, "Box plot 'values' must be a non-empty list"
    
    elif chart_type == "histogram":
        if "values" not in data:
            return False, "Histogram requires 'values' key"
        if not isinstance(data["values"], list) or len(data["values"]) == 0:
            return False, "Histogram 'values' must be a non-empty list"
    
    return True, None


def _create_bar_chart(data: Dict[str, Any], config: Dict[str, Any]) -> go.Figure:
    """Create bar chart figure"""
    fig = go.Figure()
    
    orientation = data.get("orientation", "v")  # 'v' for vertical, 'h' for horizontal
    color_scheme = config.get("color_scheme", "plotly")
    
    # **BULLY!** Only set marker_color if it's not the default "plotly" string!
    # "plotly" is a scheme name, not a CSS color, so we let Plotly use its defaults!
    marker_kwargs = {}
    if color_scheme and color_scheme != "plotly":
        marker_kwargs["marker_color"] = color_scheme

    if orientation == "v":
        fig.add_trace(go.Bar(
            x=data["labels"],
            y=data["values"],
            name=data.get("series_name", "Data"),
            **marker_kwargs
        ))
        fig.update_layout(
            xaxis_title=config.get("x_label", ""),
            yaxis_title=config.get("y_label", "")
        )
    else:
        fig.add_trace(go.Bar(
            x=data["values"],
            y=data["labels"],
            orientation='h',
            name=data.get("series_name", "Data"),
            **marker_kwargs
        ))
        fig.update_layout(
            xaxis_title=config.get("x_label", ""),
            yaxis_title=config.get("y_label", "")
        )
    
    return fig


def _create_line_chart(data: Dict[str, Any], config: Dict[str, Any]) -> go.Figure:
    """Create line chart figure"""
    fig = go.Figure()
    
    if "series" in data:
        # Multiple series
        for series in data["series"]:
            fig.add_trace(go.Scatter(
                x=series["x"],
                y=series["y"],
                mode='lines+markers',
                name=series.get("name", "Series")
            ))
    else:
        # Single series
        fig.add_trace(go.Scatter(
            x=data["x"],
            y=data["y"],
            mode='lines+markers',
            name=data.get("series_name", "Data")
        ))
    
    fig.update_layout(
        xaxis_title=config.get("x_label", ""),
        yaxis_title=config.get("y_label", "")
    )
    
    return fig


def _create_pie_chart(data: Dict[str, Any], config: Dict[str, Any]) -> go.Figure:
    """Create pie chart figure"""
    fig = go.Figure(data=[go.Pie(
        labels=data["labels"],
        values=data["values"]
    )])
    
    return fig


def _create_scatter_plot(data: Dict[str, Any], config: Dict[str, Any]) -> go.Figure:
    """Create scatter plot figure"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data["x"],
        y=data["y"],
        mode='markers',
        name=data.get("series_name", "Data Points")
    ))
    
    fig.update_layout(
        xaxis_title=config.get("x_label", ""),
        yaxis_title=config.get("y_label", "")
    )
    
    return fig


def _create_area_chart(data: Dict[str, Any], config: Dict[str, Any]) -> go.Figure:
    """Create area chart figure"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data["x"],
        y=data["y"],
        mode='lines',
        fill='tonexty',
        name=data.get("series_name", "Data")
    ))
    
    fig.update_layout(
        xaxis_title=config.get("x_label", ""),
        yaxis_title=config.get("y_label", "")
    )
    
    return fig


def _create_heatmap(data: Dict[str, Any], config: Dict[str, Any]) -> go.Figure:
    """Create heatmap figure"""
    color_scheme = config.get("color_scheme", "Viridis")
    if color_scheme == "plotly":
        color_scheme = "Viridis"
        
    fig = go.Figure(data=go.Heatmap(
        z=data["z"],
        x=data.get("x_labels", None),
        y=data.get("y_labels", None),
        colorscale=color_scheme
    ))
    
    return fig


def _create_box_plot(data: Dict[str, Any], config: Dict[str, Any]) -> go.Figure:
    """Create box plot figure"""
    fig = go.Figure()
    
    if "labels" in data and len(data["labels"]) == len(data["values"]):
        # Multiple groups
        for i, label in enumerate(data["labels"]):
            fig.add_trace(go.Box(
                y=data["values"][i] if isinstance(data["values"][i], list) else [data["values"][i]],
                name=label
            ))
    else:
        # Single group
        fig.add_trace(go.Box(y=data["values"]))
    
    fig.update_layout(
        yaxis_title=config.get("y_label", "")
    )
    
    return fig


def _create_histogram(data: Dict[str, Any], config: Dict[str, Any]) -> go.Figure:
    """Create histogram figure"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=data["values"],
        nbinsx=data.get("bins", None)
    ))
    
    fig.update_layout(
        xaxis_title=config.get("x_label", ""),
        yaxis_title=config.get("y_label", "Frequency")
    )
    
    return fig


def _encode_chart_to_base64(fig: go.Figure, width: int = 800, height: int = 600) -> str:
    """Encode chart figure to base64 PNG string"""
    try:
        # Update figure size
        fig.update_layout(width=width, height=height)
        
        # Convert to image bytes
        img_bytes = to_image(fig, format="png", width=width, height=height)
        
        # Encode to base64
        base64_str = base64.b64encode(img_bytes).decode('utf-8')
        
        # Return as data URI
        return f"data:image/png;base64,{base64_str}"
    except Exception as e:
        logger.error(f"Error encoding chart to base64: {e}")
        raise


def _chart_to_html(fig: go.Figure, width: int = 800, height: int = 600) -> str:
    """Convert chart figure to HTML string"""
    try:
        # Update figure size
        fig.update_layout(width=width, height=height)
        
        # Convert to HTML - ROOSEVELT'S STABLE CDN FIX
        html_str = to_html(
            fig, 
            include_plotlyjs='https://cdn.plot.ly/plotly-2.35.2.min.js', 
            div_id="chart",
            full_html=True
        )
        
        return html_str
    except Exception as e:
        logger.error(f"Error converting chart to HTML: {e}")
        raise


def _encode_chart_to_svg(fig: go.Figure, width: int = 800, height: int = 600) -> str:
    """Encode chart figure to SVG string"""
    try:
        # Update figure size
        fig.update_layout(width=width, height=height)
        
        # Convert to SVG bytes
        svg_bytes = to_image(fig, format="svg", width=width, height=height)
        
        # Return as string
        return svg_bytes.decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding chart to SVG: {e}")
        return ""


async def create_chart(
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
        
    Returns:
        Dict with success status, output format, and chart_data
    """
    try:
        logger.info(f"Creating {chart_type} chart with title: {title}")
        
        # Validate chart type
        if chart_type not in SUPPORTED_CHART_TYPES:
            return {
                "success": False,
                "error": f"Unsupported chart type: {chart_type}. Supported types: {', '.join(SUPPORTED_CHART_TYPES)}"
            }
        
        # Validate data structure
        is_valid, error_msg = _validate_chart_data(chart_type, data)
        if not is_valid:
            return {
                "success": False,
                "error": error_msg
            }
        
        # Prepare config
        config = {
            "title": title,
            "x_label": x_label,
            "y_label": y_label,
            "color_scheme": color_scheme,
            "width": width,
            "height": height
        }
        
        # Create chart based on type
        chart_creators = {
            "bar": _create_bar_chart,
            "line": _create_line_chart,
            "pie": _create_pie_chart,
            "scatter": _create_scatter_plot,
            "area": _create_area_chart,
            "heatmap": _create_heatmap,
            "box_plot": _create_box_plot,
            "histogram": _create_histogram
        }
        
        creator_func = chart_creators[chart_type]
        fig = creator_func(data, config)
        
        # Add title if provided
        if title:
            fig.update_layout(title=title)
        
        # Calculate metadata
        data_points = 0
        series_count = 1
        
        if chart_type == "bar" or chart_type == "pie":
            data_points = len(data.get("values", []))
        elif chart_type == "line":
            if "series" in data:
                series_count = len(data["series"])
                data_points = sum(len(s.get("x", [])) for s in data["series"])
            else:
                data_points = len(data.get("x", []))
        elif chart_type in ["scatter", "area"]:
            data_points = len(data.get("x", []))
        elif chart_type == "heatmap":
            data_points = sum(len(row) for row in data.get("z", []))
        elif chart_type in ["box_plot", "histogram"]:
            if isinstance(data.get("values", []), list):
                if isinstance(data["values"][0], list):
                    series_count = len(data["values"])
                    data_points = sum(len(v) for v in data["values"])
                else:
                    data_points = len(data["values"])
        
        # Generate output
        if interactive:
            # Generate HTML
            chart_data = _chart_to_html(fig, width, height)
            output_format = "html"
        else:
            # Generate base64 PNG
            chart_data = _encode_chart_to_base64(fig, width, height)
            output_format = "base64_png"
        
        # Add static SVG if requested
        static_svg = None
        if include_static:
            static_svg = _encode_chart_to_svg(fig, width, height)
        
        logger.info(f"Chart created successfully: {chart_type}, format: {output_format}, static: {include_static}")
        
        result = {
            "success": True,
            "chart_type": chart_type,
            "output_format": output_format,
            "chart_data": chart_data,
            "metadata": {
                "width": width,
                "height": height,
                "data_points": data_points,
                "series_count": series_count
            }
        }
        
        if static_svg:
            result["static_svg"] = static_svg
            result["static_format"] = "svg"
            
        return result
        
    except Exception as e:
        logger.error(f"Error creating chart: {e}")
        return {
            "success": False,
            "error": str(e)
        }

