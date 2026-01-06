"""
Visualization Subgraph

Reusable subgraph for generating charts and graphs from research data.
Can be used by:
- Full Research Agent (visualizing research results)
- Any agent needing chart generation

Inputs:
- query: The visualization request or original query
- messages: Conversation history for context
- metadata: Optional metadata (user_id, etc.)
- research_data: Optional research findings to visualize

Outputs:
- visualization_result: Chart generation result with metadata
- chart_type: Type of chart generated
- chart_data: Chart rendering data (HTML, base64, etc.)
- success: Whether chart generation was successful
"""

import logging
import json
import re
from typing import Dict, Any
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage

from orchestrator.agents.base_agent import BaseAgent
from orchestrator.models.visualization_models import VisualizationAnalysis
from orchestrator.tools.visualization_tools import create_chart_tool

logger = logging.getLogger(__name__)


# Use Dict[str, Any] for compatibility with any agent state
VisualizationSubgraphState = Dict[str, Any]


def _validate_chart_data_quality(chart_type: str, chart_data: Dict[str, Any], query: str) -> bool:
    """
    Validate that chart data has sufficient value for visualization.
    
    Returns True only if:
    - Data has sufficient data points (minimum 3-5 depending on chart type)
    - Data shows meaningful variation or patterns
    - Chart would provide value beyond what text can convey
    - User explicitly requested visualization (overrides strict checks)
    """
    if not chart_data:
        return False
    
    # Check for explicit visualization request in query (overrides strict validation)
    query_lower = query.lower()
    explicit_viz_keywords = ["graph", "chart", "plot", "visualize", "visualization", "show me a", "create a chart", "make a graph"]
    is_explicit_request = any(keyword in query_lower for keyword in explicit_viz_keywords)
    
    # For explicit requests, be more lenient but still check for minimum data
    min_data_points = 2 if is_explicit_request else 3
    
    # Validate based on chart type
    if chart_type in ["bar", "pie"]:
        labels = chart_data.get("labels", [])
        values = chart_data.get("values", [])
        
        if not labels or not values:
            return False
        
        if len(labels) < min_data_points or len(values) < min_data_points:
            logger.debug(f"Bar/pie chart: insufficient data points ({len(labels)} labels, {len(values)} values)")
            return False
        
        # Check for meaningful variation (not all same values)
        if len(set(values)) == 1 and not is_explicit_request:
            logger.debug(f"Bar/pie chart: all values are the same ({values[0]}) - no visualization value")
            return False
        
        return True
    
    elif chart_type in ["line", "scatter", "area"]:
        x = chart_data.get("x", [])
        y = chart_data.get("y", [])
        series = chart_data.get("series", [])
        
        # Check single series format
        if x and y:
            if len(x) < min_data_points or len(y) < min_data_points:
                logger.debug(f"Line/scatter/area chart: insufficient data points ({len(x)} x, {len(y)} y)")
                return False
            
            if len(x) != len(y):
                logger.debug(f"Line/scatter/area chart: mismatched array lengths")
                return False
            
            # Check for meaningful variation
            if len(set(y)) == 1 and not is_explicit_request:
                logger.debug(f"Line/scatter/area chart: all y values are the same - no visualization value")
                return False
            
            return True
        
        # Check multi-series format
        if series and isinstance(series, list):
            if len(series) == 0:
                return False
            
            # Check if any series has sufficient data
            has_valid_series = False
            for s in series:
                if isinstance(s, dict):
                    s_x = s.get("x", [])
                    s_y = s.get("y", [])
                    if len(s_x) >= min_data_points and len(s_y) >= min_data_points and len(s_x) == len(s_y):
                        has_valid_series = True
                        break
            
            return has_valid_series
        
        return False
    
    elif chart_type == "heatmap":
        z = chart_data.get("z", [])
        if not z or not isinstance(z, list):
            return False
        
        # Heatmap needs at least 2x2 grid
        if len(z) < 2:
            return False
        
        row_lengths = [len(row) for row in z if isinstance(row, list)]
        if not row_lengths or min(row_lengths) < 2:
            return False
        
        return True
    
    elif chart_type in ["histogram", "box_plot"]:
        values = chart_data.get("values", [])
        if not values or len(values) < min_data_points:
            logger.debug(f"Histogram/box_plot: insufficient data points ({len(values) if values else 0})")
            return False
        
        return True
    
    # Unknown chart type - be conservative
    logger.warning(f"Unknown chart type for validation: {chart_type}")
    return False


def _build_visualization_analysis_prompt(user_request: str, research_data: str, conversation_context: str) -> str:
    """Build prompt for LLM to analyze visualization needs and extract data"""
    
    return f"""You are a Data Visualization Specialist - an expert at identifying when data would benefit from charts and extracting the right data points.

**MISSION**: Analyze research data and determine the best chart type, then extract structured data points for visualization.

**CRITICAL VALUE ASSESSMENT**: Only create charts if:
1. The data has sufficient data points (minimum 3-5 depending on chart type)
2. The data shows meaningful variation, patterns, or relationships
3. The visualization would provide SIGNIFICANT value beyond what text can convey
4. The user explicitly requested a chart/graph/visualization (overrides strict checks but still needs minimum data)

**DO NOT create charts for:**
- Single data points or very few values (< 3 data points)
- Data where all values are the same (no variation)
- Data where text would be clearer than a chart
- Exploratory queries without clear visualization intent

**USER REQUEST**: {user_request}

**RESEARCH DATA TO VISUALIZE**:
{research_data[:2000] if research_data else "No research data provided"}

**CONVERSATION CONTEXT**:
{conversation_context}

**STRUCTURED OUTPUT REQUIRED**:

You MUST respond with valid JSON matching this schema:
{{
    "chart_type": "bar|line|pie|scatter|area|heatmap|box_plot|histogram",
    "title": "Chart title that summarizes what the visualization shows",
    "x_label": "X-axis label (for charts with axes, null for pie charts)",
    "y_label": "Y-axis label (for charts with axes, null for pie charts)",
    "data": {{
        "labels": ["label1", "label2"] (for bar/pie),
        "values": [10, 20] (for bar/pie/histogram),
        "x": [1, 2, 3] (for line/scatter/area),
        "y": [10, 20, 30] (for line/scatter/area),
        "series": [{{"x": [1,2], "y": [10,20], "name": "Series1"}}] (for multi-series line),
        "z": [[1,2],[3,4]] (for heatmap - 2D array)
    }},
    "confidence": 0.9,
    "reasoning": "Explanation of why this chart type and data structure were chosen"
}}

**CONFIDENCE SCORING**:
- confidence >= 0.8: High confidence - sufficient data, clear patterns, meaningful value
- confidence 0.5-0.79: Medium confidence - some data but may be borderline
- confidence < 0.5: Low confidence - insufficient data or low visualization value (DO NOT create chart)

If data is insufficient or would not provide meaningful value, set confidence < 0.5 and return empty/null data.

**CHART TYPE SELECTION GUIDE**:
- **bar**: Comparisons between categories (e.g., "Compare sales by region")
- **line**: Trends over time (e.g., "Show population growth over years")
- **pie**: Proportions/percentages (e.g., "Market share by company")
- **scatter**: Correlations between two variables (e.g., "Price vs Quality")
- **area**: Cumulative trends (e.g., "Total revenue over time")
- **heatmap**: 2D patterns/correlations (e.g., "Temperature by month and location")
- **box_plot**: Distributions/statistics (e.g., "Salary distribution by department")
- **histogram**: Frequency distributions (e.g., "Age distribution")

**DATA EXTRACTION RULES**:
1. Extract ONLY numerical data that was actually provided in the research
2. Use clear, descriptive labels
3. Ensure data arrays have matching lengths
4. For time series, extract dates/values chronologically
5. For comparisons, extract categories and their values
6. **CRITICAL**: Only extract data if there are sufficient data points (minimum 3-5 depending on chart type)
7. **CRITICAL**: Only extract data if visualization would provide meaningful value (patterns, trends, comparisons)
8. **CRITICAL**: If data is insufficient or would not provide value, return null/empty data and set confidence < 0.5

**CRITICAL**:
1. **STRUCTURED JSON ONLY** - No plain text responses!
2. **Use actual data** - Only data provided in research_data
3. **Valid chart type** - Must match one of the supported types
4. **Valid data structure** - Data format must match chart_type requirements
5. **Data quality validation** - Only create charts if:
   - There are sufficient data points (minimum 3-5 depending on chart type)
   - Data shows meaningful variation or patterns
   - Visualization would provide value beyond text
   - If data is insufficient, set confidence < 0.5 and return empty/null data
6. **Value assessment** - Ask: "Would this chart provide meaningful insight?" If no, don't create it.

**JSON RESPONSE EXAMPLE**:
```json
{{
    "chart_type": "bar",
    "title": "GDP by Country (2023)",
    "x_label": "Country",
    "y_label": "GDP (Trillions USD)",
    "data": {{
        "labels": ["USA", "China", "Japan", "Germany"],
        "values": [26.9, 17.7, 4.2, 4.1]
    }},
    "confidence": 0.95,
    "reasoning": "Bar chart best represents categorical comparison of GDP values"
}}
```
"""


async def analyze_visualization_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze research data and determine visualization needs"""
    try:
        logger.info("Analyzing visualization needs from research data...")
        
        query = state.get("query", "")
        messages = state.get("messages", [])
        research_data = state.get("research_data", "")
        
        # Extract conversation context
        if not messages:
            conversation_context = "No previous conversation context available."
        else:
            recent_messages = messages[-5:] if len(messages) > 5 else messages
            context_parts = []
            for i, msg in enumerate(recent_messages):
                if hasattr(msg, 'content'):
                    role = "ASSISTANT" if hasattr(msg, 'type') and msg.type == "ai" else "USER"
                    content = msg.content
                    context_parts.append(f"{i+1}. {role}: {content}")
            conversation_context = "\n".join(context_parts)
        
        # If no research_data provided, try to extract from messages
        # **BULLY!** Enhanced extraction for follow-up visualization requests!
        if not research_data:
            # Look for research findings in recent messages (prioritize assistant messages with data)
            for msg in reversed(messages[-10:]):
                if hasattr(msg, 'content'):
                    content = msg.content
                    # Check if this looks like research results or contains structured data
                    is_assistant = (hasattr(msg, 'type') and msg.type == "ai") or (hasattr(msg, 'role') and msg.role == "assistant")
                    has_data_indicators = any(keyword in content.lower() for keyword in [
                        "research", "findings", "data", "results", "analysis", 
                        "table", "statistics", "stats", "numbers", "figures",
                        "national debt", "gdp", "population", "percentage", "%"
                    ])
                    has_structured_data = "|" in content or "\t" in content or any(char.isdigit() for char in content[:500])
                    
                    # Prioritize assistant messages with data indicators or structured content
                    if is_assistant and len(content) > 200 and (has_data_indicators or has_structured_data):
                        research_data = content
                        logger.info(f"Extracted research data from previous assistant message ({len(content)} chars)")
                        break
            
            # If still no data, try any message with substantial content
            if not research_data:
                for msg in reversed(messages[-10:]):
                    if hasattr(msg, 'content'):
                        content = msg.content
                        if len(content) > 300:  # Substantial content
                            research_data = content
                            logger.info(f"Extracted data from message ({len(content)} chars) as fallback")
                            break
        
        if not research_data or len(research_data) < 50:
            logger.warning("Insufficient research data for visualization")
            return {
                "visualization_analysis": None,
                "visualization_error": "Insufficient research data for visualization",
                # Preserve critical state keys
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
        
        # Build analysis prompt
        analysis_prompt = _build_visualization_analysis_prompt(query, research_data, conversation_context)
        
        # Get LLM for analysis
        base_agent = BaseAgent("visualization_subgraph")
        llm = base_agent._get_llm(temperature=0.1, state=state)
        datetime_context = base_agent._get_datetime_context()
        
        # Build messages
        analysis_messages = [
            SystemMessage(content="You are a data visualization specialist. Always respond with valid JSON."),
            SystemMessage(content=datetime_context)
        ]
        
        if messages:
            analysis_messages.extend(messages)
        
        analysis_messages.append(HumanMessage(content=analysis_prompt))
        
        # Call LLM
        response = await llm.ainvoke(analysis_messages)
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        # Extract JSON from response
        text = response_content.strip()
        
        # Extract JSON from markdown code blocks
        if '```json' in text:
            match = re.search(r'```json\s*\n([\s\S]*?)\n```', text)
            if match:
                text = match.group(1).strip()
        elif '```' in text:
            match = re.search(r'```\s*\n([\s\S]*?)\n```', text)
            if match:
                text = match.group(1).strip()
        
        # Extract JSON object
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            text = json_match.group(0)
        
        # Parse JSON
        try:
            analysis_dict = json.loads(text)
            # Validate with Pydantic
            visualization_analysis = VisualizationAnalysis(**analysis_dict)
            
            logger.info(f"Visualization analysis complete: {visualization_analysis.chart_type}, confidence: {visualization_analysis.confidence}")
            
            return {
                "visualization_analysis": visualization_analysis.dict(),
                "chart_type": visualization_analysis.chart_type,
                "chart_title": visualization_analysis.title,
                "chart_x_label": visualization_analysis.x_label,
                "chart_y_label": visualization_analysis.y_label,
                "chart_data": visualization_analysis.data,
                # Preserve critical state keys
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse visualization analysis: {e}")
            return {
                "visualization_analysis": None,
                "visualization_error": f"Failed to parse visualization analysis: {str(e)}",
                # Preserve critical state keys
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
        
    except Exception as e:
        logger.error(f"Visualization analysis failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "visualization_analysis": None,
            "visualization_error": str(e),
            # Preserve critical state keys
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }


async def generate_chart_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate chart using visualization tool"""
    try:
        logger.info("Generating chart from visualization analysis...")
        
        visualization_analysis = state.get("visualization_analysis")
        
        if not visualization_analysis:
            error_msg = state.get("visualization_error", "No visualization analysis available")
            logger.warning(f"Cannot generate chart: {error_msg}")
            return {
                "visualization_result": {
                    "success": False,
                    "error": error_msg
                },
                "chart_type": None,
                "chart_data": None,
                # Preserve critical state keys
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
        
        # Extract chart parameters
        chart_type = visualization_analysis.get("chart_type")
        title = visualization_analysis.get("title", "")
        x_label = visualization_analysis.get("x_label")
        y_label = visualization_analysis.get("y_label")
        data = visualization_analysis.get("data", {})
        
        if not chart_type or not data:
            return {
                "visualization_result": {
                    "success": False,
                    "error": "Missing chart_type or data in visualization analysis"
                },
                "chart_type": None,
                "chart_data": None,
                # Preserve critical state keys
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
        
        # Check confidence - if low, data may not be meaningful
        confidence = visualization_analysis.get("confidence", 1.0)
        if confidence < 0.5:
            logger.info(f"Low confidence ({confidence}) in visualization analysis - skipping chart creation")
            return {
                "visualization_result": {
                    "success": False,
                    "error": "Insufficient data quality for meaningful visualization"
                },
                "chart_type": None,
                "chart_data": None,
                # Preserve critical state keys
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
        
        # Validate data quality before creating chart
        data_is_meaningful = _validate_chart_data_quality(chart_type, data, state.get("query", ""))
        
        if not data_is_meaningful:
            logger.info(f"Data quality validation failed for {chart_type} chart - skipping creation")
            return {
                "visualization_result": {
                    "success": False,
                    "error": "Insufficient data points or low visualization value. Chart would not provide meaningful insight."
                },
                "chart_type": None,
                "chart_data": None,
                # Preserve critical state keys
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
        
        # Call visualization tool
        chart_result = await create_chart_tool(
            chart_type=chart_type,
            data=data,
            title=title,
            x_label=x_label or "",
            y_label=y_label or "",
            interactive=True,
            color_scheme="plotly",
            width=800,
            height=600,
            include_static=True
        )
        
        if chart_result.get("success"):
            logger.info(f"Chart generated successfully: {chart_type}, format: {chart_result.get('output_format')}")
        else:
            logger.error(f"Chart generation failed: {chart_result.get('error')}")
        
        return {
            "visualization_result": {
                "success": chart_result.get("success", False),
                "chart_type": chart_type,
                "title": title,
                "output_format": chart_result.get("output_format"),
                "chart_data": chart_result.get("chart_data"),
                "static_visualization_data": chart_result.get("static_svg"),
                "static_format": chart_result.get("static_format"),
                "error": chart_result.get("error")
            },
            "chart_type": chart_type,
            "chart_data": chart_result.get("chart_data"),
            "chart_output_format": chart_result.get("output_format"),
            "static_visualization_data": chart_result.get("static_svg"),
            "static_format": chart_result.get("static_format"),
            # Preserve critical state keys
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }
        
    except Exception as e:
        logger.error(f"Chart generation failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "visualization_result": {
                "success": False,
                "error": str(e)
            },
            "chart_type": None,
            "chart_data": None,
            # Preserve critical state keys
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }


def build_visualization_subgraph(checkpointer) -> StateGraph:
    """
    Build visualization subgraph for generating charts and graphs from research data
    
    This subgraph analyzes research data, determines the best chart type, and generates
    interactive charts using the backend visualization service.
    
    Expected state inputs:
    - query: str - The visualization request or original query
    - messages: List (optional) - Conversation history for context
    - metadata: Dict[str, Any] (optional) - Metadata for checkpointing and user model selection
    - research_data: str (optional) - Research findings to visualize (will be extracted from messages if not provided)
    
    Returns state with:
    - visualization_result: Dict[str, Any] - Chart generation result with success status
    - chart_type: str - Type of chart generated
    - chart_data: str - Chart rendering data (HTML, base64, etc.)
    - chart_output_format: str - Output format (html, png, svg, etc.)
    """
    subgraph = StateGraph(Dict[str, Any])
    
    # Add nodes
    subgraph.add_node("analyze_visualization", analyze_visualization_node)
    subgraph.add_node("generate_chart", generate_chart_node)
    
    # Set entry point
    subgraph.set_entry_point("analyze_visualization")
    
    # Linear flow: analyze_visualization -> generate_chart -> END
    subgraph.add_edge("analyze_visualization", "generate_chart")
    subgraph.add_edge("generate_chart", END)
    
    return subgraph.compile(checkpointer=checkpointer)

