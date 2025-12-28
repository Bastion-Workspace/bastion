"""
Formatting Detection Utilities

Intelligent LLM-based detection of when research data would benefit from structured formatting.
Analyzes user intent and data characteristics to recommend routing to data formatting and/or visualization.
"""

import logging
import json
import re
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config.settings import settings

logger = logging.getLogger(__name__)


async def detect_formatting_need(user_query: str, research_response: str) -> Optional[str]:
    """
    Detect if research results would benefit from structured formatting
    
    Uses LLM to intelligently analyze:
    1. User preferences (explicit format requests, conversational context)
    2. Data characteristics (comparative, statistical, hierarchical)
    3. Value assessment (would formatting significantly improve comprehension?)
    
    Args:
        user_query: Original user query
        research_response: Research findings/response
        
    Returns:
        "data_formatting" if formatting recommended, None otherwise
        
    Note: This is a legacy function for backward compatibility.
    Use detect_post_processing_needs() for structured recommendations.
    """
    try:
        # Skip if no substantial research data
        if not research_response or len(research_response) < 100:
            logger.debug("Skipping formatting detection: insufficient data")
            return None
            
        # Use new structured detection
        recommendations = await detect_post_processing_needs(user_query, research_response)
        
        if recommendations and (recommendations.get("table_recommended") or recommendations.get("timeline_recommended")):
            logger.info("Formatting recommended")
            return "data_formatting"
        
        return None
        
    except Exception as e:
        logger.error(f"Format detection failed: {e}")
        return None


async def detect_post_processing_needs(user_query: str, research_response: str) -> Optional[Dict[str, Any]]:
    """
    Detect what post-processing enhancements would benefit the research results
    
    Uses LLM to intelligently analyze:
    1. User preferences (explicit format requests, conversational context)
    2. Data characteristics (comparative, statistical, hierarchical, temporal)
    3. Value assessment (would formatting/visualization significantly improve comprehension?)
    
    Args:
        user_query: Original user query
        research_response: Research findings/response
        
    Returns:
        Dict with recommendations:
        {
            "table_recommended": bool,
            "chart_recommended": bool,
            "timeline_recommended": bool,
            "formatting_reasoning": str
        }
        None if no post-processing needed
    """
    try:
        # Skip if no substantial research data
        if not research_response or len(research_response) < 100:
            logger.debug("Skipping post-processing detection: insufficient data")
            return None
        
        # **BULLY!** Fast-path check for explicit visualization requests!
        # This catches follow-up queries like "Can you graph those stats?" immediately
        query_lower = user_query.lower()
        explicit_visualization_keywords = [
            "graph", "chart", "plot", "visualize", "visualization", "diagram",
            "show me a graph", "create a chart", "make a graph", "draw a chart",
            "can you graph", "can you chart", "graph it", "chart it", "plot it",
            "show a graph", "show a chart", "display a graph", "display a chart"
        ]
        has_explicit_visualization_request = any(keyword in query_lower for keyword in explicit_visualization_keywords)
        
        if has_explicit_visualization_request:
            logger.info(f"Explicit visualization request detected in query: '{user_query}'")
            # Return immediate recommendation for chart
            return {
                "table_recommended": False,
                "chart_recommended": True,
                "timeline_recommended": False,
                "formatting_reasoning": f"User explicitly requested visualization: '{user_query}'"
            }
        
        # Use LLM to intelligently detect post-processing needs
        recommendations = await _llm_analyze_post_processing_needs(user_query, research_response)
        
        if recommendations and (recommendations.get("table_recommended") or 
                               recommendations.get("chart_recommended") or 
                               recommendations.get("timeline_recommended")):
            logger.info(f"Post-processing recommended: table={recommendations.get('table_recommended')}, "
                       f"chart={recommendations.get('chart_recommended')}, "
                       f"timeline={recommendations.get('timeline_recommended')}")
            return recommendations
        
        return None
        
    except Exception as e:
        logger.error(f"Post-processing detection failed: {e}")
        return None


async def _llm_analyze_formatting_need(user_query: str, research_findings: str) -> Optional[str]:
    """Use LLM to analyze if research data would benefit from structured formatting (legacy function)"""
    try:
        recommendations = await _llm_analyze_post_processing_needs(user_query, research_findings)
        if recommendations:
            if recommendations.get("table_recommended"):
                return "table"
            elif recommendations.get("chart_recommended"):
                return "chart"
            elif recommendations.get("timeline_recommended"):
                return "timeline"
        return None
    except Exception as e:
        logger.error(f"LLM formatting analysis failed: {e}")
        return None


async def _llm_analyze_post_processing_needs(user_query: str, research_findings: str) -> Optional[Dict[str, Any]]:
    """Use LLM to analyze what post-processing enhancements would benefit the research data"""
    try:
        # **BULLY!** Enhanced detection for explicit visualization requests!
        # Check for explicit visualization keywords first (fast path)
        query_lower = user_query.lower()
        explicit_visualization_keywords = [
            "graph", "chart", "plot", "visualize", "visualization", "diagram",
            "show me a graph", "create a chart", "make a graph", "draw a chart",
            "can you graph", "can you chart", "graph it", "chart it", "plot it"
        ]
        has_explicit_visualization_request = any(keyword in query_lower for keyword in explicit_visualization_keywords)
        
        # Enhanced user-preference-aware post-processing analysis prompt
        analysis_prompt = f"""You are an Intelligent Post-Processing Specialist. Analyze research data and determine what presentation enhancements would improve comprehension.

USER QUERY: "{user_query}"
RESEARCH FINDINGS: {research_findings[:1500]}...

**CRITICAL**: If the user explicitly requested a graph, chart, or visualization (e.g., "graph those stats", "can you chart this", "show me a graph"), you MUST recommend chart_recommended=true even if the data seems simple.

DECISION FRAMEWORK:

1. USER PREFERENCE ANALYSIS:
- Did user explicitly request "graph", "chart", "plot", "visualize", or "visualization"?
- Did user explicitly request "no table/formatting/chart"?
- Did user ask for specific format (paragraph, explanation, summary)?
- Is this a conversational query suggesting narrative preference?
- Context: Follow-up question where formatting might be jarring?

2. DATA VALUE ASSESSMENT:
- Would structured format SIGNIFICANTLY improve comprehension?
- Is this genuinely comparative/statistical data with multiple dimensions?
- Does tabular format provide clear advantage over prose?
- Are there trends/patterns that would benefit from visualization?
- Is there sufficient structured data to warrant formatting/visualization?

3. APPROPRIATENESS ANALYSIS:
- **Table**: Multi-dimensional comparisons, statistics, rankings, categorical data
- **Chart**: Trends over time, relationships between variables, numerical patterns, distributions, OR ANY EXPLICIT USER REQUEST FOR VISUALIZATION
- **Timeline**: Historical/chronological events, sequential processes
- **Both Table + Chart**: Data with both categorical comparisons AND trends (e.g., "Compare sales by region and show growth over time")
- **Text**: Simple answers, conversational responses, narratives

DECISION RULES:
- **EXPLICIT VISUALIZATION REQUESTS ALWAYS WIN**: If user says "graph", "chart", "plot", etc., recommend chart_recommended=true
- EXPLICIT USER PREFERENCE OVERRIDES data characteristics
- Only recommend when it provides SIGNIFICANT value (unless explicitly requested)
- Respect conversational context and user intent
- Prioritize user experience over data perfectionism
- **CONTENT ANALYSIS QUERIES**: Queries asking "what does X say" or "key insights from Y" prefer narrative analysis
- **COMPARATIVE QUERIES**: Queries asking "compare A vs B" or "which has more X" benefit from tables
- **TREND QUERIES**: Queries asking "how has X changed" or "show trends" benefit from charts
- **BOTH**: Data with both comparisons AND trends can benefit from both table and chart

EXAMPLES:
- "Compare debt by country" + comparative data → TABLE only
- "Show GDP growth over time" + time series data → CHART only
- "Can you graph those stats?" + any data → CHART (explicit request!)
- "Graph the national debt data" + any data → CHART (explicit request!)
- "Compare sales by region and show growth trends" + both → TABLE + CHART
- "Create a timeline for the history" + historical data → TIMELINE
- "Tell me about debt, no tables please" + any data → NONE
- "Who has the most debt?" + simple data → NONE (overkill, unless user explicitly requests chart)
- "Explain the situation" + any data → NONE (narrative preferred)

RESPOND WITH ONLY valid JSON matching this schema:
{{
    "table_recommended": boolean,
    "chart_recommended": boolean,
    "timeline_recommended": boolean,
    "formatting_reasoning": "Brief explanation of recommendations"
}}

Response:"""

        # Use fast model for quick decision
        llm = ChatOpenAI(
            model=settings.FAST_MODEL,
            openai_api_key=settings.OPENROUTER_API_KEY,
            openai_api_base=settings.OPENROUTER_BASE_URL,
            temperature=0.1,
            max_tokens=200
        )
        
        response = await llm.ainvoke([
            SystemMessage(content="You are a data presentation specialist. Always respond with valid JSON only."),
            HumanMessage(content=analysis_prompt)
        ])
        
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
            recommendations = json.loads(text)
            logger.info(f"Post-processing analysis: {recommendations}")
            return recommendations
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse post-processing recommendations: {e}")
            return None
        
    except Exception as e:
        logger.error(f"LLM post-processing analysis failed: {e}")
        return None





