"""
Tool Categories for Dynamic Tool Loading
Kiro-style categorization system for on-demand tool provisioning
"""

from enum import Enum
from typing import Dict, List


class ToolCategory(str, Enum):
    """Categories for dynamic tool loading"""
    SEARCH_LOCAL = "search_local"  # Document/entity search
    SEARCH_WEB = "search_web"  # Web search/crawl
    DOCUMENT_OPS = "document_ops"  # Get/update documents
    ANALYSIS = "analysis"  # Content analysis
    MATH = "math"  # Calculations
    WEATHER = "weather"  # Weather queries
    AWS_PRICING = "aws_pricing"  # AWS cost tools
    ORG_FILES = "org_files"  # Org-mode search
    MESSAGING = "messaging"  # Room messaging
    FILE_CREATION = "file_creation"  # File/folder ops
    EXPANSION = "expansion"  # Query expansion
    IMAGE_GENERATION = "image_generation"  # Image creation
    WEBSITE_CRAWL = "website_crawl"  # Website crawling


# Keyword mappings for category detection
CATEGORY_KEYWORDS: Dict[ToolCategory, List[str]] = {
    ToolCategory.WEATHER: [
        "weather", "temperature", "forecast", "climate", "rain", "snow",
        "sunny", "cloudy", "wind", "humidity", "precipitation", "storm"
    ],
    ToolCategory.MATH: [
        "calculate", "compute", "formula", "math", "equation", "solve",
        "arithmetic", "algebra", "geometry", "statistics", "percentage"
    ],
    ToolCategory.AWS_PRICING: [
        "aws", "cost", "pricing", "ec2", "s3", "lambda", "rds", "cloudwatch",
        "billing", "estimate", "price", "region", "instance", "service cost"
    ],
    ToolCategory.SEARCH_WEB: [
        "search web", "look up", "find online", "web search", "internet",
        "online", "browse", "crawl", "scrape", "fetch from web"
    ],
    ToolCategory.ORG_FILES: [
        "org file", "org mode", "todo", "task", "org", "inbox", "project entry"
    ],
    ToolCategory.MESSAGING: [
        "message", "chat", "send message", "room", "messaging", "notify"
    ],
    ToolCategory.FILE_CREATION: [
        "create file", "new file", "create folder", "new folder", "directory"
    ],
    ToolCategory.IMAGE_GENERATION: [
        "generate image", "create image", "draw", "picture", "image", "art"
    ],
    ToolCategory.WEBSITE_CRAWL: [
        "crawl website", "crawl site", "ingest website", "capture website"
    ],
    ToolCategory.ANALYSIS: [
        "analyze", "analysis", "compare", "contrast", "evaluate", "assess"
    ],
    ToolCategory.DOCUMENT_OPS: [
        "get document", "read document", "update document", "edit document"
    ],
}


# Core tools per agent (always loaded)
AGENT_CORE_TOOLS: Dict[str, List[str]] = {
    "research_agent": [
        "search_local",
        "expand_query",
        "search_conversation_cache"
    ],
    "chat_agent": [
        "search_conversation_cache"
    ],
    # content_analysis_agent removed - migrated to llm-orchestrator gRPC service
    # story_analysis_agent removed - migrated to llm-orchestrator gRPC service
    "fiction_editing_agent": [
        "search_local",
        "get_document"
    ],
    "weather_agent": [
        "get_weather"
    ],
    "calculate_agent": [
        "calculate"
    ],
}


def get_category_keywords(category: ToolCategory) -> List[str]:
    """Get keywords for a category"""
    return CATEGORY_KEYWORDS.get(category, [])


def detect_categories_from_query(query: str) -> List[ToolCategory]:
    """
    Detect tool categories needed based on query keywords
    
    Args:
        query: User query string
        
    Returns:
        List of detected categories
    """
    query_lower = query.lower()
    detected = []
    
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in query_lower for keyword in keywords):
            detected.append(category)
    
    return detected


def get_core_tools_for_agent(agent_type: str) -> List[str]:
    """Get core tools that are always loaded for an agent"""
    return AGENT_CORE_TOOLS.get(agent_type, [])


