"""
Dynamic Tool Analyzer
Analyzes queries to determine which tool categories are needed for dynamic loading
"""

import logging
from typing import Dict, Any, List, Optional

from services.langgraph_tools.tool_categories import (
    ToolCategory,
    detect_categories_from_query,
    get_core_tools_for_agent
)
from services.langgraph_tools.centralized_tool_registry import AgentType

logger = logging.getLogger(__name__)


class DynamicToolAnalyzer:
    """Analyzes queries to determine needed tool categories"""
    
    def __init__(self):
        """Initialize the analyzer"""
        pass
    
    async def analyze_tool_needs(
        self,
        query: str,
        agent_type: AgentType,
        conversation_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze query to determine which tool categories are needed
        
        Args:
            query: User query string
            agent_type: Type of agent processing the query
            conversation_context: Optional conversation history/context
            
        Returns:
            {
                "core_categories": [...],  # Always needed for this agent
                "conditional_categories": [...],  # Triggered by query
                "confidence": 0.85,
                "reasoning": "Query mentions AWS costs..."
            }
        """
        agent_type_str = agent_type.value
        
        # Get core categories (always loaded)
        core_tools = get_core_tools_for_agent(agent_type_str)
        core_categories = self._map_tools_to_categories(core_tools)
        
        # Detect conditional categories from query
        detected_categories = detect_categories_from_query(query)
        
        # Filter out categories already in core
        conditional_categories = [
            cat for cat in detected_categories 
            if cat not in core_categories
        ]
        
        # Check conversation context for hints
        if conversation_context:
            previous_tools = conversation_context.get("previous_tools_used", [])
            if previous_tools:
                # If web search was used before, likely needed again
                if any("web" in tool.lower() for tool in previous_tools):
                    if ToolCategory.SEARCH_WEB not in conditional_categories:
                        conditional_categories.append(ToolCategory.SEARCH_WEB)
        
        # Build reasoning
        reasoning_parts = []
        if conditional_categories:
            reasoning_parts.append(
                f"Query triggers categories: {', '.join([c.value for c in conditional_categories])}"
            )
        else:
            reasoning_parts.append("Query requires only core tools")
        
        # Calculate confidence based on keyword matches
        confidence = self._calculate_confidence(query, conditional_categories)
        
        return {
            "core_categories": core_categories,
            "conditional_categories": conditional_categories,
            "confidence": confidence,
            "reasoning": "; ".join(reasoning_parts),
            "core_tools": core_tools
        }
    
    def _map_tools_to_categories(self, tool_names: List[str]) -> List[ToolCategory]:
        """Map tool names to their categories"""
        categories = []
        
        # Tool name to category mapping
        tool_category_map = {
            "search_local": ToolCategory.SEARCH_LOCAL,
            "get_document": ToolCategory.DOCUMENT_OPS,
            "search_web": ToolCategory.SEARCH_WEB,
            "search_and_crawl": ToolCategory.SEARCH_WEB,
            "crawl_web_content": ToolCategory.SEARCH_WEB,
            "get_weather": ToolCategory.WEATHER,
            "calculate": ToolCategory.MATH,
            "get_aws_service_pricing": ToolCategory.AWS_PRICING,
            "compare_aws_regions": ToolCategory.AWS_PRICING,
            "estimate_aws_costs": ToolCategory.AWS_PRICING,
            "estimate_aws_workload": ToolCategory.AWS_PRICING,
            "search_org_files": ToolCategory.ORG_FILES,
            "list_org_todos": ToolCategory.ORG_FILES,
            "search_org_by_tag": ToolCategory.ORG_FILES,
            "expand_query": ToolCategory.EXPANSION,
            "search_conversation_cache": ToolCategory.SEARCH_LOCAL,
            "analyze_documents": ToolCategory.ANALYSIS,
            "send_room_message": ToolCategory.MESSAGING,
            "get_user_rooms": ToolCategory.MESSAGING,
            "create_file": ToolCategory.FILE_CREATION,
            "create_folder": ToolCategory.FILE_CREATION,
            "generate_image": ToolCategory.IMAGE_GENERATION,
            "crawl_website": ToolCategory.WEBSITE_CRAWL,
        }
        
        for tool_name in tool_names:
            category = tool_category_map.get(tool_name)
            if category and category not in categories:
                categories.append(category)
        
        return categories
    
    def _calculate_confidence(
        self,
        query: str,
        conditional_categories: List[ToolCategory]
    ) -> float:
        """
        Calculate confidence score for category detection
        
        Args:
            query: User query
            conditional_categories: Detected categories
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not conditional_categories:
            return 1.0  # High confidence when only core tools needed
        
        query_lower = query.lower()
        query_length = len(query.split())
        
        # More keywords matched = higher confidence
        from services.langgraph_tools.tool_categories import CATEGORY_KEYWORDS
        
        total_matches = 0
        for category in conditional_categories:
            keywords = CATEGORY_KEYWORDS.get(category, [])
            matches = sum(1 for kw in keywords if kw in query_lower)
            total_matches += matches
        
        # Normalize confidence
        # More matches relative to query length = higher confidence
        if query_length > 0:
            confidence = min(1.0, total_matches / query_length * 0.5 + 0.5)
        else:
            confidence = 0.7  # Default for very short queries
        
        return confidence


# Global analyzer instance
_analyzer_instance: Optional[DynamicToolAnalyzer] = None


def get_dynamic_tool_analyzer() -> DynamicToolAnalyzer:
    """Get global analyzer instance"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = DynamicToolAnalyzer()
    return _analyzer_instance

