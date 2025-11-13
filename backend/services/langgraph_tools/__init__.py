"""
LangGraph Tools Package
Modular tool implementations for LangGraph agents

**ROOSEVELT'S MODERNIZATION NOTICE**: 
This package now uses CentralizedToolRegistry for all tool management.
Legacy LangGraphToolRegistry has been retired.
"""

from .unified_search_tools import UnifiedSearchTools
from .web_content_tools import WebContentTools
from .crawl4ai_web_tools import Crawl4AIWebTools
from .content_analysis_tools import ContentAnalysisTools
from .math_tools import MathTools
from .weather_tools import WeatherTools

# **ROOSEVELT'S MODERNIZED TOOL ACCESS**
# Use CentralizedToolRegistry instead of legacy LangGraphToolRegistry

def get_tool_registry():
    """
    **DEPRECATED**: Use get_tool_registry() from centralized_tool_registry instead
    
    This function is maintained for backward compatibility only.
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ DEPRECATED: langgraph_tools.__init__.get_tool_registry() is deprecated")
    logger.warning("🎯 USE INSTEAD: from services.langgraph_tools.centralized_tool_registry import get_tool_registry")
    
    # Redirect to modern registry
    from .centralized_tool_registry import get_tool_registry as get_modern_registry
    import asyncio
    return asyncio.get_event_loop().run_until_complete(get_modern_registry())

def get_tool_registry_lazy():
    """
    **DEPRECATED**: Use get_tool_registry() from centralized_tool_registry instead
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ DEPRECATED: get_tool_registry_lazy() is deprecated")
    return get_tool_registry()