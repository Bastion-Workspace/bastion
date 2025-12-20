"""
Orchestrator Tools - LangGraph tools using backend gRPC services
"""

from orchestrator.tools.document_tools import (
    search_documents_tool,
    search_documents_structured,
    get_document_content_tool,
    search_within_document_tool,
    DOCUMENT_TOOLS
)

from orchestrator.tools.web_tools import (
    search_web_tool,
    search_web_structured,
    crawl_web_content_tool,
    WEB_TOOLS
)

from orchestrator.tools.enhancement_tools import (
    expand_query_tool,
    search_conversation_cache_tool,
    ENHANCEMENT_TOOLS
)

from orchestrator.tools.segment_search_tools import (
    search_segments_across_documents_tool,
    extract_relevant_content_section,
    SEGMENT_SEARCH_TOOLS
)

from orchestrator.tools.information_analysis_tools import (
    analyze_information_needs_tool,
    generate_project_aware_queries_tool,
    INFORMATION_ANALYSIS_TOOLS
)

from orchestrator.tools.math_tools import (
    calculate_expression_tool,
    list_available_formulas_tool,
    MATH_TOOLS
)

from orchestrator.tools.math_formulas import (
    evaluate_formula_tool
)

from orchestrator.tools.unit_conversions import (
    convert_units_tool
)

__all__ = [
    # Document tools
    'search_documents_tool',
    'search_documents_structured',
    'get_document_content_tool',
    'search_within_document_tool',
    'DOCUMENT_TOOLS',
    # Web tools
    'search_web_tool',
    'search_web_structured',
    'crawl_web_content_tool',
    'WEB_TOOLS',
    # Enhancement tools
    'expand_query_tool',
    'search_conversation_cache_tool',
    'ENHANCEMENT_TOOLS',
    # Segment search tools
    'search_segments_across_documents_tool',
    'extract_relevant_content_section',
    'SEGMENT_SEARCH_TOOLS',
    # Information analysis tools
    'analyze_information_needs_tool',
    'generate_project_aware_queries_tool',
    'INFORMATION_ANALYSIS_TOOLS',
    # Math tools
    'calculate_expression_tool',
    'evaluate_formula_tool',
    'convert_units_tool',
    'list_available_formulas_tool',
    'MATH_TOOLS'
]

