"""
Research agent state definitions: ResearchState TypedDict and ResearchRound enum.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict


class ResearchRound(str, Enum):
    """Research round tracking"""
    QUICK_ANSWER_CHECK = "quick_answer_check"
    CACHE_CHECK = "cache_check"
    INITIAL_LOCAL = "initial_local"
    ROUND_2_GAP_FILLING = "round_2_gap_filling"
    ROUND_2_PARALLEL = "round_2_parallel"
    WEB_ROUND_1 = "web_round_1"
    ASSESS_WEB_ROUND_1 = "assess_web_round_1"
    GAP_ANALYSIS_WEB = "gap_analysis_web"
    WEB_ROUND_2 = "web_round_2"
    FINAL_SYNTHESIS = "final_synthesis"


class ResearchState(TypedDict, total=False):
    """Complete state for sophisticated research workflow"""
    # Query info
    query: str
    original_query: str
    expanded_queries: List[str]
    key_entities: List[str]
    messages: List[Any]
    shared_memory: Dict[str, Any]
    quick_answer_provided: bool
    quick_answer_content: str
    skip_quick_answer: bool
    quick_vector_results: List[Dict[str, Any]]
    quick_vector_relevance: Optional[str]
    current_round: str
    cache_hit: bool
    cached_context: str
    round1_results: Dict[str, Any]
    round1_sufficient: bool
    gap_analysis: Dict[str, Any]
    identified_gaps: List[str]
    round2_results: Dict[str, Any]
    round2_sufficient: bool
    web_round1_results: Dict[str, Any]
    web_round1_sufficient: bool
    web_permission_granted: bool
    web_gap_analysis: Dict[str, Any]
    web_identified_gaps: List[str]
    web_round2_results: Dict[str, Any]
    web_search_results: Dict[str, Any]
    query_type: Optional[str]
    query_type_detection: Dict[str, Any]
    should_present_options: bool
    num_options: Optional[int]
    final_response: str
    citations: List[Dict[str, Any]]
    sources_used: List[str]
    routing_recommendation: Optional[str]
    formatting_recommendations: Optional[Dict[str, Any]]
    visualization_results: Optional[Dict[str, Any]]
    formatted_output: Optional[str]
    format_type: Optional[str]
    full_doc_analysis_needed: bool
    document_ids_to_analyze: List[str]
    analysis_queries: List[str]
    full_doc_insights: Dict[str, Any]
    documents_analyzed: List[str]
    full_doc_decision_reasoning: str
    has_attachments: bool
    attachments: List[Dict[str, Any]]
    attached_images: List[Dict[str, Any]]
    attachment_analysis: Optional[Dict[str, Any]]
    attachment_analysis_results: Optional[Dict[str, Any]]
    attachment_processed: bool
    research_tier: Optional[str]
    force_web_search: bool
    skip_sufficiency_check: bool
    research_complete: bool
    error: str
    metadata: Dict[str, Any]
    user_id: str
    research_findings: Dict[str, Any]
    round1_assessment: Dict[str, Any]
    sources_found: List[Dict[str, Any]]
    skill_config: Dict[str, Any]
