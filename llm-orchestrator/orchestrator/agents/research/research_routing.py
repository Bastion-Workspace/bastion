"""
Routing functions for the research workflow conditional edges.
Each function takes state and returns the edge key (str).
"""

import logging
from typing import Any, Dict

from orchestrator.agents.research.research_state import ResearchState

logger = logging.getLogger(__name__)


def route_from_research_subgraph(state: ResearchState) -> str:
    """Route after research subgraph completes"""
    if state.get("cache_hit") and state.get("cached_context"):
        logger.info("Research subgraph: Cache hit - using cached research")
        return "use_cache"

    if state.get("round1_sufficient"):
        logger.info("Research subgraph: Research sufficient - checking if full docs needed")
        return "sufficient"

    gap_analysis = state.get("gap_analysis", {})
    needs_local = gap_analysis.get("needs_local_search", False)
    needs_web = gap_analysis.get("needs_web_search", False)

    if needs_local or needs_web:
        logger.info(
            "Research subgraph: Needs Round 2 (local=%s, web=%s) - checking if full docs needed",
            needs_local, needs_web,
        )
        return "needs_round2"

    logger.info("Research subgraph: No additional searches needed - checking if full docs needed")
    return "sufficient"


def route_from_full_doc_decision(state: ResearchState) -> str:
    """Route after full document analysis decision"""
    if state.get("full_doc_analysis_needed"):
        logger.info("Full doc decision: Will analyze full documents")
        return "analyze_full_docs"
    logger.info("Full doc decision: Skipping full document analysis")
    return "skip_full_docs"


def route_from_gap_analysis_check(state: ResearchState) -> str:
    """Route after gap analysis check (with or without full doc insights). Skill config can disable round2."""
    skill_config = state.get("skill_config", {})
    if not skill_config.get("round2", True) or not skill_config.get("gap_analysis", True):
        logger.info("Gap analysis check: Skill config has round2=False or gap_analysis=False - proceeding to synthesis")
        return "proceed_to_synthesis"

    gap_analysis = state.get("gap_analysis", {})
    needs_local = gap_analysis.get("needs_local_search", False)
    needs_web = gap_analysis.get("needs_web_search", False)

    if needs_local or needs_web:
        logger.info("Gap analysis check: Needs Round 2 (local=%s, web=%s)", needs_local, needs_web)
        return "needs_round2"

    logger.info("Gap analysis check: No additional searches needed - proceeding to synthesis")
    return "proceed_to_synthesis"


def route_from_quick_answer(state: ResearchState) -> str:
    """Route from quick answer check: provide quick answer, process attachments, or proceed to full research"""
    if state.get("quick_answer_provided") and state.get("quick_answer_content"):
        logger.info("Quick answer provided - short-circuiting to response")
        return "quick_answer"

    shared_memory = state.get("shared_memory", {})
    attached_images = shared_memory.get("attached_images", [])
    if attached_images:
        logger.info("Found %d attached image(s) - processing for face identification", len(attached_images))
        return "process_attachments"

    logger.info("Proceeding to full research workflow")
    return "full_research"


def route_from_synthesis(state: ResearchState) -> str:
    """Route from synthesis: check if post-processing is needed"""
    formatting_recommendations = state.get("formatting_recommendations")
    if formatting_recommendations and (
        formatting_recommendations.get("table_recommended")
        or formatting_recommendations.get("chart_recommended")
        or formatting_recommendations.get("timeline_recommended")
    ):
        logger.info("Routing to post-processing (formatting and/or visualization)")
        return "post_process"
    return "complete"
