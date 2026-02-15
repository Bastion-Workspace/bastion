"""
Skill-level configuration for research engine (Phase 5).
Maps skill_name to config dict used by nodes to gate behavior.
"""

from typing import Any, Dict

# Default config used when skill is "research" or unknown (current behavior)
DEFAULT_RESEARCH_CONFIG: Dict[str, Any] = {
    "skip_quick_answer": False,
    "local_search": True,
    "web_search": True,
    "gap_analysis": True,
    "full_doc_analysis": True,
    "round2": True,
    "synthesis_style": "comprehensive",
}

SKILL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "research": {
        "skip_quick_answer": False,
        "local_search": True,
        "web_search": True,
        "gap_analysis": True,
        "full_doc_analysis": True,
        "round2": True,
        "synthesis_style": "comprehensive",
    },
    "knowledge_builder": {
        "skip_quick_answer": True,
        "local_search": True,
        "web_search": True,
        "gap_analysis": True,
        "full_doc_analysis": True,
        "round2": True,
        "synthesis_style": "verification",
    },
    "security_analysis": {
        "skip_quick_answer": True,
        "local_search": False,
        "web_search": True,
        "gap_analysis": False,
        "full_doc_analysis": False,
        "round2": False,
        "synthesis_style": "security_report",
    },
    "site_crawl": {
        "skip_quick_answer": True,
        "local_search": False,
        "web_search": True,
        "gap_analysis": False,
        "full_doc_analysis": False,
        "round2": False,
        "synthesis_style": "extraction",
    },
    "website_crawler": {
        "skip_quick_answer": True,
        "local_search": False,
        "web_search": True,
        "gap_analysis": False,
        "full_doc_analysis": False,
        "round2": False,
        "synthesis_style": "ingestion",
    },
}


def get_research_skill_config(skill_name: str) -> Dict[str, Any]:
    """Return skill config for the given skill name. Defaults to full research behavior."""
    config = SKILL_CONFIGS.get(skill_name, DEFAULT_RESEARCH_CONFIG)
    return config.copy()
