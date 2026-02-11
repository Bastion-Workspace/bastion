"""
Skills package - declarative skill definitions and registry for LLM-primary routing.

Usage:
    from orchestrator.skills import get_skill_registry, load_all_skills
    from orchestrator.skills.skill_schema import Skill, EngineType
    from orchestrator.skills.skill_llm_selector import llm_select_skill

    load_all_skills()
    registry = get_skill_registry()
    eligible, instant_route = registry.filter_eligible(query, editor_context, conversation_context)
    if instant_route:
        skill_name = instant_route
    elif eligible:
        skill_name = await llm_select_skill(eligible, query, editor_context, conversation_context)
    else:
        skill_name = "chat"
    skill = registry.get(skill_name)
"""

from orchestrator.skills.skill_registry import (
    ScoredSkill,
    get_skill_registry,
)
from orchestrator.skills.skill_schema import (
    EngineType,
    Skill,
)
from orchestrator.skills.definitions import load_all_skills

__all__ = [
    "EngineType",
    "Skill",
    "ScoredSkill",
    "get_skill_registry",
    "load_all_skills",
]
