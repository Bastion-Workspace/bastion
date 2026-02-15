"""
Skill definitions - declarative skill definitions per engine.

Load all definition modules and register skills with the global registry.
"""

import logging
from typing import List

from orchestrator.skills.skill_registry import get_skill_registry
from orchestrator.skills.skill_schema import Skill

from .automation_skills import AUTOMATION_SKILLS
from .conversational_skills import CONVERSATIONAL_SKILLS
from .editor_skills import EDITOR_SKILLS
from .research_skills import RESEARCH_SKILLS

logger = logging.getLogger(__name__)

_skills_loaded = False
_loaded_skills_list: List[Skill] = []


def load_all_skills() -> List[Skill]:
    """Load all skill definitions into the registry. Idempotent; safe to call multiple times."""
    global _skills_loaded, _loaded_skills_list
    if _skills_loaded:
        return _loaded_skills_list
    registry = get_skill_registry()
    all_skills: List[Skill] = []
    all_skills.extend(AUTOMATION_SKILLS)
    all_skills.extend(CONVERSATIONAL_SKILLS)
    all_skills.extend(EDITOR_SKILLS)
    all_skills.extend(RESEARCH_SKILLS)
    for s in all_skills:
        registry.register(s)
    _skills_loaded = True
    _loaded_skills_list = all_skills
    logger.info("Loaded %d skills into registry", len(all_skills))
    return all_skills


__all__ = [
    "load_all_skills",
    "AUTOMATION_SKILLS",
    "CONVERSATIONAL_SKILLS",
    "EDITOR_SKILLS",
    "RESEARCH_SKILLS",
]
