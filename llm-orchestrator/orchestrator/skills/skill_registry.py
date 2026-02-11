"""
Skill Registry - Central registry and eligibility filter for LLM-primary routing.

Filter layer answers "which skills CAN handle this?" (hard gates only).
LLM selection answers "which skill SHOULD handle this?" (intent).
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from orchestrator.skills.skill_schema import EngineType, Skill

logger = logging.getLogger(__name__)

# Short greetings: instant route to chat, skip LLM
GREETING_WORDS = frozenset([
    "hi", "hello", "hey", "howdy", "hiya", "yo", "greetings", "good morning",
    "good afternoon", "good evening", "good day", "how are you", "what's up",
    "sup", "hi there", "hello there",
])


@dataclass
class ScoredSkill:
    """Skill with optional score/reason; kept for compatibility with callers that expect it."""

    skill: Skill
    score: float = 0.0
    reason: str = ""


def _is_greeting(query: str) -> bool:
    """True if query is a short greeting so chat can be chosen instantly."""
    q = query.lower().strip()
    if not q or len(q) > 60:
        return False
    words = q.split()
    if len(words) > 3:
        return False
    return q in GREETING_WORDS or (words and words[0] in GREETING_WORDS)


def _editor_type_from_context(editor_context: Optional[Dict[str, Any]]) -> Optional[str]:
    """Extract normalized editor type from context."""
    if not editor_context:
        return None
    editor_type = (editor_context.get("type") or "").strip().lower()
    if not editor_type:
        fn = (editor_context.get("filename") or "").lower()
        lang = (editor_context.get("language") or "").lower()
        if fn.endswith(".org") or lang == "org":
            return "org"
    return editor_type or None


class SkillRegistry:
    """Singleton registry of skills with eligibility filtering (no scoring)."""

    _instance: Optional["SkillRegistry"] = None

    def __init__(self) -> None:
        self._skills: Dict[str, Skill] = {}

    @classmethod
    def get_instance(cls) -> "SkillRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, skill: Skill) -> None:
        """Register a skill by name."""
        self._skills[skill.name] = skill
        logger.debug("Registered skill: %s (engine=%s)", skill.name, skill.engine)

    def register_many(self, skills: List[Skill]) -> None:
        for s in skills:
            self.register(s)

    def get(self, name: str) -> Optional[Skill]:
        """Get skill by name. Accepts both agent-style (e.g. weather_agent) and short (weather)."""
        if name in self._skills:
            return self._skills[name]
        if name.endswith("_agent") and name[:-6] in self._skills:
            return self._skills[name[:-6]]
        return self._skills.get(name)

    def get_for_engine(self, engine: EngineType) -> List[Skill]:
        """Return all skills for an engine."""
        return [s for s in self._skills.values() if s.engine == engine]

    def filter_eligible(
        self,
        query: str,
        editor_context: Optional[Dict[str, Any]] = None,
        conversation_context: Optional[Dict[str, Any]] = None,
        editor_preference: str = "prefer",
        has_image_context: bool = False,
    ) -> Tuple[List[Skill], Optional[str]]:
        """
        Return (eligible_skills, instant_route).

        instant_route is set for greetings ("chat") so the caller skips LLM.
        Otherwise instant_route is None and the caller runs LLM selection.

        Hard gates only: internal_only, requires_editor, editor_type mismatch,
        requires_image_context.
        """
        if not self._skills:
            return [], None

        # Greeting -> instant route to chat
        if _is_greeting(query):
            chat_skill = self.get("chat_agent") or self.get("chat")
            if chat_skill:
                logger.info("Skill filter: greeting -> chat (instant)")
                return [chat_skill], "chat"
            return [], None

        editor_type = _editor_type_from_context(editor_context)
        eligible: List[Skill] = []

        # Document types that have a dedicated editor skill (fiction_editing, outline_editing, etc.).
        # When user has such a document open, "generate" means "generate in this document", not "create new file".
        editor_document_types = frozenset(
            et
            for s in self._skills.values()
            if s.editor_types and s.engine == EngineType.EDITOR
            for et in s.editor_types
        )

        for skill in self._skills.values():
            if skill.internal_only:
                continue
            if skill.requires_image_context and not has_image_context:
                continue
            if skill.requires_editor and not editor_context:
                continue
            if skill.editor_types and editor_context is not None:
                if editor_type not in skill.editor_types:
                    continue
            # document_creator creates new files/folders; when an editor doc is open, prefer editor skills.
            if skill.name == "document_creator" and editor_context and editor_type in editor_document_types:
                continue
            eligible.append(skill)

        return eligible, None


def get_skill_registry() -> SkillRegistry:
    """Return the singleton skill registry (caller must load definitions)."""
    return SkillRegistry.get_instance()
