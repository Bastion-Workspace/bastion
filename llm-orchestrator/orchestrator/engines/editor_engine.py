"""
Editor Engine - Unified editor for all document types (outline, character, rules, style, fiction, article, etc.).

Evolved from WritingAssistantAgent. Routes to appropriate subgraphs based on
active_editor.frontmatter.type or skill_name when provided.
When skill_name is provided (e.g. from skill discovery), it can override or guide routing.
"""

import logging
from typing import Any, Dict, List, Optional

from orchestrator.agents.writing_assistant_agent import WritingAssistantAgent

logger = logging.getLogger(__name__)


class EditorEngine(WritingAssistantAgent):
    """
    Editor engine: same workflow as WritingAssistantAgent, with skill_name support for dispatch.
    Subgraphs are still loaded and routed by active_editor.frontmatter.type; skill_name
    is passed in metadata for continuity and future skill-based routing.
    """

    def __init__(self) -> None:
        super().__init__()
        self.agent_type = "editor_engine"
        logger.info("Editor engine initialized")

    async def process(
        self,
        query: str,
        metadata: Optional[Dict[str, Any]] = None,
        messages: Optional[List[Any]] = None,
        skill_name: Optional[str] = None,
        cancellation_token: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Process editor request. Routes by active_editor type; skill_name stored in metadata for continuity."""
        meta = metadata or {}
        if skill_name:
            meta = {**meta, "skill_name": skill_name}
        return await super().process(query=query, metadata=meta, messages=messages)
