"""
Research Engine - Wrapper for FullResearchAgent with skill_name support.

Evolved from FullResearchAgent. Accepts skill_name for dispatch; passes it in metadata.
Future: skill_config (depth, web search on/off, output format) can be read from skill definition.
"""

import logging
from typing import Any, Dict, List, Optional

from orchestrator.agents.full_research_agent import get_full_research_agent

logger = logging.getLogger(__name__)


class ResearchEngine:
    """
    Research engine: delegates to FullResearchAgent with skill_name in metadata.
    Keeps the same workflow; skill_name is used for continuity and future skill_config.
    """

    def __init__(self) -> None:
        self._agent = None
        logger.info("Research engine initialized")

    def _get_agent(self):
        if self._agent is None:
            self._agent = get_full_research_agent()
        return self._agent

    async def process(
        self,
        query: str,
        metadata: Optional[Dict[str, Any]] = None,
        messages: Optional[List[Any]] = None,
        skill_name: Optional[str] = None,
        cancellation_token: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Process research request. skill_name stored in metadata for continuity and future skill_config."""
        meta = metadata or {}
        if skill_name:
            meta = {**meta, "skill_name": skill_name}
        agent = self._get_agent()
        return await agent.process(query=query, metadata=meta, messages=messages)
