"""
Conversational Engine - Wrapper for ChatAgent with skill_name support.

Evolved from ChatAgent. Accepts skill_name for dispatch; passes it in metadata.
Handles general chat, org_content, and story_analysis skills.
"""

import logging
from typing import Any, Dict, List, Optional

from orchestrator.agents.chat_agent import ChatAgent

logger = logging.getLogger(__name__)


class ConversationalEngine:
    """
    Conversational engine: delegates to ChatAgent with skill_name in metadata.
    Keeps the same workflow; skill_name is used for continuity.
    """

    def __init__(self) -> None:
        self._agent = None
        logger.info("Conversational engine initialized")

    def _get_agent(self):
        if self._agent is None:
            self._agent = ChatAgent()
        return self._agent

    async def process(
        self,
        query: str,
        metadata: Optional[Dict[str, Any]] = None,
        messages: Optional[List[Any]] = None,
        skill_name: Optional[str] = None,
        cancellation_token: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Process conversational request. skill_name stored in metadata for continuity."""
        meta = metadata or {}
        if skill_name:
            meta = {**meta, "skill_name": skill_name}
        agent = self._get_agent()
        return await agent.process(query=query, metadata=meta, messages=messages)
