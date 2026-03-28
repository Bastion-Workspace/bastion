"""
LangGraph Agents - minimal backend agent surface.

Import specific modules directly when needed (e.g. BaseAgent).
RSS polling runs in services.rss_polling_service (Celery), not here.
"""

from services.langgraph_agents.base_agent import BaseAgent

__all__ = [
    "BaseAgent",
]
