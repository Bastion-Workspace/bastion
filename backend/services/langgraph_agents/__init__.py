"""
LangGraph Agents - Roosevelt's "Clean Cavalry" Agents
Minimal agent set for bulletproof functionality
"""

# Core agents removed from top-level to avoid dependency bloat in Celery Beat/Flower
# Import these directly from their modules when needed:
# from services.langgraph_agents.rss_agent import RSSAgent
# from services.langgraph_agents.base_agent import BaseAgent

__all__ = [
    "BaseAgent",
    "RSSAgent"
]
