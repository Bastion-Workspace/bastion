"""
Re-export FullResearchAgent from research package for backward compatibility.
"""

from orchestrator.agents.research import FullResearchAgent, get_full_research_agent

__all__ = ["FullResearchAgent", "get_full_research_agent"]
