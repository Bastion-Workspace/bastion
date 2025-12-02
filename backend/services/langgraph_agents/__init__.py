"""
LangGraph Agents - Roosevelt's "Clean Cavalry" Agents
Minimal agent set for bulletproof functionality
"""

# Core agents only - ROOSEVELT'S CLEAN CAVALRY
# ChatAgent removed - migrated to llm-orchestrator gRPC service
from .weather_agent import WeatherAgent
from .data_formatting_agent import DataFormattingAgent
from .rss_agent import RSSAgent

# Base agent for future expansion
from .base_agent import BaseAgent

__all__ = [
    "BaseAgent",
    # ChatAgent removed - migrated to llm-orchestrator gRPC service
    "WeatherAgent",
    "DataFormattingAgent", 
    "RSSAgent"
]