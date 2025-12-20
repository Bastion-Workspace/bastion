"""
LangGraph Agents - Roosevelt's "Clean Cavalry" Agents
Minimal agent set for bulletproof functionality
"""

# Core agents only - ROOSEVELT'S CLEAN CAVALRY
# ChatAgent removed - migrated to llm-orchestrator gRPC service
# WeatherAgent removed - migrated to llm-orchestrator gRPC service
# DataFormattingAgent removed - migrated to llm-orchestrator gRPC service
from .rss_agent import RSSAgent

# Base agent for future expansion
from .base_agent import BaseAgent

__all__ = [
    "BaseAgent",
    # ChatAgent removed - migrated to llm-orchestrator gRPC service
    # WeatherAgent removed - migrated to llm-orchestrator gRPC service
    # DataFormattingAgent removed - migrated to llm-orchestrator gRPC service
    "RSSAgent"
]