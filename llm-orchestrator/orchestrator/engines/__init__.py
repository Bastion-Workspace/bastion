"""
Engines package - execution entry points for the tiered skill architecture.

Exports:
- UnifiedDispatcher: route discovery -> CustomAgentRunner dispatch
"""

from orchestrator.engines.unified_dispatch import UnifiedDispatcher, get_unified_dispatcher

__all__ = [
    "UnifiedDispatcher",
    "get_unified_dispatcher",
]
