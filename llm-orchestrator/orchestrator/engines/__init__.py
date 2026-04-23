"""
Engines package - execution entry points for the tiered skill architecture.

Exports:
- UnifiedDispatcher: route discovery -> CustomAgentRunner dispatch

Lazy-import unified_dispatch so submodules (e.g. deep_agent_executor) can be imported in
lightweight test / tooling contexts without generated protos on PYTHONPATH.
"""

from typing import Any

__all__ = [
    "UnifiedDispatcher",
    "get_unified_dispatcher",
]


def __getattr__(name: str) -> Any:
    if name == "UnifiedDispatcher":
        from orchestrator.engines.unified_dispatch import UnifiedDispatcher

        return UnifiedDispatcher
    if name == "get_unified_dispatcher":
        from orchestrator.engines.unified_dispatch import get_unified_dispatcher

        return get_unified_dispatcher
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
