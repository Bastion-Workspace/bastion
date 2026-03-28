"""
Engines package - execution engines for the tiered skill architecture.

Engines are generic workflows that execute skills. Exports:
- ResearchEngine: (Phase 5 - evolved from FullResearchAgent)
- ConversationalEngine: (Phase 6 - evolved from ChatAgent)
- UnifiedDispatcher: (Phase 3 - replaces grpc_service if/elif chain)
"""

from orchestrator.engines.conversational_engine import ConversationalEngine
from orchestrator.engines.research_engine import ResearchEngine
from orchestrator.engines.unified_dispatch import UnifiedDispatcher, get_unified_dispatcher

__all__ = [
    "ConversationalEngine",
    "ResearchEngine",
    "UnifiedDispatcher",
    "get_unified_dispatcher",
]
