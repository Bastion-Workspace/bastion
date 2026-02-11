"""
Engines package - execution engines for the tiered skill architecture.

Engines are generic workflows that execute skills. Exports:
- AutomationEngine: tool-calling + system prompt + LLM for simple skills
- EditorEngine: (Phase 4 - evolved from WritingAssistantAgent)
- ResearchEngine: (Phase 5 - evolved from FullResearchAgent)
- ConversationalEngine: (Phase 6 - evolved from ChatAgent)
- PlanEngine: (Phase 7 - compound query execution)
- UnifiedDispatcher: (Phase 3 - replaces grpc_service if/elif chain)
"""

from orchestrator.engines.automation_engine import AutomationEngine
from orchestrator.engines.conversational_engine import ConversationalEngine
from orchestrator.engines.editor_engine import EditorEngine
from orchestrator.engines.plan_engine import PlanEngine
from orchestrator.engines.plan_models import ExecutionPlan, PlanStep
from orchestrator.engines.research_engine import ResearchEngine
from orchestrator.engines.unified_dispatch import UnifiedDispatcher, get_unified_dispatcher

__all__ = [
    "AutomationEngine",
    "ConversationalEngine",
    "EditorEngine",
    "ExecutionPlan",
    "PlanEngine",
    "PlanStep",
    "ResearchEngine",
    "UnifiedDispatcher",
    "get_unified_dispatcher",
]
