"""
Fiction Agent Subgraphs

Reusable subgraphs for fiction editing workflows:
- Context Preparation: Chapter detection, reference loading, scope analysis
- Validation: Continuity tracking, outline sync, consistency checks
- Generation: Context assembly, prompt building, LLM calls, output validation
- Resolution: Operation resolution with progressive search, validation, finalization
"""

from .fiction_context_subgraph import build_context_preparation_subgraph
from .fiction_validation_subgraph import build_validation_subgraph
from .fiction_generation_subgraph import build_generation_subgraph
from .fiction_resolution_subgraph import build_resolution_subgraph

__all__ = [
    "build_context_preparation_subgraph",
    "build_validation_subgraph",
    "build_generation_subgraph",
    "build_resolution_subgraph"
]

