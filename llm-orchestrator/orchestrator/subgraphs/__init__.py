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
from .fiction_book_generation_subgraph import build_book_generation_subgraph
from .intelligent_document_retrieval_subgraph import (
    build_intelligent_document_retrieval_subgraph,
    retrieve_documents_intelligently
)
from .research_workflow_subgraph import build_research_workflow_subgraph
from .fact_verification_subgraph import build_fact_verification_subgraph
from .knowledge_document_subgraph import build_knowledge_document_subgraph
from .gap_analysis_subgraph import build_gap_analysis_subgraph
from .web_research_subgraph import build_web_research_subgraph
from .assessment_subgraph import build_assessment_subgraph
from .full_document_analysis_subgraph import build_full_document_analysis_subgraph

__all__ = [
    "build_context_preparation_subgraph",
    "build_validation_subgraph",
    "build_generation_subgraph",
    "build_resolution_subgraph",
    "build_book_generation_subgraph",
    "build_intelligent_document_retrieval_subgraph",
    "retrieve_documents_intelligently",
    "build_research_workflow_subgraph",
    "build_fact_verification_subgraph",
    "build_knowledge_document_subgraph",
    "build_gap_analysis_subgraph",
    "build_web_research_subgraph",
    "build_assessment_subgraph",
    "build_full_document_analysis_subgraph"
]

