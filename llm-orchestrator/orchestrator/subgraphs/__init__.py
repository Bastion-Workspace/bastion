"""
Fiction Agent Subgraphs

Reusable subgraphs for fiction editing workflows:
- Context Preparation: Chapter detection, reference loading, scope analysis
- Validation: Outline sync, consistency checks
- Generation: Context assembly, prompt building, LLM calls, output validation
- Resolution: Operation resolution with progressive search, validation, finalization
"""

from .fiction_validation_subgraph import build_validation_subgraph
from .fiction_generation_subgraph import build_generation_subgraph
from .fiction_book_generation_subgraph import build_book_generation_subgraph
from .intelligent_document_retrieval_subgraph import (
    build_intelligent_document_retrieval_subgraph,
    retrieve_documents_intelligently
)
from .research_workflow_subgraph import build_research_workflow_subgraph
from .collection_search_subgraph import execute_collection_search
from .fact_verification_subgraph import build_fact_verification_subgraph
from .knowledge_document_subgraph import build_knowledge_document_subgraph
from .gap_analysis_subgraph import build_gap_analysis_subgraph
from .web_research_subgraph import build_web_research_subgraph
from .assessment_subgraph import build_assessment_subgraph
from .full_document_analysis_subgraph import build_full_document_analysis_subgraph
from .entity_relationship_subgraph import build_entity_relationship_subgraph
from .data_formatting_subgraph import build_data_formatting_subgraph
from .visualization_subgraph import build_visualization_subgraph
from .diagramming_subgraph import build_diagramming_subgraph
from .proofreading_subgraph import build_proofreading_subgraph
from .image_search_subgraph import (
    build_image_search_subgraph,
    search_images_intelligently
)
from .style_editing_subgraph import build_style_editing_subgraph
from .character_development_subgraph import build_character_development_subgraph
from .outline_editing_subgraph import build_outline_editing_subgraph
from .nonfiction_outline_subgraph import build_nonfiction_outline_subgraph
from .rules_editing_subgraph import build_rules_editing_subgraph
from .article_writing_subgraph import build_article_writing_subgraph
from .podcast_script_subgraph import build_podcast_script_subgraph
from .location_management_subgraph import build_location_management_subgraph
from .route_planning_subgraph import build_route_planning_subgraph
from .attachment_analysis_subgraph import build_attachment_analysis_subgraph

__all__ = [
    "build_validation_subgraph",
    "build_generation_subgraph",
    "build_book_generation_subgraph",
    "build_fiction_editing_subgraph",
    "build_intelligent_document_retrieval_subgraph",
    "retrieve_documents_intelligently",
    "build_research_workflow_subgraph",
    "execute_collection_search",
    "execute_factual_query",
    "build_fact_verification_subgraph",
    "build_knowledge_document_subgraph",
    "build_gap_analysis_subgraph",
    "build_web_research_subgraph",
    "build_assessment_subgraph",
    "build_full_document_analysis_subgraph",
    "build_entity_relationship_subgraph",
    "build_data_formatting_subgraph",
    "build_visualization_subgraph",
    "build_diagramming_subgraph",
    "build_proofreading_subgraph",
    "build_image_search_subgraph",
    "search_images_intelligently",
    "build_style_editing_subgraph",
    "build_character_development_subgraph",
    "build_outline_editing_subgraph",
    "build_nonfiction_outline_subgraph",
    "build_rules_editing_subgraph",
    "build_article_writing_subgraph",
    "build_podcast_script_subgraph",
    "build_location_management_subgraph",
    "build_route_planning_subgraph",
    "build_attachment_analysis_subgraph",
    "build_proposal_requirement_analyzer_subgraph",
    "build_proposal_section_generator_subgraph",
    "build_proposal_compliance_validator_subgraph",
    "build_proposal_editor_operations_subgraph",
]

