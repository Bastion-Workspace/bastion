"""
Proposal Generation Subgraphs

Reusable LangGraph subgraphs for proposal generation workflows:
- Requirement Analyzer: Parse RFI/RFP and extract structured requirements
- Section Generator: Generate proposal sections using requirements + company knowledge + style
- Compliance Validator: Validate proposal against RFI/RFP requirements
- Editor Operations: Generate targeted edit operations for existing proposals
"""

from .proposal_requirement_analyzer_subgraph import build_proposal_requirement_analyzer_subgraph
from .proposal_section_generator_subgraph import build_proposal_section_generator_subgraph
from .proposal_compliance_validator_subgraph import build_proposal_compliance_validator_subgraph
from .proposal_editor_operations_subgraph import build_proposal_editor_operations_subgraph

__all__ = [
    "build_proposal_requirement_analyzer_subgraph",
    "build_proposal_section_generator_subgraph",
    "build_proposal_compliance_validator_subgraph",
    "build_proposal_editor_operations_subgraph",
]
