"""
Proposal Generation Pydantic Models

Structured data models for proposal generation workflows, providing type-safe
validation and serialization of proposal-related data.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class ProposalRequirement(BaseModel):
    """Represents an extracted requirement from RFI/RFP document"""
    requirement_id: str = Field(..., description="Unique identifier for requirement")
    question: str = Field(..., description="Full text of the requirement/question")
    section_name: Optional[str] = Field(
        None, 
        description="Proposal section that addresses this requirement"
    )
    mandatory: bool = Field(
        True,
        description="Whether this requirement is mandatory or optional"
    )
    compliance_critical: bool = Field(
        False,
        description="Whether this requirement is critical for compliance evaluation"
    )
    category: Optional[str] = Field(
        None,
        description="Category (e.g., 'technical', 'commercial', 'legal', 'operational')"
    )
    priority: int = Field(1, description="Priority level (1=highest)")
    notes: Optional[str] = Field(None, description="Additional context about requirement")


class RequirementAnalysisResult(BaseModel):
    """Result of RFI/RFP document analysis"""
    requirements: List[ProposalRequirement] = Field(
        ...,
        description="List of extracted requirements"
    )
    requirement_index: Dict[str, str] = Field(
        ...,
        description="Mapping of requirement_id to proposed section_name"
    )
    total_requirements: int = Field(..., description="Total count of requirements")
    mandatory_count: int = Field(..., description="Count of mandatory requirements")
    categories_found: List[str] = Field(
        default_factory=list,
        description="Unique categories detected in requirements"
    )
    analysis_summary: str = Field(
        ...,
        description="Human-readable summary of RFI/RFP structure"
    )
    parsing_confidence: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1) of RFI/RFP parsing"
    )


class ProposalSection(BaseModel):
    """Generated proposal section with metadata"""
    section_name: str = Field(..., description="Name of the proposal section")
    content: str = Field(..., description="Generated section content")
    word_count: int = Field(..., description="Number of words in section")
    requirement_ids: List[str] = Field(
        default_factory=list,
        description="IDs of requirements this section addresses"
    )
    coverage_percentage: float = Field(
        100.0,
        ge=0.0,
        le=100.0,
        description="Estimated % of requirements covered"
    )
    quality_score: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Quality assessment (0-1)"
    )
    style_applied: bool = Field(
        False,
        description="Whether style guide has been applied"
    )


class ComplianceMatrixEntry(BaseModel):
    """Entry in compliance matrix mapping requirement to section"""
    requirement_id: str = Field(..., description="ID of requirement")
    requirement_text: str = Field(..., description="Text of requirement")
    addressed_by_section: Optional[str] = Field(
        None,
        description="Section name that addresses requirement"
    )
    coverage_percentage: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="Estimated coverage percentage (0-100)"
    )
    status: str = Field(
        default="pending",
        description="Status: 'addressed', 'partial', 'pending', 'missing'"
    )
    notes: Optional[str] = Field(None, description="Additional notes on coverage")


class ProposalValidationResult(BaseModel):
    """Result of proposal validation against RFI/RFP"""
    compliance_matrix: List[ComplianceMatrixEntry] = Field(
        ...,
        description="Detailed matrix of requirements vs. proposal sections"
    )
    addressed_requirements: List[str] = Field(
        default_factory=list,
        description="IDs of requirements that are addressed"
    )
    missing_requirements: List[str] = Field(
        default_factory=list,
        description="IDs of requirements that are missing"
    )
    partial_requirements: List[str] = Field(
        default_factory=list,
        description="IDs of requirements only partially addressed"
    )
    completeness_score: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="Overall proposal completeness (0-100%)"
    )
    compliance_status: str = Field(
        default="incomplete",
        description="Status: 'complete', 'mostly_complete', 'incomplete'"
    )
    validation_summary: str = Field(
        ...,
        description="Human-readable validation summary"
    )
    critical_gaps: List[str] = Field(
        default_factory=list,
        description="Critical requirements that are missing"
    )
    timestamp: str = Field(..., description="ISO format timestamp of validation")


class ProposalGenerationRequest(BaseModel):
    """Request parameters for proposal generation"""
    customer_name: str = Field(..., description="Name of customer/prospect")
    proposal_type: str = Field(
        default="commercial_services",
        description="Type of proposal (commercial_services, technical, hybrid)"
    )
    pr_req_document_id: Optional[str] = Field(
        None,
        description="Document ID of RFI/RFP requirements"
    )
    pr_style_document_id: Optional[str] = Field(
        None,
        description="Document ID of style template"
    )
    company_knowledge_id: Optional[str] = Field(
        None,
        description="Document ID with company products/services"
    )
    target_length_words: int = Field(
        default=8000,
        description="Target proposal length in words"
    )
    sections: Optional[List[str]] = Field(
        None,
        description="Specific sections to generate (if None, auto-detect from RFI/RFP)"
    )


class ProposalGenerationResponse(BaseModel):
    """Response from proposal generation"""
    task_status: str = Field(..., description="Status: complete, partial, error")
    proposal_content: Optional[str] = Field(None, description="Generated proposal markdown")
    validation_result: Optional[ProposalValidationResult] = Field(
        None,
        description="Proposal validation against RFI/RFP"
    )
    sections_generated: List[ProposalSection] = Field(
        default_factory=list,
        description="Individual sections that were generated"
    )
    word_count: int = Field(0, description="Total word count of generated proposal")
    completeness_score: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="Overall proposal completeness score"
    )
    missing_sections: List[str] = Field(
        default_factory=list,
        description="Sections not yet generated"
    )
    errors: Optional[List[str]] = Field(None, description="Any errors that occurred")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about proposal generation"
    )
