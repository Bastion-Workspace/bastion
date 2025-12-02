"""
Pydantic models for structured general project agent outputs
Type-safe models for file routing and content structure extraction
"""

from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any


class GeneralProjectFileRouteItem(BaseModel):
    """Individual file routing item for general project agent content routing"""
    content_type: Literal["requirements", "design", "specifications", "tasks", "notes", "general"] = Field(
        description="Type of content to route"
    )
    target_file: str = Field(
        description="Target filename or 'project_plan' for project plan document"
    )
    target_document_id: Optional[str] = Field(
        default=None,
        description="Document ID of target file (if known)"
    )
    section: str = Field(
        description="Section name within the target file (match existing sections if updating similar content)"
    )
    content: str = Field(
        description="The formatted content to save to the target file section"
    )
    action: Literal["append", "replace", "remove"] = Field(
        default="append",
        description="Action to take: append (add to section), replace (update section), remove (delete section)"
    )
    create_new_file: bool = Field(
        default=False,
        description="Whether to create a new reference file (true if content is substantial and doesn't fit existing files)"
    )
    suggested_filename: Optional[str] = Field(
        default=None,
        description="Suggested filename for new file (required if create_new_file is true, e.g., 'hvac-specifications.md')"
    )
    file_summary: Optional[str] = Field(
        default=None,
        description="Brief description of the new file's purpose (required if create_new_file is true)"
    )


class GeneralProjectUnifiedContentPlan(BaseModel):
    """
    Unified model combining content extraction and routing in single LLM call.
    This eliminates the need for separate extraction and routing steps.
    """
    routing: List[GeneralProjectFileRouteItem] = Field(
        default_factory=list,
        description="List of routing decisions - each item includes both content and where to save it"
    )


class GeneralProjectDecision(BaseModel):
    """Structured project decision for general projects"""
    decision_id: str = Field(
        description="Unique identifier for this decision"
    )
    timestamp: str = Field(
        description="ISO timestamp when decision was made"
    )
    decision_type: Literal["requirement", "design_choice", "specification", "approach", "tradeoff", "other"] = Field(
        description="Type of decision made"
    )
    decision_summary: str = Field(
        description="Brief summary of the decision"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed information about the decision"
    )
    replaced_item: Optional[str] = Field(
        default=None,
        description="Item/approach being replaced (if any)"
    )
    reason: Optional[str] = Field(
        default=None,
        description="Reason for the decision if mentioned"
    )
    alternatives_considered: List[str] = Field(
        default_factory=list,
        description="List of alternatives that were considered"
    )
    documented_in: List[str] = Field(
        default_factory=list,
        description="List of files where this decision should be documented"
    )
    supersedes: List[str] = Field(
        default_factory=list,
        description="List of decision IDs that this decision supersedes"
    )


class GeneralProjectDocumentationInconsistency(BaseModel):
    """Documentation inconsistency for general projects"""
    file: str = Field(
        description="File path where inconsistency exists"
    )
    section: str = Field(
        description="Section name where inconsistency exists"
    )
    issue_type: Literal["mismatch", "outdated", "missing", "contradiction", "missing_documentation"] = Field(
        description="Type of inconsistency"
    )
    description: str = Field(
        description="Description of the inconsistency"
    )
    severity: Literal["low", "medium", "high"] = Field(
        description="Severity of the inconsistency"
    )
    documented_value: Optional[str] = Field(
        default=None,
        description="Value currently documented"
    )
    actual_value: Optional[str] = Field(
        default=None,
        description="Actual/correct value"
    )
    related_decision_id: Optional[str] = Field(
        default=None,
        description="Decision ID related to this inconsistency"
    )
    suggested_fix: str = Field(
        description="Suggested fix for the inconsistency"
    )


class GeneralProjectDocumentationVerificationResult(BaseModel):
    """Documentation verification result for general projects"""
    verification_status: Literal["consistent", "inconsistent", "needs_review"] = Field(
        description="Overall verification status"
    )
    inconsistencies: List[GeneralProjectDocumentationInconsistency] = Field(
        default_factory=list,
        description="List of inconsistencies found"
    )
    missing_documentation: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of missing documentation items (values may be strings or lists)"
    )
    outdated_sections: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of outdated sections (values may be strings or lists)"
    )
    completeness_score: float = Field(
        description="Completeness score (0.0-1.0)"
    )
    consistency_score: float = Field(
        description="Consistency score (0.0-1.0)"
    )
    reasoning: str = Field(
        default="",
        description="Explanation of verification results"
    )


class GeneralProjectDocumentationMaintenanceItem(BaseModel):
    """Documentation maintenance item for general projects"""
    file: str = Field(
        description="File path to maintain"
    )
    section: str = Field(
        description="Section name to maintain"
    )
    action: Literal["update", "remove", "archive"] = Field(
        description="Action to take"
    )
    reason: str = Field(
        description="Reason for maintenance"
    )
    suggested_content: Optional[str] = Field(
        default=None,
        description="Suggested content for update (if action is update)"
    )


class GeneralProjectDocumentationMaintenancePlan(BaseModel):
    """Documentation maintenance plan for general projects"""
    maintenance_items: List[GeneralProjectDocumentationMaintenanceItem] = Field(
        default_factory=list,
        description="List of maintenance operations to perform"
    )
    priority: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Priority of maintenance operations"
    )
    reasoning: str = Field(
        default="",
        description="Explanation of maintenance plan"
    )


