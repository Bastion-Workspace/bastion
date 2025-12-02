"""
Pydantic models for structured electronics agent outputs
Type-safe models for file routing and content structure extraction
"""

from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any


class FileRouteItem(BaseModel):
    """Individual file routing item for electronics agent content routing"""
    content_type: Literal["current_state", "new_plans", "components", "code", "calculations", "general"] = Field(
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
        description="Suggested filename for new file (required if create_new_file is true, e.g., 'component-specs.md')"
    )
    file_summary: Optional[str] = Field(
        default=None,
        description="Brief description of the new file's purpose (required if create_new_file is true)"
    )


class FileRoutingPlan(BaseModel):
    """Structured output for electronics agent file routing decisions - LangGraph compatible"""
    routing: List[FileRouteItem] = Field(
        default_factory=list,
        description="List of routing decisions for content to files"
    )


class ComponentReplacement(BaseModel):
    """Structured information about a component replacement"""
    old_component_name: str = Field(
        description="Name/part number of the component being replaced (e.g., 'AQV252G', 'Teensy 4.1')"
    )
    new_component_name: str = Field(
        description="Name/part number of the replacement component (e.g., 'ADW221S', 'ESP32')"
    )
    replacement_type: Literal["direct_replacement", "alternative_selected", "upgrade", "downgrade"] = Field(
        description="Type of replacement: direct_replacement (explicit 'instead of X use Y'), alternative_selected (chose Y over X), upgrade/downgrade (better/worse version)"
    )
    reason: Optional[str] = Field(
        default=None,
        description="Reason for replacement if mentioned (e.g., 'cost', 'density', 'availability')"
    )


class ContentStructure(BaseModel):
    """Structured content extraction for electronics agent - LangGraph compatible"""
    current_state: str = Field(
        default="",
        description="Information about what currently exists (formatted as reference documentation)"
    )
    new_plans: str = Field(
        default="",
        description="Recommendations and plans for what to build/do (formatted as reference documentation)"
    )
    components: str = Field(
        default="",
        description="Component specifications, part numbers, values (formatted as reference documentation)"
    )
    code: str = Field(
        default="",
        description="Code snippets, programming details, firmware requirements (formatted as reference documentation)"
    )
    calculations: str = Field(
        default="",
        description="Calculations, formulas, values, specifications (formatted as reference documentation)"
    )
    general: str = Field(
        default="",
        description="Everything else that doesn't fit the above categories (formatted as reference documentation)"
    )
    component_replacements: List[ComponentReplacement] = Field(
        default_factory=list,
        description="List of component replacements detected in the conversation. Empty list if no replacements detected."
    )


class UnifiedContentPlan(BaseModel):
    """
    Unified model combining content extraction and routing in single LLM call.
    This eliminates the need for separate extraction and routing steps.
    """
    routing: List[FileRouteItem] = Field(
        default_factory=list,
        description="List of routing decisions - each item includes both content and where to save it"
    )


class QueryTypeAnalysis(BaseModel):
    """Structured output for query type detection - LangGraph compatible"""
    query_type: Literal["circuit_design", "embedded_code", "component_selection", "calculation", "troubleshooting", "general"] = Field(
        description="Type of electronics query based on semantic intent"
    )
    confidence: float = Field(
        description="Confidence in query type classification (0.0-1.0). Note: Anthropic API doesn't support min/max constraints on number types."
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of why this query type was chosen"
    )


class ProjectPlanAnalysis(BaseModel):
    """Structured output for project plan detection and action - LangGraph compatible"""
    has_project_plan: bool = Field(
        description="Whether a project plan is currently open in the active editor"
    )
    project_plan_needed: bool = Field(
        description="Whether a project plan document is needed for this query"
    )
    project_plan_action: Optional[Literal["create", "open", "plan", "create_file", None]] = Field(
        default=None,
        description="Action to take regarding project plan: 'create' for new project, 'open' to open existing, 'plan' to plan structure, 'create_file' for file approval, None if no action"
    )
    is_new_project_request: bool = Field(
        default=False,
        description="Whether user is requesting to create a new electronics project"
    )
    is_explicit_save_request: bool = Field(
        default=False,
        description="Whether user explicitly requested saving content (e.g., 'save all', 'save what we discussed')"
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of the project plan analysis decision"
    )


class SearchNeedAnalysis(BaseModel):
    """Structured output for search need detection - LangGraph compatible"""
    should_search: bool = Field(
        description="Whether semantic search is needed to answer the query"
    )
    has_explicit_search: bool = Field(
        default=False,
        description="Whether user explicitly requested search (e.g., 'search for', 'find')"
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of why search is or isn't needed"
    )


class FollowUpAnalysis(BaseModel):
    """Structured output for follow-up question detection - LangGraph compatible"""
    is_follow_up: bool = Field(
        description="Whether this query is a follow-up to previous conversation"
    )
    needs_previous_context: bool = Field(
        default=False,
        description="Whether this follow-up needs previous conversation context"
    )
    referenced_topics: List[str] = Field(
        default_factory=list,
        description="List of topics/concepts referenced from previous conversation"
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of follow-up analysis"
    )


class ContentConflictAnalysis(BaseModel):
    """Structured output for content conflict detection - LangGraph compatible"""
    has_conflicts: bool = Field(
        description="Whether new content conflicts with existing content"
    )
    conflict_details: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of conflicts with fields: type, description, severity, existing_content, new_content"
    )
    can_auto_resolve: bool = Field(
        default=False,
        description="Whether conflicts can be automatically resolved"
    )
    resolution_strategy: Optional[str] = Field(
        default=None,
        description="Suggested resolution strategy if conflicts exist"
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of conflict analysis"
    )


class ResponseQualityAnalysis(BaseModel):
    """Structured output for response quality validation - LangGraph compatible"""
    quality_score: float = Field(
        description="Overall quality score (0.0-1.0). Note: Anthropic API doesn't support min/max constraints on number types."
    )
    is_complete: bool = Field(
        description="Whether response is complete and addresses the query"
    )
    missing_information: List[str] = Field(
        default_factory=list,
        description="List of missing information or gaps"
    )
    needs_refinement: bool = Field(
        default=False,
        description="Whether response needs refinement"
    )
    refinement_suggestions: List[str] = Field(
        default_factory=list,
        description="Suggestions for improving the response"
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of quality assessment"
    )


class IncrementalUpdateAnalysis(BaseModel):
    """Structured output for incremental update detection - LangGraph compatible"""
    is_incremental: bool = Field(
        description="Whether this is an update to existing information vs new content"
    )
    update_type: Optional[Literal["correction", "addition", "refinement", "expansion"]] = Field(
        default=None,
        description="Type of update if incremental"
    )
    existing_content_references: List[str] = Field(
        default_factory=list,
        description="References to existing content being updated"
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of update analysis"
    )


class ComponentCompatibilityAnalysis(BaseModel):
    """Structured output for component compatibility checking - LangGraph compatible"""
    components_checked: List[str] = Field(
        default_factory=list,
        description="List of component names/IDs checked"
    )
    compatibility_issues: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of compatibility issues with fields: component1, component2, issue_type, description, severity"
    )
    all_compatible: bool = Field(
        description="Whether all components are compatible"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for resolving compatibility issues"
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of compatibility analysis"
    )


class ProjectDecision(BaseModel):
    """Structured representation of a project decision made during conversation"""
    decision_id: str = Field(
        description="Unique identifier for this decision (e.g., 'dec_001')"
    )
    timestamp: str = Field(
        description="ISO timestamp when decision was made"
    )
    decision_type: Literal["component_selection", "design_choice", "specification", "architecture", "calculation", "other"] = Field(
        description="Type of decision made"
    )
    decision_summary: str = Field(
        description="Brief summary of the decision (e.g., 'Use ADW221S for oscillator switching')"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed information about the decision (component names, values, specifications, etc.)"
    )
    replaced_item: Optional[str] = Field(
        default=None,
        description="Item being replaced (e.g., 'AQV252G' if replacing a component)"
    )
    reason: Optional[str] = Field(
        default=None,
        description="Reason for the decision (e.g., 'cost and density advantages')"
    )
    alternatives_considered: List[str] = Field(
        default_factory=list,
        description="List of alternatives that were considered but not chosen"
    )
    documented_in: List[str] = Field(
        default_factory=list,
        description="List of files where this decision should be documented (e.g., ['./component_specs.md'])"
    )
    supersedes: List[str] = Field(
        default_factory=list,
        description="List of decision_ids that this decision supersedes"
    )


class DocumentationMaintenanceItem(BaseModel):
    """Structured representation of a documentation maintenance action"""
    file: str = Field(
        description="Target file for the maintenance action (e.g., './component_specs.md')"
    )
    section: str = Field(
        description="Section name or identifier within the file"
    )
    action: Literal["remove", "update", "verify", "archive"] = Field(
        description="Type of maintenance action needed"
    )
    reason: str = Field(
        description="Reason for this maintenance action (e.g., 'Component replaced with ADW221S')"
    )
    confidence: float = Field(
        description="Confidence level (0.0-1.0) that this action is needed"
    )
    related_decision_id: Optional[str] = Field(
        default=None,
        description="Decision ID that triggered this maintenance action"
    )
    suggested_content: Optional[str] = Field(
        default=None,
        description="Suggested replacement content if action is 'update'"
    )


class DocumentationMaintenancePlan(BaseModel):
    """Structured plan for documentation maintenance operations"""
    maintenance_items: List[DocumentationMaintenanceItem] = Field(
        default_factory=list,
        description="List of documentation maintenance actions to perform"
    )
    priority: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Overall priority of maintenance operations"
    )
    reasoning: str = Field(
        default="",
        description="Explanation of why these maintenance actions are needed"
    )


class DocumentationInconsistency(BaseModel):
    """Structured representation of a documentation inconsistency"""
    file: str = Field(
        description="File where inconsistency was found"
    )
    section: str = Field(
        description="Section where inconsistency exists"
    )
    issue_type: Literal["component_mismatch", "specification_conflict", "outdated_information", "missing_documentation", "contradiction"] = Field(
        description="Type of inconsistency detected"
    )
    description: str = Field(
        description="Detailed description of the inconsistency"
    )
    severity: Literal["high", "medium", "low"] = Field(
        description="Severity of the inconsistency"
    )
    documented_value: Optional[str] = Field(
        default=None,
        description="Value currently documented"
    )
    actual_value: Optional[str] = Field(
        default=None,
        description="Actual value from decisions/conversation"
    )
    related_decision_id: Optional[str] = Field(
        default=None,
        description="Decision ID related to this inconsistency"
    )
    suggested_fix: str = Field(
        description="Suggested fix for the inconsistency"
    )


class DocumentationVerificationResult(BaseModel):
    """Structured result of documentation consistency verification"""
    verification_status: Literal["consistent", "inconsistent", "needs_review"] = Field(
        description="Overall status of documentation consistency"
    )
    inconsistencies: List[DocumentationInconsistency] = Field(
        default_factory=list,
        description="List of inconsistencies found"
    )
    missing_documentation: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of missing documentation items with fields: file, section, required_for"
    )
    outdated_sections: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of outdated sections with fields: file, section, reason, last_updated"
    )
    completeness_score: float = Field(
        description="Completeness score (0.0-1.0) indicating how complete the documentation is"
    )
    consistency_score: float = Field(
        description="Consistency score (0.0-1.0) indicating how consistent the documentation is with decisions"
    )
    reasoning: str = Field(
        default="",
        description="Explanation of verification results"
    )

