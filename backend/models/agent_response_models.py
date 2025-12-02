"""
Pydantic Models for Structured Agent Responses - Roosevelt's "Type Safety" 
Best-of-breed structured outputs for LLM responses with validation
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Literal, Dict, Any
from enum import Enum
import json
from .podcast_models import PodcastScriptResponse, get_podcast_structured_output

# Agent Types
AgentType = Literal["research", "chat", "coding", "direct", "orchestrator", "data_formatting"]

# Task Status
class TaskStatus(str, Enum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    PERMISSION_REQUIRED = "permission_required"
    ERROR = "error"

# Orchestrator Decision Types  
class DecisionType(str, Enum):
    SINGLE_AGENT = "single_agent"
    AGENT_CHAIN = "agent_chain"
    PARALLEL_AGENTS = "parallel_agents"
    PERMISSION_GRANTED = "permission_granted"
    WORKFLOW = "workflow"
    SYNTHESIS = "synthesis"

# Permission Status
class PermissionStatus(str, Enum):
    GRANTED = "granted"
    DENIED = "denied"
    PENDING = "pending"


class CitationSource(BaseModel):
    """Individual citation source for numbered references"""
    id: int = Field(description="Citation number for in-line references like (1), (2)")
    title: str = Field(description="Document title or webpage title")
    type: Literal["document", "webpage", "book"] = Field(description="Source type")
    url: Optional[str] = Field(default=None, description="URL for web sources")
    author: Optional[str] = Field(default=None, description="Author if available")
    date: Optional[str] = Field(default=None, description="Publication date if available")
    excerpt: Optional[str] = Field(default=None, description="Relevant excerpt from source")


class ResearchTaskResult(BaseModel):
    """Structured output for Research Agent responses - LangGraph compatible"""
    task_status: TaskStatus = Field(
        description="Whether the research task is complete, incomplete, or requires permission"
    )
    findings: str = Field(
        description="Research findings with in-line citations in (1), (2) format linking to citations list"
    )
    citations: List[CitationSource] = Field(
        default_factory=list,
        description="Numbered citation sources referenced in findings using (1), (2) format"
    )
    sources_searched: List[str] = Field(
        default_factory=list,
        description="Types of sources searched (local, entities, web)"
    )
    permission_request: Optional[str] = Field(
        default=None,
        description="Permission request message if web search needed"
    )
    confidence_level: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence in the completeness of the research (0.0-1.0)"
    )
    next_steps: Optional[str] = Field(
        default=None,
        description="Suggested next steps if task incomplete"
    )


class ResearchAssessmentResult(BaseModel):
    """Structured output for research quality assessment - evaluating if results are sufficient"""
    sufficient: bool = Field(
        description="Whether the search results are sufficient to answer the query comprehensively"
    )
    has_relevant_info: bool = Field(
        description="Whether the results contain relevant information"
    )
    missing_info: List[str] = Field(
        default_factory=list,
        description="List of specific information gaps that need to be filled"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the assessment (0.0-1.0)"
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of the assessment decision"
    )


class ResearchGapAnalysis(BaseModel):
    """Structured output for research gap analysis - identifying what information is missing"""
    missing_entities: List[str] = Field(
        default_factory=list,
        description="Specific entities, people, facts, or concepts that are missing from results"
    )
    suggested_queries: List[str] = Field(
        default_factory=list,
        description="Targeted search queries that could fill the identified gaps"
    )
    needs_web_search: bool = Field(
        default=False,
        description="Whether web search would likely help fill the gaps"
    )
    gap_severity: Literal["minor", "moderate", "severe"] = Field(
        default="moderate",
        description="How significant the gaps are for answering the query"
    )
    reasoning: str = Field(
        default="",
        description="Explanation of why these gaps exist and how to fill them"
    )


class OrchestratorDecision(BaseModel):
    """Structured output for Orchestrator Agent decisions - LangGraph compatible"""
    decision_type: DecisionType = Field(
        description="Type of orchestration decision"
    )
    primary_agent: AgentType = Field(
        description="Primary agent to handle the request"
    )
    reasoning: str = Field(
        description="Explanation for the routing decision"
    )
    agent_chain: Optional[List[AgentType]] = Field(
        default=None,
        description="Sequence of agents for chain execution"
    )
    parallel_agents: Optional[List[AgentType]] = Field(
        default=None,
        description="Agents to execute in parallel"
    )
    permission_granted: bool = Field(
        default=False,
        description="Whether permission was granted for pending operation"
    )
    operation_id: Optional[str] = Field(
        default=None,
        description="ID of operation being approved/executed"
    )
    synthesis_instructions: Optional[str] = Field(
        default=None,
        description="Instructions for synthesizing multi-agent results"
    )


class PermissionRequest(BaseModel):
    """Structured permission request from agents - LangGraph compatible"""
    permission_type: Literal["web_search", "file_access", "api_call"] = Field(
        description="Type of permission being requested"
    )
    justification: str = Field(
        description="Why this permission is needed"
    )
    scope: str = Field(
        description="What specifically will be accessed (e.g., 'Hans Wilsdorf biographical research')"
    )
    alternative_available: bool = Field(
        default=False,
        description="Whether alternative methods are available"
    )
    urgency: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Urgency of the permission request"
    )


class PermissionAnalysisResult(BaseModel):
    """Structured permission analysis result from Permission Intelligence Agent - LangGraph compatible"""
    permission_status: str  # "granted", "request_permission", "denied"
    confidence_level: float = 0.8
    reasoning: str
    permission_type: str  # "explicit", "implicit", "continuation", "inherited"
    research_legitimacy: str  # "standard_research", "specialized", "unclear"
    recommended_tools: List[str] = []
    scope: str = "current_query"  # "current_query", "conversation", "session"


class IntentClassificationResult(BaseModel):
    """Structured output for Intent Classification Agent responses - LangGraph compatible"""
    intent_type: str = Field(
        description="The classified intent type (RESEARCH, CODING, CHAT, WEATHER, etc.)"
    )
    action_intent: str = Field(
        description="Semantic action type: observation, generation, modification, query, analysis, management"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the classification decision (0.0-1.0)"
    )
    reasoning: str = Field(
        description="Explanation of the classification decision"
    )
    routing_recommendation: str = Field(
        description="Recommended routing destination"
    )
    context_analysis: Dict[str, Any] = Field(
        default_factory=dict,
        description="Analysis of conversation context and flow"
    )
    target_agent: str = Field(
        description="Target agent for routing"
    )
    requires_context_preservation: bool = Field(
        default=False,
        description="Whether conversation context should be preserved"
    )

class ChatResponse(BaseModel):
    """Structured output for Chat Agent responses - LangGraph compatible"""
    message: str = Field(
        description="Conversational response to the user"
    )
    follow_up_suggestions: Optional[List[str]] = Field(
        default=None,
        description="Suggested follow-up questions or topics"
    )
    escalation_needed: bool = Field(
        default=False,
        description="Whether this should be escalated to specialized agents"
    )
    escalation_reason: Optional[str] = Field(
        default=None,
        description="Reason for escalation if needed"
    )
    # ROOSEVELT'S NATURAL COLLABORATION: Removed collaboration fields
    # Let the LLM handle collaboration decisions with full conversation context


class CodingResponse(BaseModel):
    """Structured output for Coding Agent responses - LangGraph compatible"""
    code: Optional[str] = Field(
        default=None,
        description="Generated code solution"
    )
    explanation: str = Field(
        description="Explanation of the solution approach"
    )
    language: Optional[str] = Field(
        default=None,
        description="Programming language used"
    )
    dependencies: Optional[List[str]] = Field(
        default=None,
        description="Required dependencies or imports"
    )
    test_examples: Optional[str] = Field(
        default=None,
        description="Example usage or test cases"
    )
    task_status: TaskStatus = Field(
        description="Status of the coding task"
    )


class DirectResponse(BaseModel):
    """Structured output for Direct Agent responses - LangGraph compatible"""
    answer: str = Field(
        description="Direct factual answer"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the answer accuracy"
    )
    source: Optional[str] = Field(
        default=None,
        description="Source of the information"
    )
    task_status: TaskStatus = Field(
        description="Status of the direct query task"
    )


class WeatherResponse(BaseModel):
    """Structured output for Weather Agent responses - LangGraph compatible"""
    weather_data: str = Field(
        description="Formatted weather information response"
    )
    location: str = Field(
        description="Location queried for weather"
    )
    request_type: Literal["current", "forecast", "location_request"] = Field(
        description="Type of weather request (current conditions, forecast, or location clarification needed)"
    )
    units: str = Field(
        description="Units used (imperial/metric/kelvin)"
    )
    task_status: TaskStatus = Field(
        description="Status of the weather query task"
    )
    confidence: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Confidence in weather data accuracy"
    )
    data_source: str = Field(
        default="OpenWeatherMap",
        description="Source of weather data"
    )
    cached_data: bool = Field(
        default=False,
        description="Whether response used cached data"
    )
    recommendations: Optional[str] = Field(
        default=None,
        description="Weather-based activity recommendations"
    )
    collaboration_suggestion: Optional[str] = Field(
        default=None,
        description="Suggested research collaboration based on weather context"
    )
    collaboration_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in collaboration suggestion relevance"
    )


class RSSManagementResult(BaseModel):
    """RSS management operation result"""
    task_status: Literal["complete", "incomplete", "permission_required", "error"]
    response: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    rss_operations: List[Dict[str, Any]] = Field(default_factory=list)


class DataFormattingResult(BaseModel):
    """Structured output for Data Formatting Agent responses - LangGraph compatible"""
    task_status: TaskStatus = Field(
        description="Status of the data formatting task"
    )
    formatted_output: str = Field(
        description="The formatted data (tables, lists, structured content)"
    )
    format_type: Literal["markdown_table", "markdown_list", "structured_text", "comparative_analysis", "timeline", "chronological_timeline", "visual_timeline", "error"] = Field(
        description="Type of formatting applied to the data"
    )
    confidence_level: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the formatting quality (0.0-1.0)"
    )
    data_sources_used: List[str] = Field(
        default_factory=list,
        description="Sources of data used in the formatting"
    )
    formatting_notes: Optional[str] = Field(
        default=None,
        description="Additional notes about the formatting process"
    )
    
    class Config:
        json_encoders = {
            TaskStatus: lambda v: v.value
        }


# ==========================
# Org Project Capture Models
# ==========================

class OrgProjectCaptureIntent(BaseModel):
    """Structured intent for capturing a project into Org inbox."""
    title: str = Field(description="Project title")
    description: Optional[str] = Field(default=None, description="Short project description / goal")
    target_date: Optional[str] = Field(default=None, description="Org timestamp like <YYYY-MM-DD Dow> for project target date")
    tags: List[str] = Field(default_factory=list, description=":tag: list for the project headline (e.g., :project:) ")
    initial_tasks: List[str] = Field(default_factory=list, description="Up to 5 starter TODO child items")
    clarification_needed: bool = Field(default=False, description="Whether more info is needed from user")
    missing_fields: List[str] = Field(default_factory=list, description="Which fields are missing and requested")
    clarification_question: Optional[str] = Field(default=None, description="Single consolidated question for missing fields")
    preview_block: Optional[str] = Field(default=None, description="Preview of the org block the agent will write")


class OrgProjectCaptureResult(BaseModel):
    """Result of project capture with preview and write details."""
    task_status: TaskStatus = Field(description="Status of the project capture task")
    message: str = Field(description="Human-readable summary of what happened")
    path: Optional[str] = Field(default=None, description="Path to inbox.org where the project was written")
    preview_block: Optional[str] = Field(default=None, description="Preview shown to the user before writing")
    written_block: Optional[str] = Field(default=None, description="Exact block written to file if committed")
    line_start_index: Optional[int] = Field(default=None, description="Start line index of inserted block in inbox.org")
    line_end_index: Optional[int] = Field(default=None, description="End line index of inserted block in inbox.org")


class OrgInboxAction(str, Enum):
    ADD = "add"
    LIST = "list"
    TOGGLE = "toggle"
    UPDATE = "update"


class OrgInboxResult(BaseModel):
    """Structured output for Org Inbox Agent responses - LangGraph compatible"""
    task_status: TaskStatus = Field(description="Status of the org inbox task")
    action: OrgInboxAction = Field(description="Performed action")
    message: str = Field(description="Summary message for the action")
    path: Optional[str] = Field(default=None, description="Path to inbox.org")
    items: Optional[List[Dict[str, Any]]] = Field(default=None, description="Items returned for list action")
    updated_index: Optional[int] = Field(default=None, description="Index of updated line")
    new_line: Optional[str] = Field(default=None, description="The updated line content")


class OrgInboxInterpretation(BaseModel):
    """Agent interpretation for org inbox add/update requests"""
    title: str = Field(description="Concise title parsed from user message")
    entry_kind: Literal["checkbox", "todo", "event", "contact"] = Field(description="What to create in org-mode")
    schedule: Optional[str] = Field(default=None, description="Org-mode <YYYY-MM-DD Dow> timestamp if any")
    deadline: Optional[str] = Field(default=None, description="Org-mode DEADLINE timestamp if any")
    repeater: Optional[str] = Field(default=None, description="Org-mode repeater like +1w, .+1m if any")
    suggested_tags: List[str] = Field(default_factory=list, description="LLM-suggested tags to apply")
    clarification_needed: bool = Field(default=False, description="Whether we need to ask the user a question")
    clarification_question: Optional[str] = Field(default=None, description="Question to disambiguate user intent")
    assistant_confirmation: Optional[str] = Field(default=None, description="Conversational confirmation line to present to the user")
    contact_properties: Optional[Dict[str, str]] = Field(default=None, description="Contact properties like EMAIL, PHONE, BIRTHDAY, COMPANY, etc.")


class OrgSettings(BaseModel):
    """Effective org-mode settings (env + overrides)."""
    todo_sequence: List[str] = Field(description="Ordered list of TODO states, first is default active state")
    default_tags: List[str] = Field(default_factory=list, description="Default tag suggestions")
    suggest_tags: bool = Field(default=True, description="Whether to suggest tags")
    tag_suggestion_mode: Literal["local", "llm", "hybrid"] = Field(default="local", description="Suggestion engine")
    tag_autocommit_confidence: float = Field(ge=0.0, le=1.0, default=0.8, description="Confidence threshold to auto-apply tags")


class TagSuggestion(BaseModel):
    tag: str = Field(description="Suggested tag")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    source: Literal["local", "llm", "hybrid"] = Field(description="Suggestion source")
    rationale: Optional[str] = Field(default=None, description="Short reasoning for the suggestion")


class TagSuggestionResult(BaseModel):
    task_status: TaskStatus = Field(description="Status of the suggestion task")
    suggestions: List[TagSuggestion] = Field(default_factory=list, description="Ordered tag suggestions")
    applied: List[str] = Field(default_factory=list, description="Tags that were auto-applied")
    message: str = Field(description="Summary message")

class RSSMetadataRequest(BaseModel):
    """Structured metadata request for RSS feed creation - LangGraph compatible"""
    task_status: Literal["incomplete"] = Field(
        default="incomplete",
        description="Always incomplete when requesting metadata"
    )
    response: str = Field(
        description="Natural language request for missing metadata"
    )
    missing_metadata: List[str] = Field(
        description="List of required metadata fields that are missing"
    )
    feed_url: str = Field(
        description="The RSS feed URL being processed"
    )
    suggested_title: Optional[str] = Field(
        default=None,
        description="Suggested title extracted from URL (if available)"
    )
    suggested_category: Optional[str] = Field(
        default=None,
        description="Suggested category based on URL analysis (if available)"
    )
    available_categories: List[str] = Field(
        default_factory=lambda: [
            "technology", "science", "news", "business", "politics", 
            "entertainment", "sports", "health", "education", "other"
        ],
        description="Available category options for the user"
    )
    operation_id: str = Field(
        description="Unique identifier for this RSS operation"
    )


# ==========================
# Fiction Editing Structures
# ==========================

class EditorOperation(BaseModel):
    """Single editor operation to apply to the manuscript buffer.

    Uses optimistic concurrency via pre_hash to avoid applying edits to stale content.
    Progressive search with confidence scoring for precise text targeting.
    
    === OPERATION TYPES ===
    
    1. **insert_after_heading**: Add content AFTER a specific line/heading WITHOUT removing anything
       - Use when: Adding new content below headers, after existing paragraphs
       - anchor_text: EXACT header line (e.g., "## Background")
       - text: New content to insert
       - Example: Insert traits after "### Traits" header
    
    2. **replace_range**: Replace ONLY specific content, preserving headers/structure
       - Use when: Changing existing content, updating placeholder text
       - original_text: EXACT text to replace (NOT including headers above it)
       - text: New content to replace with
       - Example: Replace "- [To be developed]" with "- Analytical thinker"
    
    3. **delete_range**: Remove specific content
       - Use when: Removing placeholder text, deleting sections
       - original_text: EXACT text to delete
       - text: "" (empty)
       - Example: Delete "- [To be developed]" placeholder
    
    === CRITICAL RULES ===
    
    ⚠️ NEVER include header lines in original_text for replace_range!
       ❌ BAD:  original_text="### Traits\\n- [To be developed]"
       ✅ GOOD: original_text="- [To be developed based on story needs]"
    
    ⚠️ Use insert_after_heading + anchor_text when adding content below headers!
       ❌ BAD:  replace_range with header included
       ✅ GOOD: insert_after_heading with anchor_text="### Traits"
    
    ⚠️ Provide EXACT, VERBATIM text from file (minimum 10-20 words, complete sentences)
    """
    op_type: Literal["replace_range", "insert_after_heading", "delete_range"] = Field(
        description="Operation type: insert_after_heading (add after line), replace_range (change content), delete_range (remove content)"
    )
    start: int = Field(ge=0, description="Start character offset (inclusive)")
    end: int = Field(ge=0, description="End character offset (exclusive); for insert, end == start")
    text: str = Field(default="", description="Replacement or inserted text")
    pre_hash: str = Field(description="Hash of manuscript slice [start:end] before applying the edit")
    
    # Progressive search anchors (optional, used by resolver)
    original_text: Optional[str] = Field(
        default=None,
        description="EXACT, VERBATIM text from file to replace (for replace_range/delete_range). NEVER include headers! Minimum 10-20 words, complete sentences."
    )
    anchor_text: Optional[str] = Field(
        default=None,
        description="For insert_after_heading: EXACT, COMPLETE line to insert after (e.g., '### Traits', '## Background')"
    )
    left_context: Optional[str] = Field(
        default=None,
        description="Text immediately before the target (for inserts without exact anchor)"
    )
    right_context: Optional[str] = Field(
        default=None,
        description="Text immediately after the target (for bounded revise/delete)"
    )
    occurrence_index: Optional[int] = Field(
        default=0,
        description="Which occurrence of original_text to match (0-based, default 0 for first)"
    )
    
    note: Optional[str] = Field(default=None, description="Short rationale displayed to the user")
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Search confidence (set by resolver): 1.0=exact, 0.9=normalized, 0.8=sentence, 0.7=phrase, <0.7=weak"
    )
    
    @field_validator('original_text', mode='after')
    @classmethod
    def validate_original_text(cls, value, info):
        """Ensure original_text is provided for replace_range and delete_range."""
        if not info.data:
            return value
            
        op_type = info.data.get('op_type')
        
        # For replace_range and delete_range, original_text is REQUIRED
        if op_type in ('replace_range', 'delete_range'):
            if not value or (isinstance(value, str) and len(value.strip()) < 10):
                raise ValueError(
                    f"❌ ANCHOR REQUIRED: {op_type} operations MUST provide 'original_text' field with EXACT, VERBATIM text from manuscript (minimum 10 words). "
                    f"The LLM did not provide this required field. Copy the complete paragraph/sentence you want to modify!"
                )
        
        return value
    
    @field_validator('anchor_text', mode='after')
    @classmethod
    def validate_anchor_text(cls, value, info):
        """Ensure anchor_text is provided for insert_after_heading."""
        if not info.data:
            return value
            
        op_type = info.data.get('op_type')
        
        # For insert_after_heading, anchor_text is REQUIRED  
        if op_type == 'insert_after_heading':
            if not value or (isinstance(value, str) and len(value.strip()) < 3):
                raise ValueError(
                    f"❌ ANCHOR REQUIRED: insert_after_heading operations MUST provide 'anchor_text' field with EXACT line/paragraph to insert after. "
                    f"For prose: use complete sentences (10+ words). For structural documents: use exact marker like '---', '# Heading', etc. (3+ chars). "
                    f"The LLM did not provide this required field. Copy the complete line that should come BEFORE your new text!"
                )
        
        return value


class ManuscriptEdit(BaseModel):
    """Structured, validated edit plan for fiction manuscript changes."""
    target_filename: str = Field(description="The manuscript filename the operations target")
    operations: List[EditorOperation] = Field(default_factory=list, description="List of editor operations")
    scope: Literal["paragraph", "chapter", "multi_chapter"] = Field(description="Declared scope of edits")
    chapter_index: Optional[int] = Field(default=None, description="Zero-based chapter index for primary scope")
    safety: Literal["low", "medium", "high"] = Field(default="medium", description="Risk level of changes")
    summary: str = Field(description="Human-readable summary of planned changes for HITL review")
    clarifying_questions: Optional[List[str]] = Field(
        default=None,
        description="Questions to ask the user for clarification when request is ambiguous or requires author input for quality"
    )


# ==========================
# Universal Document Edit Proposal System
# ==========================

class ContentEdit(BaseModel):
    """Simple content-based edit proposal for bulk updates.
    
    Use this for:
    - Appending content to documents
    - Replacing entire sections
    - Inserting content at specific positions
    
    Prefer EditorOperation for precise, targeted edits with validation.
    """
    edit_mode: Literal["append", "replace", "insert_at"] = Field(
        description="Edit mode: append (add to end), replace (replace entire content), insert_at (insert at position)"
    )
    content: str = Field(description="New content to add/replace/insert")
    insert_position: Optional[int] = Field(
        default=None,
        description="For insert_at mode: character position to insert at (None = append to end)"
    )
    note: Optional[str] = Field(
        default=None,
        description="Human-readable rationale for this edit, shown to user in preview"
    )


class DocumentEditProposal(BaseModel):
    """Universal document edit proposal - any agent can use this.
    
    Supports two edit modes:
    1. **operations**: Precise, targeted edits using EditorOperation objects (best for fiction, rules editing)
    2. **content**: Simple bulk updates using ContentEdit (best for project docs, bulk updates)
    
    The frontend will show appropriate preview based on edit_type.
    """
    document_id: str = Field(description="Document ID to edit")
    edit_type: Literal["operations", "content"] = Field(
        description="Edit type: 'operations' for precise EditorOperation edits, 'content' for simple ContentEdit"
    )
    
    # For operation-based edits
    operations: Optional[List[EditorOperation]] = Field(
        default=None,
        description="List of EditorOperation objects (required if edit_type='operations')"
    )
    
    # For content-based edits
    content_edit: Optional[ContentEdit] = Field(
        default=None,
        description="ContentEdit object (required if edit_type='content')"
    )
    
    # Metadata
    agent_name: str = Field(description="Name of agent proposing this edit")
    summary: str = Field(description="Human-readable summary of proposed changes")
    requires_preview: bool = Field(
        default=True,
        description="If False and edit is small, frontend may auto-apply. If True, always show preview."
    )
    
    @field_validator('operations', mode='after')
    @classmethod
    def validate_operations(cls, value, info):
        """Ensure operations are provided for operation-based edits."""
        if not info.data:
            return value
        edit_type = info.data.get('edit_type')
        if edit_type == 'operations' and (not value or len(value) == 0):
            raise ValueError("operations field is required when edit_type='operations'")
        return value
    
    @field_validator('content_edit', mode='after')
    @classmethod
    def validate_content_edit(cls, value, info):
        """Ensure content_edit is provided for content-based edits."""
        if not info.data:
            return value
        edit_type = info.data.get('edit_type')
        if edit_type == 'content' and not value:
            raise ValueError("content_edit field is required when edit_type='content'")
        return value


# LangGraph Structured Output Functions
# These functions return the Pydantic models directly for LangGraph's structured output system

def get_research_structured_output() -> ResearchTaskResult:
    """Get ResearchTaskResult model for LangGraph structured output"""
    return ResearchTaskResult

def get_orchestrator_structured_output() -> OrchestratorDecision:
    """Get OrchestratorDecision model for LangGraph structured output"""
    return OrchestratorDecision

def get_data_formatting_structured_output() -> DataFormattingResult:
    """Get DataFormattingResult model for LangGraph structured output"""
    return DataFormattingResult

def get_chat_structured_output() -> ChatResponse:
    """Get ChatResponse model for LangGraph structured output"""
    return ChatResponse

def get_coding_structured_output() -> CodingResponse:
    """Get CodingResponse model for LangGraph structured output"""
    return CodingResponse

def get_direct_structured_output() -> DirectResponse:
    """Get DirectResponse model for LangGraph structured output"""
    return DirectResponse

def get_permission_request_structured_output() -> PermissionRequest:
    """Get PermissionRequest model for LangGraph structured output"""
    return PermissionRequest

def get_rss_metadata_request_structured_output() -> RSSMetadataRequest:
    """Get RSSMetadataRequest model for LangGraph structured output"""
    return RSSMetadataRequest

def get_intent_classification_structured_output() -> IntentClassificationResult:
    """Get IntentClassificationResult model for LangGraph structured output"""
    return IntentClassificationResult


# ==========================
# Image Generation Structures
# ==========================

class ImageInfo(BaseModel):
    url: str
    filename: str
    path: str
    width: int
    height: int
    format: Literal["png", "jpg", "jpeg", "webp"]


class ImageGenerationResult(BaseModel):
    task_status: TaskStatus = Field(description="Status of the image generation task")
    prompt: str = Field(description="Original prompt used to generate images")
    model_used: str = Field(description="OpenRouter model used for generation")
    images: List[ImageInfo] = Field(default_factory=list, description="Generated images")
    message: Optional[str] = Field(default=None, description="Human-readable summary")


def get_image_generation_structured_output() -> ImageGenerationResult:
    return ImageGenerationResult


# ==========================
# Website Crawler Structures
# ==========================

class WebsiteCrawlerResponse(BaseModel):
    """Structured output for Website Crawler Agent responses - LangGraph compatible"""
    task_status: TaskStatus = Field(
        description="Status of the website crawl task"
    )
    response: str = Field(
        description="Human-readable summary of crawl results"
    )
    website_url: Optional[str] = Field(
        default=None,
        description="URL of the crawled website"
    )
    pages_crawled: int = Field(
        default=0,
        ge=0,
        description="Number of pages successfully crawled"
    )
    pages_stored: int = Field(
        default=0,
        ge=0,
        description="Number of pages stored and vectorized"
    )
    pages_failed: int = Field(
        default=0,
        ge=0,
        description="Number of pages that failed to crawl"
    )
    max_depth_reached: int = Field(
        default=0,
        ge=0,
        description="Maximum depth level reached during crawl"
    )
    crawl_session_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for this crawl session"
    )
    elapsed_time_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Total time taken for the crawl operation"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the crawl"
    )


def get_website_crawler_structured_output() -> WebsiteCrawlerResponse:
    """Get WebsiteCrawlerResponse model for LangGraph structured output"""
    return WebsiteCrawlerResponse


# ==========================
# Entertainment Agent Structures
# ==========================

class EntertainmentResponse(BaseModel):
    """Structured output for Entertainment Agent responses - LangGraph compatible"""
    task_status: TaskStatus = Field(
        description="Status of the entertainment query task"
    )
    response: str = Field(
        description="Formatted entertainment information or recommendations"
    )
    content_type: Literal["movie", "tv_show", "tv_episode", "mixed", "recommendation"] = Field(
        description="Type of entertainment content in response"
    )
    items_found: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Structured list of found movies/shows with metadata"
    )
    recommendations: Optional[List[str]] = Field(
        default=None,
        description="Recommended titles based on query"
    )
    comparison_summary: Optional[str] = Field(
        default=None,
        description="Summary for comparison requests"
    )
    confidence: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Confidence in response accuracy"
    )


def get_entertainment_structured_output() -> EntertainmentResponse:
    """Get EntertainmentResponse model for LangGraph structured output"""
    return EntertainmentResponse


# ==========================
# Podcast Script Structures
# ==========================

def get_podcast_script_structured_output() -> PodcastScriptResponse:
    """Get PodcastScriptResponse model for LangGraph structured output"""
    return PodcastScriptResponse


# ==========================
# Outline Clarification Structures
# ==========================

class OutlineClarificationRequest(BaseModel):
    """Structured clarification request from Outline Agent - LangGraph compatible
    
    Used when the agent needs more information to properly develop the outline.
    """
    task_status: Literal["incomplete"] = Field(
        default="incomplete",
        description="Always incomplete when requesting clarification"
    )
    clarification_needed: bool = Field(
        default=True,
        description="Always true for clarification requests"
    )
    questions: List[str] = Field(
        description="Specific questions to ask the user to flesh out the story"
    )
    context: str = Field(
        description="Context explaining what section/aspect needs clarification"
    )
    missing_elements: List[str] = Field(
        default_factory=list,
        description="List of story elements that are unclear or missing"
    )
    suggested_direction: Optional[str] = Field(
        default=None,
        description="Optional suggestion for direction if user wants guidance"
    )
    section_affected: Optional[str] = Field(
        default=None,
        description="Which section of the outline needs clarification (e.g., 'Chapter 3', 'Overall Synopsis')"
    )
    confidence_without_clarification: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How confident the agent would be proceeding without clarification (0.0-1.0)"
    )


def get_outline_clarification_structured_output() -> OutlineClarificationRequest:
    """Get OutlineClarificationRequest model for LangGraph structured output"""
    return OutlineClarificationRequest


# ==========================
# Content Analysis & Comparison Structures
# ==========================

class DocumentSummary(BaseModel):
    """Individual document summary for multi-document comparison - LangGraph compatible
    
    **ROOSEVELT'S STRUCTURED SUMMARIZATION**: Each document gets a comprehensive summary
    capturing key details for comparison analysis.
    """
    title: str = Field(description="Document title or filename")
    summary: str = Field(description="Comprehensive summary (300-500 words) capturing essence and key details")
    main_topics: List[str] = Field(
        default_factory=list,
        description="Primary topics or themes discussed in the document"
    )
    key_arguments: List[str] = Field(
        default_factory=list,
        description="Main arguments, findings, or claims made"
    )
    key_data_points: List[str] = Field(
        default_factory=list,
        description="Important data, dates, facts, or statistics mentioned"
    )
    author_perspective: Optional[str] = Field(
        default=None,
        description="Author's perspective, stance, or conclusions"
    )
    unique_insights: List[str] = Field(
        default_factory=list,
        description="Unique or notable aspects that distinguish this document"
    )
    document_id: Optional[str] = Field(default=None, description="Source document ID")
    original_length: int = Field(default=0, ge=0, description="Original document length in characters")
    confidence: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Confidence in summary quality (0.0-1.0)"
    )


class ComparisonAnalysisResult(BaseModel):
    """Structured output for multi-document comparison analysis - LangGraph compatible
    
    **ROOSEVELT'S STRUCTURED COMPARISON**: Best-of-breed comparative analysis with
    structured fields for similarities, differences, conflicts, and recommendations.
    """
    task_status: TaskStatus = Field(description="Status of the comparison task")
    
    # Core Comparison Fields
    key_similarities: List[str] = Field(
        default_factory=list,
        description="Themes, ideas, or patterns that appear across documents"
    )
    key_differences: List[str] = Field(
        default_factory=list,
        description="How documents differ in approach, focus, conclusions, or perspectives"
    )
    conflicts_contradictions: List[str] = Field(
        default_factory=list,
        description="Contradictory claims, conflicting information, or competing viewpoints"
    )
    unique_contributions: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Unique perspectives or insights each document offers (keyed by document title)"
    )
    
    # Synthesis & Analysis
    overall_assessment: str = Field(
        description="Synthesized collective picture that emerges from all documents together"
    )
    dominant_themes: List[str] = Field(
        default_factory=list,
        description="Most prominent themes or patterns across the collection"
    )
    gaps_missing_perspectives: List[str] = Field(
        default_factory=list,
        description="Important gaps or missing perspectives in the collective analysis"
    )
    
    # Recommendations
    reading_recommendations: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Which documents to read first and why (list of {title, reason} dicts)"
    )
    follow_up_questions: List[str] = Field(
        default_factory=list,
        description="Suggested follow-up questions for deeper analysis"
    )
    
    # Metadata
    documents_compared: int = Field(default=0, ge=0, description="Number of documents analyzed")
    document_titles: List[str] = Field(default_factory=list, description="Titles of compared documents")
    comparison_strategy: Literal["direct", "summarize_then_compare"] = Field(
        description="Strategy used for comparison"
    )
    confidence_level: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Overall confidence in comparison quality (0.0-1.0)"
    )
    
    # Optional document summaries (if using summarization strategy)
    document_summaries: Optional[List[DocumentSummary]] = Field(
        default=None,
        description="Individual document summaries if summarization strategy was used"
    )


def get_document_summary_structured_output() -> DocumentSummary:
    """Get DocumentSummary model for LangGraph structured output"""
    return DocumentSummary


def get_comparison_analysis_structured_output() -> ComparisonAnalysisResult:
    """Get ComparisonAnalysisResult model for LangGraph structured output"""
    return ComparisonAnalysisResult


# ==========================
# Electronics Agent File Routing Structures
# ==========================

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


class FileRoutingPlan(BaseModel):
    """Structured output for electronics agent file routing decisions - LangGraph compatible"""
    routing: List[FileRouteItem] = Field(
        default_factory=list,
        description="List of routing decisions for content to files"
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


def get_file_routing_plan_structured_output() -> FileRoutingPlan:
    """Get FileRoutingPlan model for LangGraph structured output"""
    return FileRoutingPlan


def get_content_structure_structured_output() -> ContentStructure:
    """Get ContentStructure model for LangGraph structured output"""
    return ContentStructure


class EmailDraft(BaseModel):
    """Structured email draft"""
    recipients: List[str] = Field(description="List of recipient email addresses")
    cc: Optional[List[str]] = Field(default=[], description="List of CC email addresses")
    bcc: Optional[List[str]] = Field(default=[], description="List of BCC email addresses")
    subject: str = Field(description="Email subject line")
    body_text: str = Field(description="Plain text email body")
    body_html: Optional[str] = Field(default=None, description="HTML email body (optional)")
    from_email: str = Field(description="Sender email address (user's verified email)")
    from_name: str = Field(description="Sender display name")
    confidence: float = Field(description="Agent's confidence in the draft quality (0.0-1.0)", ge=0.0, le=1.0)
    context_sources: Optional[List[str]] = Field(default=[], description="What conversation content was used in the email")


class EmailResponse(BaseModel):
    """Structured output for email agent"""
    task_status: TaskStatus = Field(description="Task completion status")
    draft: Optional[EmailDraft] = Field(default=None, description="Email draft (if available)")
    send_status: Literal["draft", "sent", "failed", "cancelled", "verification_required", "rate_limited"] = Field(
        description="Current status of the email sending process"
    )
    message: str = Field(description="Natural language response to user")
    error: Optional[str] = Field(default=None, description="Error message if task failed")
    rate_limit_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Rate limit information (remaining emails, reset time, etc.)"
    )
