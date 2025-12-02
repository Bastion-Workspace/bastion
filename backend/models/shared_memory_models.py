"""
Pydantic Models for Shared Memory - Roosevelt's "Type Safety" Architecture
Bulletproof schemas for agent communication and state persistence
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task completion status enumeration"""
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    PERMISSION_REQUIRED = "permission_required"
    ERROR = "error"


class ResearchFindingEntry(BaseModel):
    """Single research finding entry in shared memory"""
    findings: str = Field(..., description="Research findings content")
    sources_searched: List[str] = Field(default_factory=list, description="Sources that were searched")
    confidence_level: float = Field(0.0, ge=0.0, le=1.0, description="Confidence level of findings")
    task_status: str = Field("complete", description="Task completion status")
    next_steps: Optional[str] = Field(None, description="Next steps if task incomplete")
    tools_used: List[str] = Field(default_factory=list, description="Tools used for research")
    citations: List[Dict[str, Any]] = Field(default_factory=list, description="Citation information")
    synthesis_mode: bool = Field(False, description="Whether this was synthesized from context")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="When this finding was created")


class FormattedReportEntry(BaseModel):
    """Single formatted report entry in shared memory"""
    report_content: str = Field(..., description="Formatted report content")
    template_used: str = Field(..., description="Template name used for formatting")
    template_id: Optional[str] = Field(None, description="Template ID used")
    sections_completed: int = Field(0, description="Number of sections successfully completed")
    source_data: Dict[str, Any] = Field(default_factory=dict, description="Metadata about source data used")
    agent: str = Field(..., description="Agent that created this entry")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="When this report was created")


class DataSufficiency(BaseModel):
    """Data sufficiency assessment for research decisions"""
    local_result_count: int = Field(0, ge=0, description="Number of local results found")
    local_data_available: bool = Field(False, description="Whether local data is available")
    web_search_needed: bool = Field(False, description="Whether web search is recommended")
    confidence_level: float = Field(0.0, ge=0.0, le=1.0, description="Overall confidence in available data")


class AgentHandoff(BaseModel):
    """Record of agent handoff in conversation"""
    from_agent: str = Field(..., description="Agent that initiated handoff")
    to_agent: str = Field(..., description="Agent that received handoff")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="When handoff occurred")
    query: str = Field("", description="Query associated with handoff")
    reason: Optional[str] = Field(None, description="Reason for handoff")


class SearchHistoryEntry(BaseModel):
    """Single search history entry"""
    tool: str = Field(..., description="Tool that was used")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="When search occurred")
    result_count: int = Field(0, ge=0, description="Number of results returned")
    success: bool = Field(False, description="Whether search was successful")
    query: Optional[str] = Field(None, description="Search query if available")


class SharedMemory(BaseModel):
    """Complete shared memory structure for agent communication"""
    search_results: Dict[str, Any] = Field(default_factory=dict, description="Results from search operations")
    conversation_context: Dict[str, Any] = Field(default_factory=dict, description="Context for ongoing conversation")
    agent_handoffs: List[AgentHandoff] = Field(default_factory=list, description="Record of agent handoffs")
    data_sufficiency: DataSufficiency = Field(default_factory=DataSufficiency, description="Assessment of available data")
    research_findings: Dict[str, ResearchFindingEntry] = Field(default_factory=dict, description="Research results by topic")
    formatted_reports: Dict[str, FormattedReportEntry] = Field(default_factory=dict, description="Generated reports by template")
    search_history: List[SearchHistoryEntry] = Field(default_factory=list, description="History of search operations")
    last_agent: Optional[str] = Field(None, description="Last primary agent used in conversation for continuity tracking")
    primary_agent_selected: Optional[str] = Field(None, description="Primary agent currently selected by intent classifier")
    
    # Optional fields for specific use cases
    web_search_permission: bool = Field(False, description="Whether web search permission is granted")
    approved_operation: Optional[Dict[str, Any]] = Field(None, description="Approved operation details")
    pending_operations: List[Dict[str, Any]] = Field(default_factory=list, description="Operations pending user approval")

    class Config:
        """Pydantic configuration"""
        extra = "allow"  # Allow additional fields for flexibility
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AgentNodeOutput(BaseModel):
    """Standard output format for agent nodes"""
    agent_results: Dict[str, Any] = Field(..., description="Agent processing results")
    shared_memory: SharedMemory = Field(..., description="Updated shared memory")
    latest_response: Optional[str] = Field(None, description="Latest response content")
    
    # Optional fields for specific agents
    chat_agent_response: Optional[str] = Field(None, description="Chat agent specific response")
    permission_request: Optional[Dict[str, Any]] = Field(None, description="Permission request if needed")
    is_complete: bool = Field(False, description="Whether processing is complete")

    class Config:
        """Pydantic configuration"""
        extra = "allow"  # Allow additional fields for agent-specific data


def validate_shared_memory(shared_memory_dict: Dict[str, Any]) -> SharedMemory:
    """Validate and convert dict to SharedMemory model with error handling"""
    try:
        return SharedMemory(**shared_memory_dict)
    except Exception as e:
        logger.warning(f"⚠️ Shared memory validation failed: {e}")
        # Return default SharedMemory with preserved data where possible
        safe_memory = SharedMemory()
        
        # Preserve what we can
        if "search_results" in shared_memory_dict:
            safe_memory.search_results = shared_memory_dict["search_results"]
        if "conversation_context" in shared_memory_dict:
            safe_memory.conversation_context = shared_memory_dict["conversation_context"]
        if "research_findings" in shared_memory_dict:
            # Validate each research finding
            validated_findings = {}
            for key, finding in shared_memory_dict["research_findings"].items():
                try:
                    if isinstance(finding, dict):
                        validated_findings[key] = ResearchFindingEntry(**finding)
                    else:
                        logger.warning(f"⚠️ Skipping invalid research finding: {key}")
                except Exception as finding_error:
                    logger.warning(f"⚠️ Failed to validate research finding {key}: {finding_error}")
            safe_memory.research_findings = validated_findings
        
        return safe_memory


def merge_shared_memory(base: SharedMemory, updates: SharedMemory) -> SharedMemory:
    """Deep merge two SharedMemory objects, with updates taking precedence"""
    try:
        # Start with base
        merged_dict = base.dict()
        updates_dict = updates.dict()
        
        # Deep merge specific fields
        for key, value in updates_dict.items():
            if key == "research_findings":
                # Merge research findings
                merged_dict["research_findings"].update(value)
            elif key == "agent_handoffs":
                # Append new handoffs
                merged_dict["agent_handoffs"].extend(value)
            elif key == "search_history":
                # Append new search history
                merged_dict["search_history"].extend(value)
            elif key == "search_results":
                # Deep merge search results
                merged_dict["search_results"].update(value)
            elif key == "data_sufficiency":
                # Update data sufficiency fields
                merged_dict["data_sufficiency"].update(value.dict() if hasattr(value, 'dict') else value)
            else:
                # Direct update for other fields
                merged_dict[key] = value
        
        return SharedMemory(**merged_dict)
        
    except Exception as e:
        logger.error(f"❌ Shared memory merge failed: {e}")
        # Return updates as fallback
        return updates
