"""
Agent Response Contract - Standardized response structure for all agents

This module defines the standard response contract that all agents should follow.
It provides type safety and consistency across the codebase.

During migration, agents can gradually adopt this contract while maintaining
backward compatibility through unified extraction helpers in grpc_service.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from enum import Enum
from datetime import datetime


class TaskStatus(str, Enum):
    """Task completion status"""
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    PERMISSION_REQUIRED = "permission_required"
    ERROR = "error"
    REJECTED = "rejected"  # skill cannot handle this query; orchestration may retry with fallback


class ManuscriptEditMetadata(BaseModel):
    """Metadata about document edits for workspace integration"""
    target_filename: str = Field(description="Target document filename")
    scope: str = Field(description="Edit scope: 'chapter', 'document', 'section'")
    summary: str = Field(description="Human-readable summary of the edit")
    chapter_index: Optional[int] = Field(default=None, description="Chapter index if scope is 'chapter'")
    safety: Optional[str] = Field(default=None, description="Safety level: 'safe', 'review_recommended'")


class AgentResponse(BaseModel):
    """
    Standard response contract for ALL agents
    
    This contract ensures consistent response structure across all agents,
    making it easier for grpc_service to extract data and for the frontend
    to handle responses predictably.
    
    Required fields must be provided by all agents.
    Optional fields are used based on agent type and capabilities.
    """
    
    # Required fields (all agents)
    response: str = Field(description="Natural language response for chat sidebar")
    task_status: TaskStatus = Field(description="Task completion status")
    agent_type: str = Field(description="Agent identifier (e.g., 'fiction_editing_agent')")
    timestamp: str = Field(description="ISO timestamp of response generation")
    
    # Optional: Workspace editing
    editor_operations: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="List of editor operations to apply (insert, replace, delete)"
    )
    manuscript_edit: Optional[ManuscriptEditMetadata] = Field(
        default=None,
        description="Metadata about the edit (summary, scope, target filename)"
    )
    failed_operations: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Operations that couldn't be automatically resolved (require manual placement)"
    )
    
    # Optional: Analysis/structured data
    structured_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured analysis results (patterns, insights, calculations, frequencies)"
    )
    
    # Optional: Visualizations
    static_visualization_data: Optional[str] = Field(
        default=None,
        description="Visualization data (chart HTML, base64 image, etc.)"
    )
    static_format: Optional[str] = Field(
        default=None,
        description="Visualization format: 'html', 'base64_png', 'mermaid_code', 'svg'"
    )
    diagram_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Diagram generation results (Mermaid code, SVG, etc.)"
    )
    chart_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Chart generation results (Vega-Lite JSON, HTML, etc.)"
    )
    
    # Optional: Images
    images: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="List of images with metadata (url, alt_text, type, etc.). Images can also be embedded in response text as markdown for backward compatibility."
    )
    
    # Optional: Research/citations
    citations: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Citation sources with metadata (title, url, author, excerpt)"
    )
    sources: Optional[List[str]] = Field(
        default=None,
        description="List of source identifiers or URLs"
    )
    
    # Optional: Errors/warnings
    error: Optional[str] = Field(
        default=None,
        description="Error message if task_status is 'error'"
    )
    warnings: Optional[List[str]] = Field(
        default=None,
        description="Warning messages about potential issues"
    )
    validation_notices: Optional[List[str]] = Field(
        default=None,
        description="Validation notices (consistency warnings, etc.)"
    )
    
    # Optional: Metadata
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0-1.0)"
    )
    processing_time: Optional[float] = Field(
        default=None,
        description="Processing time in seconds"
    )
    
    # Agent-specific extensions allowed during migration
    class Config:
        extra = "allow"  # Allow agent-specific fields during migration period
    
    @classmethod
    def from_legacy_format(cls, result: Dict[str, Any], agent_type: str) -> "AgentResponse":
        """
        Create AgentResponse from legacy agent response format
        
        This helper method normalizes legacy response formats into the standard contract.
        Used during migration period when agents haven't been updated yet.
        
        Args:
            result: Legacy agent result dictionary
            agent_type: Agent identifier
            
        Returns:
            AgentResponse instance with normalized data
        """
        # Extract response text (try multiple locations)
        response_text = (
            result.get("response", "") if isinstance(result.get("response"), str) else
            result.get("final_response", "") if isinstance(result.get("final_response"), str) else
            result.get("response", {}).get("response", "") if isinstance(result.get("response"), dict) else
            result.get("response", {}).get("message", "") if isinstance(result.get("response"), dict) else
            result.get("messages", [{}])[-1].content if result.get("messages") and hasattr(result.get("messages", [])[-1], "content") else
            str(result.get("messages", [""])[-1]) if result.get("messages") else
            "Response generated"
        )
        
        # Extract task_status
        task_status_str = result.get("task_status", "complete")
        try:
            task_status = TaskStatus(task_status_str)
        except ValueError:
            task_status = TaskStatus.COMPLETE
        
        # Extract timestamp
        timestamp = result.get("timestamp") or datetime.now().isoformat()
        
        # Extract editor operations (check multiple locations)
        editor_operations = (
            result.get("editor_operations") or
            result.get("agent_results", {}).get("editor_operations") or
            result.get("response", {}).get("editor_operations") if isinstance(result.get("response"), dict) else None
        )
        
        # Extract manuscript_edit (check multiple locations)
        manuscript_edit_raw = (
            result.get("manuscript_edit") or
            result.get("agent_results", {}).get("manuscript_edit") or
            result.get("response", {}).get("manuscript_edit") if isinstance(result.get("response"), dict) else None
        )
        
        manuscript_edit = None
        if manuscript_edit_raw:
            try:
                manuscript_edit = ManuscriptEditMetadata(**manuscript_edit_raw) if isinstance(manuscript_edit_raw, dict) else None
            except Exception:
                manuscript_edit = None
        
        # Extract failed_operations
        failed_operations = (
            result.get("failed_operations") or
            result.get("response", {}).get("failed_operations") if isinstance(result.get("response"), dict) else None
        )
        
        # Extract structured_data (from various agent-specific locations)
        structured_data = (
            result.get("structured_data") or
            result.get("agent_results", {}).get("structured_data") or
            result.get("analysis_results") or
            result.get("patterns_found") or
            {}
        )
        
        # Extract visualization data
        static_visualization_data = (
            result.get("static_visualization_data") or
            result.get("response", {}).get("static_visualization_data") if isinstance(result.get("response"), dict) else None
        )
        static_format = (
            result.get("static_format") or
            result.get("response", {}).get("static_format") if isinstance(result.get("response"), dict) else None
        )
        diagram_result = result.get("diagram_result")
        chart_result = result.get("chart_result")
        
        # Extract images (from structured_images, structured_response, or agent_results)
        images_raw = (
            result.get("images") or
            result.get("structured_images") or
            result.get("agent_results", {}).get("structured_response", {}).get("images") or
            result.get("agent_results", {}).get("images")
        )
        images = None
        if images_raw and isinstance(images_raw, list):
            # Normalize image objects to standard format
            images = []
            for img in images_raw:
                if isinstance(img, dict):
                    images.append({
                        "url": img.get("url") or img.get("image_url") or img.get("src"),
                        "alt_text": img.get("alt_text") or img.get("alt") or None,
                        "type": img.get("type") or "generated",
                        "metadata": img.get("metadata") or {}
                    })
                elif isinstance(img, str):
                    # Simple URL string - convert to structured format
                    images.append({
                        "url": img,
                        "alt_text": None,
                        "type": "generated",
                        "metadata": {}
                    })
        
        # Extract citations and sources
        citations = result.get("citations") or result.get("agent_results", {}).get("citations")
        sources = result.get("sources") or result.get("agent_results", {}).get("sources")
        
        # Extract errors/warnings
        error = result.get("error") or result.get("response", {}).get("error") if isinstance(result.get("response"), dict) else None
        warnings = result.get("warnings") or result.get("response", {}).get("warnings") if isinstance(result.get("response"), dict) else None
        validation_notices = result.get("validation_notices") or result.get("consistency_warnings") or result.get("reference_warnings")
        
        # Extract metadata
        confidence = result.get("confidence") or result.get("agent_results", {}).get("confidence")
        processing_time = result.get("processing_time")
        
        return cls(
            response=response_text,
            task_status=task_status,
            agent_type=agent_type,
            timestamp=timestamp,
            editor_operations=editor_operations if editor_operations else None,
            manuscript_edit=manuscript_edit,
            failed_operations=failed_operations if failed_operations else None,
            structured_data=structured_data if structured_data else None,
            static_visualization_data=static_visualization_data,
            static_format=static_format,
            diagram_result=diagram_result,
            chart_result=chart_result,
            images=images if images else None,
            citations=citations if citations else None,
            sources=sources if sources else None,
            error=error,
            warnings=warnings if warnings else None,
            validation_notices=validation_notices if validation_notices else None,
            confidence=confidence,
            processing_time=processing_time
        )
  
        # Extract metadata
        confidence = result.get("confidence") or result.get("agent_results", {}).get("confidence")
        processing_time = result.get("processing_time")
        
        return cls(
            response=response_text,
            task_status=task_status,
            agent_type=agent_type,
            timestamp=timestamp,
            editor_operations=editor_operations if editor_operations else None,
            manuscript_edit=manuscript_edit,
            failed_operations=failed_operations if failed_operations else None,
            structured_data=structured_data if structured_data else None,
            static_visualization_data=static_visualization_data,
            static_format=static_format,
            diagram_result=diagram_result,
            chart_result=chart_result,
            citations=citations if citations else None,
            sources=sources if sources else None,
            error=error,
            warnings=warnings if warnings else None,
            validation_notices=validation_notices if validation_notices else None,
            confidence=confidence,
            processing_time=processing_time
        )
