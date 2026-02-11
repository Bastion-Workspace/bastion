"""
Org-Mode Quick Capture Models
Pydantic models for Emacs-style org-capture functionality
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class OrgCaptureTemplate(BaseModel):
    """
    Template for quick capture entries
    
    Example templates:
    - TODO: Creates a TODO entry
    - Note: Creates a simple note
    - Journal: Creates a dated journal entry
    """
    type: str = Field(..., description="Template type: 'todo', 'note', 'journal', 'meeting'")
    heading_level: int = Field(default=1, description="Heading level (* or ** or ***)")
    include_timestamp: bool = Field(default=True, description="Include timestamp in entry")
    todo_state: Optional[str] = Field(default="TODO", description="TODO state for todo templates")
    tags: Optional[list[str]] = Field(default=None, description="Default tags to apply")
    
    class Config:
        json_schema_extra = {
            "example": {
                "type": "todo",
                "heading_level": 2,
                "include_timestamp": True,
                "todo_state": "TODO",
                "tags": ["inbox"]
            }
        }


class OrgCaptureRequest(BaseModel):
    """
    Request to capture content to inbox.org
    """
    content: str = Field(..., description="The content to capture")
    template_type: str = Field(default="note", description="Template type: 'todo', 'note', 'journal', 'meeting'")
    tags: Optional[list[str]] = Field(default=None, description="Tags to apply")
    priority: Optional[str] = Field(None, description="Priority for TODO items (A, B, C)")
    scheduled: Optional[str] = Field(None, description="Scheduled date (YYYY-MM-DD)")
    deadline: Optional[str] = Field(None, description="Deadline date (YYYY-MM-DD)")
    entry_date: Optional[str] = Field(
        None,
        description="Date for journal entry (YYYY-MM-DD). If None, uses today."
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "content": "Review quarterly reports",
                "template_type": "todo",
                "tags": ["work", "review"],
                "priority": "A",
                "scheduled": "2025-10-21"
            }
        }


class OrgCaptureResponse(BaseModel):
    """
    Response from capture operation
    """
    success: bool = Field(..., description="Whether capture was successful")
    message: str = Field(..., description="Success or error message")
    entry_preview: Optional[str] = Field(None, description="Preview of captured entry")
    file_path: Optional[str] = Field(None, description="Path to inbox.org")
    line_number: Optional[int] = Field(None, description="Line number where entry was added")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Successfully captured to inbox.org",
                "entry_preview": "** TODO Review quarterly reports :work:review:",
                "file_path": "uploads/Users/admin/inbox.org",
                "line_number": 42
            }
        }



