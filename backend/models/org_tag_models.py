"""
Pydantic models for org-mode tagging operations
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class OrgTagRequest(BaseModel):
    """Request to add/update tags on an org entry"""
    file_path: str = Field(..., description="Relative path to org file (e.g., 'inbox.org' or 'Projects/work.org')")
    line_number: int = Field(..., description="Line number of the heading to tag (1-indexed)")
    tags: List[str] = Field(..., description="List of tags to add (e.g., ['@outside', 'urgent'])")
    replace_existing: bool = Field(default=False, description="If True, replace all existing tags. If False, merge with existing tags")


class OrgTagResponse(BaseModel):
    """Response from tagging operation"""
    success: bool
    message: str
    updated_line: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    tags_applied: Optional[List[str]] = None











