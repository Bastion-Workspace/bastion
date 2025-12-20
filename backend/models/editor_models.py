"""
Editor Suggestion Models
Structured request/response for inline editor suggestions.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class EditorSuggestionRequest(BaseModel):
    """Payload from the editor to request a suggestion near the cursor."""
    prefix: str = Field(description="Up to N characters before cursor (context)")
    suffix: str = Field(default="", description="Up to M characters after cursor (context)")
    filename: Optional[str] = Field(default=None, description="Current filename, e.g., notes.md")
    language: Optional[str] = Field(default="markdown", description="Language mode of the buffer")
    cursor_offset: Optional[int] = Field(default=None, description="Cursor offset in the full document")
    frontmatter: Optional[Dict[str, Any]] = Field(default=None, description="Parsed frontmatter (YAML) if present")
    max_chars: int = Field(default=300, ge=1, le=400, description="Maximum suggested characters to return")
    temperature: float = Field(default=0.25, ge=0.0, le=1.0, description="Creativity for continuation")


class EditorSuggestionResponse(BaseModel):
    """Suggestion response for inline ghost text."""
    suggestion: str = Field(description="Suggestion text to display as ghost text (no leading whitespace changes)")
    confidence: float = Field(default=0.7, ge=0.0, le=1.0, description="Model confidence heuristic")
    model_used: Optional[str] = Field(default=None, description="OpenRouter model used for this suggestion")


