"""
Proofreading Models - Roosevelt's "Square Deal" for structured corrections

Defines typed outputs for proofreading operations so we never rely on string
matching hanky-panky. BULLY!
"""

from __future__ import annotations

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


class ProofreadingTaskStatus(str):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    PERMISSION_REQUIRED = "permission_required"
    ERROR = "error"


class CorrectionEntry(BaseModel):
    """Single correction suggestion."""
    original_text: str = Field(description="Exact, verbatim text to be replaced from the source")
    changed_to: str = Field(description="Replacement text")
    explanation: Optional[str] = Field(default=None, description="Brief rationale when the change might be unclear")
    scope: Literal["word", "phrase", "clause", "sentence", "paragraph", "duplicate"] = Field(
        default="sentence", description="Granularity of the correction"
    )


class ProofreadingMode(str):
    CLARITY = "clarity"
    COMPLIANCE = "compliance"
    ACCURACY = "accuracy"


class ProofreadingResult(BaseModel):
    """Structured output for proofreading results."""
    task_status: str = Field(description="complete | incomplete | permission_required | error")
    mode: Optional[str] = Field(default=None, description="clarity | compliance | accuracy")
    summary: Optional[str] = Field(default=None, description="High-level summary of issues found and fixed")
    corrections: List[CorrectionEntry] = Field(default_factory=list, description="Ordered list of corrections")
    style_guide_used: Optional[str] = Field(default=None, description="Filename or identifier of style guide if applied")
    consistency_checks: Optional[List[str]] = Field(default_factory=list, description="Notes on consistency validations performed")
    permission_request: Optional[str] = Field(default=None, description="Deprecated for this agent; no web permissions are requested")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata (e.g., counts, timings)")
    # ROOSEVELT'S EDITOR INTEGRATION: Editor operations for applying corrections
    editor_operations: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Editor operations for applying corrections to the document")


def get_proofreading_structured_output() -> ProofreadingResult:
    return ProofreadingResult


