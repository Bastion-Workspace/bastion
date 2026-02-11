"""
Editor Operation Models for Fiction and Outline Editing Workflows

Provides Pydantic models for structured editor operations with validation.
Used by fiction_editing_agent and outline_editing_subgraph for type-safe operation handling.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


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
    op_type: Literal["replace_range", "insert_after_heading", "insert_after", "delete_range"] = Field(
        description="Operation type: insert_after (add after text), insert_after_heading (add after heading), replace_range (change content), delete_range (remove content)"
    )
    
    # Resolved positions (set by resolver, not LLM)
    start: Optional[int] = Field(
        default=None,
        ge=0,
        description="Start character offset (inclusive). Set by resolver after operation resolution."
    )
    end: Optional[int] = Field(
        default=None,
        ge=0,
        description="End character offset (exclusive); for insert, end == start. Set by resolver after operation resolution."
    )
    pre_hash: Optional[str] = Field(
        default=None,
        description="Hash of manuscript slice [start:end] before applying the edit. Set by resolver."
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Search confidence (set by resolver): 1.0=exact, 0.9=normalized, 0.8=sentence, 0.7=phrase, <0.7=weak"
    )
    
    # Content fields (provided by LLM)
    text: str = Field(default="", description="Replacement or inserted text")
    
    # Progressive search anchors (provided by LLM, used by resolver)
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
        """Ensure anchor_text is provided for insert_after_heading, except for empty files."""
        if not info.data:
            return value
            
        op_type = info.data.get('op_type')
        
        # For insert_after_heading/insert_after, anchor_text is REQUIRED (except for empty files - handled in resolver)
        if op_type in ('insert_after_heading', 'insert_after'):
            # Allow empty anchor_text - resolver will check if file is empty
            # If file is not empty and anchor_text is missing, resolver will fail appropriately
            if value and isinstance(value, str) and len(value.strip()) < 3:
                raise ValueError(
                    f"❌ INVALID ANCHOR: anchor_text must be at least 3 characters if provided. "
                    f"For prose: use complete sentences (10+ words). For structural documents: use exact marker like '---', '# Heading', etc. (3+ chars)."
                )
        
        return value


class ManuscriptEdit(BaseModel):
    """Structured, validated edit plan for fiction and outline manuscript changes.
    
    Used by both fiction_editing_agent and outline_editing_subgraph for type-safe
    operation planning and validation.
    """
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
    
    @field_validator('operations', mode='after')
    @classmethod
    def validate_operations(cls, value):
        """Validate operations list - allow empty for questions with no edits needed."""
        # Empty operations are allowed for questions where analysis shows no edits needed
        # The summary field will contain the answer/analysis
        return value

