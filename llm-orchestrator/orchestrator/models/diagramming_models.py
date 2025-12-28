"""
Diagramming Models - Structured outputs for diagram generation

Pydantic models for diagram analysis and results
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any, List


class DiagramAnalysis(BaseModel):
    """Structured output for LLM analysis of diagramming needs"""
    diagram_type: Literal[
        "mermaid_flowchart",
        "mermaid_sequence",
        "mermaid_state",
        "mermaid_gantt",
        "mermaid_class",
        "mermaid_er",
        "circuit_ascii",
        "pin_table",
        "block_diagram"
    ] = Field(
        description="Type of diagram that best represents the request"
    )
    title: str = Field(
        description="Diagram title that summarizes what the diagram shows"
    )
    description: str = Field(
        description="Description of what the diagram represents"
    )
    diagram_content: str = Field(
        description="Generated diagram content (Mermaid syntax, ASCII art, or markdown table)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the diagram (components, connections, etc.)"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the diagram analysis (0.0-1.0)"
    )
    reasoning: str = Field(
        description="Explanation of why this diagram type and content were chosen"
    )


class DiagramResult(BaseModel):
    """Final diagram result with validation"""
    success: bool = Field(
        description="Whether diagram generation was successful"
    )
    diagram_type: str = Field(
        description="Type of diagram that was generated"
    )
    title: str = Field(
        description="Diagram title"
    )
    diagram_format: Literal["mermaid", "ascii", "markdown_table"] = Field(
        description="Format of the diagram content"
    )
    diagram_content: Optional[str] = Field(
        default=None,
        description="Diagram content (Mermaid syntax, ASCII art, or markdown table)"
    )
    validation_errors: Optional[List[str]] = Field(
        default=None,
        description="List of validation errors if any"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if generation failed"
    )
