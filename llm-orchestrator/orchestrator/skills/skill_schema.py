"""
Skill Schema - Declarative skill definitions for the tiered skill architecture.

Skills replace per-agent capability declarations with a unified, data-driven registry.
Tools and subgraphs are validated at load time where possible; gRPC-backed tools
are allowed for runtime resolution.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class EngineType(str, Enum):
    """Execution engine that runs this skill."""

    CONVERSATIONAL = "conversational"
    AUTOMATION = "automation"
    EDITOR = "editor"
    RESEARCH = "research"


class Skill(BaseModel):
    """
    Declarative skill definition.

    Replaces per-agent classes for simple skills; complex skills reference
    subgraphs and optional context loaders.
    """

    # Identity
    name: str = Field(
        description="Unique skill identifier; matches agent_type for dispatch (e.g. weather, fiction_editing)"
    )
    description: str = Field(
        description="Human-readable description for LLM skill selection and discovery"
    )

    # Routing
    engine: EngineType = Field(description="Which execution engine runs this skill")
    domains: List[str] = Field(
        default_factory=list,
        description="Matching domains (e.g. weather, fiction, general)",
    )
    actions: List[str] = Field(
        default_factory=list,
        description="Supported action intents: observation, generation, modification, analysis, query, management",
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Trigger keywords for deterministic matching",
    )
    priority: int = Field(
        default=50,
        ge=0,
        le=100,
        description="Disambiguation priority (higher wins when scores tie)",
    )

    # Editor gating
    editor_types: List[str] = Field(
        default_factory=list,
        description="Supported editor document types (fiction, outline, electronics, etc.)",
    )
    requires_editor: bool = Field(
        default=False,
        description="Hard requirement for active editor; no keyword bypass for editing agents",
    )
    editor_preference: str = Field(
        default="none",
        description="'prefer' boosts when editor present; 'require' is hard gate; 'none' no editor signal",
    )
    context_boost: int = Field(
        default=0,
        description="Score boost when editor context matches (0-25 typical)",
    )

    # Execution
    tools: List[str] = Field(
        default_factory=list,
        description="Tool names this skill needs (resolved at runtime from orchestrator.tools or gRPC)",
    )
    subgraphs: List[str] = Field(
        default_factory=list,
        description="Subgraph builder names for complex skills (e.g. context_preparation, fiction_generation)",
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Skill-specific system prompt for simple skills",
    )
    context_loader: Optional[str] = Field(
        default=None,
        description="Dotted path to context loader function for complex skills (e.g. orchestrator.skills.loaders.fiction_loader)",
    )

    # Behavior flags (migrated from AGENT_CAPABILITIES)
    override_continuity: bool = Field(
        default=False,
        description="Explicit requests override conversation continuity (e.g. research)",
    )
    requires_explicit_keywords: bool = Field(
        default=False,
        description="Only match when explicit keywords present (e.g. story_analysis)",
    )
    supports_hitl: bool = Field(
        default=False,
        description="Skill may trigger human-in-the-loop confirmation",
    )
    requires_image_context: bool = Field(
        default=False,
        description="Skill requires attached image context (e.g. image description)",
    )
    internal_only: bool = Field(
        default=False,
        description="If True, skill is not eligible for user routing (called by other agents only)",
    )
    stateless: bool = Field(
        default=False,
        description="If True, the LLM does not see conversation history (look_back_limit=0). Messages and responses still persist in history; only the prompt sent to the LLM omits prior turns. Use for action skills like org_capture to avoid re-executing past actions.",
    )

    @field_validator("name")
    @classmethod
    def name_lowercase_snake(cls, v: str) -> str:
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("name must be alphanumeric with optional underscores or hyphens")
        return v

    @field_validator("editor_preference")
    @classmethod
    def editor_preference_valid(cls, v: str) -> str:
        if v not in ("none", "prefer", "require"):
            raise ValueError("editor_preference must be one of: none, prefer, require")
        return v
