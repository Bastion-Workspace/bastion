"""
Pydantic models for the Compound Query Planner (Phase 7).

ExecutionPlan serves double duty: when is_compound=False it carries a single skill
and confidence (same as today's selector); when is_compound=True it carries steps.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """A single step in a compound execution plan."""

    step_id: int
    skill_name: str = Field(
        default="",
        description="Skill name (mutually exclusive with fragment_name)",
    )
    fragment_name: str = Field(
        default="",
        description="Subgraph fragment name",
    )
    sub_query: str = Field(description="Focused query for this skill")
    depends_on: List[int] = Field(default_factory=list)
    context_keys: List[str] = Field(default_factory=list)
    tool_packs: List[str] = Field(
        default_factory=list,
        description="Optional tool packs to augment this step",
    )


class ExecutionPlan(BaseModel):
    """Compound query execution plan or single-skill selection."""

    is_compound: bool = Field(description="True if query needs multiple skills")
    skill: Optional[str] = Field(default=None, description="Single skill if not compound")
    confidence: float = Field(default=0.0)
    steps: List[PlanStep] = Field(default_factory=list)
    reasoning: str = Field(default="")
