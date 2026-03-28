"""
Skill acquisition tool - Search Agent Factory skills by capability and acquire their tools mid-loop.

When included in an llm_agent step's available_tools, the agent can call this to discover and
load additional skills (procedure + required_tools) during ReAct iterations.
"""

import logging
from typing import List

from pydantic import BaseModel, Field

from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)


class SearchAndAcquireSkillsInputs(BaseModel):
    """Required inputs for search_and_acquire_skills."""
    query: str = Field(
        description="Capability description to search for (e.g. 'create documents', 'send email', 'search calendar')"
    )


class SearchAndAcquireSkillsParams(BaseModel):
    """Optional parameters."""
    max_skills: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum number of skills to acquire from search results (1-5)",
    )


class SearchAndAcquireSkillsOutputs(BaseModel):
    """Typed outputs; internal _acquire_skills, _acquired_tools, _skill_guidance are in the raw dict for the loop."""
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def search_and_acquire_skills_tool(
    query: str,
    max_skills: int = 3,
    user_id: str = "system",
) -> dict:
    """
    Search Agent Factory skills by capability description and acquire matching tools and procedures.
    When called from an llm_agent step, the ReAct loop merges the new tools and guidance for subsequent iterations.
    """
    from orchestrator.backend_tool_client import get_backend_tool_client

    try:
        client = await get_backend_tool_client()
        limit = min(max(1, max_skills), 5)
        discovered = await client.search_skills(
            user_id=user_id,
            query=(query or "").strip(),
            limit=limit,
            score_threshold=0.5,
        )
    except Exception as e:
        logger.warning("search_and_acquire_skills search_skills failed: %s", e)
        return {
            "formatted": f"Skill search failed: {e}.",
            "_acquire_skills": False,
        }

    if not discovered:
        return {
            "formatted": f"No skills found matching '{query}'.",
            "_acquire_skills": False,
        }

    skill_ids = [
        h.get("id") or h.get("skill_id")
        for h in discovered
        if h.get("id") or h.get("skill_id")
    ]
    try:
        skills = await client.get_skills_by_ids(user_id, skill_ids)
    except Exception as e:
        logger.warning("search_and_acquire_skills get_skills_by_ids failed: %s", e)
        return {
            "formatted": f"Failed to load skills: {e}.",
            "_acquire_skills": False,
        }

    guidance_parts: List[str] = []
    acquired_tools: List[str] = []
    skill_names: List[str] = []
    for s in skills:
        name = s.get("name") or s.get("slug") or "Skill"
        procedure = (s.get("procedure") or "").strip()
        skill_names.append(name)
        if procedure:
            guidance_parts.append(f"## Skill: {name}\n{procedure}")
        acquired_tools.extend(s.get("required_tools") or [])

    acquired_tools_deduped = list(dict.fromkeys(acquired_tools))
    guidance = "\n\n".join(guidance_parts)

    tools_summary = ", ".join(acquired_tools_deduped) if acquired_tools_deduped else "none"
    formatted = (
        f"Acquired skills: {', '.join(skill_names)}. "
        f"New tools available: {tools_summary}."
    )

    return {
        "formatted": formatted,
        "_acquire_skills": True,
        "_acquired_tools": acquired_tools_deduped,
        "_skill_guidance": guidance,
    }


register_action(
    name="search_and_acquire_skills",
    category="agent",
    description=(
        "Search Agent Factory skills by capability (e.g. 'create documents', 'send email') and acquire "
        "their tools and procedures for the rest of this run. Use when you need capabilities you don't have."
    ),
    inputs_model=SearchAndAcquireSkillsInputs,
    params_model=SearchAndAcquireSkillsParams,
    outputs_model=SearchAndAcquireSkillsOutputs,
    tool_function=search_and_acquire_skills_tool,
)
