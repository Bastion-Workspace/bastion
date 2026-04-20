"""
Skill acquisition tool - Search Agent Factory skills by capability and acquire their tools mid-loop.

When included in an llm_agent step's available_tools, the agent can call this to discover and
load additional skills (procedure + required_tools) during ReAct iterations.
"""

import logging
from typing import Any, Dict, List, Optional

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
        le=10,
        description="Maximum number of skills to acquire from search results (1-10)",
    )


class SearchAndAcquireSkillsOutputs(BaseModel):
    """Typed outputs; internal _acquire_skills, _acquired_tools, _skill_guidance are in the raw dict for the loop."""
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def search_and_acquire_skills_tool(
    query: str,
    max_skills: int = 3,
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Search Agent Factory skills by capability description and acquire matching tools and procedures.
    When called from an llm_agent step, the ReAct loop merges the new tools and guidance for subsequent iterations.
    """
    # Lazy import: avoids circular import during package init (tool_resolution ↔ tools/__init__).
    from orchestrator.engines.tool_resolution import (
        parse_connections_map,
        scope_connection_bound_tool_name,
    )
    from orchestrator.backend_tool_client import get_backend_tool_client

    cmap = parse_connections_map(_pipeline_metadata)
    active_types_arg: Optional[List[str]] = None
    if cmap:
        active_types_arg = sorted(cmap.keys())

    try:
        client = await get_backend_tool_client()
        limit = min(max(1, max_skills), 10)
        q = (query or "").strip()
        discovered = await client.search_skills(
            user_id=user_id,
            query=q,
            limit=limit,
            score_threshold=0.5,
            active_connection_types=active_types_arg,
        )
    except Exception as e:
        logger.warning("search_and_acquire_skills search_skills failed: %s", e)
        return {
            "formatted": f"Skill search failed: {e}.",
            "_acquire_skills": False,
        }

    if not discovered:
        logger.info(
            "search_and_acquire_skills: no hits query=%r max_skills=%s active_conn_types=%s",
            (query or "")[:500],
            limit,
            active_types_arg,
        )
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
        for t in s.get("required_tools") or []:
            acquired_tools.append(scope_connection_bound_tool_name(str(t), cmap))

    acquired_tools_deduped = list(dict.fromkeys(acquired_tools))
    guidance = "\n\n".join(guidance_parts)

    tools_summary = ", ".join(acquired_tools_deduped) if acquired_tools_deduped else "none"
    formatted = (
        f"Acquired skills: {', '.join(skill_names)}. "
        f"New tools available: {tools_summary}."
    )
    _hit_summary = [
        f"{(h.get('slug') or h.get('name') or h.get('id'))}:{float(h.get('similarity_score') or h.get('score') or 0):.3f}"
        for h in discovered
    ]
    logger.info(
        "search_and_acquire_skills: query=%r max_skills=%s hits=%s skills=%s tools=%s",
        (query or "")[:500],
        limit,
        _hit_summary,
        skill_names,
        acquired_tools_deduped,
    )

    _acquired_skill_infos = [
        {
            "skill_id": str(s.get("id") or ""),
            "skill_slug": s.get("slug") or "",
            "skill_version": s.get("version") or 1,
        }
        for s in skills
    ]

    return {
        "formatted": formatted,
        "_acquire_skills": True,
        "_acquired_tools": acquired_tools_deduped,
        "_skill_guidance": guidance,
        "_acquired_skill_infos": _acquired_skill_infos,
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


# ---------------------------------------------------------------------------
# acquire_skill — direct slug-based skill loading (manifest companion)
# ---------------------------------------------------------------------------


class AcquireSkillInputs(BaseModel):
    """Required inputs for acquire_skill."""
    slug: str = Field(
        description="Skill slug to load (from the skill catalog, e.g. 'document-search', 'email-compose')"
    )


class AcquireSkillOutputs(BaseModel):
    """Typed outputs for acquire_skill."""
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def acquire_skill_tool(
    slug: str,
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Load a skill by slug and acquire its tools and procedure.
    Deterministic alternative to search_and_acquire_skills when you know the skill slug
    (e.g. from the skill catalog in the capability manifest).
    """
    from orchestrator.engines.tool_resolution import (
        parse_connections_map,
        scope_connection_bound_tool_name,
    )
    from orchestrator.backend_tool_client import get_backend_tool_client

    slug_clean = (slug or "").strip().lower()
    if not slug_clean:
        return {
            "formatted": "No slug provided.",
            "_acquire_skills": False,
        }

    cmap = parse_connections_map(_pipeline_metadata)

    try:
        client = await get_backend_tool_client()
        skill = await client.get_skill_by_slug(user_id, slug_clean)
    except Exception as e:
        logger.warning("acquire_skill get_skill_by_slug(%s) failed: %s", slug_clean, e)
        return {
            "formatted": f"Failed to load skill '{slug_clean}': {e}.",
            "_acquire_skills": False,
        }

    if not skill:
        logger.info("acquire_skill: slug=%r not found for user=%s", slug_clean, user_id)
        return {
            "formatted": f"Skill '{slug_clean}' not found.",
            "_acquire_skills": False,
        }

    name = skill.get("name") or skill.get("slug") or "Skill"
    procedure = (skill.get("procedure") or "").strip()
    acquired_tools: List[str] = []
    for t in skill.get("required_tools") or []:
        acquired_tools.append(scope_connection_bound_tool_name(str(t), cmap))
    acquired_tools_deduped = list(dict.fromkeys(acquired_tools))

    guidance = f"## Skill: {name}\n{procedure}" if procedure else ""
    tools_summary = ", ".join(acquired_tools_deduped) if acquired_tools_deduped else "none"
    formatted = f"Acquired skill: {name}. New tools available: {tools_summary}."

    logger.info(
        "acquire_skill: slug=%r name=%s tools=%s",
        slug_clean,
        name,
        acquired_tools_deduped,
    )

    return {
        "formatted": formatted,
        "_acquire_skills": True,
        "_acquired_tools": acquired_tools_deduped,
        "_skill_guidance": guidance,
        "_acquired_skill_infos": [{
            "skill_id": str(skill.get("id") or ""),
            "skill_slug": skill.get("slug") or "",
            "skill_version": skill.get("version") or 1,
        }],
    }


register_action(
    name="acquire_skill",
    category="agent",
    description=(
        "Load a skill by its slug from the skill catalog and acquire its tools and procedure. "
        "Use when you know the exact skill slug (e.g. from the capability manifest)."
    ),
    inputs_model=AcquireSkillInputs,
    outputs_model=AcquireSkillOutputs,
    tool_function=acquire_skill_tool,
)
