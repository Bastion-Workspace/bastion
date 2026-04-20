"""Agent skills CRUD/search orchestration for gRPC — returns plain dicts (no protobuf)."""

import json
from typing import Any, Dict, List, Optional


async def op_get_skills_by_ids(skill_ids: List[str]) -> Dict[str, Any]:
    from services.agent_skills_service import get_skills_by_ids

    skills = await get_skills_by_ids(skill_ids)
    return {"success": True, "skills_json": json.dumps(skills), "error": None}


async def op_search_skills(
    *,
    user_id: str,
    query: str,
    limit: int,
    score_threshold: float,
    active_connection_types: Optional[List[str]],
) -> Dict[str, Any]:
    from services.skill_vector_service import search_skills

    results = await search_skills(
        user_id=user_id,
        query=query,
        limit=limit,
        score_threshold=score_threshold,
        active_connection_types=active_connection_types,
    )
    return {"success": True, "skills_json": json.dumps(results), "error": None}


async def op_list_skills(
    *, user_id: str, category: Optional[str], include_builtin: bool
) -> Dict[str, Any]:
    from services.agent_skills_service import list_skills

    skills = await list_skills(user_id, category=category, include_builtin=include_builtin)
    return {"success": True, "skills_json": json.dumps(skills), "error": None}


async def op_list_skill_summaries(
    *, user_id: str, include_builtin: bool
) -> Dict[str, Any]:
    from services.agent_skills_service import list_skill_summaries

    summaries = await list_skill_summaries(user_id, include_builtin=include_builtin)
    return {"success": True, "summaries_json": json.dumps(summaries), "error": None}


async def op_get_skill_by_slug(*, user_id: str, slug: str) -> Dict[str, Any]:
    from services.agent_skills_service import get_skill_by_slug

    if not slug:
        return {
            "success": False,
            "skill_json": "",
            "error": "slug is required",
        }
    skill = await get_skill_by_slug(slug, user_id=user_id)
    if not skill:
        return {
            "success": False,
            "skill_json": "",
            "error": f"Skill '{slug}' not found",
        }
    return {"success": True, "skill_json": json.dumps(skill), "error": None}


async def op_get_candidate_for_slug(*, user_id: str, slug: str) -> Dict[str, Any]:
    from services.agent_skills_service import get_candidate_for_slug

    if not slug:
        return {"success": True, "has_candidate": False, "skill_json": "", "error": None}
    candidate = await get_candidate_for_slug(slug, user_id=user_id)
    if not candidate:
        return {"success": True, "has_candidate": False, "skill_json": "", "error": None}
    return {
        "success": True,
        "has_candidate": True,
        "skill_json": json.dumps(candidate),
        "error": None,
    }


async def op_get_skills_by_slugs(
    *, user_id: str, slugs: List[str]
) -> Dict[str, Any]:
    from services.agent_skills_service import get_skills_by_slugs

    skills = await get_skills_by_slugs(slugs, user_id=user_id)
    return {"success": True, "skills_json": json.dumps(skills), "error": None}


async def op_create_skill(
    *,
    user_id: str,
    name: str,
    slug: str,
    procedure: str,
    required_tools: List[str],
    optional_tools: List[str],
    description: Optional[str],
    category: Optional[str],
    tags: List[str],
) -> Dict[str, Any]:
    from services.agent_skills_service import create_skill

    skill = await create_skill(
        user_id=user_id,
        name=name,
        slug=slug,
        procedure=procedure,
        required_tools=required_tools,
        optional_tools=optional_tools,
        description=description,
        category=category,
        tags=tags,
    )
    return {
        "success": True,
        "skill_id": str(skill.get("id", "")),
        "skill_json": json.dumps(skill),
        "error": None,
    }


async def op_update_skill(
    *,
    skill_id: str,
    user_id: str,
    procedure: Optional[str],
    improvement_rationale: Optional[str],
    evidence_metadata: Optional[Dict[str, Any]],
    name: Optional[str],
    description: Optional[str],
    category: Optional[str],
    required_tools: Optional[List[str]],
    optional_tools: Optional[List[str]],
) -> Dict[str, Any]:
    from services.agent_skills_service import update_skill

    skill = await update_skill(
        skill_id=skill_id,
        user_id=user_id,
        procedure=procedure,
        improvement_rationale=improvement_rationale,
        evidence_metadata=evidence_metadata,
        name=name,
        description=description,
        category=category,
        required_tools=required_tools,
        optional_tools=optional_tools,
    )
    return {
        "success": True,
        "skill_id": str(skill.get("id", "")),
        "version": skill.get("version", 1),
        "skill_json": json.dumps(skill),
        "error": None,
    }
