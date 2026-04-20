"""
Deep-clone service for Agent Factory artifacts.

Clones skills (with depends_on resolution), playbooks (with skill_id remapping),
and full agent profiles (playbook + skills + profile) into a target user's workspace.
"""

import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from services import agent_skills_service, agent_factory_service

logger = logging.getLogger(__name__)


async def deep_clone_skill(
    skill_id: str,
    target_user_id: str,
    *,
    _id_map: Optional[Dict[str, str]] = None,
    _visited: Optional[set] = None,
) -> Tuple[Optional[str], Dict[str, str]]:
    """
    Clone a skill into the target user's workspace, including depends_on dependencies.
    Returns (new_skill_id, skill_id_map) -- map of old_id -> new_id for all cloned skills.
    """
    from services.database_manager.database_helpers import fetch_one

    if _id_map is None:
        _id_map = {}
    if _visited is None:
        _visited = set()

    if skill_id in _id_map:
        return _id_map[skill_id], _id_map
    if skill_id in _visited:
        return None, _id_map
    _visited.add(skill_id)

    skill = await agent_skills_service.get_skill(skill_id)
    if not skill:
        return None, _id_map

    if skill.get("is_builtin"):
        _id_map[skill_id] = skill_id
        return skill_id, _id_map

    for dep_slug in (skill.get("depends_on") or []):
        source_user_id = skill.get("user_id")
        dep_skill = await agent_skills_service.get_skill_by_slug(dep_slug, source_user_id)
        if dep_skill and not dep_skill.get("is_builtin"):
            dep_id = dep_skill.get("id")
            if dep_id and dep_id not in _id_map:
                await deep_clone_skill(dep_id, target_user_id, _id_map=_id_map, _visited=_visited)

    slug = skill.get("slug", "")
    base_slug = slug
    if base_slug.endswith("-copy"):
        base_slug = base_slug[:-5] or "skill"
    target_slug = base_slug

    existing = await agent_skills_service.get_skill_by_slug(target_slug, target_user_id)
    if existing:
        suffix = 1
        while True:
            candidate = f"{base_slug}-copy{'-' + str(suffix) if suffix > 1 else ''}"
            ex = await agent_skills_service.get_skill_by_slug(candidate, target_user_id)
            if not ex:
                target_slug = candidate
                break
            suffix += 1

    name = skill.get("name", slug)
    if not name.endswith(" (Copy)"):
        name = f"{name} (Copy)"

    remapped_deps = []
    for dep_slug in (skill.get("depends_on") or []):
        remapped_deps.append(dep_slug)

    try:
        created = await agent_skills_service.create_skill(
            target_user_id,
            name=name,
            slug=target_slug,
            procedure=skill.get("procedure", ""),
            required_tools=skill.get("required_tools"),
            optional_tools=skill.get("optional_tools"),
            description=skill.get("description"),
            category=skill.get("category"),
            inputs_schema=skill.get("inputs_schema"),
            outputs_schema=skill.get("outputs_schema"),
            examples=skill.get("examples"),
            tags=skill.get("tags"),
            required_connection_types=skill.get("required_connection_types"),
            is_core=skill.get("is_core", False),
            depends_on=remapped_deps,
        )
        new_id = str(created.get("id", ""))
        _id_map[skill_id] = new_id
        return new_id, _id_map
    except ValueError as e:
        logger.warning("Skill clone slug conflict for %s: %s", target_slug, e)
        existing = await agent_skills_service.get_skill_by_slug(target_slug, target_user_id)
        if existing:
            new_id = str(existing.get("id", ""))
            _id_map[skill_id] = new_id
            return new_id, _id_map
        return None, _id_map


def _remap_playbook_skill_ids(definition: Dict[str, Any], skill_id_map: Dict[str, str]) -> Dict[str, Any]:
    """Rewrite skill_ids/skills in playbook step definitions using the ID map."""
    definition = json.loads(json.dumps(definition))
    steps = definition.get("steps") or []
    for step in steps:
        if not isinstance(step, dict):
            continue
        for key in ("skill_ids", "skills"):
            if key not in step:
                continue
            orig = step[key]
            if not isinstance(orig, list):
                continue
            new_ids = []
            for sid in orig:
                s = (sid or "").strip() if isinstance(sid, str) else None
                if not s:
                    continue
                mapped = skill_id_map.get(s, s)
                if mapped and mapped not in new_ids:
                    new_ids.append(mapped)
            step[key] = new_ids
    return definition


async def deep_clone_playbook(
    playbook_id: str,
    target_user_id: str,
    *,
    skill_id_map: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[str], Dict[str, str]]:
    """
    Clone a playbook and all its referenced skills into the target user's workspace.
    Returns (new_playbook_id, skill_id_map).
    """
    from services.database_manager.database_helpers import fetch_one, fetch_value

    if skill_id_map is None:
        skill_id_map = {}

    pb = await fetch_one(
        "SELECT * FROM custom_playbooks WHERE id = $1",
        playbook_id,
    )
    if not pb:
        return None, skill_id_map

    definition = pb.get("definition") or {}
    if isinstance(definition, str):
        try:
            definition = json.loads(definition)
        except (json.JSONDecodeError, TypeError):
            definition = {}

    steps = definition.get("steps") or [] if isinstance(definition, dict) else []
    skill_ids_in_playbook: List[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        for sid in (step.get("skill_ids") or step.get("skills") or []):
            if sid and isinstance(sid, str) and sid not in skill_ids_in_playbook:
                skill_ids_in_playbook.append(sid)

    for sid in skill_ids_in_playbook:
        if sid not in skill_id_map:
            await deep_clone_skill(sid, target_user_id, _id_map=skill_id_map)

    if isinstance(definition, dict) and skill_id_map:
        definition = _remap_playbook_skill_ids(definition, skill_id_map)

    base_name = (pb.get("name") or "Playbook").strip()
    if base_name.endswith(" (Copy)"):
        base_name = base_name[:-7].strip() or "Playbook"
    name = f"{base_name} (Copy)"
    suffix = 0
    while True:
        candidate = f"{name}_{suffix}" if suffix else name
        existing = await fetch_one(
            "SELECT id FROM custom_playbooks WHERE user_id = $1 AND name = $2",
            target_user_id,
            candidate,
        )
        if not existing:
            name = candidate
            break
        suffix += 1

    defn, _, _ = await agent_factory_service.validate_and_remediate_playbook_models_for_user(
        target_user_id, definition
    )

    triggers = pb.get("triggers") or []
    tags = list(pb.get("tags") or [])
    required_connectors = list(pb.get("required_connectors") or [])

    new_id = await fetch_value(
        """
        INSERT INTO custom_playbooks (
            user_id, name, description, version, definition, triggers,
            is_template, category, tags, required_connectors
        ) VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, false, $7, $8, $9)
        RETURNING id
        """,
        target_user_id,
        name,
        pb.get("description") or "",
        pb.get("version", "1.0"),
        json.dumps(defn) if isinstance(defn, (dict, list)) else defn,
        json.dumps(triggers) if isinstance(triggers, (dict, list)) else triggers,
        pb.get("category"),
        tags,
        required_connectors,
    )
    return str(new_id) if new_id else None, skill_id_map


async def deep_clone_agent_profile(
    profile_id: str,
    target_user_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Clone an agent profile, its default playbook, and all referenced skills
    into the target user's workspace. Returns the new profile dict.
    """
    from services.database_manager.database_helpers import fetch_one

    profile = await fetch_one(
        "SELECT * FROM agent_profiles WHERE id = $1",
        profile_id,
    )
    if not profile:
        return None

    skill_id_map: Dict[str, str] = {}
    new_playbook_id: Optional[str] = None
    pb_id = profile.get("default_playbook_id")
    if pb_id:
        new_playbook_id, skill_id_map = await deep_clone_playbook(
            str(pb_id), target_user_id, skill_id_map=skill_id_map
        )

    handle = (profile.get("handle") or "imported").strip() or "imported"
    base_handle = handle[:90]
    suffix = 0
    while True:
        candidate = f"{base_handle}_{suffix}" if suffix else base_handle
        existing = await fetch_one(
            "SELECT id FROM agent_profiles WHERE user_id = $1 AND handle = $2",
            target_user_id,
            candidate,
        )
        if not existing:
            handle = candidate
            break
        suffix += 1

    name = (profile.get("name") or "Agent").strip()
    if not name.endswith(" (Copy)"):
        name = f"{name} (Copy)"

    def _ensure_json(val, fallback=None):
        if fallback is None:
            fallback = {}
        if val is None:
            return fallback
        if isinstance(val, (dict, list)):
            return val
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, (dict, list)):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
        return fallback

    persona_id_val = None
    persona_id = profile.get("persona_id")
    if persona_id:
        persona_row = await fetch_one(
            "SELECT id, is_builtin, user_id FROM personas WHERE id = $1", persona_id
        )
        if persona_row:
            if persona_row.get("is_builtin") or str(persona_row.get("user_id", "")) == target_user_id:
                persona_id_val = str(persona_row["id"])

    create_payload: Dict[str, Any] = {
        "name": name,
        "handle": handle,
        "description": profile.get("description"),
        "is_active": profile.get("is_active", True),
        "model_preference": profile.get("model_preference"),
        "model_source": profile.get("model_source"),
        "model_provider_type": profile.get("model_provider_type"),
        "max_research_rounds": profile.get("max_research_rounds", 3),
        "system_prompt_additions": profile.get("system_prompt_additions"),
        "knowledge_config": _ensure_json(profile.get("knowledge_config"), {}),
        "default_playbook_id": new_playbook_id,
        "default_run_context": profile.get("default_run_context") or "interactive",
        "default_approval_policy": profile.get("default_approval_policy") or "require",
        "journal_config": _ensure_json(profile.get("journal_config"), {}),
        "team_config": _ensure_json(profile.get("team_config"), {}),
        "watch_config": _ensure_json(profile.get("watch_config"), {}),
        "prompt_history_enabled": profile.get(
            "prompt_history_enabled", profile.get("chat_history_enabled", False)
        ),
        "chat_history_lookback": profile.get("chat_history_lookback", 10),
        "summary_threshold_tokens": profile.get("summary_threshold_tokens", 5000),
        "summary_keep_messages": profile.get("summary_keep_messages", 10),
        "persona_mode": profile.get("persona_mode") or "none",
        "persona_id": persona_id_val,
        "include_user_context": profile.get("include_user_context", False),
        "include_datetime_context": profile.get("include_datetime_context", True),
        "include_user_facts": profile.get("include_user_facts", False),
        "include_facts_categories": _ensure_json(profile.get("include_facts_categories"), []),
        "use_themed_memory": profile.get("use_themed_memory", True),
        "include_agent_memory": profile.get("include_agent_memory", False),
        "auto_routable": False,
        "chat_visible": profile.get("chat_visible", True),
        "category": profile.get("category"),
        "data_workspace_config": _ensure_json(profile.get("data_workspace_config"), {}),
        "allowed_connections": _ensure_json(profile.get("allowed_connections"), []),
    }

    if persona_id_val:
        create_payload["persona_mode"] = "specific"
    elif create_payload["persona_mode"] == "specific":
        create_payload["persona_mode"] = "none"

    created = await agent_factory_service.create_profile(target_user_id, create_payload)
    return created
