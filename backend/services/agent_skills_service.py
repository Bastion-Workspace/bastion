"""
Agent Factory Skills service - CRUD, version management, and built-in skill seeding.

Skills are procedural knowledge injected into LLM steps (profile, playbook, step levels).
"""

import json
import logging
from typing import Any, Dict, List, Optional

from services.builtin_skill_definitions import BUILTIN_SKILL_DEFINITIONS

logger = logging.getLogger(__name__)


def _ensure_json_obj(val: Any, fallback: Any = None) -> Any:
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


def _row_to_skill(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not row:
        return {}
    ownership = row.get("ownership") or "owned"
    if row.get("is_builtin"):
        ownership = "builtin"
    return {
        "id": str(row["id"]),
        "user_id": row.get("user_id"),
        "name": row["name"],
        "slug": row["slug"],
        "description": row.get("description"),
        "category": row.get("category"),
        "procedure": row.get("procedure") or "",
        "required_tools": list(row["required_tools"]) if row.get("required_tools") else [],
        "required_connection_types": list(row["required_connection_types"])
        if row.get("required_connection_types")
        else [],
        "optional_tools": list(row["optional_tools"]) if row.get("optional_tools") else [],
        "inputs_schema": _ensure_json_obj(row.get("inputs_schema"), {}),
        "outputs_schema": _ensure_json_obj(row.get("outputs_schema"), {}),
        "examples": _ensure_json_obj(row.get("examples"), []),
        "tags": list(row["tags"]) if row.get("tags") else [],
        "is_builtin": row.get("is_builtin", False),
        "is_core": row.get("is_core", False),
        "is_locked": row.get("is_locked", False),
        "version": row.get("version", 1),
        "parent_skill_id": str(row["parent_skill_id"]) if row.get("parent_skill_id") else None,
        "improvement_rationale": row.get("improvement_rationale"),
        "depends_on": list(row["depends_on"]) if row.get("depends_on") else [],
        "is_candidate": row.get("is_candidate", False),
        "candidate_weight": row.get("candidate_weight", 0),
        "evidence_metadata": _ensure_json_obj(row.get("evidence_metadata"), {}),
        "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
        "updated_at": row.get("updated_at").isoformat() if row.get("updated_at") else None,
        "ownership": ownership,
        "owner_username": row.get("owner_username"),
        "owner_display_name": row.get("owner_display_name"),
    }


async def list_skills(
    user_id: str,
    category: Optional[str] = None,
    include_builtin: bool = True,
) -> List[Dict[str, Any]]:
    """Return user skills, shared skills, and optionally built-in skills, optionally filtered by category."""
    from services.database_manager.database_helpers import fetch_all

    _shared_clause = """
        OR EXISTS (
            SELECT 1 FROM agent_artifact_shares _sh
            WHERE _sh.artifact_type = 'skill'
              AND _sh.artifact_id = sk.id
              AND _sh.shared_with_user_id = $1
        )
    """
    _ownership_expr = """
        CASE
            WHEN sk.is_builtin = true THEN 'builtin'
            WHEN sk.user_id = $1 THEN 'owned'
            ELSE 'shared'
        END AS ownership,
        u_owner.username AS owner_username,
        u_owner.display_name AS owner_display_name
    """
    _join = "LEFT JOIN users u_owner ON u_owner.user_id = sk.user_id"

    if include_builtin and category:
        rows = await fetch_all(
            f"""
            SELECT sk.*, {_ownership_expr}
            FROM agent_skills sk {_join}
            WHERE (sk.user_id = $1 OR sk.is_builtin = true {_shared_clause})
              AND (sk.category = $2 OR sk.category IS NULL AND $2 IS NULL)
            ORDER BY sk.is_builtin ASC, sk.name ASC
            """,
            user_id,
            category,
        )
    elif include_builtin:
        rows = await fetch_all(
            f"""
            SELECT sk.*, {_ownership_expr}
            FROM agent_skills sk {_join}
            WHERE sk.user_id = $1 OR sk.is_builtin = true {_shared_clause}
            ORDER BY sk.is_builtin ASC, sk.name ASC
            """,
            user_id,
        )
    elif category:
        rows = await fetch_all(
            f"""
            SELECT sk.*, {_ownership_expr}
            FROM agent_skills sk {_join}
            WHERE (sk.user_id = $1 {_shared_clause})
              AND (sk.category = $2 OR sk.category IS NULL AND $2 IS NULL)
            ORDER BY sk.name ASC
            """,
            user_id,
            category,
        )
    else:
        rows = await fetch_all(
            f"""
            SELECT sk.*, {_ownership_expr}
            FROM agent_skills sk {_join}
            WHERE sk.user_id = $1 {_shared_clause}
            ORDER BY sk.name ASC
            """,
            user_id,
        )
    return [_row_to_skill(r) for r in rows]


def _row_to_skill_summary(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Lightweight skill representation for manifest/catalog (no procedure, schemas, or examples)."""
    if not row:
        return {}
    return {
        "id": str(row["id"]),
        "slug": row.get("slug") or "",
        "name": row.get("name") or "",
        "description": row.get("description") or "",
        "category": row.get("category") or "",
        "tags": list(row["tags"]) if row.get("tags") else [],
        "required_connection_types": list(row["required_connection_types"])
        if row.get("required_connection_types")
        else [],
        "is_builtin": row.get("is_builtin", False),
        "is_core": row.get("is_core", False),
    }


async def list_skill_summaries(
    user_id: str,
    include_builtin: bool = True,
) -> List[Dict[str, Any]]:
    """Return lightweight skill summaries (no procedure/schemas) for catalog/manifest injection."""
    from services.database_manager.database_helpers import fetch_all

    cols = "id, slug, name, description, category, tags, required_connection_types, is_builtin, is_core"
    if include_builtin:
        rows = await fetch_all(
            f"""
            SELECT {cols} FROM agent_skills
            WHERE user_id = $1 OR is_builtin = true
            ORDER BY is_builtin ASC, name ASC
            """,
            user_id,
        )
    else:
        rows = await fetch_all(
            f"""
            SELECT {cols} FROM agent_skills
            WHERE user_id = $1
            ORDER BY name ASC
            """,
            user_id,
        )
    return [_row_to_skill_summary(r) for r in rows]


async def get_skill(skill_id: str) -> Optional[Dict[str, Any]]:
    """Get a single skill by ID. Returns None if not found."""
    from services.database_manager.database_helpers import fetch_one

    row = await fetch_one(
        "SELECT * FROM agent_skills WHERE id = $1::uuid",
        skill_id,
    )
    return _row_to_skill(row) if row else None


async def get_skill_by_slug(slug: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get the latest version of a skill by slug. Prefer user's version if user_id given."""
    from services.database_manager.database_helpers import fetch_one

    if user_id:
        row = await fetch_one(
            """
            SELECT * FROM agent_skills
            WHERE slug = $1 AND (user_id = $2 OR is_builtin = true)
            ORDER BY user_id ASC NULLS LAST, version DESC
            LIMIT 1
            """,
            slug,
            user_id,
        )
    else:
        row = await fetch_one(
            """
            SELECT * FROM agent_skills
            WHERE slug = $1 AND is_builtin = true
            ORDER BY version DESC
            LIMIT 1
            """,
            slug,
        )
    return _row_to_skill(row) if row else None


def _is_valid_uuid(s: str) -> bool:
    """Return True if s is a valid UUID string."""
    if not s or not isinstance(s, str):
        return False
    s = s.strip()
    if len(s) != 36:
        return False
    try:
        import uuid
        uuid.UUID(s)
        return True
    except (ValueError, TypeError):
        return False


async def get_skills_by_slugs(slugs: List[str], user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Batch fetch latest version of skills by slug. Prefer user's version if user_id given."""
    if not slugs:
        return []
    results: List[Dict[str, Any]] = []
    seen: set = set()
    for slug in slugs:
        s = (slug or "").strip().lower()
        if not s or s in seen:
            continue
        seen.add(s)
        skill = await get_skill_by_slug(s, user_id=user_id)
        if skill:
            results.append(skill)
    return results


async def get_skills_by_ids(skill_ids: List[str]) -> List[Dict[str, Any]]:
    """Batch fetch skills by IDs. Returns list in same order as requested; missing IDs omitted. Non-UUID entries are skipped."""
    from services.database_manager.database_helpers import fetch_all

    if not skill_ids:
        return []
    seen = set()
    ordered = []
    for sid in skill_ids:
        sid = (sid or "").strip()
        if not sid or sid in seen or not _is_valid_uuid(sid):
            continue
        seen.add(sid)
        ordered.append(sid)
    if not ordered:
        return []
    placeholders = ", ".join(f"${i + 1}::uuid" for i in range(len(ordered)))
    rows = await fetch_all(
        f"SELECT * FROM agent_skills WHERE id IN ({placeholders})",
        *ordered,
    )
    by_id = {str(r["id"]): _row_to_skill(r) for r in rows}
    return [by_id[sid] for sid in ordered if sid in by_id]


async def create_skill(
    user_id: str,
    name: str,
    slug: str,
    procedure: str,
    required_tools: Optional[List[str]] = None,
    optional_tools: Optional[List[str]] = None,
    description: Optional[str] = None,
    category: Optional[str] = None,
    inputs_schema: Optional[Dict[str, Any]] = None,
    outputs_schema: Optional[Dict[str, Any]] = None,
    examples: Optional[List[Any]] = None,
    tags: Optional[List[str]] = None,
    required_connection_types: Optional[List[str]] = None,
    is_core: bool = False,
    depends_on: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Create a user skill. Slug must be unique per user."""
    from services.database_manager.database_helpers import fetch_one

    slug_clean = (slug or "").strip().lower().replace(" ", "-")[:100]
    if not slug_clean:
        raise ValueError("slug is required")
    existing = await fetch_one(
        "SELECT id FROM agent_skills WHERE user_id = $1 AND slug = $2",
        user_id,
        slug_clean,
    )
    if existing:
        raise ValueError("Slug already in use for this user")

    row = await fetch_one(
        """
        INSERT INTO agent_skills (
            user_id, name, slug, description, category, procedure,
            required_tools, required_connection_types, optional_tools, inputs_schema, outputs_schema,
            examples, tags, is_builtin, is_locked, version, is_core, depends_on
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11::jsonb, $12::jsonb, $13, false, false, 1, $14, $15)
        RETURNING *
        """,
        user_id,
        (name or slug_clean).strip()[:255],
        slug_clean,
        (description or "")[:5000] or None,
        (category or "")[:100] or None,
        (procedure or "").strip(),
        list(required_tools or []),
        list(required_connection_types or []),
        list(optional_tools or []),
        json.dumps(inputs_schema or {}),
        json.dumps(outputs_schema or {}),
        json.dumps(examples or []),
        list(tags or []),
        bool(is_core),
        list(depends_on or []),
    )
    skill_dict = _row_to_skill(row)
    try:
        from services.skill_vector_service import embed_skill
        await embed_skill(skill_dict)
    except Exception as e:
        logger.warning("embed_skill after create_skill failed: %s", e)
    return skill_dict


async def update_skill(
    skill_id: str,
    user_id: str,
    procedure: Optional[str] = None,
    required_tools: Optional[List[str]] = None,
    optional_tools: Optional[List[str]] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    category: Optional[str] = None,
    inputs_schema: Optional[Dict[str, Any]] = None,
    outputs_schema: Optional[Dict[str, Any]] = None,
    examples: Optional[List[Any]] = None,
    tags: Optional[List[str]] = None,
    improvement_rationale: Optional[str] = None,
    evidence_metadata: Optional[Dict[str, Any]] = None,
    required_connection_types: Optional[List[str]] = None,
    is_core: Optional[bool] = None,
    depends_on: Optional[List[str]] = None,
    as_candidate: bool = False,
) -> Dict[str, Any]:
    """Update a user skill by creating a new version. Parent skill id set to previous version.
    When as_candidate=True, creates a candidate version (A/B testing) without removing the old version's vector."""
    from services.database_manager.database_helpers import fetch_one, execute

    row = await fetch_one(
        "SELECT * FROM agent_skills WHERE id = $1::uuid AND user_id = $2 AND NOT is_builtin",
        skill_id,
        user_id,
    )
    if not row:
        raise ValueError("Skill not found or not editable")
    prev_id = str(row["id"])
    version = (row.get("version") or 1) + 1
    name_val = (name or row["name"] or "").strip()[:255]
    procedure_val = (procedure if procedure is not None else row.get("procedure") or "").strip()
    required_tools_val = required_tools if required_tools is not None else list(row.get("required_tools") or [])
    optional_tools_val = optional_tools if optional_tools is not None else list(row.get("optional_tools") or [])
    description_val = description if description is not None else row.get("description")
    category_val = category if category is not None else row.get("category")
    inputs_schema_val = inputs_schema if inputs_schema is not None else _ensure_json_obj(row.get("inputs_schema"), {})
    outputs_schema_val = outputs_schema if outputs_schema is not None else _ensure_json_obj(row.get("outputs_schema"), {})
    examples_val = examples if examples is not None else _ensure_json_obj(row.get("examples"), [])
    tags_val = tags if tags is not None else list(row.get("tags") or [])
    improvement_rationale_val = improvement_rationale or None
    evidence_metadata_val = evidence_metadata if evidence_metadata is not None else _ensure_json_obj(row.get("evidence_metadata"), {})
    required_connection_types_val = (
        required_connection_types
        if required_connection_types is not None
        else list(row.get("required_connection_types") or [])
    )
    is_core_val = is_core if is_core is not None else row.get("is_core", False)
    depends_on_val = depends_on if depends_on is not None else list(row.get("depends_on") or [])

    new_row = await fetch_one(
        """
        INSERT INTO agent_skills (
            user_id, name, slug, description, category, procedure,
            required_tools, required_connection_types, optional_tools, inputs_schema, outputs_schema,
            examples, tags, is_builtin, is_locked, version, parent_skill_id,
            improvement_rationale, evidence_metadata, is_core, depends_on,
            is_candidate, candidate_weight
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11::jsonb, $12::jsonb, $13, false, false, $14, $15::uuid, $16, $17::jsonb, $18, $19, $20, $21)
        RETURNING *
        """,
        user_id,
        name_val,
        row["slug"],
        (description_val or "")[:5000] or None,
        (category_val or "")[:100] or None,
        procedure_val,
        required_tools_val,
        required_connection_types_val,
        optional_tools_val,
        json.dumps(inputs_schema_val),
        json.dumps(outputs_schema_val),
        json.dumps(examples_val),
        tags_val,
        version,
        prev_id,
        improvement_rationale_val,
        json.dumps(evidence_metadata_val),
        bool(is_core_val),
        depends_on_val,
        bool(as_candidate),
        10 if as_candidate else 0,
    )
    new_skill = _row_to_skill(new_row)
    try:
        from services.skill_vector_service import embed_skill, remove_skill_vector
        if not as_candidate:
            await remove_skill_vector(prev_id)
        await embed_skill(new_skill)
    except Exception as e:
        logger.warning("skill vector update failed: %s", e)
    return new_skill


async def delete_skill(skill_id: str, user_id: str) -> None:
    """Delete a user skill. Built-in skills cannot be deleted."""
    from services.database_manager.database_helpers import execute

    try:
        from services.skill_vector_service import remove_skill_vector
        await remove_skill_vector(skill_id)
    except Exception as e:
        logger.warning("remove_skill_vector before delete failed: %s", e)

    await execute(
        "DELETE FROM agent_skills WHERE id = $1::uuid AND user_id = $2 AND NOT is_builtin",
        skill_id,
        user_id,
    )


async def get_candidate_for_slug(slug: str, user_id: str) -> Optional[Dict[str, Any]]:
    """Return the candidate version of a skill (if any) for a given slug."""
    from services.database_manager.database_helpers import fetch_one
    row = await fetch_one(
        """
        SELECT * FROM agent_skills
        WHERE slug = $1 AND (user_id = $2 OR is_builtin = true)
          AND is_candidate = true
        ORDER BY version DESC
        LIMIT 1
        """,
        slug, user_id,
    )
    return _row_to_skill(row) if row else None


async def promote_candidate(candidate_id: str, user_id: str) -> Dict[str, Any]:
    """Promote a candidate to active: set is_candidate=false, remove old active version's vector."""
    from services.database_manager.database_helpers import fetch_one, execute

    candidate = await fetch_one(
        """
        SELECT * FROM agent_skills
        WHERE id = $1::uuid AND user_id = $2 AND is_candidate = true
        """,
        candidate_id, user_id,
    )
    if not candidate:
        raise ValueError(f"No candidate found with id '{candidate_id}'")

    slug = candidate["slug"]
    active = await fetch_one(
        """
        SELECT * FROM agent_skills
        WHERE slug = $1 AND user_id = $2 AND is_candidate = false
        ORDER BY version DESC LIMIT 1
        """,
        slug, user_id,
    )

    await execute(
        "UPDATE agent_skills SET is_candidate = false, candidate_weight = 0 WHERE id = $1::uuid",
        str(candidate["id"]),
    )

    if active:
        try:
            from services.skill_vector_service import remove_skill_vector
            await remove_skill_vector(str(active["id"]))
        except Exception as e:
            logger.warning("remove_skill_vector during promotion failed: %s", e)

    return _row_to_skill(await fetch_one(
        "SELECT * FROM agent_skills WHERE id = $1::uuid", str(candidate["id"])
    ))


async def reject_candidate(candidate_id: str, user_id: str) -> None:
    """Reject a candidate: delete it and its vector."""
    from services.database_manager.database_helpers import fetch_one, execute

    candidate = await fetch_one(
        """
        SELECT id FROM agent_skills
        WHERE id = $1::uuid AND user_id = $2 AND is_candidate = true
        """,
        candidate_id, user_id,
    )
    if not candidate:
        raise ValueError(f"No candidate found with id '{candidate_id}'")

    cid = str(candidate["id"])
    try:
        from services.skill_vector_service import remove_skill_vector
        await remove_skill_vector(cid)
    except Exception as e:
        logger.warning("remove_skill_vector during rejection failed: %s", e)

    await execute("DELETE FROM agent_skills WHERE id = $1::uuid", cid)


async def set_candidate_weight(candidate_id: str, weight: int, user_id: str) -> Dict[str, Any]:
    """Adjust candidate traffic split (0-100)."""
    from services.database_manager.database_helpers import fetch_one, execute

    weight = max(0, min(100, weight))
    candidate = await fetch_one(
        """
        SELECT * FROM agent_skills
        WHERE id = $1::uuid AND user_id = $2 AND is_candidate = true
        """,
        candidate_id, user_id,
    )
    if not candidate:
        raise ValueError(f"No candidate found with id '{candidate_id}'")

    await execute(
        "UPDATE agent_skills SET candidate_weight = $1 WHERE id = $2::uuid",
        weight, str(candidate["id"]),
    )
    return _row_to_skill(await fetch_one(
        "SELECT * FROM agent_skills WHERE id = $1::uuid", str(candidate["id"])
    ))


async def list_promotion_recommendations(
    status: str = "pending",
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Return skill promotion/demotion recommendations filtered by status."""
    from services.database_manager.database_helpers import fetch_all

    rows = await fetch_all(
        """
        SELECT r.*, a.name AS skill_name
        FROM skill_promotion_recommendations r
        LEFT JOIN agent_skills a ON a.id = r.skill_id
        WHERE r.status = $1
        ORDER BY r.created_at DESC
        LIMIT $2
        """,
        status, limit,
    )
    out = []
    for r in rows or []:
        item = dict(r)
        for k in ("created_at", "resolved_at"):
            if item.get(k):
                item[k] = item[k].isoformat()
        if isinstance(item.get("evidence"), str):
            import json as _json
            try:
                item["evidence"] = _json.loads(item["evidence"])
            except Exception:
                pass
        out.append(item)
    return out


async def apply_recommendation(rec_id: int, user_id: str) -> Dict[str, Any]:
    """Apply a pending recommendation (promote or demote the skill)."""
    from services.database_manager.database_helpers import fetch_one, execute

    rec = await fetch_one(
        "SELECT * FROM skill_promotion_recommendations WHERE id = $1 AND status = 'pending'",
        rec_id,
    )
    if not rec:
        raise ValueError(f"Recommendation {rec_id} not found or already resolved")

    skill_id = str(rec["skill_id"])
    action = rec["action"]
    new_is_core = action == "promote"

    await execute(
        "UPDATE agent_skills SET is_core = $1 WHERE id = $2::uuid",
        new_is_core, skill_id,
    )
    await execute(
        "UPDATE skill_promotion_recommendations SET status = 'applied', resolved_at = NOW() WHERE id = $1",
        rec_id,
    )

    skill = await fetch_one("SELECT * FROM agent_skills WHERE id = $1::uuid", skill_id)
    return _row_to_skill(skill) if skill else {"id": skill_id, "is_core": new_is_core}


async def dismiss_recommendation(rec_id: int) -> None:
    """Dismiss a pending recommendation without applying it."""
    from services.database_manager.database_helpers import fetch_one, execute

    rec = await fetch_one(
        "SELECT id FROM skill_promotion_recommendations WHERE id = $1 AND status = 'pending'",
        rec_id,
    )
    if not rec:
        raise ValueError(f"Recommendation {rec_id} not found or already resolved")

    await execute(
        "UPDATE skill_promotion_recommendations SET status = 'dismissed', resolved_at = NOW() WHERE id = $1",
        rec_id,
    )


def resolve_skills_for_step(step_skill_ids: Optional[List[str]] = None) -> List[str]:
    """Dedupe step-level skill IDs. Skills are assigned only at step level."""
    seen = set()
    out = []
    for sid in step_skill_ids or []:
        s = (sid or "").strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


async def list_skill_versions(skill_id: str, user_id: str) -> List[Dict[str, Any]]:
    """Return version history for a skill (current first)."""
    from services.database_manager.database_helpers import fetch_all

    rows = await fetch_all(
        """
        SELECT * FROM agent_skills
        WHERE id = $1::uuid AND (user_id = $2 OR is_builtin = true)
        ORDER BY version DESC
        """,
        skill_id,
        user_id,
    )
    if not rows:
        return []
    slug = rows[0].get("slug")
    all_versions = await fetch_all(
        """
        SELECT * FROM agent_skills
        WHERE slug = $1 AND (user_id = $2 OR is_builtin = true)
        ORDER BY version DESC
        """,
        slug,
        user_id,
    )
    return [_row_to_skill(r) for r in all_versions]


async def revert_skill_to_version(skill_id: str, version_id: str, user_id: str) -> Dict[str, Any]:
    """Create a new version with content from a previous version (rollback)."""
    from services.database_manager.database_helpers import fetch_one

    target = await fetch_one(
        "SELECT * FROM agent_skills WHERE id = $1::uuid",
        version_id,
    )
    if not target:
        raise ValueError("Version not found")
    current = await fetch_one(
        "SELECT * FROM agent_skills WHERE id = $1::uuid AND user_id = $2 AND NOT is_builtin",
        skill_id,
        user_id,
    )
    if not current:
        raise ValueError("Skill not found or not editable")
    if target.get("slug") != current.get("slug"):
        raise ValueError("Version does not belong to this skill")
    return await update_skill(
        skill_id,
        user_id,
        procedure=target.get("procedure"),
        required_tools=list(target.get("required_tools") or []),
        required_connection_types=list(target.get("required_connection_types") or []),
        optional_tools=list(target.get("optional_tools") or []),
        name=target.get("name"),
        description=target.get("description"),
        category=target.get("category"),
        inputs_schema=_ensure_json_obj(target.get("inputs_schema"), {}),
        outputs_schema=_ensure_json_obj(target.get("outputs_schema"), {}),
        examples=_ensure_json_obj(target.get("examples"), []),
        tags=list(target.get("tags") or []),
        improvement_rationale="Reverted to version " + str(target.get("version")),
    )


async def seed_builtin_skills() -> None:
    """Upsert built-in skills (idempotent, keyed by slug). Call at backend startup."""
    from services.database_manager.database_helpers import fetch_one, execute

    retired_slugs = [
        "org-todo-management",
        "email-composition",
        "multi-source-research",
        "dictionary",
        "entertainment",
        "image-description",
        "fiction-editing",
        "nonfiction-outline-editing",
        "character-development",
        "rules-editing",
        "style-editing",
        "series-editing",
        "electronics",
        "general-project",
        "podcast-script",
        "proofreading",
        "article-writing",
        "content-analysis",
        "research",
        "knowledge-builder",
        "effective-document-search",
        "thorough-web-search",
        "site-crawl",
        "website-crawler",
        "help",
    ]
    for slug in retired_slugs:
        await execute(
            "DELETE FROM agent_skills WHERE slug = $1 AND is_builtin = true",
            slug,
        )

    builtins = BUILTIN_SKILL_DEFINITIONS
    for s in builtins:
        existing = await fetch_one(
            "SELECT id, procedure FROM agent_skills WHERE slug = $1 AND is_builtin = true",
            s["slug"],
        )
        tags = s.get("tags") or []
        evidence_metadata = s.get("evidence_metadata") or {}
        req_conn = list(s.get("required_connection_types") or [])
        is_core = s.get("is_core", True)
        depends_on = list(s.get("depends_on") or [])
        if existing:
            await execute(
                """
                UPDATE agent_skills SET
                    name = $2, description = $3, category = $4, procedure = $5,
                    required_tools = $6, required_connection_types = $7, optional_tools = $8,
                    tags = $9, evidence_metadata = $10::jsonb, is_core = $11,
                    depends_on = $12, updated_at = NOW()
                WHERE id = $1
                """,
                existing["id"],
                s["name"],
                s["description"],
                s["category"],
                s["procedure"],
                s.get("required_tools", []),
                req_conn,
                s.get("optional_tools", []),
                tags,
                json.dumps(evidence_metadata),
                is_core,
                depends_on,
            )
            logger.debug("Updated built-in skill: %s", s["slug"])
        else:
            await execute(
                """
                INSERT INTO agent_skills (
                    user_id, name, slug, description, category, procedure,
                    required_tools, required_connection_types, optional_tools, tags,
                    evidence_metadata, is_builtin, is_locked, version, is_core, depends_on
                ) VALUES (NULL, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, true, true, 1, $11, $12)
                """,
                s["name"],
                s["slug"],
                s["description"],
                s["category"],
                s["procedure"],
                s.get("required_tools", []),
                req_conn,
                s.get("optional_tools", []),
                tags,
                json.dumps(evidence_metadata),
                is_core,
                depends_on,
            )
            logger.info("Seeded built-in skill: %s", s["slug"])


# ---------------------------------------------------------------------------
# Skill execution metrics
# ---------------------------------------------------------------------------


async def record_skill_execution_events(
    events: List[Dict[str, Any]],
    user_id: str,
    agent_profile_id: Optional[str] = None,
) -> int:
    """Batch-insert skill execution events. Returns count of rows inserted."""
    from utils.shared_db_pool import execute, fetch_one

    if not events:
        return 0
    inserted = 0
    for ev in events:
        skill_id = (ev.get("skill_id") or "").strip()
        skill_slug = (ev.get("skill_slug") or "").strip()
        if not skill_id or not skill_slug:
            continue
        exists = await fetch_one(
            "SELECT 1 FROM agent_skills WHERE id = $1::uuid",
            skill_id,
        )
        if not exists:
            continue
        try:
            await execute(
                """
                INSERT INTO skill_execution_events
                    (skill_id, skill_slug, agent_profile_id, step_name,
                     user_id, discovery_method, tool_calls_made, success, skill_version)
                VALUES ($1::uuid, $2, $3::uuid, $4, $5, $6, $7, $8, $9)
                """,
                skill_id,
                skill_slug,
                agent_profile_id if agent_profile_id else None,
                ev.get("step_name") or "",
                user_id,
                ev.get("discovery_method") or "explicit",
                ev.get("tool_calls_made") or 0,
                ev.get("success"),
                ev.get("skill_version") or 1,
            )
            inserted += 1
        except Exception as exc:
            logger.debug("Skipping skill execution event for %s: %s", skill_slug, exc)
    return inserted


async def get_skill_metrics(skill_id: str) -> Dict[str, Any]:
    """Return aggregated metrics for a single skill from the materialized view."""
    from utils.shared_db_pool import fetch_one

    row = await fetch_one(
        "SELECT * FROM skill_usage_stats WHERE skill_id = $1::uuid",
        skill_id,
    )
    if not row:
        return {
            "skill_id": skill_id,
            "total_uses": 0, "unique_users": 0, "unique_agents": 0,
            "success_rate": None, "avg_execution_ms": None,
            "last_used_at": None, "uses_last_7d": 0, "uses_last_30d": 0,
        }
    return {
        "skill_id": str(row["skill_id"]),
        "skill_slug": row.get("skill_slug") or "",
        "total_uses": row.get("total_uses") or 0,
        "unique_users": row.get("unique_users") or 0,
        "unique_agents": row.get("unique_agents") or 0,
        "success_rate": float(row["success_rate"]) if row.get("success_rate") is not None else None,
        "avg_execution_ms": float(row["avg_execution_ms"]) if row.get("avg_execution_ms") is not None else None,
        "last_used_at": row["last_used_at"].isoformat() if row.get("last_used_at") else None,
        "uses_last_7d": row.get("uses_last_7d") or 0,
        "uses_last_30d": row.get("uses_last_30d") or 0,
    }


async def get_skills_metrics_summary(limit: int = 20) -> Dict[str, Any]:
    """Return top-used and lowest-success-rate skills from the materialized view."""
    from utils.shared_db_pool import fetch_all

    top_used = await fetch_all(
        "SELECT * FROM skill_usage_stats ORDER BY total_uses DESC LIMIT $1",
        limit,
    )
    low_success = await fetch_all(
        """
        SELECT * FROM skill_usage_stats
        WHERE total_uses >= 5
        ORDER BY success_rate ASC NULLS LAST
        LIMIT $1
        """,
        limit,
    )
    def _row_to_stat(row):
        return {
            "skill_id": str(row["skill_id"]),
            "skill_slug": row.get("skill_slug") or "",
            "total_uses": row.get("total_uses") or 0,
            "unique_users": row.get("unique_users") or 0,
            "unique_agents": row.get("unique_agents") or 0,
            "success_rate": float(row["success_rate"]) if row.get("success_rate") is not None else None,
            "avg_execution_ms": float(row["avg_execution_ms"]) if row.get("avg_execution_ms") is not None else None,
            "last_used_at": row["last_used_at"].isoformat() if row.get("last_used_at") else None,
            "uses_last_7d": row.get("uses_last_7d") or 0,
            "uses_last_30d": row.get("uses_last_30d") or 0,
        }
    return {
        "top_used": [_row_to_stat(r) for r in (top_used or [])],
        "low_success_rate": [_row_to_stat(r) for r in (low_success or [])],
    }
