"""
Agent Artifact Sharing service -- user-to-user "use" sharing of agent profiles,
playbooks, and skills with transitive dependency cascading.
"""

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

VALID_ARTIFACT_TYPES = frozenset({"agent_profile", "playbook", "skill"})


def _is_valid_uuid(val: str) -> bool:
    try:
        uuid.UUID(val)
        return True
    except (ValueError, AttributeError):
        return False


async def share_artifact(
    artifact_type: str,
    artifact_id: str,
    owner_user_id: str,
    target_user_id: str,
    *,
    is_transitive: bool = False,
    parent_share_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Create a share grant and cascade transitively to dependencies.
    Returns the share row dict, or None if the artifact doesn't exist.
    """
    from services.database_manager.database_helpers import fetch_one, fetch_all

    if artifact_type not in VALID_ARTIFACT_TYPES:
        raise ValueError(f"Invalid artifact_type: {artifact_type}")
    if not _is_valid_uuid(artifact_id):
        raise ValueError(f"Invalid artifact_id: {artifact_id}")
    if owner_user_id == target_user_id:
        raise ValueError("Cannot share an artifact with yourself")

    table = _artifact_table(artifact_type)
    owner_row = await fetch_one(
        f"SELECT id FROM {table} WHERE id = $1 AND user_id = $2",
        artifact_id,
        owner_user_id,
    )
    if not owner_row:
        return None

    parent_param = uuid.UUID(parent_share_id) if parent_share_id else None
    row = await fetch_one(
        """
        INSERT INTO agent_artifact_shares
            (artifact_type, artifact_id, owner_user_id, shared_with_user_id, is_transitive, parent_share_id)
        VALUES ($1, $2::uuid, $3, $4, $5, $6::uuid)
        ON CONFLICT (artifact_type, artifact_id, shared_with_user_id) DO UPDATE
            SET parent_share_id = COALESCE(EXCLUDED.parent_share_id, agent_artifact_shares.parent_share_id),
                is_transitive = EXCLUDED.is_transitive
        RETURNING *
        """,
        artifact_type,
        artifact_id,
        owner_user_id,
        target_user_id,
        is_transitive,
        parent_param,
    )
    if not row:
        return None

    share_id = str(row["id"])

    if artifact_type == "agent_profile":
        await _cascade_agent_profile(artifact_id, owner_user_id, target_user_id, share_id)
    elif artifact_type == "playbook":
        await _cascade_playbook(artifact_id, owner_user_id, target_user_id, share_id)
    elif artifact_type == "skill":
        await _cascade_skill_deps(artifact_id, owner_user_id, target_user_id, share_id)

    return _row_to_share(row)


async def revoke_share(share_id: str, owner_user_id: str) -> bool:
    """Revoke a share. ON DELETE CASCADE handles transitive children."""
    from services.database_manager.database_helpers import execute

    if not _is_valid_uuid(share_id):
        return False
    result = await execute(
        "DELETE FROM agent_artifact_shares WHERE id = $1 AND owner_user_id = $2",
        share_id,
        owner_user_id,
    )
    return True


async def list_shares_by_owner(owner_user_id: str) -> List[Dict[str, Any]]:
    """List non-transitive shares created by the owner."""
    from services.database_manager.database_helpers import fetch_all

    rows = await fetch_all(
        """
        SELECT s.*, u.username AS shared_with_username, u.display_name AS shared_with_display_name
        FROM agent_artifact_shares s
        JOIN users u ON u.user_id = s.shared_with_user_id
        WHERE s.owner_user_id = $1 AND s.is_transitive = false
        ORDER BY s.created_at DESC
        """,
        owner_user_id,
    )
    return [_row_to_share(r) for r in rows]


async def list_shares_with_user(user_id: str) -> List[Dict[str, Any]]:
    """List non-transitive shares where the user is a recipient."""
    from services.database_manager.database_helpers import fetch_all

    rows = await fetch_all(
        """
        SELECT s.*, u.username AS owner_username, u.display_name AS owner_display_name
        FROM agent_artifact_shares s
        JOIN users u ON u.user_id = s.owner_user_id
        WHERE s.shared_with_user_id = $1 AND s.is_transitive = false
        ORDER BY s.created_at DESC
        """,
        user_id,
    )
    return [_row_to_share(r) for r in rows]


async def list_shares_for_artifact(
    artifact_type: str, artifact_id: str
) -> List[Dict[str, Any]]:
    """List all direct recipients of a specific artifact."""
    from services.database_manager.database_helpers import fetch_all

    rows = await fetch_all(
        """
        SELECT s.*, u.username AS shared_with_username, u.display_name AS shared_with_display_name
        FROM agent_artifact_shares s
        JOIN users u ON u.user_id = s.shared_with_user_id
        WHERE s.artifact_type = $1 AND s.artifact_id = $2 AND s.is_transitive = false
        ORDER BY s.created_at DESC
        """,
        artifact_type,
        artifact_id,
    )
    return [_row_to_share(r) for r in rows]


async def can_user_access_artifact(
    user_id: str, artifact_type: str, artifact_id: str
) -> bool:
    """Check if user owns or has a share grant (direct or transitive) for the artifact."""
    from services.database_manager.database_helpers import fetch_one

    table = _artifact_table(artifact_type)
    owned = await fetch_one(
        f"SELECT id FROM {table} WHERE id = $1 AND user_id = $2",
        artifact_id,
        user_id,
    )
    if owned:
        return True

    shared = await fetch_one(
        """
        SELECT id FROM agent_artifact_shares
        WHERE artifact_type = $1 AND artifact_id = $2 AND shared_with_user_id = $3
        """,
        artifact_type,
        artifact_id,
        user_id,
    )
    return shared is not None


async def refresh_transitive_shares(
    artifact_type: str, artifact_id: str, owner_user_id: str
) -> None:
    """
    Re-sync transitive child shares after a playbook or profile is edited.
    Adds missing skill shares and removes stale ones.
    """
    from services.database_manager.database_helpers import fetch_all, execute

    parent_shares = await fetch_all(
        """
        SELECT id, shared_with_user_id FROM agent_artifact_shares
        WHERE artifact_type = $1 AND artifact_id = $2 AND owner_user_id = $3 AND is_transitive = false
        """,
        artifact_type,
        artifact_id,
        owner_user_id,
    )
    if not parent_shares:
        return

    needed_skill_ids: List[str] = []
    needed_playbook_id: Optional[str] = None

    if artifact_type == "agent_profile":
        needed_playbook_id, needed_skill_ids = await _collect_profile_deps(
            artifact_id, owner_user_id
        )
    elif artifact_type == "playbook":
        needed_skill_ids = await _collect_playbook_skill_ids(artifact_id, owner_user_id)

    for parent in parent_shares:
        target_user_id = parent["shared_with_user_id"]
        parent_sid = str(parent["id"])

        existing_children = await fetch_all(
            "SELECT artifact_type, artifact_id FROM agent_artifact_shares WHERE parent_share_id = $1",
            parent["id"],
        )
        existing_set = {(r["artifact_type"], str(r["artifact_id"])) for r in existing_children}

        needed_set: set = set()
        if needed_playbook_id:
            needed_set.add(("playbook", needed_playbook_id))
        for sid in needed_skill_ids:
            needed_set.add(("skill", sid))

        for atype, aid in needed_set - existing_set:
            await share_artifact(atype, aid, owner_user_id, target_user_id,
                                 is_transitive=True, parent_share_id=parent_sid)

        stale = existing_set - needed_set
        if stale:
            for atype, aid in stale:
                await execute(
                    """
                    DELETE FROM agent_artifact_shares
                    WHERE parent_share_id = $1 AND artifact_type = $2 AND artifact_id = $3
                    """,
                    parent["id"],
                    atype,
                    aid,
                )


def shared_access_predicate(artifact_type: str, table_alias: str, user_param: str) -> str:
    """Return a SQL OR clause for sharing-aware queries."""
    return f"""
        EXISTS (
            SELECT 1 FROM agent_artifact_shares _sh
            WHERE _sh.artifact_type = '{artifact_type}'
              AND _sh.artifact_id = {table_alias}.id
              AND _sh.shared_with_user_id = {user_param}
        )
    """


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _artifact_table(artifact_type: str) -> str:
    return {
        "agent_profile": "agent_profiles",
        "playbook": "custom_playbooks",
        "skill": "agent_skills",
    }[artifact_type]


def _row_to_share(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "id": str(row["id"]),
        "artifact_type": row["artifact_type"],
        "artifact_id": str(row["artifact_id"]),
        "owner_user_id": row["owner_user_id"],
        "shared_with_user_id": row["shared_with_user_id"],
        "is_transitive": row.get("is_transitive", False),
        "parent_share_id": str(row["parent_share_id"]) if row.get("parent_share_id") else None,
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
    }
    if row.get("shared_with_username"):
        out["shared_with_username"] = row["shared_with_username"]
        out["shared_with_display_name"] = row.get("shared_with_display_name")
    if row.get("owner_username"):
        out["owner_username"] = row["owner_username"]
        out["owner_display_name"] = row.get("owner_display_name")
    return out


async def _cascade_agent_profile(
    profile_id: str, owner_user_id: str, target_user_id: str, parent_share_id: str
) -> None:
    from services.database_manager.database_helpers import fetch_one

    profile = await fetch_one(
        "SELECT default_playbook_id FROM agent_profiles WHERE id = $1 AND user_id = $2",
        profile_id,
        owner_user_id,
    )
    if not profile:
        return
    pb_id = profile.get("default_playbook_id")
    if pb_id:
        await share_artifact(
            "playbook", str(pb_id), owner_user_id, target_user_id,
            is_transitive=True, parent_share_id=parent_share_id,
        )


async def _cascade_playbook(
    playbook_id: str, owner_user_id: str, target_user_id: str, parent_share_id: str
) -> None:
    skill_ids = await _collect_playbook_skill_ids(playbook_id, owner_user_id)
    for sid in skill_ids:
        await share_artifact(
            "skill", sid, owner_user_id, target_user_id,
            is_transitive=True, parent_share_id=parent_share_id,
        )


async def _cascade_skill_deps(
    skill_id: str, owner_user_id: str, target_user_id: str, parent_share_id: str
) -> None:
    """Share skills referenced in depends_on (resolved in owner's namespace)."""
    from services.database_manager.database_helpers import fetch_all

    rows = await fetch_all(
        "SELECT depends_on FROM agent_skills WHERE id = $1",
        skill_id,
    )
    if not rows or not rows[0].get("depends_on"):
        return
    slugs = list(rows[0]["depends_on"])
    if not slugs:
        return
    for slug in slugs:
        dep_rows = await fetch_all(
            "SELECT id FROM agent_skills WHERE slug = $1 AND (user_id = $2 OR is_builtin = true) LIMIT 1",
            slug,
            owner_user_id,
        )
        for dep in dep_rows:
            dep_id = str(dep["id"])
            is_builtin_check = await fetch_all(
                "SELECT is_builtin FROM agent_skills WHERE id = $1", dep_id
            )
            if is_builtin_check and is_builtin_check[0].get("is_builtin"):
                continue
            await share_artifact(
                "skill", dep_id, owner_user_id, target_user_id,
                is_transitive=True, parent_share_id=parent_share_id,
            )


async def _collect_playbook_skill_ids(playbook_id: str, owner_user_id: str) -> List[str]:
    """Extract non-builtin skill IDs from a playbook's steps."""
    from services.database_manager.database_helpers import fetch_one, fetch_all

    pb = await fetch_one(
        "SELECT definition FROM custom_playbooks WHERE id = $1 AND user_id = $2",
        playbook_id,
        owner_user_id,
    )
    if not pb:
        return []
    defn = pb.get("definition") or {}
    if isinstance(defn, str):
        try:
            defn = json.loads(defn)
        except (json.JSONDecodeError, TypeError):
            return []
    steps = defn.get("steps") or []
    skill_ids: List[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        for sid in (step.get("skill_ids") or step.get("skills") or []):
            if sid and isinstance(sid, str) and sid not in skill_ids and _is_valid_uuid(sid):
                skill_ids.append(sid)

    if not skill_ids:
        return []

    # Filter out builtins
    placeholders = ", ".join(f"${i + 1}::uuid" for i in range(len(skill_ids)))
    builtin_rows = await fetch_all(
        f"SELECT id FROM agent_skills WHERE id IN ({placeholders}) AND is_builtin = true",
        *skill_ids,
    )
    builtin_set = {str(r["id"]) for r in builtin_rows}
    return [sid for sid in skill_ids if sid not in builtin_set]


async def _collect_profile_deps(
    profile_id: str, owner_user_id: str
) -> tuple:
    """Returns (playbook_id_or_None, [skill_ids])."""
    from services.database_manager.database_helpers import fetch_one

    profile = await fetch_one(
        "SELECT default_playbook_id FROM agent_profiles WHERE id = $1 AND user_id = $2",
        profile_id,
        owner_user_id,
    )
    if not profile:
        return None, []
    pb_id = profile.get("default_playbook_id")
    if not pb_id:
        return None, []
    pb_id_str = str(pb_id)
    skill_ids = await _collect_playbook_skill_ids(pb_id_str, owner_user_id)
    return pb_id_str, skill_ids
