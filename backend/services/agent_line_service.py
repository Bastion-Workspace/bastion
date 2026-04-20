"""
Agent Line service - CRUD and queries for autonomous agent lines and org chart.

Used by the REST API (agent_line_api / agent_factory_api) and gRPC tool handlers.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

GOVERNANCE_MODES = frozenset({"hierarchical", "committee", "round_robin", "consensus"})


def normalize_governance_mode(mode: Optional[str]) -> str:
    m = (mode or "hierarchical").strip().lower()
    return m if m in GOVERNANCE_MODES else "hierarchical"


def _ensure_json_obj(val: Any, fallback: Any = None) -> Any:
    if fallback is None:
        fallback = {}
    if val is None:
        return fallback
    if isinstance(val, (dict, list)):
        return val
    if isinstance(val, str):
        try:
            import json
            parsed = json.loads(val)
            if isinstance(parsed, (dict, list)):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return fallback


def _normalize_pack_entries(raw: Any) -> List[Dict[str, str]]:
    """Convert legacy ["name"] to [{"pack": "name", "mode": "full"}] for backward compatibility."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        return []
    result = []
    for entry in raw:
        if isinstance(entry, dict) and entry.get("pack"):
            mode = (entry.get("mode") or "full").strip().lower()
            result.append({
                "pack": str(entry["pack"]),
                "mode": "read" if mode == "read" else "full",
            })
        elif isinstance(entry, str) and entry.strip():
            result.append({"pack": entry.strip(), "mode": "full"})
    return result


def _row_to_line(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not row:
        return {}
    out = {
        "id": str(row["id"]),
        "line_id": str(row["id"]),
        "user_id": row["user_id"],
        "name": row["name"],
        "description": row.get("description"),
        "mission_statement": row.get("mission_statement"),
        "status": row.get("status", "active"),
        "heartbeat_config": _ensure_json_obj(row.get("heartbeat_config"), {}),
        "governance_policy": _ensure_json_obj(row.get("governance_policy"), {}),
        "governance_mode": normalize_governance_mode(row.get("governance_mode")),
        "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
        "updated_at": row.get("updated_at").isoformat() if row.get("updated_at") else None,
    }
    if "next_beat_at" in row:
        out["next_beat_at"] = row["next_beat_at"].isoformat() if row.get("next_beat_at") else None
    if "last_beat_at" in row:
        out["last_beat_at"] = row["last_beat_at"].isoformat() if row.get("last_beat_at") else None
    if "budget_config" in row:
        out["budget_config"] = _ensure_json_obj(row.get("budget_config"), {})
    if "active_celery_task_id" in row:
        out["active_celery_task_id"] = row.get("active_celery_task_id")
    if "handle" in row:
        out["handle"] = row.get("handle")
    if "team_tool_packs" in row:
        raw = _ensure_json_obj(row.get("team_tool_packs"), [])
        out["team_tool_packs"] = _normalize_pack_entries(raw)
    if "team_skill_ids" in row:
        out["team_skill_ids"] = _ensure_json_obj(row.get("team_skill_ids"), [])
    if "reference_config" in row:
        out["reference_config"] = _ensure_json_obj(row.get("reference_config"), {})
    if "data_workspace_config" in row:
        out["data_workspace_config"] = _ensure_json_obj(row.get("data_workspace_config"), {})
    return out


def normalize_data_workspace_config(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Normalize agent line data_workspace_config: workspaces with access, optional schema injection.
    """
    base = _ensure_json_obj(cfg, {})
    out: Dict[str, Any] = {
        "workspaces": [],
        "auto_inject_schema": bool(base.get("auto_inject_schema", False)),
        "context_instructions": (base.get("context_instructions") or "").strip()[:20000],
    }
    raw_ws = base.get("workspaces")
    if isinstance(raw_ws, list):
        seen: set = set()
        for item in raw_ws:
            if not isinstance(item, dict):
                continue
            wid = item.get("workspace_id")
            if not wid:
                continue
            wid = str(wid).strip()
            if not wid or wid in seen:
                continue
            seen.add(wid)
            acc = (item.get("access") or "read").strip().lower()
            access = "read_write" if acc == "read_write" else "read"
            out["workspaces"].append({"workspace_id": wid, "access": access})
    return out


def normalize_reference_config(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Normalize agent line reference_config: folders/documents lists and access rules.
    Global entries are always read-only; team entries honor declared read or read_write.
    """
    base = _ensure_json_obj(cfg, {})
    _ls = base.get("load_strategy")
    load_strategy = (
        (_ls or "full").strip().lower() if isinstance(_ls, str) else "full"
    )
    out: Dict[str, Any] = {
        "folders": [],
        "documents": [],
        "load_strategy": load_strategy,
    }
    if out["load_strategy"] not in ("full", "metadata_first"):
        out["load_strategy"] = "full"
    for key in ("folders", "documents"):
        raw = base.get(key)
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, dict):
                continue
            entry = dict(item)
            ct = (entry.get("collection_type") or "user").strip().lower()
            if ct not in ("user", "global", "team"):
                ct = "user"
            entry["collection_type"] = ct
            if ct == "global":
                entry["access"] = "read"
            else:
                acc = (entry.get("access") or "read").strip().lower()
                entry["access"] = "read_write" if acc == "read_write" else "read"
            if key == "folders":
                fid = entry.get("folder_id")
                if not fid:
                    continue
                entry["folder_id"] = str(fid).strip()
                entry["name"] = (entry.get("name") or entry["folder_id"])[:500]
                if ct == "team" and entry.get("team_id"):
                    entry["team_id"] = str(entry["team_id"]).strip()
                out["folders"].append(entry)
            else:
                did = entry.get("document_id")
                if not did:
                    continue
                entry["document_id"] = str(did).strip()
                entry["title"] = (entry.get("title") or entry["document_id"])[:500]
                if ct == "team" and entry.get("team_id"):
                    entry["team_id"] = str(entry["team_id"]).strip()
                out["documents"].append(entry)
    return out


def _row_to_membership(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not row:
        return {}
    out = {
        "id": str(row["id"]),
        "line_id": str(row["line_id"]),
        "agent_profile_id": str(row["agent_profile_id"]),
        "role": row.get("role", "worker"),
        "reports_to": str(row["reports_to"]) if row.get("reports_to") else None,
        "hire_approved": row.get("hire_approved", False),
        "hire_approved_at": row.get("hire_approved_at").isoformat() if row.get("hire_approved_at") else None,
        "joined_at": row.get("joined_at").isoformat() if row.get("joined_at") else None,
    }
    if "agent_name" in row:
        out["agent_name"] = row.get("agent_name")
    if "agent_handle" in row:
        out["agent_handle"] = row.get("agent_handle")
    if "agent_description" in row:
        out["agent_description"] = row.get("agent_description")
    if "color" in row and row.get("color"):
        out["color"] = row.get("color")
    if "additional_tools" in row:
        out["additional_tools"] = _ensure_json_obj(row.get("additional_tools"), [])
    return out


async def create_line(
    user_id: str,
    name: str,
    description: Optional[str] = None,
    mission_statement: Optional[str] = None,
    status: str = "active",
    heartbeat_config: Optional[Dict[str, Any]] = None,
    governance_policy: Optional[Dict[str, Any]] = None,
    governance_mode: Optional[str] = None,
    reference_config: Optional[Dict[str, Any]] = None,
    data_workspace_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a new agent team."""
    from services.database_manager.database_helpers import fetch_one, execute

    ref = normalize_reference_config(reference_config)
    dw = normalize_data_workspace_config(data_workspace_config)
    gmode = normalize_governance_mode(governance_mode)
    await execute(
        """
        INSERT INTO agent_lines (user_id, name, description, mission_statement, status, heartbeat_config, governance_policy, governance_mode, reference_config, data_workspace_config)
        VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8, $9::jsonb, $10::jsonb)
        """,
        user_id,
        name,
        description or "",
        mission_statement or "",
        status,
        json.dumps(_ensure_json_obj(heartbeat_config)),
        json.dumps(_ensure_json_obj(governance_policy)),
        gmode,
        json.dumps(ref),
        json.dumps(dw),
    )
    row = await fetch_one(
        "SELECT * FROM agent_lines WHERE user_id = $1 AND name = $2 ORDER BY created_at DESC LIMIT 1",
        user_id,
        name,
    )
    line_id = str(row["id"])
    await refresh_line_next_beat_at(line_id)
    row2 = await fetch_one("SELECT * FROM agent_lines WHERE id = $1 AND user_id = $2", line_id, user_id)
    return _row_to_line(row2)


async def refresh_line_next_beat_at(line_id: str) -> None:
    """Set next_beat_at from heartbeat_config (UTC). Clears when heartbeat disabled or no periodic schedule."""
    from services.database_manager.database_helpers import fetch_one, execute
    from services.celery_tasks.team_heartbeat_utils import _heartbeat_enabled
    from services.line_heartbeat_schedule import compute_next_beat_at

    row = await fetch_one("SELECT id, heartbeat_config FROM agent_lines WHERE id = $1", line_id)
    if not row:
        return
    cfg = _ensure_json_obj(row.get("heartbeat_config"), {})
    next_at = None
    if _heartbeat_enabled(cfg):
        next_at = compute_next_beat_at(cfg)
    await execute(
        "UPDATE agent_lines SET next_beat_at = $1, updated_at = NOW() WHERE id = $2",
        next_at,
        line_id,
    )


async def update_line(
    line_id: str,
    user_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    mission_statement: Optional[str] = None,
    status: Optional[str] = None,
    heartbeat_config: Optional[Dict[str, Any]] = None,
    governance_policy: Optional[Dict[str, Any]] = None,
    governance_mode: Optional[str] = None,
    budget_config: Optional[Dict[str, Any]] = None,
    handle: Optional[str] = None,
    team_tool_packs: Optional[List[Any]] = None,
    team_skill_ids: Optional[List[str]] = None,
    reference_config: Optional[Dict[str, Any]] = None,
    data_workspace_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Update an existing team. Only provided fields are updated."""
    from services.database_manager.database_helpers import fetch_one, execute
    from services.celery_tasks.team_heartbeat_utils import _heartbeat_enabled

    existing = await fetch_one("SELECT * FROM agent_lines WHERE id = $1 AND user_id = $2", line_id, user_id)
    if not existing:
        raise ValueError("Team not found")

    forced_status: Optional[str] = None
    heartbeat_config_merged: Optional[Dict[str, Any]] = None
    if heartbeat_config is not None:
        old_cfg = _ensure_json_obj(existing.get("heartbeat_config"), {})
        new_cfg = _ensure_json_obj(heartbeat_config, {})
        heartbeat_config_merged = new_cfg
        if _heartbeat_enabled(old_cfg) and not _heartbeat_enabled(heartbeat_config_merged):
            task_id = existing.get("active_celery_task_id")
            if task_id:
                try:
                    from services.celery_app import celery_app

                    celery_app.control.revoke(str(task_id), terminate=True)
                    logger.info("Revoked in-flight heartbeat (heartbeat disabled via update): %s", task_id)
                except Exception as e:
                    logger.warning("Failed to revoke Celery task %s: %s", task_id, e)
            await set_line_active_celery_task_id(line_id, None)
            forced_status = "paused"
        elif not _heartbeat_enabled(old_cfg) and _heartbeat_enabled(heartbeat_config_merged):
            forced_status = "active"

    updates = ["updated_at = NOW()"]
    args = []
    pos = 1
    if handle is not None:
        updates.append(f"handle = ${pos}")
        args.append(handle.strip() if isinstance(handle, str) and handle.strip() else None)
        pos += 1
    if name is not None:
        updates.append(f"name = ${pos}")
        args.append(name)
        pos += 1
    if description is not None:
        updates.append(f"description = ${pos}")
        args.append(description)
        pos += 1
    if mission_statement is not None:
        updates.append(f"mission_statement = ${pos}")
        args.append(mission_statement)
        pos += 1
    status_to_apply = forced_status if forced_status is not None else status
    if status_to_apply is not None:
        updates.append(f"status = ${pos}")
        args.append(status_to_apply)
        pos += 1
    if heartbeat_config is not None:
        updates.append(f"heartbeat_config = ${pos}::jsonb")
        args.append(json.dumps(_ensure_json_obj(heartbeat_config_merged)))
        pos += 1
    if governance_policy is not None:
        updates.append(f"governance_policy = ${pos}::jsonb")
        args.append(json.dumps(_ensure_json_obj(governance_policy)))
        pos += 1
    if governance_mode is not None:
        updates.append(f"governance_mode = ${pos}")
        args.append(normalize_governance_mode(governance_mode))
        pos += 1
    if budget_config is not None:
        updates.append(f"budget_config = ${pos}::jsonb")
        args.append(json.dumps(_ensure_json_obj(budget_config)))
        pos += 1
    if team_tool_packs is not None:
        updates.append(f"team_tool_packs = ${pos}::jsonb")
        args.append(json.dumps(_normalize_pack_entries(_ensure_json_obj(team_tool_packs, []))))
        pos += 1
    if team_skill_ids is not None:
        updates.append(f"team_skill_ids = ${pos}::jsonb")
        args.append(json.dumps(_ensure_json_obj(team_skill_ids, [])))
        pos += 1
    if reference_config is not None:
        updates.append(f"reference_config = ${pos}::jsonb")
        args.append(json.dumps(normalize_reference_config(reference_config)))
        pos += 1
    if data_workspace_config is not None:
        updates.append(f"data_workspace_config = ${pos}::jsonb")
        args.append(json.dumps(normalize_data_workspace_config(data_workspace_config)))
        pos += 1

    if len(args) == 0:
        return _row_to_line(existing)

    args.append(line_id)
    await execute(
        f"UPDATE agent_lines SET {', '.join(updates)} WHERE id = ${pos} AND user_id = ${pos + 1}",
        *args,
        user_id,
    )
    if heartbeat_config is not None:
        await refresh_line_next_beat_at(line_id)
    row = await fetch_one("SELECT * FROM agent_lines WHERE id = $1", line_id)
    return _row_to_line(row)


async def set_line_active_celery_task_id(line_id: str, task_id: Optional[str]) -> None:
    """Set or clear the active Celery task ID for a team (used for heartbeat revocation)."""
    from services.database_manager.database_helpers import execute

    await execute(
        "UPDATE agent_lines SET active_celery_task_id = $1, updated_at = NOW() WHERE id = $2",
        task_id,
        line_id,
    )


async def check_line_budget(line_id: str, user_id: str) -> tuple[bool, bool]:
    """
    Check if team is within budget. Returns (allowed, over_limit).
    Uses team budget_config (monthly_limit_usd, enforce_hard_limit) and aggregated member spend.
    """
    from services.database_manager.database_helpers import fetch_one

    row = await fetch_one(
        "SELECT budget_config FROM agent_lines WHERE id = $1 AND user_id = $2",
        line_id,
        user_id,
    )
    if not row:
        return True, False
    cfg = _ensure_json_obj(row.get("budget_config"), {})
    limit = cfg.get("monthly_limit_usd")
    if limit is None:
        return True, False
    try:
        limit = float(limit)
    except (TypeError, ValueError):
        return True, False
    if limit <= 0:
        return True, False
    budget = await get_line_budget_summary(line_id, user_id)
    spend = float(budget.get("total_current_period_spend_usd") or 0)
    enforce = cfg.get("enforce_hard_limit", True)
    if enforce and spend >= limit:
        return False, True
    return True, False


async def delete_line(line_id: str, user_id: str) -> None:
    """Delete a team and all its memberships (CASCADE)."""
    from services.database_manager.database_helpers import execute

    await execute("DELETE FROM agent_lines WHERE id = $1 AND user_id = $2", line_id, user_id)


async def list_lines(user_id: str) -> List[Dict[str, Any]]:
    """List all teams for the user with member counts."""
    from services.database_manager.database_helpers import fetch_all

    rows = await fetch_all(
        """
        SELECT t.*, COUNT(m.id)::int AS member_count
        FROM agent_lines t
        LEFT JOIN agent_line_memberships m ON m.line_id = t.id
        WHERE t.user_id = $1
        GROUP BY t.id
        ORDER BY t.updated_at DESC
        """,
        user_id,
    )
    result = []
    for r in rows:
        team = _row_to_line(r)
        team["member_count"] = r.get("member_count", 0)
        result.append(team)
    return result


async def get_line(line_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """Get a team by id with full membership list (with agent names/handles)."""
    from services.database_manager.database_helpers import fetch_one, fetch_all

    row = await fetch_one("SELECT * FROM agent_lines WHERE id = $1 AND user_id = $2", line_id, user_id)
    if not row:
        return None
    team = _row_to_line(row)
    members = await fetch_all(
        """
        SELECT m.*, p.name AS agent_name, p.handle AS agent_handle,
               p.description AS agent_description
        FROM agent_line_memberships m
        JOIN agent_profiles p ON p.id = m.agent_profile_id
        WHERE m.line_id = $1
        ORDER BY m.joined_at
        """,
        line_id,
    )
    team["members"] = [_row_to_membership(r) for r in members]
    return team


AGENT_COLOR_PALETTE = [
    "#1976d2", "#00897b", "#43a047", "#7b1fa2", "#e65100",
    "#3949ab", "#d81b60", "#00838f", "#f57c00", "#c62828",
]


async def add_member(
    line_id: str,
    user_id: str,
    agent_profile_id: str,
    role: str = "worker",
    reports_to: Optional[str] = None,
) -> Dict[str, Any]:
    """Add an agent to a team. Agent must belong to the same user. Auto-assigns color from palette."""
    from services.database_manager.database_helpers import fetch_one, execute, fetch_all

    team = await fetch_one("SELECT id FROM agent_lines WHERE id = $1 AND user_id = $2", line_id, user_id)
    if not team:
        raise ValueError("Team not found")
    profile = await fetch_one(
        "SELECT id FROM agent_profiles WHERE id = $1 AND user_id = $2",
        agent_profile_id,
        user_id,
    )
    if not profile:
        raise ValueError("Agent profile not found")
    existing = await fetch_one(
        "SELECT id, color FROM agent_line_memberships WHERE line_id = $1 AND agent_profile_id = $2",
        line_id,
        agent_profile_id,
    )
    if existing:
        color = existing.get("color")
        if not color:
            count = await fetch_one(
                "SELECT COUNT(*)::int AS c FROM agent_line_memberships WHERE line_id = $1", line_id
            )
            color = AGENT_COLOR_PALETTE[(count.get("c", 0) - 1) % len(AGENT_COLOR_PALETTE)]
            await execute(
                "UPDATE agent_line_memberships SET color = $1, role = $2, reports_to = $3 WHERE id = $4",
                color, role, reports_to, existing["id"],
            )
        else:
            await execute(
                "UPDATE agent_line_memberships SET role = $1, reports_to = $2::uuid WHERE id = $3",
                role, reports_to, existing["id"],
            )
    else:
        count = await fetch_one(
            "SELECT COUNT(*)::int AS c FROM agent_line_memberships WHERE line_id = $1", line_id
        )
        color = AGENT_COLOR_PALETTE[(count.get("c", 0)) % len(AGENT_COLOR_PALETTE)]
        await execute(
            """
            INSERT INTO agent_line_memberships (line_id, agent_profile_id, role, reports_to, color)
            VALUES ($1, $2, $3, $4::uuid, $5)
            """,
            line_id,
            agent_profile_id,
            role,
            reports_to,
            color,
        )
    row = await fetch_one(
        """
        SELECT m.*, p.name AS agent_name, p.handle AS agent_handle
        FROM agent_line_memberships m
        JOIN agent_profiles p ON p.id = m.agent_profile_id
        WHERE m.line_id = $1 AND m.agent_profile_id = $2
        """,
        line_id,
        agent_profile_id,
    )
    return _row_to_membership(row)


async def remove_member(line_id: str, user_id: str, agent_profile_id: str) -> None:
    """Remove an agent from a team."""
    from services.database_manager.database_helpers import execute, fetch_one

    team = await fetch_one("SELECT id FROM agent_lines WHERE id = $1 AND user_id = $2", line_id, user_id)
    if not team:
        raise ValueError("Team not found")
    await execute(
        "DELETE FROM agent_line_memberships WHERE line_id = $1 AND agent_profile_id = $2",
        line_id,
        agent_profile_id,
    )


async def remove_member_by_membership_id(line_id: str, user_id: str, membership_id: str) -> None:
    """Remove a member by membership id."""
    from services.database_manager.database_helpers import execute, fetch_one

    row = await fetch_one(
        """
        SELECT m.agent_profile_id FROM agent_line_memberships m
        JOIN agent_lines t ON t.id = m.line_id
        WHERE m.id = $1 AND m.line_id = $2 AND t.user_id = $3
        """,
        membership_id,
        line_id,
        user_id,
    )
    if not row:
        raise ValueError("Membership not found")
    await execute("DELETE FROM agent_line_memberships WHERE id = $1", membership_id)


async def update_member(
    line_id: str,
    user_id: str,
    membership_id: str,
    role: Optional[str] = None,
    reports_to: Optional[str] = None,
    additional_tools: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Update a membership's role, reports_to, or additional_tools."""
    from services.database_manager.database_helpers import fetch_one, execute

    existing = await fetch_one(
        """
        SELECT m.* FROM agent_line_memberships m
        JOIN agent_lines t ON t.id = m.line_id
        WHERE m.id = $1 AND m.line_id = $2 AND t.user_id = $3
        """,
        membership_id,
        line_id,
        user_id,
    )
    if not existing:
        raise ValueError("Membership not found")
    updates = []
    args = []
    pos = 1
    if role is not None:
        updates.append(f"role = ${pos}")
        args.append(role)
        pos += 1
    if reports_to is not None:
        updates.append(f"reports_to = ${pos}::uuid")
        args.append(reports_to)
        pos += 1
    if additional_tools is not None:
        updates.append(f"additional_tools = ${pos}::jsonb")
        args.append(json.dumps(_ensure_json_obj(additional_tools, [])))
        pos += 1
    if not updates:
        return _row_to_membership(existing)
    args.append(membership_id)
    await execute(
        f"UPDATE agent_line_memberships SET {', '.join(updates)} WHERE id = ${pos}",
        *args,
    )
    row = await fetch_one(
        """
        SELECT m.*, p.name AS agent_name, p.handle AS agent_handle
        FROM agent_line_memberships m
        JOIN agent_profiles p ON p.id = m.agent_profile_id
        WHERE m.id = $1
        """,
        membership_id,
    )
    return _row_to_membership(row)


def _build_org_chart_tree(members: List[Dict[str, Any]], parent_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Build a tree of members by reports_to. parent_id None = roots."""
    nodes = []
    for m in members:
        if (m.get("reports_to") or None) == parent_id:
            child_list = _build_org_chart_tree(members, m["id"])
            node = {
                "id": m["id"],
                "membership_id": m["id"],
                "agent_profile_id": m["agent_profile_id"],
                "agent_name": m.get("agent_name"),
                "agent_handle": m.get("agent_handle"),
                "agent_description": m.get("agent_description"),
                "role": m.get("role", "worker"),
                "reports_to": m.get("reports_to"),
                "color": m.get("color"),
                "children": child_list,
            }
            nodes.append(node)
    return nodes


async def get_org_chart(line_id: str, user_id: str) -> List[Dict[str, Any]]:
    """Return org chart as a tree (roots = no reports_to). Each node has id, agent info, role, children."""
    from services.database_manager.database_helpers import fetch_all

    team = await get_line(line_id, user_id)
    if not team:
        return []
    members = team.get("members", [])
    return _build_org_chart_tree(members, None)


async def get_line_budget_summary(line_id: str, user_id: str) -> Dict[str, Any]:
    """Aggregate budget (limit and spend) across all agents in the team."""
    from services.database_manager.database_helpers import fetch_all, fetch_one

    team = await fetch_one("SELECT id FROM agent_lines WHERE id = $1 AND user_id = $2", line_id, user_id)
    if not team:
        raise ValueError("Team not found")
    rows = await fetch_all(
        """
        SELECT b.agent_profile_id, p.name AS agent_name, p.handle AS agent_handle,
               b.monthly_limit_usd, b.current_period_spend_usd, b.current_period_start
        FROM agent_line_memberships m
        JOIN agent_profiles p ON p.id = m.agent_profile_id
        LEFT JOIN agent_budgets b ON b.agent_profile_id = m.agent_profile_id
        WHERE m.line_id = $1
        """,
        line_id,
    )
    total_limit = 0
    total_spend = 0
    per_agent = []
    for r in rows:
        limit = float(r["monthly_limit_usd"]) if r.get("monthly_limit_usd") is not None else None
        spend = float(r.get("current_period_spend_usd") or 0)
        if limit is not None:
            total_limit += limit
        total_spend += spend
        per_agent.append({
            "agent_profile_id": str(r["agent_profile_id"]),
            "agent_name": r.get("agent_name"),
            "agent_handle": r.get("agent_handle"),
            "monthly_limit_usd": limit,
            "current_period_spend_usd": spend,
            "current_period_start": r.get("current_period_start").isoformat() if r.get("current_period_start") else None,
        })
    return {
        "line_id": line_id,
        "total_monthly_limit_usd": total_limit if total_limit else None,
        "total_current_period_spend_usd": total_spend,
        "per_agent": per_agent,
    }


async def get_line_dispatch_mode(line_id: str, user_id: str) -> str:
    """Return normalized governance_mode for the line."""
    team = await get_line(line_id, user_id)
    if not team:
        return "hierarchical"
    return normalize_governance_mode(team.get("governance_mode"))


async def advance_round_robin_leader(line_id: str, user_id: str) -> None:
    """After a round-robin heartbeat cycle, advance current_leader_idx for the next run."""
    team = await get_line(line_id, user_id)
    if not team or normalize_governance_mode(team.get("governance_mode")) != "round_robin":
        return
    gov = _ensure_json_obj(team.get("governance_policy"), {})
    order = gov.get("rotation_order") or []
    if not isinstance(order, list) or len(order) == 0:
        return
    idx = int(gov.get("current_leader_idx") or 0)
    gov["current_leader_idx"] = (idx + 1) % len(order)
    await update_line(line_id, user_id, governance_policy=gov)


async def get_heartbeat_agents(line_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """
    Build a dispatch plan for team heartbeat / line chat primary agent.

    Returns dict with: mode, user_id, line_id, leader_agent_profile_id, participants (profile ids),
    workers_dispatched_after, governance_policy snapshot, and mode-specific fields.
    """
    team = await get_line(line_id, user_id)
    if not team:
        return None
    uid = str(team["user_id"])
    mode = normalize_governance_mode(team.get("governance_mode"))
    members: List[Dict[str, Any]] = team.get("members") or []
    profile_ids = [str(m["agent_profile_id"]) for m in members]
    if not profile_ids:
        return None
    gov = _ensure_json_obj(team.get("governance_policy"), {})

    base: Dict[str, Any] = {
        "mode": mode,
        "user_id": uid,
        "line_id": line_id,
        "workers_dispatched_after": True,
        "members": members,
        "governance_policy": gov,
        "chair_agent_id": None,
        "quorum_count": None,
        "quorum_pct": 60,
        "tiebreaker_agent_id": None,
        "rotation_cycle_index": None,
        "vote_timeout_seconds": 300,
    }

    if mode == "hierarchical":
        ceo = next((m for m in members if not m.get("reports_to")), None)
        if not ceo:
            ceo = members[0]
        base["leader_agent_profile_id"] = str(ceo["agent_profile_id"])
        base["participants"] = [base["leader_agent_profile_id"]]
        return base

    if mode == "committee":
        chair = str(gov.get("chair_agent_id") or "").strip()
        if chair and chair not in profile_ids:
            chair = ""
        lead = chair if chair else profile_ids[0]
        q_raw = gov.get("quorum_count")
        try:
            q_count = int(q_raw) if q_raw is not None else None
        except (TypeError, ValueError):
            q_count = None
        base["leader_agent_profile_id"] = lead
        base["participants"] = list(profile_ids)
        base["chair_agent_id"] = chair if chair else None
        base["quorum_count"] = q_count
        return base

    if mode == "round_robin":
        mid_by_id = {str(m["id"]): m for m in members}
        order = [str(x) for x in (gov.get("rotation_order") or [])]
        valid_order = [x for x in order if x in mid_by_id]
        if not valid_order:
            valid_order = [str(m["id"]) for m in members]
            gov = dict(gov)
            gov["rotation_order"] = valid_order
            gov["current_leader_idx"] = int(gov.get("current_leader_idx") or 0)
            await update_line(line_id, user_id, governance_policy=gov)
            team = await get_line(line_id, user_id) or team
            members = team.get("members") or []
            mid_by_id = {str(m["id"]): m for m in members}
            gov = _ensure_json_obj(team.get("governance_policy"), {})
            valid_order = [str(x) for x in (gov.get("rotation_order") or [])]
        idx = int(gov.get("current_leader_idx") or 0) % len(valid_order)
        mem = mid_by_id.get(valid_order[idx])
        if not mem:
            return None
        base["leader_agent_profile_id"] = str(mem["agent_profile_id"])
        base["participants"] = [str(mid_by_id[mid]["agent_profile_id"]) for mid in valid_order if mid in mid_by_id]
        base["rotation_cycle_index"] = idx
        return base

    if mode == "consensus":
        try:
            qpct = int(gov.get("quorum_pct") or 60)
        except (TypeError, ValueError):
            qpct = 60
        qpct = max(1, min(100, qpct))
        tb = str(gov.get("tiebreaker_agent_id") or "").strip()
        if tb and tb not in profile_ids:
            tb = ""
        lead = tb if tb else profile_ids[0]
        base["leader_agent_profile_id"] = lead
        base["participants"] = list(profile_ids)
        base["quorum_pct"] = qpct
        base["tiebreaker_agent_id"] = tb if tb else None
        try:
            base["vote_timeout_seconds"] = int(gov.get("vote_timeout_seconds") or 300)
        except (TypeError, ValueError):
            base["vote_timeout_seconds"] = 300
        base["workers_dispatched_after"] = False
        return base

    base["leader_agent_profile_id"] = profile_ids[0]
    base["participants"] = list(profile_ids)
    return base


async def get_ceo_membership_id(line_id: str) -> Optional[str]:
    """Return the membership id of the team's CEO (root of org chart). Used by heartbeat. One root only."""
    from services.database_manager.database_helpers import fetch_one

    row = await fetch_one(
        """
        SELECT id FROM agent_line_memberships
        WHERE line_id = $1 AND reports_to IS NULL
        ORDER BY joined_at ASC
        LIMIT 1
        """,
        line_id,
    )
    return str(row["id"]) if row else None


async def get_ceo_agent_for_heartbeat(line_id: str) -> Optional[Dict[str, Any]]:
    """Return primary line leader agent_profile_id and user_id (governance-aware). None if no members."""
    from services.database_manager.database_helpers import fetch_one

    row = await fetch_one("SELECT user_id FROM agent_lines WHERE id = $1", line_id)
    if not row:
        return None
    plan = await get_heartbeat_agents(line_id, str(row["user_id"]))
    if not plan or not plan.get("leader_agent_profile_id"):
        return None
    return {"agent_profile_id": str(plan["leader_agent_profile_id"]), "user_id": row["user_id"]}


async def get_worker_agents_with_pending_tasks(
    line_id: str, user_id: str
) -> List[Dict[str, Any]]:
    """Non-root agents who have at least one task in 'assigned' and none in 'in_progress'.
    Avoids dispatching an agent who is already working (task in progress)."""
    from services.database_manager.database_helpers import fetch_all, fetch_one

    team = await fetch_one("SELECT id FROM agent_lines WHERE id = $1 AND user_id = $2", line_id, user_id)
    if not team:
        return []
    rows = await fetch_all(
        """
        SELECT DISTINCT m.agent_profile_id, p.name AS agent_name, p.handle AS agent_handle,
               p.description AS agent_description,
               rm.agent_profile_id AS reports_to_agent_id,
               rp.name AS reports_to_agent_name,
               rp.description AS reports_to_agent_description
        FROM agent_line_memberships m
        JOIN agent_profiles p ON p.id = m.agent_profile_id
        JOIN agent_tasks t ON t.line_id = m.line_id
            AND t.assigned_agent_id = m.agent_profile_id
            AND t.status = 'assigned'
        LEFT JOIN agent_line_memberships rm ON rm.id = m.reports_to
        LEFT JOIN agent_profiles rp ON rp.id = rm.agent_profile_id
        WHERE m.line_id = $1 AND m.reports_to IS NOT NULL
        AND NOT EXISTS (
            SELECT 1 FROM agent_tasks t2
            WHERE t2.line_id = m.line_id
              AND t2.assigned_agent_id = m.agent_profile_id
              AND t2.status = 'in_progress'
        )
        """,
        line_id,
    )
    return [
        {
            "agent_profile_id": str(r["agent_profile_id"]),
            "agent_name": r.get("agent_name"),
            "agent_handle": r.get("agent_handle"),
            "agent_description": r.get("agent_description"),
            "reports_to_agent_id": str(r["reports_to_agent_id"]) if r.get("reports_to_agent_id") else None,
            "reports_to_agent_name": r.get("reports_to_agent_name"),
            "reports_to_agent_description": r.get("reports_to_agent_description"),
        }
        for r in rows
    ]


async def apply_autonomous_heartbeat_run_quota(line_id: str, user_id: str) -> None:
    """After a successful scheduled heartbeat: increment autonomous_runs_completed; disable at max_autonomous_runs."""
    from services.database_manager.database_helpers import fetch_one, execute
    from services.celery_tasks.team_heartbeat_utils import _send_team_notification

    row = await fetch_one(
        "SELECT id, name, heartbeat_config FROM agent_lines WHERE id = $1 AND user_id = $2",
        line_id,
        user_id,
    )
    if not row:
        return
    cfg = _ensure_json_obj(row.get("heartbeat_config"), {})
    raw_max = cfg.get("max_autonomous_runs")
    if raw_max is None or raw_max == "":
        return
    try:
        max_n = int(raw_max)
    except (TypeError, ValueError):
        return
    if max_n <= 0:
        return
    try:
        done = int(cfg.get("autonomous_runs_completed") or 0)
    except (TypeError, ValueError):
        done = 0
    done += 1
    cfg["autonomous_runs_completed"] = done
    disabled = False
    if done >= max_n:
        cfg["enabled"] = False
        disabled = True
    await update_line(line_id, user_id, heartbeat_config=cfg)
    if disabled:
        await execute("UPDATE agent_lines SET next_beat_at = NULL WHERE id = $1", line_id)
        await _send_team_notification(
            user_id,
            line_id,
            row.get("name") or "Line",
            "heartbeat_run_limit_reached",
            message=f"Autonomous heartbeat stopped after {max_n} run(s). Re-enable or raise the limit in line settings.",
        )


async def update_line_beat_timestamps(
    line_id: str, last_beat_at: Optional[datetime] = None, next_beat_at: Optional[datetime] = None
) -> None:
    """Update last_beat_at and/or next_beat_at for a team (heartbeat scheduling)."""
    from services.database_manager.database_helpers import execute

    updates = []
    args = []
    pos = 1
    if last_beat_at is not None:
        updates.append(f"last_beat_at = ${pos}")
        args.append(last_beat_at)
        pos += 1
    if next_beat_at is not None:
        updates.append(f"next_beat_at = ${pos}")
        args.append(next_beat_at)
        pos += 1
    if not updates:
        return
    args.append(line_id)
    await execute(f"UPDATE agent_lines SET {', '.join(updates)} WHERE id = ${pos}", *args)


async def get_line_chat_context(line_id: str, user_id: str) -> str:
    """Build a compact text summary of the team for chat context (e.g. when user asks @team-handle what's new)."""
    team = await get_line(line_id, user_id)
    if not team:
        return "Team not found."
    from services import agent_goal_service, agent_task_service, agent_message_service

    name = team.get("name", "Team")
    parts = [f"Team: {name}", ""]

    try:
        budget = await get_line_budget_summary(line_id, user_id)
        spend = budget.get("total_current_period_spend_usd")
        limit = budget.get("total_monthly_limit_usd")
        if spend is not None:
            parts.append(f"Budget this period: ${float(spend):.2f}" + (f" / ${float(limit):.2f}" if limit else "") + " USD")
            parts.append("")
    except Exception:
        pass

    try:
        goals = await agent_goal_service.get_goal_tree(line_id, user_id)

        def count_status(nodes, status):
            n = 0
            for g in nodes or []:
                if g.get("status") == status:
                    n += 1
                n += count_status(g.get("children", []), status)
            return n

        active = count_status(goals, "active")
        completed = count_status(goals, "completed")
        parts.append(f"Goals: {active} active, {completed} completed")
        parts.append("")
    except Exception:
        parts.append("Goals: (unavailable)")
        parts.append("")

    try:
        tasks = await agent_task_service.list_line_tasks(line_id, user_id)
        todo = sum(1 for t in tasks if t.get("status") == "todo")
        in_progress = sum(1 for t in tasks if t.get("status") == "in_progress")
        done = sum(1 for t in tasks if t.get("status") == "done")
        parts.append(f"Tasks: {todo} todo, {in_progress} in progress, {done} done")
        parts.append("")
    except Exception:
        parts.append("Tasks: (unavailable)")
        parts.append("")

    try:
        summary = await agent_message_service.get_line_timeline_summary(line_id, user_id)
        msg_today = summary.get("message_count_today", 0)
        last_at = summary.get("last_activity_at")
        parts.append(f"Timeline: {msg_today} messages today")
        if last_at:
            parts.append(f"Last activity: {last_at}")
        parts.append("")
    except Exception:
        parts.append("Timeline: (unavailable)")
        parts.append("")

    parts.append(f"Team dashboard: /agent-factory/teams/{line_id}")
    return "\n".join(parts).strip()
