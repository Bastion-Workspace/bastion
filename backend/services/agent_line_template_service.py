"""
Load built-in agent line templates and instantiate lines (goals, workspace seed, members).
"""

import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).resolve().parent / "line_templates"


def _load_all_templates() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not _TEMPLATE_DIR.is_dir():
        return out
    for path in sorted(_TEMPLATE_DIR.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                continue
            tid = (data.get("id") or path.stem).strip()
            if tid:
                out[tid] = data
        except Exception as e:
            logger.warning("Skipping line template %s: %s", path, e)
    return out


def list_template_summaries() -> List[Dict[str, str]]:
    """Lightweight catalog for UI."""
    rows = []
    for tid, tpl in sorted(_load_all_templates().items()):
        rows.append(
            {
                "id": tid,
                "title": (tpl.get("title") or tid).strip(),
                "description": (tpl.get("description") or "").strip(),
            }
        )
    return rows


async def _assert_profile_owned(user_id: str, profile_id: str) -> None:
    from services.database_manager.database_helpers import fetch_one

    row = await fetch_one(
        "SELECT id FROM agent_profiles WHERE id = $1::uuid AND user_id = $2",
        profile_id,
        user_id,
    )
    if not row:
        raise ValueError(f"Agent profile not found or not owned by user: {profile_id}")


async def instantiate_line_from_template(
    user_id: str,
    template_id: str,
    name: str,
    ceo_agent_profile_id: str,
    handle: Optional[str] = None,
    member_agent_profile_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Create a line from a built-in template: line row, workspace seed, CEO + optional workers, goals."""
    from services import agent_goal_service, agent_line_service, agent_workspace_service

    templates = _load_all_templates()
    tpl = templates.get((template_id or "").strip())
    if not tpl:
        raise ValueError(f"Unknown template_id: {template_id}")

    await _assert_profile_owned(user_id, ceo_agent_profile_id)
    mids = [str(x).strip() for x in (member_agent_profile_ids or []) if x]
    for pid in mids:
        if pid == ceo_agent_profile_id:
            continue
        await _assert_profile_owned(user_id, pid)

    hb = copy.deepcopy(tpl.get("heartbeat_config")) if isinstance(tpl.get("heartbeat_config"), dict) else {}
    if not isinstance(hb, dict):
        hb = {}

    line = await agent_line_service.create_line(
        user_id=user_id,
        name=name.strip(),
        description=(tpl.get("description") or "").strip() or None,
        mission_statement=(tpl.get("mission_statement") or "").strip() or None,
        status="active",
        heartbeat_config=hb,
        governance_policy=tpl.get("governance_policy") if isinstance(tpl.get("governance_policy"), dict) else None,
        governance_mode=tpl.get("governance_mode"),
        reference_config=tpl.get("reference_config") if isinstance(tpl.get("reference_config"), dict) else None,
        data_workspace_config=tpl.get("data_workspace_config") if isinstance(tpl.get("data_workspace_config"), dict) else None,
    )
    line_id = str(line["id"])

    seed = tpl.get("workspace_seed")
    if isinstance(seed, dict):
        for key, val in seed.items():
            if not isinstance(key, str) or not key.strip():
                continue
            text = val if isinstance(val, str) else json.dumps(val, default=str)
            try:
                await agent_workspace_service.set_workspace_entry(
                    line_id, key.strip(), text or "", user_id, updated_by_agent_id=None
                )
            except Exception as e:
                logger.warning("Workspace seed failed for key %s: %s", key, e)

    ceo_mem = await agent_line_service.add_member(
        line_id, user_id, ceo_agent_profile_id, role="ceo", reports_to=None
    )
    ceo_membership_id = ceo_mem.get("id")
    max_workers = int(tpl.get("max_worker_slots") or 8)
    for pid in mids[:max_workers]:
        if pid == ceo_agent_profile_id:
            continue
        try:
            await agent_line_service.add_member(
                line_id, user_id, pid, role="worker", reports_to=ceo_membership_id
            )
        except Exception as e:
            logger.warning("add_member from template failed for %s: %s", pid, e)

    for parent in tpl.get("goals") or []:
        if not isinstance(parent, dict) or not (parent.get("title") or "").strip():
            continue
        try:
            g = await agent_goal_service.create_goal(
                line_id,
                user_id,
                title=parent["title"].strip()[:500],
                description=(parent.get("description") or "")[:5000] or None,
                parent_goal_id=None,
                status="active",
                progress_pct=int(parent.get("progress_pct") or 0),
            )
            gid = g.get("id")
            for ch in parent.get("children") or []:
                if not isinstance(ch, dict) or not (ch.get("title") or "").strip():
                    continue
                await agent_goal_service.create_goal(
                    line_id,
                    user_id,
                    title=ch["title"].strip()[:500],
                    description=(ch.get("description") or "")[:5000] or None,
                    parent_goal_id=gid,
                    status="active",
                    progress_pct=int(ch.get("progress_pct") or 0),
                )
        except Exception as e:
            logger.warning("Goal creation from template failed: %s", e)

    skill_ids = tpl.get("team_skill_ids")
    if isinstance(skill_ids, list) and skill_ids:
        try:
            await agent_line_service.update_line(
                line_id,
                user_id,
                team_skill_ids=[str(x) for x in skill_ids if x],
            )
        except Exception as e:
            logger.warning("team_skill_ids from template failed: %s", e)

    if handle and isinstance(handle, str) and handle.strip():
        try:
            await agent_line_service.update_line(line_id, user_id, handle=handle.strip()[:100])
        except Exception as e:
            logger.warning("handle from template failed: %s", e)

    full = await agent_line_service.get_line(line_id, user_id)
    return full if full else line
