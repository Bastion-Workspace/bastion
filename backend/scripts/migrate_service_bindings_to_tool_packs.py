"""
One-time migration: copy agent_service_bindings (email/calendar) into default playbook
tool_packs on llm_agent / deep_agent steps.

Run inside the backend container when DATABASE_URL is set:
    python scripts/migrate_service_bindings_to_tool_packs.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncpg

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _normalize_tool_packs(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for e in raw:
        if isinstance(e, dict) and e.get("pack"):
            out.append({
                "pack": e["pack"],
                "mode": "read" if e.get("mode") == "read" else "full",
                **({"connections": list(e["connections"])} if isinstance(e.get("connections"), list) else {}),
            })
        elif isinstance(e, str) and e.strip():
            out.append({"pack": e.strip(), "mode": "full"})
    return out


def _merge_external_connections(
    packs: List[Dict[str, Any]],
    pack_name: str,
    ids: List[int],
) -> List[Dict[str, Any]]:
    if not ids:
        return packs
    idx = next((i for i, p in enumerate(packs) if p.get("pack") == pack_name), None)
    if idx is None:
        packs.append({"pack": pack_name, "mode": "full", "connections": list(ids)})
        return packs
    row = dict(packs[idx])
    conns = list(row.get("connections") or [])
    if not isinstance(conns, list):
        conns = []
    s = {int(x) for x in conns if x is not None}
    for c in ids:
        s.add(int(c))
    row["connections"] = sorted(s)
    packs[idx] = row
    return packs


def _mutate_step(
    step: Dict[str, Any],
    email_ids: List[int],
    calendar_ids: List[int],
) -> bool:
    st = step.get("step_type") or step.get("type") or ""
    if st not in ("llm_agent", "deep_agent"):
        return False
    raw = step.get("tool_packs")
    packs = _normalize_tool_packs(raw)
    before = json.dumps(packs, sort_keys=True)
    if email_ids:
        packs = _merge_external_connections(packs, "email", email_ids)
    if calendar_ids:
        packs = _merge_external_connections(packs, "calendar", calendar_ids)
    after = json.dumps(packs, sort_keys=True)
    if before != after:
        step["tool_packs"] = packs
        return True
    return False


def _walk_steps(steps: Any, email_ids: List[int], calendar_ids: List[int]) -> bool:
    changed = False
    if not isinstance(steps, list):
        return False
    for step in steps:
        if not isinstance(step, dict):
            continue
        if _mutate_step(step, email_ids, calendar_ids):
            changed = True
        for key in ("then_steps", "else_steps", "body_steps", "parallel_steps", "steps"):
            nested = step.get(key)
            if isinstance(nested, list) and _walk_steps(nested, email_ids, calendar_ids):
                changed = True
    return changed


async def run() -> None:
    conn = await asyncpg.connect(settings.DATABASE_URL)
    try:
        bindings = await conn.fetch(
            """
            SELECT agent_profile_id, service_type, connection_id
            FROM agent_service_bindings
            WHERE is_enabled = true AND service_type IN ('email', 'calendar')
            """
        )
        by_profile: Dict[str, Dict[str, List[int]]] = {}
        for r in bindings:
            pid = str(r["agent_profile_id"])
            st = (r["service_type"] or "").strip()
            cid = int(r["connection_id"])
            by_profile.setdefault(pid, {}).setdefault(st, []).append(cid)

        updated_playbooks = 0
        for profile_id, groups in by_profile.items():
            row = await conn.fetchrow(
                "SELECT default_playbook_id FROM agent_profiles WHERE id = $1::uuid",
                profile_id,
            )
            if not row or not row["default_playbook_id"]:
                continue
            pb_id = row["default_playbook_id"]
            pb = await conn.fetchrow(
                "SELECT id, definition FROM custom_playbooks WHERE id = $1::uuid",
                pb_id,
            )
            if not pb:
                continue
            definition = pb["definition"]
            if isinstance(definition, str):
                try:
                    definition = json.loads(definition)
                except json.JSONDecodeError:
                    logger.warning("Skip playbook %s: invalid JSON definition", pb_id)
                    continue
            if not isinstance(definition, dict):
                continue
            steps = definition.get("steps")
            email_ids = sorted(set(groups.get("email", [])))
            cal_ids = sorted(set(groups.get("calendar", [])))
            if _walk_steps(steps, email_ids, cal_ids):
                await conn.execute(
                    "UPDATE custom_playbooks SET definition = $2::jsonb, updated_at = NOW() WHERE id = $1::uuid",
                    pb_id,
                    json.dumps(definition),
                )
                updated_playbooks += 1
                logger.info("Updated playbook %s for profile %s", pb_id, profile_id)

        logger.info("Migration complete. Updated %s playbook(s).", updated_playbooks)
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(run())
