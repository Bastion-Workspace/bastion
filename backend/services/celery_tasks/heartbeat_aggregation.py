"""
Committee synthesis helpers and consensus proposal parsing / tally / execution.
"""

import json
import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _extract_json_array(text: str) -> List[Dict[str, Any]]:
    """Parse a JSON array from model output; supports markdown code fence."""
    if not text or not text.strip():
        return []
    raw = text.strip()
    if raw.startswith("```"):
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if m:
            raw = m.group(1).strip()
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict) and "actions" in data and isinstance(data["actions"], list):
            return [x for x in data["actions"] if isinstance(x, dict)]
    except json.JSONDecodeError:
        pass
    return []


def _action_signature(act: Dict[str, Any]) -> str:
    """Stable key for tallying similar proposals across agents."""
    at = (act.get("action_type") or act.get("action") or "create_task").strip().lower()
    title = (act.get("title") or act.get("description") or "").strip()[:200]
    tgt = str(act.get("assigned_agent_id") or act.get("target_agent") or "").strip()
    return f"{at}|{tgt}|{title}"


def tally_consensus_proposals(
    agent_responses: List[Tuple[str, str]],
    quorum_pct: int,
    _tiebreaker_agent_id: Optional[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    agent_responses: list of (agent_profile_id, response_text).
    Returns (approved_actions_as_representative_dicts, summary_dict).
    """
    votes_by_sig: Dict[str, List[str]] = defaultdict(list)
    sig_to_example: Dict[str, Dict[str, Any]] = {}
    n_agents = max(len(agent_responses), 1)
    threshold = max(1, int((quorum_pct / 100.0) * n_agents + 0.9999))

    for aid, text in agent_responses:
        proposals = _extract_json_array(text)
        seen_sigs: set = set()
        for p in proposals:
            sig = _action_signature(p)
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)
            votes_by_sig[sig].append(aid)
            if sig not in sig_to_example:
                sig_to_example[sig] = dict(p)

    approved: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []
    for sig, voters in votes_by_sig.items():
        ex = sig_to_example.get(sig) or {}
        row = {"signature": sig, "votes": len(voters), "voters": voters, "action": ex}
        if len(voters) >= threshold:
            approved.append(ex)
        else:
            pending.append(row)

    summary = {
        "quorum_pct": quorum_pct,
        "threshold_votes": threshold,
        "agent_count": n_agents,
        "approved_count": len(approved),
        "pending_signatures": [p["signature"] for p in pending],
    }
    return approved, summary


async def execute_consensus_actions(
    line_id: str,
    user_id: str,
    actions: List[Dict[str, Any]],
    created_by_agent_id: Optional[str],
) -> List[str]:
    """
    Execute approved consensus actions (create_task, optional send_message).
    Returns log lines for timeline.
    """
    from services import agent_line_service, agent_task_service

    lines: List[str] = []
    members = (await agent_line_service.get_line(line_id, user_id) or {}).get("members") or []
    handle_to_id = {}
    for m in members:
        h = (m.get("agent_handle") or "").strip().lstrip("@").lower()
        if h:
            handle_to_id[h] = str(m["agent_profile_id"])

    for act in actions:
        at = (act.get("action_type") or act.get("action") or "").strip().lower()
        if at in ("create_task", "task", "assign"):
            title = (act.get("title") or act.get("description") or "Consensus task")[:500]
            desc = (act.get("description") or act.get("title") or "")[:5000]
            tgt = str(act.get("assigned_agent_id") or act.get("target_agent") or "").strip()
            if not tgt:
                lines.append(f"Skipped task (no assignee): {title[:40]}")
                continue
            resolved = tgt
            if tgt.startswith("@"):
                rh = tgt[1:].lower()
                resolved = handle_to_id.get(rh) or tgt
            try:
                await agent_task_service.create_task(
                    line_id=line_id,
                    user_id=user_id,
                    title=title,
                    description=desc or None,
                    assigned_agent_id=resolved,
                    goal_id=str(act["goal_id"]) if act.get("goal_id") else None,
                    created_by_agent_id=created_by_agent_id,
                )
                lines.append(f"Created task: {title[:60]} -> {resolved[:8]}...")
            except Exception as e:
                logger.warning("execute_consensus_actions create_task failed: %s", e)
                lines.append(f"Task failed: {title[:40]} ({e})")
    return lines


def merge_committee_responses(responses: List[Tuple[str, str, str]]) -> str:
    """Concatenate (name, handle, text) tuples for chair input or timeline without chair."""
    parts = []
    for name, handle, text in responses:
        label = name or handle or "Agent"
        parts.append(f"--- {label} ---\n{(text or '').strip()}")
    return "\n\n".join(parts)
