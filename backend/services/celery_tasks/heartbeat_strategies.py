"""
Governance-mode-specific team heartbeat execution (hierarchical, committee, round_robin, consensus).
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from services.celery_tasks.team_heartbeat_context import (
    TEAM_TOOL_IDS_RULE,
    _build_heartbeat_context,
)
from services.celery_tasks.agent_tasks import _call_grpc_orchestrator_custom_agent
from services.celery_tasks.heartbeat_aggregation import (
    execute_consensus_actions,
    merge_committee_responses,
    tally_consensus_proposals,
)
from services.celery_tasks.heartbeat_delivery_contract import format_delivery_contract_appendix


async def hierarchical_heartbeat_core(
    line_id: str,
    user_id: str,
    leader_agent_profile_id: str,
    from_manual_trigger: bool,
    heartbeat_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Original CEO heartbeat: one leader run with delegation instructions.
    Returns dict: success, error, response (grpc), leader_agent_profile_id
    """
    context_text = await _build_heartbeat_context(
        line_id, user_id, leader_agent_profile_id, governance_mode="hierarchical"
    )
    query = (
        "Team heartbeat. You are the CEO (root).\n\n"
        "CRITICAL — DELEGATION (do not do workers' work yourself):\n"
        "- The ONLY way to assign work so a worker agent actually runs is to call create_task_for_agent. "
        "Writing to the workspace or sending a message documents the assignment but does NOT create a task or trigger the worker. "
        "If create_task_for_agent returns an error or success=false, the assignment is NOT complete—fix and retry.\n"
        "- When you have workers (reports), delegate: use create_task_for_agent to assign work (worker is dispatched on the next cycle), "
        "or send_to_agent(..., wait_for_response=True) to get immediate output. If you never call create_task_for_agent or send_to_agent, the worker never runs.\n"
        "- create_task_for_agent: Creates a task; the assigned agent is invoked automatically. Use for async work. Required for any deliverable you want a report to produce.\n"
        "- send_to_agent(..., wait_for_response=True): Runs the worker now and returns their response. Use when you need output this cycle.\n\n"
        "CRITICAL — GOAL PROGRESS (keep system state accurate):\n"
        "- When work advances or is completed, you MUST call report_goal_progress with the goal_id and new progress_pct (0–100). "
        "Otherwise the briefing will keep showing 0% and you will repeat the same directives. "
        "When a goal is fully done, call report_goal_progress with progress_pct=100 and then update the goal status to completed if supported.\n\n"
        f"{TEAM_TOOL_IDS_RULE}\n\n"
        "YOUR JOB THIS CYCLE: Review goals and pending tasks. Delegate execution to workers (create_task_for_agent or send_to_agent). "
        "Call report_goal_progress when work advances. Post a brief status to the timeline.\n\n"
        f"{context_text}"
        f"{format_delivery_contract_appendix(heartbeat_config)}"
    )

    result = await _call_grpc_orchestrator_custom_agent(
        agent_profile_id=leader_agent_profile_id,
        query=query,
        user_id=user_id,
        conversation_id="",
        trigger_input=context_text,
        extra_context={
            "trigger_type": "team_heartbeat",
            "line_id": line_id,
            "agent_profile_id": leader_agent_profile_id,
        },
    )
    ok = bool(result.get("success"))
    disp = (result.get("response") or result.get("message") or context_text or "").strip()
    return {
        "success": ok,
        "error": result.get("error"),
        "grpc_result": result,
        "leader_agent_profile_id": leader_agent_profile_id,
        "context_text": context_text,
        "display_response": disp,
    }


async def round_robin_heartbeat_core(
    line_id: str,
    user_id: str,
    leader_agent_profile_id: str,
    cycle_index: int,
    from_manual_trigger: bool,
    heartbeat_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Same tools as hierarchical but framed as rotating leader."""
    context_text = await _build_heartbeat_context(
        line_id,
        user_id,
        leader_agent_profile_id,
        governance_mode="round_robin",
        rotation_cycle_index=cycle_index,
    )
    query = (
        f"Team heartbeat (round-robin). You are the designated leader for this cycle (rotation index {cycle_index}). "
        "You have the same delegation authority as a CEO for this turn.\n\n"
        "CRITICAL — DELEGATION:\n"
        "- Use create_task_for_agent so assigned workers run on the next cycle; workspace text alone does not schedule them.\n"
        "- Or send_to_agent(..., wait_for_response=True) for immediate output.\n\n"
        "CRITICAL — GOAL PROGRESS:\n"
        "- Call report_goal_progress when goals advance.\n\n"
        f"{TEAM_TOOL_IDS_RULE}\n\n"
        "Review goals and tasks; delegate; update goal progress; post a brief timeline status.\n\n"
        f"{context_text}"
        f"{format_delivery_contract_appendix(heartbeat_config)}"
    )
    result = await _call_grpc_orchestrator_custom_agent(
        agent_profile_id=leader_agent_profile_id,
        query=query,
        user_id=user_id,
        conversation_id="",
        trigger_input=context_text,
        extra_context={
            "trigger_type": "team_heartbeat",
            "line_id": line_id,
            "agent_profile_id": leader_agent_profile_id,
            "governance_mode": "round_robin",
        },
    )
    ok = bool(result.get("success"))
    disp = (result.get("response") or result.get("message") or context_text or "").strip()
    return {
        "success": ok,
        "error": result.get("error"),
        "grpc_result": result,
        "leader_agent_profile_id": leader_agent_profile_id,
        "context_text": context_text,
        "display_response": disp,
    }


async def _invoke_agent_parallel(
    specs: List[Tuple[str, str, str]],
    user_id: str,
    line_id: str,
    trigger_type: str,
) -> List[Any]:
    """specs: (agent_profile_id, query, trigger_input)"""

    async def one(pid: str, q: str, tin: str) -> Dict[str, Any]:
        return await _call_grpc_orchestrator_custom_agent(
            agent_profile_id=pid,
            query=q,
            user_id=user_id,
            conversation_id="",
            trigger_input=tin,
            extra_context={
                "trigger_type": trigger_type,
                "line_id": line_id,
                "agent_profile_id": pid,
            },
        )

    coros = [one(pid, q, tin) for pid, q, tin in specs]
    return await asyncio.gather(*coros, return_exceptions=True)


async def committee_heartbeat_core(
    line_id: str,
    user_id: str,
    plan: Dict[str, Any],
    from_manual_trigger: bool,
    heartbeat_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    members: List[Dict[str, Any]] = plan.get("members") or []
    participants = plan.get("participants") or []
    chair_id = plan.get("chair_agent_id")
    leader_id = plan.get("leader_agent_profile_id")

    context_text = await _build_heartbeat_context(
        line_id, user_id, leader_id, governance_mode="committee"
    )
    member_prompt = (
        "Team heartbeat (committee). You are one of several members receiving this briefing in parallel.\n"
        "Review status, propose concrete next actions (tasks, messages, goal updates). "
        "You may use team tools (create_task_for_agent, send_to_agent, report_goal_progress) when appropriate.\n\n"
        f"{TEAM_TOOL_IDS_RULE}\n\n"
        f"{context_text}"
        f"{format_delivery_contract_appendix(heartbeat_config)}"
    )

    specs: List[Tuple[str, str, str]] = []
    pid_to_meta: Dict[str, Tuple[str, str]] = {}
    for m in members:
        pid = str(m.get("agent_profile_id") or "")
        if pid not in participants:
            continue
        name = m.get("agent_name") or m.get("agent_handle") or "Agent"
        handle = m.get("agent_handle") or ""
        pid_to_meta[pid] = (name, handle)
        specs.append((pid, member_prompt, context_text))

    results = await _invoke_agent_parallel(specs, user_id, line_id, "team_heartbeat_committee")
    collected: List[Tuple[str, str, str]] = []
    for i, r in enumerate(results):
        pid = specs[i][0]
        name, handle = pid_to_meta.get(pid, ("Agent", ""))
        if isinstance(r, Exception):
            logger.warning("committee member %s failed: %s", pid, r)
            collected.append((name, handle, f"(error: {r})"))
            continue
        if isinstance(r, dict):
            text = (r.get("response") or r.get("message") or r.get("formatted") or "").strip()
            collected.append((name, handle, text or "(no response)"))

    merged = merge_committee_responses(collected)
    final_text = merged
    final_leader = chair_id or leader_id

    chair_ok = False
    if chair_id:
        chair_prompt = (
            "You are the committee chair. Below are parallel contributions from committee members this heartbeat. "
            "Synthesize them into one coherent plan; execute or delegate remaining actions with team tools if needed; "
            "then give a concise chair summary for the timeline.\n\n"
            f"{TEAM_TOOL_IDS_RULE}\n\n"
            f"MEMBER CONTRIBUTIONS:\n{merged}"
            f"{format_delivery_contract_appendix(heartbeat_config)}"
        )
        cr = await _call_grpc_orchestrator_custom_agent(
            agent_profile_id=chair_id,
            query=chair_prompt,
            user_id=user_id,
            conversation_id="",
            trigger_input=merged[:12000],
            extra_context={
                "trigger_type": "team_heartbeat_committee_chair",
                "line_id": line_id,
                "agent_profile_id": chair_id,
            },
        )
        chair_ok = bool(cr.get("success"))
        if chair_ok:
            final_text = (cr.get("response") or cr.get("message") or merged).strip() or merged

    member_ok = any(isinstance(r, dict) and r.get("success") for r in results if not isinstance(r, Exception))
    ok = member_ok or chair_ok
    if not specs:
        ok = False
    return {
        "success": ok,
        "error": None if ok else "Committee heartbeat: no successful member or chair run",
        "grpc_result": {"committee_results": results, "chair_run": bool(chair_id)},
        "leader_agent_profile_id": final_leader,
        "context_text": context_text,
        "display_response": final_text[:15000],
    }


CONSENSUS_PROMPT_SUFFIX = (
    "\n\nRespond with ONLY a JSON array of 0 to 3 objects, each like:\n"
    '{"action_type":"create_task","title":"short title","description":"details",'
    '"assigned_agent_id":"<uuid or @handle>"}\n'
    "Use valid action_type: create_task (preferred) or send_message with target_agent_id and content.\n"
    "No markdown outside the JSON."
)


async def consensus_heartbeat_core(
    line_id: str,
    user_id: str,
    plan: Dict[str, Any],
    from_manual_trigger: bool,
    heartbeat_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    members: List[Dict[str, Any]] = plan.get("members") or []
    participants = plan.get("participants") or []
    quorum_pct = int(plan.get("quorum_pct") or 60)
    tiebreaker = plan.get("tiebreaker_agent_id")
    leader_id = plan.get("leader_agent_profile_id")

    context_text = await _build_heartbeat_context(
        line_id,
        user_id,
        leader_id,
        governance_mode="consensus",
        quorum_pct=quorum_pct,
    )
    base_q = (
        f"Team heartbeat (consensus). You are a voting member. Quorum agreement threshold: {quorum_pct}% of members.\n"
        "Review the briefing. Propose up to 3 concrete actions as JSON only (see format below).\n"
        f"{TEAM_TOOL_IDS_RULE}\n\n"
        f"{context_text}"
        f"{format_delivery_contract_appendix(heartbeat_config)}"
        f"{CONSENSUS_PROMPT_SUFFIX}"
    )

    specs: List[Tuple[str, str, str]] = []
    for m in members:
        pid = str(m.get("agent_profile_id") or "")
        if pid not in participants:
            continue
        specs.append((pid, base_q, context_text))

    results = await _invoke_agent_parallel(specs, user_id, line_id, "team_heartbeat_consensus")
    agent_responses: List[Tuple[str, str]] = []
    for i, r in enumerate(results):
        pid = specs[i][0]
        if isinstance(r, Exception):
            agent_responses.append((pid, "[]"))
            continue
        text = (r.get("response") or r.get("message") or "").strip() if isinstance(r, dict) else ""
        agent_responses.append((pid, text or "[]"))

    approved, summary = tally_consensus_proposals(agent_responses, quorum_pct, tiebreaker)
    exec_lines = await execute_consensus_actions(
        line_id,
        user_id,
        approved,
        created_by_agent_id=leader_id,
    )
    summary_json = json.dumps(summary, default=str)
    final_text = (
        f"Consensus heartbeat summary\n{summary_json}\n\n"
        + "\n".join(exec_lines)
        + "\n\n--- Raw proposals (truncated) ---\n"
        + merge_committee_responses(
            [
                (
                    next((m.get("agent_name") for m in members if str(m.get("agent_profile_id")) == aid), aid),
                    "",
                    txt[:2000],
                )
                for aid, txt in agent_responses
            ]
        )[:8000]
    )

    return {
        "success": True,
        "error": None,
        "grpc_result": {"consensus_summary": summary, "approved_count": len(approved)},
        "leader_agent_profile_id": leader_id,
        "context_text": context_text,
        "display_response": final_text[:15000],
    }
