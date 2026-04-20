"""
Team Heartbeat Tasks - Celery tasks for autonomous agent team heartbeats.

check_team_heartbeats: Beat task every 60s, finds teams with heartbeat enabled and due, enqueues execute_team_heartbeat.
execute_team_heartbeat: Loads CEO agent, builds team context summary, invokes CEO via gRPC with trigger_type=team_heartbeat.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from services.celery_app import celery_app
from services.celery_tasks.async_runner import run_async
from services.celery_tasks.team_heartbeat_context import (
    TEAM_TOOL_IDS_RULE,
    _build_heartbeat_context,
    _build_worker_context,
)
from services.celery_tasks.team_heartbeat_utils import (
    _agent_has_reports,
    _compute_next_beat_at,
    _fetch_teams_due_heartbeat,
    _fetch_teams_with_pending_worker_tasks,
    _heartbeat_enabled,
    _send_team_execution_status,
    _send_team_notification,
)

logger = logging.getLogger(__name__)


def _line_is_active(team: Optional[Dict[str, Any]]) -> bool:
    """Background autonomous work (heartbeat, worker dispatches) only runs when status is active."""
    return bool(team) and str(team.get("status") or "").lower() == "active"


def _parse_line_heartbeat_config(team: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not team:
        return {}
    raw = team.get("heartbeat_config")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw) if raw else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


@celery_app.task(name="services.celery_tasks.team_heartbeat_tasks.check_team_heartbeats")
def check_team_heartbeats() -> Dict[str, Any]:
    """Beat task: find teams due for a heartbeat, enqueue execute_team_heartbeat for each."""
    try:
        due = run_async(_fetch_teams_due_heartbeat())
        if not due:
            return {"enqueued": 0}
        enqueued = 0
        for team in due:
            execute_team_heartbeat.apply_async(args=[team["id"], team["user_id"]], countdown=2)
            enqueued += 1
        return {"enqueued": enqueued}
    except Exception as e:
        logger.exception("check_team_heartbeats failed: %s", e)
        return {"enqueued": 0, "error": str(e)}


@celery_app.task(name="services.celery_tasks.team_heartbeat_tasks.dispatch_user_post_to_ceo", soft_time_limit=300)
def dispatch_user_post_to_ceo(line_id: str, user_id: str, post_content: str, message_id: str) -> None:
    """Invoke CEO agent to respond to a user timeline post."""
    from services import agent_line_service
    from services.celery_tasks.agent_tasks import _call_grpc_orchestrator_custom_agent

    team = run_async(agent_line_service.get_line(line_id, user_id))
    ceo = run_async(agent_line_service.get_ceo_agent_for_heartbeat(line_id))
    if not team or not ceo:
        return
    query = f'User posted to team timeline: "{post_content}". Respond and delegate as appropriate using send_to_agent.'
    run_async(
        _call_grpc_orchestrator_custom_agent(
            agent_profile_id=ceo["agent_profile_id"],
            query=query,
            user_id=user_id,
            conversation_id="",
            trigger_input=post_content,
            extra_context={
                "trigger_type": "user_timeline_post",
                "line_id": line_id,
                "parent_message_id": message_id,
            },
        )
    )


@celery_app.task(bind=True, name="services.celery_tasks.team_heartbeat_tasks.execute_team_heartbeat", soft_time_limit=600)
def execute_team_heartbeat(
    self, line_id: str, user_id: str, from_manual_trigger: bool = False
) -> Dict[str, Any]:
    """Run one team heartbeat: invoke CEO agent with team context.

    from_manual_trigger: True when the user clicked \"Run heartbeat\" — runs even if autonomous scheduling is off.
    """
    from services import agent_line_service
    from services.celery_tasks.agent_tasks import _call_grpc_orchestrator_custom_agent

    team = None
    hb_cfg: Dict[str, Any] = {}
    try:
        team = run_async(agent_line_service.get_line(line_id, user_id))
        if not team:
            return {"success": False, "error": "Team not found"}

        if not _line_is_active(team):
            logger.info("Team %s heartbeat skipped: line is paused", line_id)
            return {"success": False, "error": "Line is paused; resume the line to run heartbeats"}

        allowed, over_limit = run_async(agent_line_service.check_line_budget(line_id, user_id))
        if not allowed and over_limit:
            logger.warning("Team %s heartbeat skipped: budget limit reached", line_id)
            run_async(_send_team_notification(
                user_id,
                line_id,
                team.get("name", "Team"),
                "team_budget_exceeded",
                message="Team heartbeat skipped: monthly budget limit reached.",
            ))
            return {"success": False, "error": "Team monthly budget limit reached"}

        if not from_manual_trigger and not _heartbeat_enabled(team.get("heartbeat_config")):
            logger.info("Team %s scheduled heartbeat skipped: autonomous disabled", line_id)
            return {"success": False, "error": "Autonomous heartbeat disabled"}

        hb_cfg = _parse_line_heartbeat_config(team) or {}

        task_id = self.request.id
        run_async(agent_line_service.set_line_active_celery_task_id(line_id, task_id))

        try:
            run_async(_send_team_execution_status(line_id, "running", None))
            from utils.websocket_manager import get_websocket_manager
            from datetime import datetime as _dt
            ws = get_websocket_manager()
            run_async(ws.send_to_session({
                "type": "team_execution_status",
                "line_id": line_id,
                "team_name": team.get("name", "Team"),
                "status": "running",
                "agent_id": None,
                "timestamp": _dt.now(timezone.utc).isoformat(),
            }, user_id))
        except Exception as ws_err:
            logger.debug("Send execution_status running failed: %s", ws_err)

        plan = run_async(agent_line_service.get_heartbeat_agents(line_id, user_id))
        if not plan or not plan.get("leader_agent_profile_id"):
            logger.warning("Team %s has no heartbeat leader / empty line", line_id)
            run_async(agent_line_service.set_line_active_celery_task_id(line_id, None))
            return {"success": False, "error": "No leader agent (empty line or invalid governance plan)"}

        from services.celery_tasks import heartbeat_strategies

        mode = plan.get("mode") or "hierarchical"
        if mode == "hierarchical":
            hb_result = run_async(
                heartbeat_strategies.hierarchical_heartbeat_core(
                    line_id, user_id, plan["leader_agent_profile_id"], from_manual_trigger, hb_cfg
                )
            )
        elif mode == "round_robin":
            hb_result = run_async(
                heartbeat_strategies.round_robin_heartbeat_core(
                    line_id,
                    user_id,
                    plan["leader_agent_profile_id"],
                    int(plan.get("rotation_cycle_index") or 0),
                    from_manual_trigger,
                    hb_cfg,
                )
            )
        elif mode == "committee":
            hb_result = run_async(
                heartbeat_strategies.committee_heartbeat_core(line_id, user_id, plan, from_manual_trigger, hb_cfg)
            )
        elif mode == "consensus":
            hb_result = run_async(
                heartbeat_strategies.consensus_heartbeat_core(line_id, user_id, plan, from_manual_trigger, hb_cfg)
            )
        else:
            hb_result = run_async(
                heartbeat_strategies.hierarchical_heartbeat_core(
                    line_id, user_id, plan["leader_agent_profile_id"], from_manual_trigger, hb_cfg
                )
            )

        leader_id = hb_result.get("leader_agent_profile_id")
        success = bool(hb_result.get("success"))
        context_text = hb_result.get("context_text") or ""
        full_response = (hb_result.get("display_response") or "").strip() or context_text
        result = {
            "success": success,
            "error": hb_result.get("error"),
            "response": full_response,
            "message": full_response,
        }
        ceo = {"agent_profile_id": leader_id}

        now = datetime.now(timezone.utc)
        next_at = _compute_next_beat_at((team or {}).get("heartbeat_config"))
        run_async(agent_line_service.update_line_beat_timestamps(line_id, last_beat_at=now, next_beat_at=next_at))

        if plan.get("workers_dispatched_after", True):
            dispatch_team_workers.apply_async(
                kwargs={
                    "line_id": line_id,
                    "user_id": user_id,
                    "from_manual_trigger": from_manual_trigger,
                },
                countdown=15,
            )

        if success and mode == "round_robin":
            run_async(agent_line_service.advance_round_robin_leader(line_id, user_id))

        if result.get("success") and ceo.get("agent_profile_id"):
            summary_for_memory = full_response[:1500] if len(full_response) > 1500 else full_response
            try:
                from services.database_manager.database_helpers import execute
                from utils.grpc_rls import grpc_user_rls as _hb_mem_rls
                run_async(execute(
                    """
                    INSERT INTO agent_memory (agent_profile_id, user_id, memory_key, memory_value, memory_type)
                    VALUES ($1::uuid, $2, 'last_heartbeat_summary', $3::jsonb, 'kv')
                    ON CONFLICT (agent_profile_id, memory_key)
                    DO UPDATE SET memory_value = EXCLUDED.memory_value, updated_at = NOW()
                    """,
                    ceo["agent_profile_id"],
                    user_id,
                    __import__("json").dumps({"summary": summary_for_memory, "at": now.isoformat()}),
                    rls_context=_hb_mem_rls(user_id),
                ))
            except Exception as mem_err:
                logger.debug("Store last_heartbeat_summary failed: %s", mem_err)
            try:
                from services import agent_message_service
                meta = {"source": "heartbeat", "governance_mode": mode, "heartbeat_delivery": "v1"}
                run_async(agent_message_service.create_message(
                    line_id=line_id,
                    from_agent_id=ceo["agent_profile_id"],
                    to_agent_id=None,
                    message_type="report",
                    content=full_response,
                    metadata=meta,
                    user_id=user_id,
                ))
            except Exception as msg_err:
                logger.debug("Post heartbeat to timeline failed: %s", msg_err)

        if result.get("success"):
            try:
                from services.agent_line_brief_snapshot_service import record_heartbeat_brief_snapshot

                run_async(record_heartbeat_brief_snapshot(line_id, user_id, full_response, hb_cfg))
            except Exception as snap_e:
                logger.warning("record_heartbeat_brief_snapshot failed: %s", snap_e)
            try:
                from services.agent_line_delivery_hooks import apply_post_heartbeat_delivery

                run_async(
                    apply_post_heartbeat_delivery(
                        line_id,
                        user_id,
                        team.get("name", "Team"),
                        hb_cfg,
                        full_response,
                        ceo.get("agent_profile_id") if ceo else None,
                        True,
                        None,
                    )
                )
            except Exception as del_e:
                logger.warning("apply_post_heartbeat_delivery failed: %s", del_e)

        if result.get("success") and not from_manual_trigger:
            try:
                run_async(agent_line_service.apply_autonomous_heartbeat_run_quota(line_id, user_id))
            except Exception as quota_err:
                logger.warning("apply_autonomous_heartbeat_run_quota failed: %s", quota_err)

        try:
            run_async(_send_team_execution_status(line_id, "idle", ceo.get("agent_profile_id")))
            from utils.websocket_manager import get_websocket_manager
            from datetime import datetime as _dt
            ws = get_websocket_manager()
            run_async(ws.send_to_session({
                "type": "team_execution_status",
                "line_id": line_id,
                "team_name": team.get("name", "Team"),
                "status": "idle",
                "agent_id": ceo.get("agent_profile_id"),
                "timestamp": _dt.now(timezone.utc).isoformat(),
            }, user_id))
        except Exception as ws_err:
            logger.debug("Send execution_status idle failed: %s", ws_err)

        if not result.get("success"):
            delivery = hb_cfg.get("delivery") if isinstance(hb_cfg.get("delivery"), dict) else {}
            if delivery.get("notify_on_failure", True):
                run_async(
                    _send_team_notification(
                        user_id,
                        line_id,
                        team.get("name", "Team"),
                        "heartbeat_failed",
                        message="Team heartbeat run failed.",
                        error_details=result.get("error"),
                    )
                )

        return {"success": result.get("success", False), "error": result.get("error")}
    except Exception as e:
        logger.exception("execute_team_heartbeat failed: %s", e)
        if line_id and user_id:
            crash_delivery = hb_cfg.get("delivery") if isinstance(hb_cfg.get("delivery"), dict) else {}
            if crash_delivery.get("notify_on_failure", True):
                run_async(
                    _send_team_notification(
                        user_id,
                        line_id,
                        (team.get("name", "Team") if isinstance(team, dict) else "Team"),
                        "heartbeat_failed",
                        message="Team heartbeat crashed.",
                        error_details=str(e),
                    )
                )
        return {"success": False, "error": str(e)}
    finally:
        try:
            run_async(_send_team_execution_status(line_id, "idle", None))
            from utils.websocket_manager import get_websocket_manager
            from datetime import datetime as _dt
            ws = get_websocket_manager()
            team_name = (team.get("name", "Team") if isinstance(team, dict) else "Team") if team else "Team"
            run_async(ws.send_to_session({
                "type": "team_execution_status",
                "line_id": line_id,
                "team_name": team_name,
                "status": "idle",
                "agent_id": None,
                "timestamp": _dt.now(timezone.utc).isoformat(),
            }, user_id))
        except Exception:
            pass
        run_async(agent_line_service.set_line_active_celery_task_id(line_id, None))


async def _run_worker_dispatches_parallel(
    line_id: str, user_id: str, workers: List[Dict[str, Any]], team: Dict[str, Any]
) -> List[Any]:
    """Build context and query for each worker, then invoke all agents in parallel."""
    from services.celery_tasks.agent_tasks import _call_grpc_orchestrator_custom_agent

    coros = []
    for w in workers:
        manager_name = w.get("reports_to_agent_name") or "your manager"
        has_reports = await _agent_has_reports(line_id, w["agent_profile_id"])
        context = await _build_worker_context(
            line_id, user_id, w["agent_profile_id"], manager_name, has_reports=has_reports
        )
        if has_reports:
            query = (
                "Task dispatch. You are a manager with direct reports.\n"
                f"{TEAM_TOOL_IDS_RULE}\n\n"
                "1. Call check_my_tasks for your own task queue.\n"
                "2. To delegate so a report actually runs: call create_task_for_agent (required; writing to workspace does NOT assign work). "
                "Or use send_to_agent(..., wait_for_response=True) for immediate output.\n"
                "3. Call update_task_status to mark your tasks done.\n"
                "4. Call report_goal_progress for any goals that advanced.\n"
                "5. If your task produces a deliverable (report, analysis, etc.), save it via write_to_workspace so the team can access it.\n"
                f"6. Send a brief report to your manager ({manager_name}) via send_to_agent.\n\n"
                f"{context}"
            )
        else:
            query = (
                "Task dispatch. You have tasks assigned to you.\n"
                f"{TEAM_TOOL_IDS_RULE}\n\n"
                "Call check_my_tasks to see your work queue, complete each task, "
                "call update_task_status to mark tasks done. "
                "If a task is linked to a goal (goal_id in task), call report_goal_progress with that goal_id and the appropriate progress_pct so the team briefing stays accurate. "
                "If your task produces a deliverable (report, analysis, etc.), save it via write_to_workspace so the team can access it. "
                f"Send a brief report to your manager ({manager_name}) via send_to_agent.\n\n"
                f"{context}"
            )
        coros.append(
            _call_grpc_orchestrator_custom_agent(
                agent_profile_id=w["agent_profile_id"],
                query=query,
                user_id=user_id,
                conversation_id="",
                trigger_input=context,
                extra_context={
                    "trigger_type": "worker_dispatch",
                    "line_id": line_id,
                    "agent_profile_id": w["agent_profile_id"],
                },
            )
        )
    return await asyncio.gather(*coros, return_exceptions=True)


@celery_app.task(
    bind=True,
    name="services.celery_tasks.team_heartbeat_tasks.dispatch_team_workers",
    soft_time_limit=600,
)
def dispatch_team_workers(
    self, line_id: str, user_id: str, from_manual_trigger: bool = False
) -> Dict[str, Any]:
    """Run worker dispatch for a team: invoke each non-root agent that has pending tasks in parallel.

    from_manual_trigger: must match the heartbeat that scheduled this task so manual runs still dispatch after autonomous is off.
    """
    from services import agent_line_service

    team = None
    try:
        team = run_async(agent_line_service.get_line(line_id, user_id))
        if not team:
            return {"success": False, "error": "Team not found"}

        if not _line_is_active(team):
            logger.info("Team %s worker dispatch skipped: line paused", line_id)
            return {"success": True, "dispatched": 0}

        if not from_manual_trigger and not _heartbeat_enabled(team.get("heartbeat_config")):
            logger.info("Team %s worker dispatch skipped: autonomous disabled", line_id)
            return {"success": True, "dispatched": 0}

        allowed, over_limit = run_async(agent_line_service.check_line_budget(line_id, user_id))
        if not allowed and over_limit:
            logger.warning("Team %s worker dispatch skipped: budget limit reached", line_id)
            run_async(_send_team_notification(
                user_id,
                line_id,
                team.get("name", "Team"),
                "team_budget_exceeded",
                message="Worker dispatch skipped: monthly budget limit reached.",
            ))
            return {"success": False, "error": "Team monthly budget limit reached"}

        workers = run_async(agent_line_service.get_worker_agents_with_pending_tasks(line_id, user_id))
        if not workers:
            return {"success": True, "dispatched": 0}

        try:
            run_async(_send_team_execution_status(line_id, "running", None))
            from utils.websocket_manager import get_websocket_manager
            from datetime import datetime as _dt
            ws = get_websocket_manager()
            run_async(ws.send_to_session({
                "type": "team_execution_status",
                "line_id": line_id,
                "team_name": team.get("name", "Team"),
                "status": "running",
                "agent_id": None,
                "timestamp": _dt.now(timezone.utc).isoformat(),
            }, user_id))
        except Exception as ws_err:
            logger.debug("Send execution_status running failed: %s", ws_err)

        results = run_async(_run_worker_dispatches_parallel(line_id, user_id, workers, team))
        dispatched = 0
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.warning("Worker dispatch failed for agent %s: %s", workers[i].get("agent_profile_id"), r)
            elif isinstance(r, dict) and r.get("success"):
                dispatched += 1

        try:
            run_async(_send_team_execution_status(line_id, "idle", None))
            from utils.websocket_manager import get_websocket_manager
            from datetime import datetime as _dt
            ws = get_websocket_manager()
            run_async(ws.send_to_session({
                "type": "team_execution_status",
                "line_id": line_id,
                "team_name": team.get("name", "Team"),
                "status": "idle",
                "agent_id": None,
                "timestamp": _dt.now(timezone.utc).isoformat(),
            }, user_id))
        except Exception as ws_err:
            logger.debug("Send execution_status idle failed: %s", ws_err)

        return {"success": True, "dispatched": dispatched}
    except Exception as e:
        logger.exception("dispatch_team_workers failed: %s", e)
        return {"success": False, "error": str(e), "dispatched": 0}
    finally:
        try:
            run_async(_send_team_execution_status(line_id, "idle", None))
        except Exception:
            pass


@celery_app.task(
    bind=True,
    name="services.celery_tasks.team_heartbeat_tasks.dispatch_single_worker",
    soft_time_limit=300,
)
def dispatch_single_worker(
    self, line_id: str, user_id: str, agent_profile_id: str
) -> Dict[str, Any]:
    """Dispatch a single agent who has pending tasks (e.g. after create_task_for_agent)."""
    from services import agent_line_service
    from services import agent_task_service
    from services.celery_tasks.agent_tasks import _call_grpc_orchestrator_custom_agent

    try:
        team = run_async(agent_line_service.get_line(line_id, user_id))
        if not team:
            return {"success": False, "error": "Team not found", "dispatched": 0}

        if not _line_is_active(team):
            logger.info("Team %s single-worker dispatch skipped: line paused", line_id)
            return {"success": True, "dispatched": 0}

        allowed, over_limit = run_async(agent_line_service.check_line_budget(line_id, user_id))
        if not allowed and over_limit:
            logger.warning("Team %s single-worker dispatch skipped: budget limit reached", line_id)
            run_async(_send_team_notification(
                user_id,
                line_id,
                team.get("name", "Team"),
                "team_budget_exceeded",
                message="Worker dispatch skipped: monthly budget limit reached.",
            ))
            return {"success": False, "error": "Team monthly budget limit reached", "dispatched": 0}

        tasks = run_async(agent_task_service.get_agent_work_queue(agent_profile_id, line_id, user_id))
        if not tasks:
            return {"success": True, "dispatched": 0}
        if any(t.get("status") == "in_progress" for t in tasks):
            return {"success": True, "dispatched": 0}

        members = team.get("members") or []
        me = next(
            (m for m in members if str(m.get("agent_profile_id") or "") == str(agent_profile_id)),
            None,
        )
        manager_name = "your manager"
        if me and me.get("reports_to"):
            reports_to_mid = me["reports_to"]
            manager_m = next(
                (m for m in members if str(m.get("id") or "") == str(reports_to_mid)),
                None,
            )
            if manager_m:
                manager_name = manager_m.get("agent_name") or manager_m.get("agent_handle") or manager_name

        has_reports = run_async(_agent_has_reports(line_id, agent_profile_id))
        context = run_async(_build_worker_context(
            line_id, user_id, agent_profile_id, manager_name, has_reports=has_reports
        ))
        if has_reports:
            query = (
                "Task dispatch. You are a manager with direct reports.\n"
                f"{TEAM_TOOL_IDS_RULE}\n\n"
                "1. Call check_my_tasks for your own task queue.\n"
                "2. To delegate so a report actually runs: call create_task_for_agent (required; writing to workspace does NOT assign work). "
                "Or use send_to_agent(..., wait_for_response=True) for immediate output.\n"
                "3. Call update_task_status to mark your tasks done.\n"
                "4. Call report_goal_progress for any goals that advanced.\n"
                f"5. Send a brief report to your manager ({manager_name}) via send_to_agent.\n\n"
                f"{context}"
            )
        else:
            query = (
                "Task dispatch. You have tasks assigned to you.\n"
                f"{TEAM_TOOL_IDS_RULE}\n\n"
                "Call check_my_tasks to see your work queue, complete each task, "
                "call update_task_status to mark tasks done. "
                "If a task is linked to a goal (goal_id in task), call report_goal_progress with that goal_id and the appropriate progress_pct so the team briefing stays accurate. "
                f"Send a brief report to your manager ({manager_name}) via send_to_agent.\n\n"
                f"{context}"
            )

        try:
            run_async(_send_team_execution_status(line_id, "running", agent_profile_id))
            from utils.websocket_manager import get_websocket_manager
            from datetime import datetime as _dt
            ws = get_websocket_manager()
            run_async(ws.send_to_session({
                "type": "team_execution_status",
                "line_id": line_id,
                "team_name": team.get("name", "Team"),
                "status": "running",
                "agent_id": agent_profile_id,
                "timestamp": _dt.now(timezone.utc).isoformat(),
            }, user_id))
        except Exception as ws_err:
            logger.debug("Send execution_status running failed: %s", ws_err)

        try:
            result = run_async(
                _call_grpc_orchestrator_custom_agent(
                    agent_profile_id=agent_profile_id,
                    query=query,
                    user_id=user_id,
                    conversation_id="",
                    trigger_input=context,
                    extra_context={
                        "trigger_type": "worker_dispatch",
                        "line_id": line_id,
                        "agent_profile_id": agent_profile_id,
                    },
                )
            )
            dispatched = 1 if result.get("success") else 0
            resp_text = (result.get("response") or "").strip()
            if dispatched and resp_text:
                manager_profile_id = None
                if me and me.get("reports_to"):
                    reports_to_mid = me["reports_to"]
                    manager_m = next(
                        (m for m in members if str(m.get("id") or "") == str(reports_to_mid)),
                        None,
                    )
                    if manager_m:
                        manager_profile_id = manager_m.get("agent_profile_id")
                if manager_profile_id:
                    try:
                        from services import agent_message_service
                        run_async(agent_message_service.create_message(
                            line_id=line_id,
                            from_agent_id=agent_profile_id,
                            to_agent_id=manager_profile_id,
                            message_type="report",
                            content=resp_text,
                            metadata={"trigger_type": "worker_dispatch"},
                            user_id=user_id,
                        ))
                    except Exception as post_err:
                        logger.warning("dispatch_single_worker: post response to timeline failed: %s", post_err)
        except Exception as e:
            logger.warning("Single-worker dispatch failed for agent %s: %s", agent_profile_id, e)
            dispatched = 0
        finally:
            try:
                run_async(_send_team_execution_status(line_id, "idle", None))
                from utils.websocket_manager import get_websocket_manager
                from datetime import datetime as _dt
                ws = get_websocket_manager()
                run_async(ws.send_to_session({
                    "type": "team_execution_status",
                    "line_id": line_id,
                    "team_name": team.get("name", "Team"),
                    "status": "idle",
                    "agent_id": None,
                    "timestamp": _dt.now(timezone.utc).isoformat(),
                }, user_id))
            except Exception as ws_err:
                logger.debug("Send execution_status idle failed: %s", ws_err)

        return {"success": True, "dispatched": dispatched}
    except Exception as e:
        logger.exception("dispatch_single_worker failed: %s", e)
        return {"success": False, "error": str(e), "dispatched": 0}


@celery_app.task(
    bind=True,
    name="services.celery_tasks.team_heartbeat_tasks.dispatch_worker_for_message",
    soft_time_limit=300,
    max_retries=1,
)
def dispatch_worker_for_message(
    self,
    line_id: str,
    user_id: str,
    agent_profile_id: str,
    message_id: str,
    from_agent_id: str,
) -> Dict[str, Any]:
    """Dispatch an agent to handle a new incoming message; post their response as a threaded reply."""
    from services import agent_line_service
    from services import agent_message_service
    from services.celery_tasks.agent_tasks import _call_grpc_orchestrator_custom_agent

    try:
        team = run_async(agent_line_service.get_line(line_id, user_id))
        if not team:
            return {"success": False, "error": "Team not found", "dispatched": 0}

        if not _line_is_active(team):
            logger.info("Team %s message dispatch skipped: line paused", line_id)
            return {"success": True, "dispatched": 0}

        allowed, over_limit = run_async(agent_line_service.check_line_budget(line_id, user_id))
        if not allowed and over_limit:
            logger.warning("Team %s message dispatch skipped: budget limit reached", line_id)
            run_async(_send_team_notification(
                user_id,
                line_id,
                team.get("name", "Team"),
                "team_budget_exceeded",
                message="Worker dispatch for message skipped: monthly budget limit reached.",
            ))
            return {"success": False, "error": "Team monthly budget limit reached", "dispatched": 0}

        members = team.get("members") or []
        me = next(
            (m for m in members if str(m.get("agent_profile_id") or "") == str(agent_profile_id)),
            None,
        )
        manager_name = "your manager"
        if me and me.get("reports_to"):
            reports_to_mid = me["reports_to"]
            manager_m = next(
                (m for m in members if str(m.get("id") or "") == str(reports_to_mid)),
                None,
            )
            if manager_m:
                manager_name = manager_m.get("agent_name") or manager_m.get("agent_handle") or manager_name

        has_reports = run_async(_agent_has_reports(line_id, agent_profile_id))
        context = run_async(_build_worker_context(
            line_id, user_id, agent_profile_id, manager_name, has_reports=has_reports
        ))
        query = (
            "You have received a new message from a team member.\n"
            f"{TEAM_TOOL_IDS_RULE}\n\n"
            "Call read_my_messages to see it and respond appropriately. "
            "If it contains an assignment, use create_task_for_agent so the work is actually scheduled (writing to workspace does not). Or complete it directly if you can. "
            "If your work produces a deliverable (report, analysis, etc.), save it via write_to_workspace so the team can access it. "
            f"Send a brief acknowledgment or report back to your manager ({manager_name}) via send_to_agent if needed.\n\n"
            f"{context}"
        )

        try:
            run_async(_send_team_execution_status(line_id, "running", agent_profile_id))
            from utils.websocket_manager import get_websocket_manager
            ws = get_websocket_manager()
            run_async(ws.send_to_session({
                "type": "team_execution_status",
                "line_id": line_id,
                "team_name": team.get("name", "Team"),
                "status": "running",
                "agent_id": agent_profile_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }, user_id))
        except Exception as ws_err:
            logger.debug("Send execution_status running failed: %s", ws_err)

        try:
            result = run_async(
                _call_grpc_orchestrator_custom_agent(
                    agent_profile_id=agent_profile_id,
                    query=query,
                    user_id=user_id,
                    conversation_id="",
                    trigger_input=context,
                    extra_context={
                        "trigger_type": "message_dispatch",
                        "line_id": line_id,
                        "agent_profile_id": agent_profile_id,
                    },
                )
            )
            dispatched = 1 if result.get("success") else 0
            resp_text = (result.get("response") or "").strip()
            if dispatched and resp_text and message_id and from_agent_id:
                try:
                    run_async(agent_message_service.create_message(
                        line_id=line_id,
                        from_agent_id=agent_profile_id,
                        to_agent_id=from_agent_id,
                        message_type="response",
                        content=resp_text,
                        metadata={"in_reply_to": message_id},
                        parent_message_id=message_id,
                        user_id=user_id,
                    ))
                except Exception as post_err:
                    logger.warning("dispatch_worker_for_message: post response to timeline failed: %s", post_err)
        except Exception as e:
            logger.warning("Message dispatch failed for agent %s: %s", agent_profile_id, e)
            dispatched = 0
        finally:
            try:
                run_async(_send_team_execution_status(line_id, "idle", None))
                from utils.websocket_manager import get_websocket_manager
                ws = get_websocket_manager()
                run_async(ws.send_to_session({
                    "type": "team_execution_status",
                    "line_id": line_id,
                    "team_name": team.get("name", "Team"),
                    "status": "idle",
                    "agent_id": None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }, user_id))
            except Exception as ws_err:
                logger.debug("Send execution_status idle failed: %s", ws_err)

        return {"success": True, "dispatched": dispatched}
    except Exception as e:
        logger.exception("dispatch_worker_for_message failed: %s", e)
        return {"success": False, "error": str(e), "dispatched": 0}


@celery_app.task(name="services.celery_tasks.team_heartbeat_tasks.check_worker_dispatches")
def check_worker_dispatches() -> Dict[str, Any]:
    """Beat task: find teams with pending worker tasks, enqueue dispatch_team_workers for each."""
    try:
        due = run_async(_fetch_teams_with_pending_worker_tasks())
        if not due:
            return {"enqueued": 0}
        enqueued = 0
        for team in due:
            dispatch_team_workers.apply_async(args=[team["id"], team["user_id"]], countdown=2)
            enqueued += 1
        return {"enqueued": enqueued}
    except Exception as e:
        logger.exception("check_worker_dispatches failed: %s", e)
        return {"enqueued": 0, "error": str(e)}
