"""
Agent Line Analytics API - health, analytics, and approvals endpoints.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from utils.auth_middleware import get_current_user, AuthenticatedUserResponse
from services import agent_line_service
from services.database_manager.database_helpers import fetch_all

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/lines", tags=["Agent Lines - Analytics"])


@router.get("/{line_id}/agent-health", response_model=List[Dict[str, Any]])
async def get_line_agent_health(
    line_id: str = Path(..., description="Line UUID"),
    days: int = Query(7, ge=1, le=90),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Per-agent health: runs, success rate, avg duration, last run, cost (last N days)."""
    team = await agent_line_service.get_line(line_id, current_user.user_id)
    if not team:
        raise HTTPException(status_code=404, detail="Line not found")
    members = team.get("members") or []
    if not members:
        return []
    member_ids = [m["agent_profile_id"] for m in members if m.get("agent_profile_id")]
    if not member_ids:
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    rows = await fetch_all(
        """
        SELECT
            e.agent_profile_id,
            COUNT(*)::int AS runs,
            COUNT(*) FILTER (WHERE e.status IN ('success', 'completed'))::int AS successes,
            COUNT(*) FILTER (WHERE e.status IN ('failed', 'error'))::int AS failures,
            COALESCE(AVG(e.duration_ms) FILTER (WHERE e.duration_ms IS NOT NULL), 0)::float AS avg_duration_ms,
            MAX(e.started_at) AS last_run_at,
            SUM(COALESCE(e.cost_usd, 0))::float AS total_cost_usd,
            (array_agg(e.status ORDER BY e.started_at DESC))[1] AS last_status
        FROM agent_execution_log e
        WHERE e.agent_profile_id = ANY($1::uuid[])
          AND e.user_id = $2
          AND e.started_at >= $3::timestamptz
        GROUP BY e.agent_profile_id
        """,
        member_ids,
        current_user.user_id,
        cutoff,
    )
    stats_by_agent = {str(r["agent_profile_id"]): r for r in rows}

    out = []
    for m in members:
        pid = m.get("agent_profile_id")
        if not pid:
            continue
        r = stats_by_agent.get(str(pid)) or {}
        runs = r.get("runs") or 0
        successes = r.get("successes") or 0
        last_run_at = r.get("last_run_at")
        out.append({
            "agent_profile_id": pid,
            "agent_name": m.get("agent_name") or m.get("agent_handle") or "Agent",
            "agent_color": m.get("color"),
            "runs": runs,
            "successes": successes,
            "failures": r.get("failures") or 0,
            "success_rate": (successes / runs * 100) if runs else 0,
            "avg_duration_ms": round((r.get("avg_duration_ms") or 0), 0),
            "last_run_at": last_run_at.isoformat() if last_run_at else None,
            "last_status": r.get("last_status"),
            "total_cost_usd": round(float(r.get("total_cost_usd") or 0), 4),
        })
    return out


@router.get("/{line_id}/analytics", response_model=Dict[str, Any])
async def get_line_analytics(
    line_id: str = Path(..., description="Line UUID"),
    days: int = Query(30, ge=1, le=365),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Analytics: task throughput, cost over time, goal progress, agent activity, message volume."""
    team = await agent_line_service.get_line(line_id, current_user.user_id)
    if not team:
        raise HTTPException(status_code=404, detail="Line not found")
    start_date = datetime.now(timezone.utc) - timedelta(days=days)

    task_throughput_rows = await fetch_all(
        """
        SELECT (d.d::date)::text AS date,
               COALESCE(c.cnt, 0)::int AS created,
               COALESCE(done.cnt, 0)::int AS completed
        FROM generate_series($1::timestamp, CURRENT_DATE::timestamp, '1 day'::interval) d(d)
        LEFT JOIN (
            SELECT created_at::date AS date, COUNT(*)::int AS cnt
            FROM agent_tasks WHERE line_id = $2 AND created_at >= $3::timestamptz
            GROUP BY created_at::date
        ) c ON c.date = (d.d)::date
        LEFT JOIN (
            SELECT updated_at::date AS date, COUNT(*)::int AS cnt
            FROM agent_tasks WHERE line_id = $2 AND status = 'done' AND updated_at >= $3::timestamptz
            GROUP BY updated_at::date
        ) done ON done.date = (d.d)::date
        ORDER BY d.d
        """,
        start_date,
        line_id,
        start_date,
    )
    task_throughput = [
        {"date": r["date"], "created": r["created"], "completed": r["completed"]}
        for r in task_throughput_rows
    ]

    cost_rows = await fetch_all(
        """
        SELECT (e.started_at AT TIME ZONE 'UTC')::date::text AS date,
               SUM(COALESCE(e.cost_usd, 0))::float AS cost_usd
        FROM agent_execution_log e
        INNER JOIN agent_line_memberships m ON m.agent_profile_id = e.agent_profile_id AND m.line_id = $1
        WHERE e.user_id = $2 AND e.started_at >= $3::timestamptz
        GROUP BY (e.started_at AT TIME ZONE 'UTC')::date
        ORDER BY 1
        """,
        line_id,
        current_user.user_id,
        start_date,
    )
    cost_over_time = [{"date": r["date"], "cost_usd": round(float(r["cost_usd"] or 0), 4)} for r in cost_rows]

    goal_rows = await fetch_all(
        """
        SELECT id, title, progress_pct, status
        FROM agent_line_goals
        WHERE line_id = $1
        ORDER BY priority DESC NULLS LAST, created_at DESC
        """,
        line_id,
    )
    goal_progress = [
        {
            "goal_id": str(r["id"]),
            "title": r.get("title") or "",
            "progress_pct": r.get("progress_pct") or 0,
            "status": r.get("status") or "active",
        }
        for r in goal_rows
    ]

    members = team.get("members") or []
    member_ids = [m["agent_profile_id"] for m in members if m.get("agent_profile_id")]
    agent_activity = []
    if member_ids:
        activity_rows = await fetch_all(
            """
            SELECT
                e.agent_profile_id,
                COUNT(*)::int AS runs,
                COUNT(*) FILTER (WHERE e.status IN ('success', 'completed'))::int AS successes,
                COUNT(*) FILTER (WHERE e.status IN ('failed', 'error'))::int AS failures,
                SUM(COALESCE(e.cost_usd, 0))::float AS total_cost_usd
            FROM agent_execution_log e
            WHERE e.agent_profile_id = ANY($1::uuid[])
              AND e.user_id = $2
              AND e.started_at >= $3::timestamptz
            GROUP BY e.agent_profile_id
            """,
            member_ids,
            current_user.user_id,
            start_date,
        )
        activity_by_agent = {str(r["agent_profile_id"]): r for r in activity_rows}
        for m in members:
            pid = m.get("agent_profile_id")
            if not pid:
                continue
            r = activity_by_agent.get(str(pid)) or {}
            agent_activity.append({
                "agent_name": m.get("agent_name") or m.get("agent_handle") or "Agent",
                "agent_color": m.get("color"),
                "runs": r.get("runs") or 0,
                "successes": r.get("successes") or 0,
                "failures": r.get("failures") or 0,
                "total_cost_usd": round(float(r.get("total_cost_usd") or 0), 4),
            })

    msg_rows = await fetch_all(
        """
        SELECT (created_at AT TIME ZONE 'UTC')::date::text AS date, COUNT(*)::int AS count
        FROM agent_messages
        WHERE line_id = $1 AND created_at >= $2::timestamptz
        GROUP BY (created_at AT TIME ZONE 'UTC')::date
        ORDER BY 1
        """,
        line_id,
        start_date,
    )
    message_volume = [{"date": r["date"], "count": r["count"]} for r in msg_rows]

    return {
        "task_throughput": task_throughput,
        "cost_over_time": cost_over_time,
        "goal_progress": goal_progress,
        "agent_activity": agent_activity,
        "message_volume": message_volume,
    }


@router.get("/{line_id}/approvals", response_model=List[Dict[str, Any]])
async def get_line_approvals(
    line_id: str = Path(..., description="Line UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Pending approval queue entries for this team (governance proposals)."""
    team = await agent_line_service.get_line(line_id, current_user.user_id)
    if not team:
        raise HTTPException(status_code=404, detail="Line not found")
    rows = await fetch_all(
        """
        SELECT a.id, a.agent_profile_id, a.step_name, a.prompt, a.preview_data, a.governance_type, a.status, a.created_at,
               p.name AS agent_name, p.handle AS agent_handle
        FROM agent_approval_queue a
        LEFT JOIN agent_profiles p ON p.id = a.agent_profile_id
        WHERE a.user_id = $1 AND a.status = 'pending'
          AND (a.preview_data->>'line_id') = $2
        ORDER BY a.created_at DESC
        """,
        current_user.user_id,
        line_id,
    )
    return [
        {
            "id": str(r["id"]),
            "agent_profile_id": str(r["agent_profile_id"]) if r.get("agent_profile_id") else None,
            "step_name": r.get("step_name", ""),
            "prompt": r.get("prompt", ""),
            "preview_data": r.get("preview_data") or {},
            "governance_type": r.get("governance_type") or "playbook_step",
            "status": r.get("status", "pending"),
            "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
            "agent_name": r.get("agent_name") or r.get("agent_handle") or "Agent",
        }
        for r in rows
    ]
