"""
Team Heartbeat Context - build CEO and worker briefings for agent team heartbeats.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Max length for timeline/escalation message content in heartbeat context. Do not truncate
# so agents see full messages and never post truncated content to the timeline.
HEARTBEAT_CONTEXT_MESSAGE_CONTENT_MAX = 5000

# Instruct agents on UUID usage; line_id is auto-injected in agent line context.
TEAM_TOOL_IDS_RULE = (
    "Tool IDs: line_id (agent line UUID) is provided automatically in your briefing — do not pass the line display name. "
    "create_task_for_agent: pass title, assigned_agent_id (UUID from roster/status board), and optionally goal_id to attach the task to an existing goal. "
    "You do NOT supply task_id when creating a task — the tool creates a new row and returns task_id. "
    "update_task_status / escalate_task: pass task_id from the TASKS section of this briefing. "
    "For update_task_status: only valid workflow moves (e.g. toward done); tasks already marked done or cancelled cannot change status again. "
    "report_goal_progress: pass goal_id from the GOALS section. "
    "send_to_agent: pass target_agent as @handle, agent profile UUID from the roster, or exact display name from get_team_status_board. "
    "Use UUIDs from context, not display names or titles."
)


def _format_relative_time(utc_dt, now: datetime) -> str:
    """Format a datetime as relative time (e.g. '2h ago')."""
    if not utc_dt:
        return "unknown"
    try:
        if isinstance(utc_dt, str):
            utc_dt = datetime.fromisoformat(utc_dt.replace("Z", "+00:00"))
        delta = now - utc_dt
        if delta.total_seconds() < 60:
            return "just now"
        if delta.total_seconds() < 3600:
            return f"{int(delta.total_seconds() / 60)}m ago"
        if delta.total_seconds() < 86400:
            return f"{int(delta.total_seconds() / 3600)}h ago"
        return f"{delta.days}d ago"
    except (ValueError, TypeError):
        return "unknown"


def _format_goal_tree_lines(nodes: List[Dict], agent_names: Dict[str, str], indent: int = 0) -> List[str]:
    """Format goal tree as indented lines (title, status, assignee, progress)."""
    lines = []
    for n in nodes or []:
        if n.get("status") in ("done", "cancelled"):
            continue
        title = (n.get("title") or "")[:60]
        pct = n.get("progress_pct") or 0
        aid = n.get("assigned_agent_id")
        assignee = agent_names.get(aid, "unassigned") if aid else "unassigned"
        prefix = "  " * indent
        goal_id_suffix = f" [goal_id: {n.get('id')}]" if n.get("id") else ""
        lines.append(f"{prefix}- \"{title}\" ({pct}% complete, assigned to {assignee}){goal_id_suffix}")
        children = n.get("children") or []
        if children:
            lines.extend(_format_goal_tree_lines(children, agent_names, indent + 1))
    return lines


async def _build_workspace_section(team_id: str, user_id: str, now: datetime, header: str = "WORKSPACE (shared artifacts):") -> List[str]:
    """Shared workspace context section used by both CEO and worker briefings."""
    from services import agent_workspace_service
    ws_result = await agent_workspace_service.list_workspace(team_id, user_id)
    if not ws_result.get("success") or not ws_result.get("entries"):
        return []
    lines = [header]
    for e in ws_result["entries"][:15]:
        key_name = e.get("key", "")
        by_name = e.get("updated_by_agent_name") or "unknown"
        ago = _format_relative_time(e.get("updated_at"), now) if e.get("updated_at") else "unknown"
        lines.append(f"- {key_name} (updated by {by_name}, {ago})")
    lines.append("")
    return lines


def _build_team_roster_section(
    members: List[Dict[str, Any]],
    title: str = "TEAM ROSTER (capabilities):",
    worker_agent_profile_id: Optional[str] = None,
    manager_name: Optional[str] = None,
) -> List[str]:
    """Format team member list. If worker_agent_profile_id/manager_name set, add (you)/(your manager) labels."""
    lines = [title]
    is_worker_view = worker_agent_profile_id is not None and manager_name is not None
    for m in members:
        aid = m.get("agent_profile_id", "")
        name = m.get("agent_name") or m.get("agent_handle") or "Unknown"
        desc = (m.get("agent_description") or "").strip()
        role = (m.get("role") or "worker").lower()
        if is_worker_view:
            is_me = str(aid) == str(worker_agent_profile_id)
            is_manager = name == manager_name
            label = " (you)" if is_me else (" (your manager)" if is_manager else "")
        else:
            label = ""
        if desc:
            lines.append(f"- {name}{label} ({role}): {desc}")
        else:
            suffix = ": no description set" if not is_worker_view else ""
            lines.append(f"- {name}{label} ({role}){suffix}")
    lines.append("")
    return lines


async def _build_heartbeat_context(
    team_id: str,
    user_id: str,
    ceo_agent_profile_id: Optional[str] = None,
) -> str:
    """Build a rich situational briefing for the CEO: tasks, goals, escalations, messages, agent activity, previous summary."""
    from datetime import timedelta
    from services.database_manager.database_helpers import fetch_one, fetch_all

    parts = []
    try:
        from services import agent_task_service, agent_goal_service, agent_message_service, agent_line_service
        now = datetime.now(timezone.utc)
        cutoff_24h = now - timedelta(hours=24)

        team = await agent_line_service.get_line(team_id, user_id)
        if not team:
            return "Team not found."
        members = team.get("members") or []
        agent_id_to_name = {m["agent_profile_id"]: (m.get("agent_name") or m.get("agent_handle") or "Unknown") for m in members if m.get("agent_profile_id")}

        tasks = await agent_task_service.list_line_tasks(team_id, user_id)
        pending = [t for t in tasks if t.get("status") not in ("done", "cancelled")]
        in_progress = [t for t in pending if t.get("status") == "in_progress"]
        stalled = []
        for t in in_progress:
            u = t.get("updated_at")
            if u:
                try:
                    utc_dt = datetime.fromisoformat(str(u).replace("Z", "+00:00"))
                    if (now - utc_dt) > timedelta(hours=24):
                        stalled.append(t)
                except (ValueError, TypeError):
                    pass

        parts.append("== TEAM STATUS BRIEFING ==")
        parts.append(f"Line ID: {team_id}  (use this UUID as line_id in tool context)")
        parts.append("")

        org_tree = await agent_line_service.get_org_chart(team_id, user_id)

        def _org_lines(nodes, indent: int = 0):
            lines = []
            for n in nodes or []:
                name = n.get("agent_name") or n.get("agent_handle") or "Unknown"
                role = (n.get("role") or "worker").lower()
                desc = (n.get("agent_description") or "").strip()
                children = n.get("children") or []
                prefix = "  " * indent
                role_label = f"{role}, root" if indent == 0 else role
                desc_suffix = f": {desc}" if desc else ""
                lines.append(f"{prefix}{name} ({role_label}){desc_suffix}")
                if children:
                    lines.append(f"{prefix}  reports: {', '.join((c.get('agent_name') or c.get('agent_handle') or 'Unknown') for c in children)}")
                lines.extend(_org_lines(children, indent + 1))
            return lines

        if org_tree:
            parts.append("ORG CHART (who reports to whom; delegate to managers so they assign work to their reports):")
            for line in _org_lines(org_tree):
                parts.append(line)
            parts.append("")

        parts.extend(_build_team_roster_section(members))

        if escalation_rows := await fetch_all(
            """
            SELECT id, content, created_at, from_agent_id FROM agent_messages
            WHERE line_id = $1 AND message_type = 'escalation' AND created_at >= $2
            ORDER BY created_at DESC LIMIT 10
            """,
            team_id,
            cutoff_24h,
        ):
            parts.append("ESCALATIONS (action needed):")
            for r in escalation_rows:
                raw = r.get("content") or ""
                content = raw[:HEARTBEAT_CONTEXT_MESSAGE_CONTENT_MAX].strip()
                if len(raw) > HEARTBEAT_CONTEXT_MESSAGE_CONTENT_MAX:
                    content += "..."
                from_name = agent_id_to_name.get(str(r["from_agent_id"]), "Agent") if r.get("from_agent_id") else "Agent"
                ago = _format_relative_time(r.get("created_at"), now)
                parts.append(f"- [{from_name} -> CEO] \"{content}\" ({ago})")
            parts.append("")

        if stalled:
            parts.append("STALLED TASKS (>24h no progress):")
            for t in stalled[:5]:
                title = (t.get("title") or "")[:50]
                assignee = agent_id_to_name.get(t.get("assigned_agent_id") or "", "unassigned")
                u = t.get("updated_at")
                ago = _format_relative_time(u, now) if u else "unknown"
                parts.append(f"- \"{title}\" assigned to {assignee}, in_progress since {ago}")
            parts.append("")

        parts.append(f"TASKS ({len(pending)} active):")
        for t in pending[:10]:
            title = (t.get("title") or "")[:50]
            aid = t.get("assigned_agent_id")
            assignee = agent_id_to_name.get(aid, "unassigned") if aid else "unassigned"
            status = t.get("status", "pending")
            u = t.get("updated_at")
            ago = _format_relative_time(u, now) if u else "unknown"
            stalled_mark = " (STALLED)" if t in stalled else ""
            parts.append(f"- \"{title}\" assigned to {assignee}, {status} (updated {ago}){stalled_mark}")
        if len(pending) > 10:
            parts.append(f"... and {len(pending) - 10} more")
        parts.append("")

        goals = await agent_goal_service.get_goal_tree(team_id, user_id)

        def count_and_unassigned(nodes):
            total, unassigned = 0, 0
            for n in nodes or []:
                total += 1
                if n.get("status") == "active" and not n.get("assigned_agent_id"):
                    unassigned += 1
                t, u = count_and_unassigned(n.get("children", []))
                total += t
                unassigned += u
            return total, unassigned

        total_goals, unassigned_goals = count_and_unassigned(goals)
        parts.append(f"GOALS ({total_goals} total, {unassigned_goals} unassigned active):")
        goal_lines = _format_goal_tree_lines(goals, agent_id_to_name)
        for line in goal_lines[:20]:
            parts.append(line)
        if len(goal_lines) > 20:
            parts.append(f"... and {len(goal_lines) - 20} more goals")
        parts.append("")

        member_ids = [m["agent_profile_id"] for m in members if m.get("agent_profile_id")]
        msg_count_per_agent = {}
        last_msg_per_agent = {}
        if member_ids:
            placeholders = ",".join([f"${i+2}" for i in range(len(member_ids))])
            cutoff_param = f"${len(member_ids) + 2}"
            msg_counts = await fetch_all(
                f"""
                SELECT from_agent_id, COUNT(*)::int AS c FROM agent_messages
                WHERE line_id = $1 AND from_agent_id IN ({placeholders}) AND created_at >= {cutoff_param}
                GROUP BY from_agent_id
                """,
                team_id,
                *member_ids,
                cutoff_24h,
            )
            for r in msg_counts:
                msg_count_per_agent[str(r["from_agent_id"])] = r.get("c", 0)
            last_msg = await fetch_all(
                f"""
                SELECT DISTINCT ON (from_agent_id) from_agent_id, created_at FROM agent_messages
                WHERE line_id = $1 AND from_agent_id IN ({placeholders})
                ORDER BY from_agent_id, created_at DESC
                """,
                team_id,
                *member_ids,
            )
            for r in last_msg:
                last_msg_per_agent[str(r["from_agent_id"])] = r.get("created_at")
        task_count_per_agent = {}
        last_task_per_agent = {}
        for t in pending:
            aid = t.get("assigned_agent_id")
            if not aid:
                continue
            sid = str(aid)
            task_count_per_agent[sid] = task_count_per_agent.get(sid, 0) + 1
            u = t.get("updated_at")
            if u:
                try:
                    ut = datetime.fromisoformat(str(u).replace("Z", "+00:00"))
                    if sid not in last_task_per_agent or ut > last_task_per_agent[sid]:
                        last_task_per_agent[sid] = ut
                except (ValueError, TypeError):
                    pass
        last_exec_per_agent = {}
        if member_ids:
            exec_placeholders = ",".join([f"${i+1}" for i in range(len(member_ids))])
            last_exec = await fetch_all(
                f"""
                SELECT DISTINCT ON (agent_profile_id) agent_profile_id, started_at FROM agent_execution_log
                WHERE agent_profile_id IN ({exec_placeholders})
                ORDER BY agent_profile_id, started_at DESC
                """,
                *member_ids,
            )
            for r in last_exec:
                last_exec_per_agent[str(r["agent_profile_id"])] = r.get("started_at")

        def _parse_ts(ts):
            if not ts:
                return None
            try:
                if isinstance(ts, datetime):
                    return ts
                return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            except (ValueError, TypeError):
                return None

        parts.append("RECENT AGENT ACTIVITY (last 24h):")
        for m in members:
            aid = m.get("agent_profile_id")
            if not aid:
                continue
            sid = str(aid)
            name = m.get("agent_name") or m.get("agent_handle") or "Unknown"
            msg_count = msg_count_per_agent.get(sid, 0)
            task_count = task_count_per_agent.get(sid, 0)
            last_msg_at = _parse_ts(last_msg_per_agent.get(sid))
            last_task_at = _parse_ts(last_task_per_agent.get(sid))
            last_exec_at = _parse_ts(last_exec_per_agent.get(sid))
            last_activity = max((t for t in (last_msg_at, last_task_at, last_exec_at) if t is not None), default=None)
            ago_str = _format_relative_time(last_activity, now) if last_activity else "no activity"
            idle_note = ""
            if last_activity and (now - last_activity) > timedelta(hours=24):
                idle_note = " (IDLE)"
            parts.append(f"- {name}: {msg_count} messages sent, {task_count} tasks in progress, last active {ago_str}{idle_note}")
        parts.append("")

        timeline = await agent_message_service.get_line_timeline(team_id, user_id, limit=10)
        recent = (timeline.get("items") or [])[:10]
        if recent:
            parts.append("RECENT MESSAGES (last 10):")
            for msg in recent:
                from_name = msg.get("from_agent_name") or msg.get("from_agent_handle") or "Agent"
                to_name = msg.get("to_agent_name") or msg.get("to_agent_handle") or "team"
                raw = msg.get("content") or ""
                content = raw[:HEARTBEAT_CONTEXT_MESSAGE_CONTENT_MAX].strip()
                if len(raw) > HEARTBEAT_CONTEXT_MESSAGE_CONTENT_MAX:
                    content += "..."
                created = msg.get("created_at")
                ago = _format_relative_time(created, now) if created else "unknown"
                parts.append(f"- {from_name} -> {to_name}: \"{content}\" ({ago})")
            parts.append("")

        ws_section = await _build_workspace_section(team_id, user_id, now, "WORKSPACE (shared scratchpad):")
        parts.extend(ws_section)

        approval_count = await fetch_one(
            """
            SELECT COUNT(*)::int AS c FROM agent_approval_queue
            WHERE user_id = $1 AND status = 'pending'
              AND ((preview_data->>'line_id') = $2 OR (preview_data->>'team_id') = $2)
            """,
            user_id,
            team_id,
        )
        if approval_count and (approval_count.get("c") or 0) > 0:
            parts.append(f"Pending approvals for this team: {approval_count['c']}")
            parts.append("")

        if member_ids:
            placeholders = ",".join([f"${i+2}" for i in range(len(member_ids))])
            err_rows = await fetch_all(
                f"""
                SELECT agent_profile_id, COUNT(*)::int AS cnt FROM agent_execution_log
                WHERE agent_profile_id IN ({placeholders}) AND status = 'failed' AND started_at >= $1
                GROUP BY agent_profile_id
                """,
                cutoff_24h,
                *member_ids,
            )
            if err_rows:
                parts.append(f"Agent failures (last 24h): {sum(r['cnt'] for r in err_rows)} total")
                parts.append("")

        budget = await agent_line_service.get_line_budget_summary(team_id, user_id)
        if budget and budget.get("total_current_period_spend_usd") is not None:
            parts.append(f"BUDGET: ${float(budget['total_current_period_spend_usd']):.2f} spent this period")
        else:
            parts.append("BUDGET: (no spend data)")
        parts.append("")

        if ceo_agent_profile_id:
            prev = await fetch_one(
                "SELECT memory_value FROM agent_memory WHERE agent_profile_id = $1 AND user_id = $2 AND memory_key = 'last_heartbeat_summary'",
                ceo_agent_profile_id,
                user_id,
            )
            if prev and isinstance(prev.get("memory_value"), dict) and prev["memory_value"].get("summary"):
                summary_text = prev["memory_value"]["summary"] or ""
                parts.append("Previous heartbeat: " + (summary_text[:1500] + "..." if len(summary_text) > 1500 else summary_text))
    except Exception as e:
        logger.warning("Build heartbeat context failed: %s", e)
        parts.append(f"(Context error: {e})")
    return "\n".join(parts) if parts else "No context available."


async def _build_worker_context(
    team_id: str,
    user_id: str,
    worker_agent_profile_id: str,
    manager_name: Optional[str] = None,
    has_reports: bool = False,
) -> str:
    """Build context for a worker dispatch: their task queue, messages to them, workspace. If has_reports, include goals and reports' tasks."""
    parts = []
    try:
        from services import agent_task_service, agent_message_service, agent_line_service

        now = datetime.now(timezone.utc)
        team = await agent_line_service.get_line(team_id, user_id)
        members = team.get("members") or [] if team else []

        tasks = await agent_task_service.get_agent_work_queue(
            worker_agent_profile_id, team_id, user_id
        )
        parts.append("== YOUR TASK QUEUE ==")
        parts.append(f"Line ID: {team_id}  (use this UUID as line_id in tool context)")
        parts.append("")
        if not tasks:
            parts.append("No tasks assigned.")
        else:
            for t in tasks[:15]:
                title = (t.get("title") or "")[:80]
                status = t.get("status", "assigned")
                desc = (t.get("description") or "")[:300]
                task_id = t.get("id", "")
                parts.append(f"- [{status}] \"{title}\" (id: {task_id})")
                if desc:
                    parts.append(f"  {desc}")
            if len(tasks) > 15:
                parts.append(f"... and {len(tasks) - 15} more")
        parts.append("")

        if has_reports:
            me = next((m for m in members if str(m.get("agent_profile_id") or "") == str(worker_agent_profile_id)), None)
            direct_reports = []
            report_ids = []
            if me and me.get("id"):
                my_mid = me.get("id")
                direct_reports = [m for m in members if m.get("reports_to") == my_mid]
                report_ids = [str(r["agent_profile_id"]) for r in direct_reports if r.get("agent_profile_id")]
            if direct_reports:
                parts.append("YOUR DIRECT REPORTS (to assign work so they actually run: call create_task_for_agent; write_to_workspace/send_to_agent alone do NOT schedule workers):")
                for r in direct_reports:
                    name = r.get("agent_name") or r.get("agent_handle") or r.get("agent_profile_id")
                    handle = r.get("agent_handle") or ""
                    parts.append(f"- {name}" + (f" (@{handle})" if handle else ""))
                parts.append("")
            my_and_report_ids = [str(worker_agent_profile_id)] + report_ids
            try:
                from services import agent_goal_service
                goals_tree = await agent_goal_service.get_goal_tree(team_id, user_id)
                def _goals_for_me_or_reports(nodes, agent_names, indent=0):
                    lines = []
                    for n in nodes or []:
                        if n.get("status") in ("done", "cancelled"):
                            continue
                        aid = n.get("assigned_agent_id")
                        if aid and str(aid) not in my_and_report_ids:
                            continue
                        title = (n.get("title") or "")[:60]
                        pct = n.get("progress_pct") or 0
                        assignee = agent_names.get(str(aid), "unassigned") if aid else "unassigned"
                        prefix = "  " * indent
                        lines.append(f"{prefix}- \"{title}\" ({pct}% complete, assigned to {assignee}) [goal_id: {n.get('id')}]")
                        lines.extend(_goals_for_me_or_reports(n.get("children") or [], agent_names, indent + 1))
                    return lines
                agent_id_to_name = {m.get("agent_profile_id"): (m.get("agent_name") or m.get("agent_handle") or "Unknown") for m in members if m.get("agent_profile_id")}
                goal_lines = _goals_for_me_or_reports(goals_tree, agent_id_to_name)
                if goal_lines:
                    parts.append("GOALS (assigned to you or your reports):")
                    for line in goal_lines[:15]:
                        parts.append(line)
                    if len(goal_lines) > 15:
                        parts.append(f"... and {len(goal_lines) - 15} more")
                    parts.append("")
            except Exception as g_err:
                logger.debug("Manager goal context skipped: %s", g_err)
            if report_ids:
                try:
                    all_tasks = await agent_task_service.list_line_tasks(team_id, user_id)
                    reports_pending = [t for t in all_tasks if str(t.get("assigned_agent_id") or "") in report_ids and t.get("status") not in ("done", "cancelled")]
                    if reports_pending:
                        parts.append("YOUR REPORTS' PENDING TASKS:")
                        for t in reports_pending[:10]:
                            title = (t.get("title") or "")[:60]
                            status = t.get("status", "")
                            assignee_id = t.get("assigned_agent_id")
                            assignee_name = next((r.get("agent_name") or r.get("agent_handle") for r in direct_reports if str(r.get("agent_profile_id")) == str(assignee_id)), assignee_id)
                            parts.append(f"- [{status}] \"{title}\" (assigned to {assignee_name})")
                        if len(reports_pending) > 10:
                            parts.append(f"... and {len(reports_pending) - 10} more")
                        parts.append("")
                except Exception as t_err:
                    logger.debug("Manager reports tasks context skipped: %s", t_err)

        msg_result = await agent_message_service.get_agent_messages(
            worker_agent_profile_id, team_id, user_id, limit=15, offset=0
        )
        items = msg_result.get("items") or []
        to_me = [m for m in items if str(m.get("to_agent_id") or "") == str(worker_agent_profile_id)]
        if to_me:
            parts.append("MESSAGES ADDRESSED TO YOU (recent):")
            for m in to_me[:10]:
                from_name = m.get("from_agent_name") or m.get("from_agent_handle") or "Agent"
                raw = m.get("content") or ""
                content = raw[:HEARTBEAT_CONTEXT_MESSAGE_CONTENT_MAX].strip()
                if len(raw) > HEARTBEAT_CONTEXT_MESSAGE_CONTENT_MAX:
                    content += "..."
                created = m.get("created_at")
                ago = _format_relative_time(created, now) if created else "unknown"
                parts.append(f"- From {from_name} ({ago}): \"{content}\"")
            parts.append("")

        ws_section = await _build_workspace_section(
            team_id, user_id, now,
            "WORKSPACE (shared artifacts; use read_workspace key=<key> to read content):"
        )
        parts.extend(ws_section)

        parts.extend(_build_team_roster_section(
            members,
            title="YOUR TEAM:",
            worker_agent_profile_id=worker_agent_profile_id,
            manager_name=manager_name,
        ))
        if manager_name:
            parts.append(f"Send reports via send_to_agent targeting your manager ({manager_name}).")
    except Exception as e:
        logger.warning("Build worker context failed: %s", e)
        parts.append(f"(Context error: {e})")
    return "\n".join(parts) if parts else "No context available."
