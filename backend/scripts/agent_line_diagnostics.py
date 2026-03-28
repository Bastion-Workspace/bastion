"""
Agent line diagnostics: query goals, tasks, members; optionally fix goal progress.

Usage (from repo root, with backend env):
  docker compose exec backend python scripts/agent_line_diagnostics.py
  docker compose exec backend python scripts/agent_line_diagnostics.py --fix-goals
  docker compose exec backend python scripts/agent_line_diagnostics.py --fix-goal-title "Fritzol"
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
import asyncpg


async def run_diagnostics(conn):
    """Print teams, members, goals, and tasks."""
    print("=== AGENT LINES ===\n")
    teams = await conn.fetch("""
        SELECT id, user_id, name, status,
               heartbeat_config,
               last_beat_at, next_beat_at
        FROM agent_lines
        WHERE status = 'active'
        ORDER BY name
    """)
    if not teams:
        print("No active lines.\n")
        return []

    line_ids = []
    for t in teams:
        tid = str(t["id"])
        line_ids.append(tid)
        cfg = t.get("heartbeat_config") or {}
        enabled = cfg.get("enabled", False) if isinstance(cfg, dict) else False
        print(f"Line: {t['name']} (id={tid[:8]}...)")
        print(f"  user_id={t['user_id']}  heartbeat_enabled={enabled}")
        print(f"  last_beat_at={t['last_beat_at']}  next_beat_at={t['next_beat_at']}\n")

    print("=== ORG CHART (members with reports_to) ===\n")
    members = await conn.fetch("""
        SELECT m.line_id, m.agent_profile_id, m.reports_to,
               p.name AS agent_name, p.handle AS agent_handle
        FROM agent_line_memberships m
        JOIN agent_profiles p ON p.id = m.agent_profile_id
        WHERE m.line_id = ANY($1::uuid[])
        ORDER BY m.line_id, m.reports_to NULLS FIRST, p.name
    """, line_ids)
    for m in members:
        root = " (CEO/root)" if m["reports_to"] is None else ""
        print(f"  {m['agent_name']} (@{m['agent_handle'] or '?'}){root}  line={str(m['line_id'])[:8]}...")

    print("\n=== GOALS (active, progress) ===\n")
    goals = await conn.fetch("""
        SELECT g.id, g.line_id, g.title, g.status, g.progress_pct, g.assigned_agent_id,
               p.name AS assigned_name
        FROM agent_line_goals g
        LEFT JOIN agent_profiles p ON p.id = g.assigned_agent_id
        WHERE g.line_id = ANY($1::uuid[])
          AND g.status NOT IN ('cancelled')
        ORDER BY g.line_id, g.created_at
    """, line_ids)
    for g in goals:
        print(f"  \"{g['title'][:60]}\"  progress={g['progress_pct']}%  status={g['status']}  assignee={g['assigned_name'] or 'unassigned'}  id={str(g['id'])[:8]}...")

    print("\n=== TASKS (by status, assignee) ===\n")
    tasks = await conn.fetch("""
        SELECT t.id, t.line_id, t.title, t.status, t.assigned_agent_id,
               p.name AS assigned_name
        FROM agent_tasks t
        LEFT JOIN agent_profiles p ON p.id = t.assigned_agent_id
        WHERE t.line_id = ANY($1::uuid[])
        ORDER BY t.line_id, t.status, t.created_at
    """, line_ids)
    for t in tasks:
        print(f"  [{t['status']}] \"{t['title'][:50]}\" -> {t['assigned_name'] or 'unassigned'}  id={str(t['id'])[:8]}...")

    print("\n=== WORKERS WITH PENDING TASKS (would be dispatched) ===\n")
    workers = await conn.fetch("""
        SELECT DISTINCT m.agent_profile_id, p.name AS agent_name,
               (SELECT COUNT(*) FROM agent_tasks tk
                WHERE tk.line_id = m.line_id AND tk.assigned_agent_id = m.agent_profile_id
                  AND tk.status IN ('assigned', 'in_progress')) AS pending_count
        FROM agent_line_memberships m
        JOIN agent_profiles p ON p.id = m.agent_profile_id
        JOIN agent_tasks tk ON tk.line_id = m.line_id
            AND tk.assigned_agent_id = m.agent_profile_id
            AND tk.status IN ('assigned', 'in_progress')
        WHERE m.line_id = ANY($1::uuid[]) AND m.reports_to IS NOT NULL
    """, line_ids)
    if not workers:
        print("  None. No non-CEO members have tasks in assigned/in_progress.")
    else:
        for w in workers:
            print(f"  {w['agent_name']}  pending_tasks={w['pending_count']}")

    return line_ids


async def fix_goals(conn, goal_title_substring=None):
    """
    Set progress_pct and optionally status on goals where:
    - All linked tasks are done/cancelled and progress_pct is still 0, or
    - goal_title_substring is given and matches (then set to 75% and status active).
    """
    if goal_title_substring:
        goals = await conn.fetch("""
            SELECT id, line_id, title, progress_pct, status
            FROM agent_line_goals
            WHERE status IN ('active', 'blocked')
              AND title ILIKE $1
        """, f"%{goal_title_substring}%")
    else:
        goals = await conn.fetch("""
            SELECT g.id, g.line_id, g.title, g.progress_pct, g.status,
                   (SELECT COUNT(*) FROM agent_tasks t
                    WHERE t.goal_id = g.id AND t.status NOT IN ('done', 'cancelled')) AS open_tasks,
                   (SELECT COUNT(*) FROM agent_tasks t
                    WHERE t.goal_id = g.id AND t.status IN ('done')) AS done_tasks
            FROM agent_line_goals g
            WHERE g.status IN ('active', 'blocked')
              AND g.progress_pct = 0
              AND EXISTS (SELECT 1 FROM agent_tasks t WHERE t.goal_id = g.id)
        """)

    updated = 0
    for g in goals:
        if goal_title_substring:
            new_pct = 75
            new_status = "active"
        else:
            open_t = g.get("open_tasks") or 0
            done_t = g.get("done_tasks") or 0
            total = open_t + done_t
            if total == 0:
                continue
            if open_t > 0:
                new_pct = int(100 * done_t / total)
                new_status = g["status"]
            else:
                new_pct = 100
                new_status = "completed"

        await conn.execute("""
            UPDATE agent_line_goals
            SET progress_pct = $1, status = $2, updated_at = NOW()
            WHERE id = $3
        """, new_pct, new_status, g["id"])
        updated += 1
        print(f"Updated goal \"{g['title'][:50]}\" -> progress_pct={new_pct}% status={new_status}")

    return updated


async def main():
    parser = argparse.ArgumentParser(description="Agent team diagnostics and goal progress fix")
    parser.add_argument("--fix-goals", action="store_true", help="Fix goals with 0%% progress when tasks are done")
    parser.add_argument("--fix-goal-title", type=str, metavar="SUBSTRING", help="Fix goals whose title contains this (set 75%%, active)")
    args = parser.parse_args()

    try:
        conn = await asyncpg.connect(settings.DATABASE_URL)
    except Exception as e:
        print(f"Database connection failed: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        line_ids = await run_diagnostics(conn)

        if args.fix_goals or args.fix_goal_title:
            print("\n=== APPLYING FIX ===\n")
            n = await fix_goals(conn, goal_title_substring=args.fix_goal_title if args.fix_goal_title else None)
            print(f"\nUpdated {n} goal(s).")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
