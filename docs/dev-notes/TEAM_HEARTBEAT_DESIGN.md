# Team Heartbeat Design

Autonomous agent lines can run a periodic "heartbeat" that invokes the CEO agent (root of the org chart) with a summary of pending tasks, goal progress, recent timeline activity, and budget. The CEO uses its playbook and tools to delegate, create tasks, and review work.

## Intended flow (what should happen)

1. **User clicks "Run heartbeat"** (or the beat runs on schedule). The **CEO** agent (root of the org chart) is invoked with a briefing: goals, tasks, escalations, recent timeline, member activity, budget.
2. **CEO delegates to managers** (does not assign work for managers’ teams):
   - **Goals assigned to a manager** (an agent who has reports) → CEO uses `send_to_agent` to send the brief to that **manager**. The **manager** is the one who should create tasks and assign work to their reports (using `create_task_for_agent`). The CEO does not create tasks for a manager’s direct reports. The CEO can use `send_to_agent(..., wait_for_response=True)` so the manager runs in the same heartbeat and creates tasks immediately.
   - **Goals assigned to the CEO or unassigned** → CEO may create tasks directly with `create_task_for_agent` or `delegate_goal_to_tasks`.
   - **Report** → CEO’s final response is posted to the team timeline (full text).
3. **Managers assign work**: When a manager runs (e.g. after receiving a message from the CEO, or when the user invokes them), the manager uses `create_task_for_agent` to create tasks for their direct reports. Tasks then appear on the **Tasks** tab.
4. **Workers** see tasks via `check_my_tasks` and run when you invoke them (e.g. from chat) or when the manager triggers them via `send_to_agent` (e.g. with `wait_for_response` so the worker runs in the same flow).

So: **CEO delegates to managers; managers assign work (create tasks) for their team; workers execute tasks when invoked.**

### Org awareness (subordinates, manager, peers)

- **Heartbeat briefing** includes an **ORG CHART** section so the CEO sees who reports to whom (e.g. root → managers → workers). That makes it clear which agents are managers (have reports) and who to delegate to.
- **get_team_status_board** returns for every member:
  - **direct_reports**: list of `{agent_profile_id, agent_name, agent_handle}` — the agents who report to this member. So any agent with subordinates can see who they are and use `create_task_for_agent(assigned_agent_id=...)` or `send_to_agent(@handle, ...)` to assign work.
  - **reports_to_agent_id** / **reports_to_agent_name**: the manager of this member. So agents know who to escalate to or ask for help.
  - **peers**: members with the same manager (same `reports_to`). So managers can see other managers to coordinate or delegate to (e.g. “ask peer manager to handle X”).
- When the calling agent has `agent_profile_id` in context, the tool’s formatted output adds a **“Your context”** block: *Your direct reports: …*, *Your manager: …*, *Your peers: …*. So managers are explicitly aware of their subordinates and how to assign work; any agent is aware of their manager and peers.

## Schema

- **agent_lines**: `next_beat_at TIMESTAMPTZ`, `last_beat_at TIMESTAMPTZ` (migration 090). `heartbeat_config` JSONB: `enabled`, `interval_seconds` (or `interval` in minutes), `cron_expression`, `schedule_type`.

## Scheduling

- **check_team_heartbeats**: Celery Beat task every 60s. Finds teams where `status = 'active'`, `heartbeat_config.enabled` is true, and `next_beat_at IS NULL OR next_beat_at <= NOW()`. For each, enqueues **execute_team_heartbeat(team_id, user_id)** (does not advance next_beat_at here so failed runs can retry).
- **execute_team_heartbeat**: Loads team and CEO via `get_ceo_agent_for_heartbeat(team_id)`. Builds context string via `_build_heartbeat_context` (pending tasks count, in-progress count, goals count, messages today, team spend). Invokes CEO with `_call_grpc_orchestrator_custom_agent` with `query` = "Team heartbeat. Review and act as needed.\n\n{context}" and `extra_context = { trigger_type: 'team_heartbeat', team_id, agent_profile_id }`. On completion, updates `last_beat_at = now()`, `next_beat_at = _compute_next_beat_at(heartbeat_config)`.

## Context injection

The CEO receives the summary as the main query (human message). The orchestrator also receives `metadata.trigger_type = 'team_heartbeat'` and `metadata.team_id`, so playbooks can branch on trigger type if needed.

## Next beat computation

- If `heartbeat_config.interval_seconds` (or `interval` × 60) is set, next beat = now + interval.
- Else if `heartbeat_config.cron_expression` is set, next beat = next cron time from now.
- Else no next beat (heartbeat will not run again until configured).

## Failure handling

- If the team has no CEO, the task returns an error and does not update last_beat_at/next_beat_at, so the team remains due and will be picked again on the next beat cycle.
- next_beat_at is only advanced after a successful invoke, so transient failures result in retry on the next minute.
