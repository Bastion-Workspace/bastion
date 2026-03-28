# Goal Hierarchy Design

Agent teams can have a hierarchy of goals aligned to the team mission. Goals support tree structure, assignment to agents, progress tracking, and context injection into the agent system prompt.

## Schema

- **agent_line_goals**: `id`, `line_id`, `parent_goal_id` (self-ref), `title`, `description`, `status` (active, completed, blocked, cancelled), `assigned_agent_id` (FK agent_profiles), `priority`, `progress_pct`, `due_date`, `created_at`, `updated_at`.
- Index on `(team_id, parent_goal_id)` for tree queries.

## Service (agent_goal_service)

- **create_goal** / **update_goal** / **delete_goal**: CRUD with team ownership check.
- **get_goal_tree(team_id, user_id)**: Returns root-level goals with nested `children` (recursive from flat list).
- **get_goal_ancestry(goal_id, user_id)**: Returns path from leaf goal up to root (for context injection).
- **update_progress(goal_id, user_id, progress_pct)**: Updates progress; can be extended to roll up to parents.
- **get_goals_for_agent(agent_profile_id, team_id, user_id)**: Goals assigned to an agent in that team.

## Context Injection (custom_agent_runner)

When `metadata` contains `team_id` and `agent_profile_id`, the runner:

1. Calls gRPC **GetGoalsForAgent** to get goals assigned to that agent.
2. For the first assigned goal, calls **GetGoalAncestry** to get the path to the root.
3. Formats a "GOAL CONTEXT" block (Mission → … → Your goal with status and progress).
4. Sets `pipeline_metadata["goal_context_str"]`, which is appended to the system message in `pipeline_executor._build_system_message`.

Agents then see their place in the goal tree and can use **report_goal_progress** and **list_team_goals** tools.

## Orchestrator Tools (agent_goal_tools)

- **list_team_goals(team_id)**: Returns formatted goal tree and JSON tree for downstream use.
- **report_goal_progress(goal_id, progress_pct)**: Updates progress on an assigned goal (0–100).

Both use the backend gRPC client (GetTeamGoalsTree, UpdateGoalProgress).

## API

- `GET /api/agent-factory/teams/{team_id}/goals` — goal tree.
- `POST /api/agent-factory/teams/{team_id}/goals` — create goal (body: title, description, parent_goal_id, assigned_agent_id, status, priority, progress_pct, due_date).
- `PUT /api/agent-factory/teams/{team_id}/goals/{goal_id}` — update goal.
- `DELETE /api/agent-factory/teams/{team_id}/goals/{goal_id}` — delete goal.

## Frontend

- **GoalTreeView**: Recursive tree of goals with status, progress bar, edit/delete actions.
- **GoalEditor**: Form for create/edit (title, description, parent, assigned agent, status, priority, progress %, due date). Parent and agent options passed as props (e.g. from team goals and team members).

## Progress Rollup (future)

`update_progress` could propagate progress to parent goals (e.g. average of children or weighted by priority). Currently it only updates the single goal.
