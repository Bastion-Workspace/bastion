# Agent Task System

Agent teams have a task/ticket system: tasks can be created, assigned to agents, moved through a status lifecycle, and linked to goals and message threads.

## Schema

- **agent_tasks**: `id`, `line_id` (FK agent_lines), `title`, `description`, `status` (backlog, assigned, in_progress, review, done, cancelled), `assigned_agent_id` (FK agent_profiles), `created_by_agent_id`, `goal_id` (FK agent_line_goals), `priority`, `thread_id` (FK agent_messages for discussion), `execution_id` (FK agent_execution_log when worked on), `metadata` JSONB, `due_date`, `created_at`, `updated_at`.
- Indexes: `(team_id, status)`, `(assigned_agent_id, status)`.

## Service (agent_task_service)

- **create_task** / **update_task** / **delete_task**: CRUD with team ownership.
- **assign_task(task_id, agent_profile_id, user_id)**: Sets assigned_agent_id, status to `assigned`, and creates an `agent_messages` entry of type `task_assignment`.
- **get_agent_work_queue(agent_profile_id, team_id, user_id)**: Tasks assigned to the agent, excluding done/cancelled, ordered by priority and created_at.
- **transition_task(task_id, user_id, new_status)**: State machine (backlog→assigned→in_progress→review→done; cancelled from most states).
- **list_team_tasks(team_id, user_id, status_filter, agent_filter)**.
- **get_task(task_id, user_id)**: Task plus optional `thread` (messages from get_thread(thread_id)).

## Orchestrator tools (agent_task_tools)

- **create_task_for_agent**: Create a task and assign to an agent (uses _pipeline_metadata for team_id and created_by_agent_id).
- **check_my_tasks**: Get work queue for the current agent (team_id and agent_profile_id from pipeline metadata).
- **update_task_status**: Transition task to in_progress, review, done, etc.
- **escalate_task**: Reassign task to another agent (e.g. manager) via assign_task_to_agent gRPC.

## API

- `GET /api/agent-factory/teams/{team_id}/tasks` — list with optional `status` and `agent_id` query params.
- `POST /api/agent-factory/teams/{team_id}/tasks` — create (body: title, description, assigned_agent_id, goal_id, priority, created_by_agent_id, due_date).
- `GET /api/agent-factory/teams/{team_id}/tasks/{task_id}` — get task with thread.
- `PUT /api/agent-factory/teams/{team_id}/tasks/{task_id}` — update.
- `POST /api/agent-factory/teams/{team_id}/tasks/{task_id}/assign?agent_profile_id=...` — assign.
- `POST /api/agent-factory/teams/{team_id}/tasks/{task_id}/transition?new_status=...` — transition.
- `DELETE /api/agent-factory/teams/{team_id}/tasks/{task_id}` — delete.

## Frontend

- **TaskBoard**: Kanban columns (backlog, assigned, in_progress, review, done); move cards between columns via transition; edit/delete per task.
- **TaskDetail**: Title, description, status dropdown, assigned agent, due date, and discussion thread (TimelineMessage list when thread_id is set).

## Timeline integration

Task state changes can be reflected as system or status_update messages on the team timeline; assign_task already creates a task_assignment message. Further integration (e.g. status_change messages) can be added in the service layer.
