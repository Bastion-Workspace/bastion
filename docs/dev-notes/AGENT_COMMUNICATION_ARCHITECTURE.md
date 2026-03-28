# Agent Communication and Team Timeline

> Inter-agent messaging, team timeline, and tool contracts for send_to_agent, start_agent_conversation, halt_agent_conversation.

---

## Overview

Agents in a team can send messages to each other. Messages are stored in `agent_messages` and displayed on the **team timeline**. Tools in the orchestrator persist messages via gRPC and can optionally wait for the target agent's response.

---

## Database Schema

### `agent_messages` (migration 087)

| Column | Type | Purpose |
|--------|------|---------|
| `id` | UUID PK | Message id |
| `line_id` | UUID FK agent_lines | Line scope |
| `from_agent_id` | UUID FK agent_profiles | Sender (NULL = system) |
| `to_agent_id` | UUID FK agent_profiles | Recipient (NULL = broadcast) |
| `message_type` | VARCHAR(50) | task_assignment, status_update, request, response, delegation, escalation, report, system |
| `content` | TEXT | Body |
| `metadata` | JSONB | Attachments, typed_data, execution_id |
| `parent_message_id` | UUID FK agent_messages | Threading |
| `created_at` | TIMESTAMPTZ | |

**Indexes:** `(team_id, created_at DESC)`, `(from_agent_id, created_at DESC)`, `(to_agent_id, created_at DESC)`, `(parent_message_id)`.

---

## Service Layer

**Module:** `backend/services/agent_message_service.py`

- `create_message(team_id, from_agent_id, to_agent_id, message_type, content, metadata, parent_message_id, user_id)` — Inserts message, optionally emits WebSocket `team_timeline_update`.
- `get_team_timeline(team_id, user_id, limit, offset, message_type_filter, agent_filter, since)` — Paginated list `{ items, total }`.
- `get_agent_messages(agent_profile_id, team_id, user_id, limit, offset)` — Messages from or to one agent.
- `get_thread(parent_message_id, user_id)` — Root message + replies.
- `get_team_timeline_summary(team_id, user_id)` — `message_count_today`, `active_threads`, `last_activity_at`.

**WebSocket:** On create, `get_websocket_manager().send_team_timeline_update(team_id, { type: "team_timeline_update", message })` so subscribed clients get live updates.

---

## gRPC and Orchestrator Tools

### Backend gRPC

- **CreateAgentMessage** (proto `CreateAgentMessageRequest` / `CreateAgentMessageResponse`): calls `agent_message_service.create_message`, returns `message_id` and full `message_json`.

### Orchestrator tools (`llm-orchestrator/orchestrator/tools/agent_communication_tools.py`)

| Tool | Purpose |
|------|---------|
| **send_to_agent** | Send message to another agent by @handle or profile id. Params: `team_id`, `message_type`, `wait_for_response`, `timeout_minutes`. If `wait_for_response`, invokes target via `invoke_agent_tool` and returns response. Persists to timeline when `team_id` and `from_agent_id` are in `_pipeline_metadata`. |
| **start_agent_conversation** | Start multi-turn conversation: creates root system message, then invokes first participant with seed message. |
| **halt_agent_conversation** | Set agent memory key `conversation_halt:{conversation_id}` so running conversations can check and stop. |

**Contract:** Tools use `_pipeline_metadata` for `agent_profile_id` (sender) and `team_id` (for persistence). Same-owner rule: team and agents must belong to the same user.

---

## API and WebSocket

- `GET /api/agent-factory/teams/{team_id}/timeline` — Paginated timeline (query: limit, offset, message_type, agent, since).
- `GET /api/agent-factory/teams/{team_id}/timeline/summary` — Summary stats.
- `GET /api/agent-factory/teams/{team_id}/messages/{message_id}/thread` — Thread (root + replies).
- `WS /api/ws/team-timeline/{team_id}?token=...` — Live timeline updates; server sends `team_timeline_update` with full message object on new messages.

---

## Frontend

- **TeamTimelinePage** — Route `/agent-factory/teams/:teamId/timeline`; fetches timeline, filters, subscribes to WebSocket, merges live messages.
- **TimelineMessage** — Renders one message with type icon/chip, from → to, time, content.
- **TimelineFilters** — Message type, agent, since (datetime).
- **AgentActivityPanel** — Sidebar with per-agent message counts and last activity.

---

## Security and Rate Limits

- **Ownership:** Timeline and thread endpoints enforce line `user_id` via `agent_line_service.get_line(line_id, user_id)`.
- **Same-owner:** Backend creates messages only for the requesting user's team and agents.
- Rate limits and max conversation turns can be added later (e.g. daily message cap per agent, max_turns in start_agent_conversation).

---

## References

- Plan: Autonomous Agent Lines (Phase 2)
- Teams: `docs/dev-notes/AGENT_TEAMS_ARCHITECTURE.md`
- Migration: `backend/sql/migrations/087_agent_messages.sql`
