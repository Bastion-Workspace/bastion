# Agent Lines: Architecture and Data Model

> Autonomous agent lines (Paperclip-like): org chart, inter-agent messaging, goal hierarchy, and heartbeat-driven execution. This doc covers the foundation: lines and org chart.

---

## Overview

Agent Lines let users group Agent Factory profiles into a **line** with a hierarchy (who reports to whom). Lines are the container for:

- **Org chart** — `agent_line_memberships` with `reports_to` (self-referential)
- **Inter-agent messages** — timeline of communication (Phase 2)
- **Goals** — hierarchy from mission to task (Phase 3)
- **Tasks** — ticket system assigned to agents (Phase 4)
- **Heartbeat** — CEO agent wakes on schedule and delegates (Phase 5)

This document describes the **database schema**, **service layer**, **API surface**, and **governance model** for lines and memberships only.

---

## Database Schema

### `agent_lines`

| Column | Type | Purpose |
|--------|------|---------|
| `id` | UUID PK | Line identifier |
| `user_id` | VARCHAR FK users | Owner |
| `name` | VARCHAR(255) | Display name |
| `description` | TEXT | Optional |
| `mission_statement` | TEXT | High-level goal (e.g. "Build #1 AI note-taking app to $1M MRR") |
| `status` | VARCHAR(50) | `active`, `paused`, `archived` |
| `heartbeat_config` | JSONB | `schedule_type`, `interval`, `cron_expression`, `enabled` (Phase 5) |
| `governance_policy` | JSONB | `require_hire_approval`, `require_strategy_changes`, `auto_approve_task_creation` (Phase 6) |
| `created_at`, `updated_at` | TIMESTAMPTZ | |

**Migration:** `086_agent_teams.sql` (historical); `101_agent_teams_to_agent_lines.sql` renames to `agent_lines`. Tables also in `01_init.sql` for fresh installs.

### `agent_line_memberships`

| Column | Type | Purpose |
|--------|------|---------|
| `id` | UUID PK | Membership identifier |
| `line_id` | UUID FK agent_lines ON DELETE CASCADE | Line |
| `agent_profile_id` | UUID FK agent_profiles ON DELETE CASCADE | Agent in the line |
| `role` | VARCHAR(100) | Free-form: `ceo`, `manager`, `worker`, `specialist` |
| `reports_to` | UUID FK agent_line_memberships(id) ON DELETE SET NULL | Parent in org chart (NULL = root) |
| `hire_approved` | BOOLEAN | Governance: whether this member was approved (Phase 6) |
| `hire_approved_at` | TIMESTAMPTZ | |
| `joined_at` | TIMESTAMPTZ | |

**Unique:** `(line_id, agent_profile_id)` — an agent can be in a line only once.

**Indexes:** `line_id`, `agent_profile_id`, `reports_to` for list/org-chart queries.

---

## Service Layer

**Module:** `backend/services/agent_line_service.py`

| Function | Purpose |
|----------|---------|
| `create_line(user_id, name, ...)` | Insert line, return serialized row |
| `update_line(line_id, user_id, ...)` | Partial update; only provided fields |
| `delete_line(line_id, user_id)` | Delete line (CASCADE memberships) |
| `list_lines(user_id)` | All lines with `member_count` |
| `get_line(line_id, user_id)` | Line + full `members` list (with agent name/handle) |
| `add_member(line_id, user_id, agent_profile_id, role, reports_to)` | Upsert membership |
| `remove_member(line_id, user_id, agent_profile_id)` | Delete by agent |
| `remove_member_by_membership_id(line_id, user_id, membership_id)` | Delete by membership id |
| `update_member(line_id, user_id, membership_id, role, reports_to)` | Update role/reports_to |
| `get_org_chart(line_id, user_id)` | Tree: roots = no `reports_to`, each node has `children` |
| `get_line_budget_summary(line_id, user_id)` | Aggregate `agent_budgets` for all members |
| `get_ceo_membership_id(line_id)` | Single root membership (for heartbeat CEO) |

All operations that modify data enforce **user_id**: the line must belong to the current user, and added agents must be that user's profiles.

---

## API Surface

**Prefix:** `/api/agent-factory/lines`  
**Router:** `backend/api/agent_line_api.py` (included in main with `prefix="/api/agent-factory"`).

| Method | Path | Description |
|--------|------|-------------|
| GET | `/lines` | List user's lines (with member count) |
| POST | `/lines` | Create line |
| GET | `/lines/{line_id}` | Get line + members |
| PUT | `/lines/{line_id}` | Update line |
| DELETE | `/lines/{line_id}` | Delete line |
| POST | `/lines/{line_id}/members` | Add member (body: `agent_profile_id`, `role`, `reports_to`) |
| PUT | `/lines/{line_id}/members/{membership_id}` | Update member (body: `role`, `reports_to`) |
| DELETE | `/lines/{line_id}/members/{membership_id}` | Remove by membership id |
| DELETE | `/lines/{line_id}/members/by-agent/{agent_profile_id}` | Remove by agent |
| GET | `/lines/{line_id}/org-chart` | Org chart tree |
| GET | `/lines/{line_id}/budget-summary` | Aggregated budget |

Auth: all routes use `get_current_user`; line ownership is enforced in the service.

---

## Governance Model (Foundation)

- **Line scope:** Every line has a `user_id`; only that user can CRUD the line and members.
- **Agent ownership:** Only the user's own `agent_profiles` can be added to their lines.
- **Governance policy** (Phase 6): `governance_policy` on `agent_lines` will drive which actions require approval (e.g. hire, strategy change). The column is present; behavior is implemented in Phase 6.

---

## Relationship to Agent Profiles

- **agent_profiles** remain the single source of identity and configuration (playbook, model, persona).
- A profile can belong to **multiple lines** (different memberships).
- Lines do **not** duplicate profile data; they only reference `agent_profile_id` and store `role` and `reports_to`.
- Budget and execution log stay **per profile**; line budget summary is an aggregation over members' `agent_budgets`.

---

## Frontend Components

| Component | Path | Purpose |
|-----------|------|---------|
| LineListPanel | `agent-factory/LineListPanel.js` | List lines, create button, navigate to line |
| TeamEditor | `agent-factory/TeamEditor.js` | Create/edit line (name, description, mission, status) |
| OrgChartView | `agent-factory/OrgChartView.js` | Recursive tree of members with role labels |
| TeamMembershipEditor | `agent-factory/TeamMembershipEditor.js` | Add/remove members, set role and reports_to |

**API client:** `frontend/src/services/agentFactoryService.js` — `listLines`, `createLine`, `getLine`, `updateLine`, `deleteLine`, `addLineMember`, `updateLineMember`, `removeLineMember`, `removeLineMemberByAgent`, `getLineOrgChart`, `getLineBudgetSummary`.

---

## References

- Plan: Autonomous Agent Lines (phases 1–7)
- Agent Factory: `docs/dev-notes/AGENT_FACTORY_ARCHITECTURE.md`
- Migration: `backend/postgres_init/migrations/101_agent_teams_to_agent_lines.sql`
