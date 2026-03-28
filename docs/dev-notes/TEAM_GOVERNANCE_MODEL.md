# Team Governance Model

Structural changes (hiring agents, changing strategy/playbooks) can require user approval. The approval queue is extended with a `governance_type` so the frontend and backend can handle different approval kinds.

## Schema

- **agent_approval_queue**: `governance_type VARCHAR(50) DEFAULT 'playbook_step'` (migration 091). Values: `playbook_step` (existing), `hire_agent`, `modify_team`, `exceed_budget`, `strategy_change`.

## Orchestrator tools

- **propose_hire**: Agent proposes adding a new agent to a team. Calls `park_approval` with `governance_type=hire_agent`, prompt describing the hire, and `preview_data` with `team_id`, `proposed_name`, `proposed_role`, `proposed_handle`, `reason`. On user approval, the backend can create the profile and add the membership (implementation in respond-approval flow can resolve governance_type and execute the action).

- **propose_strategy_change**: Agent proposes changing its playbook or team goals. Calls `park_approval` with `governance_type=strategy_change`, prompt with description, and `preview_data` with `description`, `scope`, `team_id`.

## ParkApproval

- **grpc_tool_service**: `ParkApproval` handler inserts into `agent_approval_queue` including `governance_type` from the request (default `playbook_step`).
- **backend_tool_client**: `park_approval(..., governance_type="playbook_step")` passes through to the gRPC request.

## Frontend / respond flow

- Pending approvals list can show `governance_type` so the user sees "Hire request" vs "Strategy change" vs "Playbook step".
- When the user approves an entry with `governance_type=hire_agent`, the backend can read `preview_data`, create an agent profile, and add the membership to the team; similarly for `strategy_change` (e.g. apply playbook update or goal changes). The exact execution logic is left to the existing or extended respond-approval API.

## Policies

- **governance_policy** on `agent_lines` can specify `require_hire_approval`, `require_strategy_changes`, etc. Agents can check these before calling propose_hire / propose_strategy_change (tool logic or playbook steps).
