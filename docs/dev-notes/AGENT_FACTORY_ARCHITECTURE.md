# Agent Factory: Architecture & Implementation Guide

> Developer reference for Bastion’s Agent Factory: how it is wired, what is implemented today, and where to change code.

---

## What Is the Agent Factory?

The Agent Factory is a **GUI-driven, no-code agent composition system** that lets users assemble custom AI agents from modular building blocks without writing code. It sits between shallow drag-and-drop tools and developer-only frameworks (LangGraph, LangChain, etc.).

**Core idea:** a custom agent **is a workflow** — a mix of deterministic tool calls, LLM steps, human-in-the-loop gates, branching, parallelism, and loops. The UI composes that workflow; the orchestrator executes it with LangGraph checkpointing and typed tool contracts.

### What It Enables (Implemented)

- Domain-specific agents with optional `@handle` for chat invocation (`handle` may be null for schedule-only agents)
- **Interactive** chat runs, **cron/interval** schedules, and **event/watch** hooks (see `watch_config` on profiles)
- **Data source connectors** (REST-oriented definitions; execution delegated to backend + connections-service)
- **Zone 4 plugins** (Trello, Notion, CalDAV, Slack API, GitHub, VictoriaMetrics) with per-profile encrypted credentials
- **Skills**: procedural text + required tools, merged into `llm_agent` / `deep_agent` prompts; optional semantic auto-discovery
- **Team-aware** tool/skill packs when `team_config` and team metadata are present
- **Data Workspace** binding via `data_workspace_config` (workspace IDs, schema inject, read-only modes)
- **Agent memory** (key/value and log-style) when `include_agent_memory` is enabled
- **Execution logging**, token/cost fields, **budgets**, **approval queue** for scheduled runs, **circuit breaker** on schedules

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Frontend (React) — /agent-factory/*                    │
│  AgentListSidebar, AgentEditor, PlaybookEditor,          │
│  DataSourceEditor, SkillEditor, ExecutionHistoryCard   │
└───────────────────────────┬─────────────────────────────┘
                            │ REST (/api/agent-factory/...)
┌───────────────────────────▼─────────────────────────────┐
│  Backend API                                             │
│  api/agent_factory_api.py — CRUD, actions catalog,       │
│    schedules, internal notify (e.g. schedule paused)     │
│  services/agent_factory_service.py — shared validation,  │
│    row mapping, helpers (also used from gRPC tools)      │
└───────────────────────────┬─────────────────────────────┘
                            │ gRPC (orchestrator stream)
┌───────────────────────────▼─────────────────────────────┐
│  LLM Orchestrator                                        │
│  orchestrator/grpc_service.py                          │
│  orchestrator/engines/unified_dispatch.py                │
│    → CustomAgentRunner (outer LangGraph)                │
│    → playbook_graph_builder (inner LangGraph, dynamic)  │
│    → pipeline_executor (steps, variables, tools)        │
│  orchestrator/utils/action_io_registry.py               │
│  orchestrator/plugins/* — BasePlugin, PluginLoader,     │
│    integrations/*                                        │
└───────────────────────────┬─────────────────────────────┘
                            │ gRPC Tools Service
┌───────────────────────────▼─────────────────────────────┐
│  backend/services/grpc_tool_service.py                  │
│  Document/vector/graph/data-workspace tools,             │
│  ExecuteConnector → connections-service (HTTP to APIs)  │
└─────────────────────────────────────────────────────────┘
```

**Three execution tiers:**

1. **Routing** — `UnifiedDispatcher` + `RouteRegistry`; built-in routes plus **auto-routable** custom agents (`custom_<handle>`).
2. **Orchestration** — `CustomAgentRunner`: load profile/playbook, enrich metadata, run inner graph, approvals, format response.
3. **Execution** — `pipeline_executor`: variable resolution, type coercion, tool/LLM/loop/branch/parallel dispatch, traces.

---

## Core Concepts

### 1. Agent Profile (`agent_profiles`)

Primary artifact per custom agent. Notable columns (see `backend/postgres_init/01_init.sql` and `_row_to_profile` in `agent_factory_service.py`):

| Field | Purpose |
|-------|---------|
| `id`, `user_id`, `name`, `handle`, `description`, `is_active` | Identity and lifecycle |
| `default_playbook_id` | Playbook run on invoke |
| `model_preference`, `model_source`, `model_provider_type` | Per-agent model; provenance for admin vs user |
| `max_research_rounds` | Cap for multi-round behavior where applicable |
| `system_prompt_additions` | Extra system instructions |
| `knowledge_config` | JSONB: vector/graph scope for retrieval |
| `journal_config` | JSONB: auto-journal behavior |
| `team_config` | JSONB: team visibility and shared resources |
| `watch_config` | JSONB: event-driven triggers |
| `chat_history_enabled`, `chat_history_lookback` | DB columns; runtime merges as `prompt_history_enabled` + lookback in `CustomAgentRunner` |
| `persona_mode`, `persona_id` | `none` / `default` / `specific` + optional `personas` row |
| `include_user_context`, `include_datetime_context` | Inject user profile snippet and clock context |
| `include_user_facts`, `include_facts_categories` | User fact store; optional category filter |
| `include_agent_memory` | Load agent memory keys into pipeline metadata |
| `auto_routable` | Eligible for semantic auto-route (see below) |
| `chat_visible` | Whether agent appears in @mention UI |
| `category` | Sidebar grouping |
| `data_workspace_config` | JSONB: workspace binding, schema inject, instructions |
| `default_run_context`, `default_approval_policy` | Run and governance defaults |
| `is_locked`, `is_builtin` | Edit locks; shipped templates |

**Handle uniqueness:** `UNIQUE (user_id, handle)` where `handle` is non-null.

### 2. Custom Playbook (`custom_playbooks`)

Workflow stored as JSONB `definition` with a top-level `steps` array. Steps may use `step_type` or alias `type` (orchestrator normalizes via `_step_type()`).

Example **tool** step:

```json
{
  "name": "search_for_documents",
  "step_type": "tool",
  "action": "search_documents",
  "inputs": {
    "query": "{query}",
    "limit": 10
  },
  "params": {
    "scope": "my_docs"
  },
  "output_key": "search_results",
  "condition": "{search_results.count} > 0"
}
```

**Validated step types** (`agent_factory_service.VALID_STEP_TYPES`, must match orchestrator):

| Type | Role |
|------|------|
| `tool` | Registry action or special `action` patterns (below) |
| `llm_task` | Single LLM call; optional structured output |
| `llm_agent` | ReAct loop with `bind_tools` |
| `deep_agent` | Multi-phase subgraph (`phases` with reason/act/search/evaluate/synthesize/refine) |
| `approval` | HITL pause; integrates with outer `approval_gate` |
| `loop` | Iterate child steps over a list |
| `parallel` | Run child steps concurrently (see implementation) |
| `branch` | Conditional subgraph routing |
| `browser_authenticate` | Browser-based auth flow node (inner graph; see `playbook_graph_builder.py`) |

**Common structural fields (non-tool steps):**

| Step types | Fields |
|------------|--------|
| `loop` | `steps` (child steps), `max_iterations` (default 3) |
| `parallel` | `parallel_steps` (list of child steps run with `asyncio.gather`) |
| `branch` | `branch_condition`, `then_steps`, `else_steps` |
| `approval` | `prompt`, optional `timeout_minutes`, `on_reject` |
| `deep_agent` | `phases` — each phase has `name`, `type` ∈ {`reason`,`act`,`search`,`evaluate`,`synthesize`,`refine`}, plus type-specific fields (see `validate_playbook_definition`) |

**Special `tool` actions (not plain registry names):**

| Pattern | Behavior |
|---------|----------|
| `connector:<connector_uuid>:<endpoint_id>` | `BackendToolClient.execute_connector` → Tools Service → outbound HTTP via connections-service; requires `agent_profile_id` in run metadata |
| `email:<connection_id>:<tool_name>` | Runs registry tool `tool_name` with `connection_id` + user binding (Microsoft Graph / email-scoped tools) |

Playbook metadata on `custom_playbooks` includes `triggers` (GIN indexed), `is_template`, `category`, `tags`, `required_connectors`.

### 3. Action I/O Registry

**Location:** `llm-orchestrator/orchestrator/utils/action_io_registry.py`

Single registry entry per tool: **inputs** model, optional **params** model, **outputs** model (must include `formatted: str`), and async **tool_function**.

Powers:

1. Workflow Composer — `GET /api/agent-factory/actions` exposes schemas for wiring UI
2. `is_type_compatible()` for wiring validation
3. Runtime — `get_action(name).tool_function` for standard tools; email/connector paths bypass or wrap as above

### 4. Type System

Pragmatic types: `text`, `number`, `boolean`, `date`, `record`, `list[T]`, `file_ref`, `any`. **`text` is the universal connector** (coercion rules in `.cursor/rules/tool-io-contracts.mdc` and registry code).

### 5. Data Source Connectors

Connector **definitions** live in `data_source_connectors`; per-profile bindings and encrypted credentials in `agent_data_sources`.

Composer surfaces endpoints as pseudo-actions `connector:<id>:<endpoint_id>`. **Runtime path:** orchestrator `pipeline_executor.execute_step` → gRPC `ExecuteConnector` on Tools Service → backend loads definition + credentials → **connections-service** performs the outbound HTTP call (Zone 3). There is no separate `ConnectorRuntime` class in the orchestrator; the docstring pattern is “execute_connector + connections-service.”

Supported connector **kinds** in product docs include REST (primary), GraphQL, scraper, file parser, database, RSS — exact coverage depends on connector editor and executor implementation in backend/connections-service.

### 6. Plugins (Zone 4)

**Location:** `llm-orchestrator/orchestrator/plugins/`

- `base_plugin.py`, `plugin_loader.py`
- `integrations/`: `trello_plugin`, `notion_plugin`, `caldav_plugin`, `slack_plugin`, `github_plugin`, `victoria_plugin` (VictoriaMetrics-oriented)

Credentials: `agent_plugin_configs` per `(agent_profile_id, plugin_name)`. At runtime, `category.startswith("plugin:")` triggers credential injection and `plugin.configure()` before invocation.

### 7. Skills (`agent_skills` table — applied via migrations)

Skills store **procedure text** and **required_tools**. Referenced by step `skill_ids` / `skills` and optionally **auto-discovered** via `search_skills` gRPC when `auto_discover_skills` is true on a step. Team-level skills can be merged from metadata (`team_skill_ids`). See `_build_augmented_tool_names` / `_resolve_and_inject_skills` in `pipeline_executor.py`.

---

## Execution Flow (End-to-End)

### Entry Points

| Path | Mechanism |
|------|-----------|
| Chat `@handle` / profile | Message metadata includes `agent_profile_id` → `grpc_service.py` → `UnifiedDispatcher.dispatch_custom_agent()` |
| Auto-route | `load_auto_routable_agents(user_id)` in `orchestrator/routes/definitions/__init__.py`: loads profiles with `auto_routable`, registers routes `custom_<handle>`, **60s TTL** per user; switching users unregisters previous custom routes |
| Schedule | Celery `scheduled_agent_tasks` → backend/orchestrator with profile id and optional `trigger_input` / schedule context |
| Events | Driven by `watch_config` and related backend machinery (see main Agent Factory docs for UX) |

### Outer LangGraph (`CustomAgentRunner`)

File: `llm-orchestrator/orchestrator/agents/custom_agent_runner.py`

```
load_profile → prepare_context → execute_pipeline → format_response
                     ↑                    ↓
                     └────────── approval_gate ← interrupt_before
                                      ↓
                               format_response
```

- **`load_profile` / `prepare_context`** — gRPC fetch profile + playbook; persona; optional history string into `inputs["history"]` when `prompt_history_enabled`; user context/facts/agent memory/data workspace hints into `pipeline_metadata`
- **`execute_pipeline`** — builds inner graph with `build_playbook_graph`, thread id `playbook_{user_id}_{playbook_id}`, same Postgres checkpointer class as outer graph, **different `thread_id`**
- **`approval_gate`** — `interrupt_before=["approval_gate"]` for HITL resume
- **`format_response`** — final structured response for streaming

### Inner LangGraph (`playbook_graph_builder.py`)

`CustomAgentRunner` compiles steps with `build_playbook_graph(steps, checkpointer=...)` and runs `graph.ainvoke(...)` (thread id `playbook_{user_id}_{playbook_id}`).

Each top-level step becomes a LangGraph node with:

1. **Condition wrapper** — skip step without running if `condition` is false
2. **Tracing** — `execution_trace` entries (input/output snapshots, duration, status, token usage, `tool_call_trace` for `llm_agent`)

**Node implementations** call into `pipeline_executor` helpers: `execute_step` (tools, connectors, email-prefixed actions), `_execute_llm_step`, `_execute_llm_agent_step`, `_execute_deep_agent_step`. **`loop`** and **`branch`** build **nested** compiled graphs via recursive `build_playbook_graph` and `ainvoke`. **`parallel`** runs `parallel_steps` with `asyncio.gather`.

**Interrupts (inner graph):** `approval` and `browser_authenticate` nodes are registered with LangGraph **`interrupt_after`** on the inner workflow so the outer runner can surface HITL UI before continuing.

**`execute_pipeline()`** (same module file) implements a **linear** interpreter over the same step types (including nested loop/branch via self-recursion). It supports `resume_after_step_name` for approval resume. It is **not** invoked by `CustomAgentRunner` today; the supported production path is the dynamic graph above.

### Pipeline executor responsibilities

File: `llm-orchestrator/orchestrator/engines/pipeline_executor.py`

- Variable resolution `{query}`, `{step.field}`, wildcards, section extracts
- Type coercion from registry input types
- `llm_agent` / `deep_agent` tool lists + skills + team augmentation
- Read-only enforcement for `query_data_workspace` when workspace access mode is `read`
- Unified dict tool results; `formatted` for LLM `ToolMessage` content

---

## Variable Reference System

Resolved against `playbook_state` + initial `inputs` (e.g. `query`, `history`, `trigger_input`). **No `eval()`.**

| Syntax | Meaning |
|--------|---------|
| `{query}` | User query |
| `{step_name}` | Whole step output dict |
| `{step_name.field}` | Field access |
| `{step_name.nested.path}` | Deep path |
| `{editor_refs_*}` | Prefix wildcard aggregation |
| `{editor_refs_DOCS_section:Heading}` | Section extract |

Conditions: comparisons, `is defined`, `matches` (regex), `AND`/`OR` — tokenized evaluator.

---

## Human-in-the-Loop (HITL)

`approval` and `browser_authenticate` use **two layers** of checkpointing:

1. **Inner** `StateGraph`: `interrupt_after=[...]` on approval/browser nodes (`build_playbook_graph` compile args)
2. **Outer** `CustomAgentRunner`: `interrupt_before=["approval_gate"]` so the runner can emit chunks and wait for user action

State carries `pending_approval` and/or `pending_auth` (browser: `screenshot`, `login_url`, `session_id`, etc.). The stream surfaces permission / browser UI payloads to the frontend; resume uses the same outer `thread_id` and checkpoint.

Approval steps support `on_reject`: `stop`, `skip`, or `continue`.

**Scheduled runs:** pending approvals can be recorded in `agent_approval_queue` for asynchronous review.

---

## Database Schema (Agent Factory–Related)

| Table | Role |
|-------|------|
| `agent_profiles` | Agent identity and configuration |
| `custom_playbooks` | Workflow definitions |
| `agent_factory_sidebar_categories` | User-defined sidebar folders |
| `data_source_connectors` | Connector templates |
| `agent_data_sources` | Profile ↔ connector + encrypted credentials |
| `agent_plugin_configs` | Per-profile plugin secrets |
| `agent_execution_log` | Runs: status, duration, tokens, cost, `metadata.execution_trace`, `trigger_type`, `schedule_id` |
| `agent_discoveries` | Structured discovery rows per execution |
| `agent_schedules` | Cron/interval, circuit breaker (`consecutive_failures`, `max_consecutive_failures`) |
| `agent_budgets` | Monthly spend caps |
| `agent_approval_queue` | Background approval rows |
| `agent_skills` | Skill library (created in migrations; may not appear in a minimal `01_init` excerpt — rely on migrated DBs) |

---

## API & Frontend Pointers

- **REST base:** `/api/agent-factory/` — see `backend/api/agent_factory_api.py`
- **Actions catalog:** `GET /actions` for composer wiring
- **Execution UI:** `frontend/src/components/agent-factory/ExecutionHistoryCard.js` (history + step trace drawer + export)
- **Main shell:** `frontend/src/components/AgentFactoryPage.js`, routes under `/agent-factory/...`

---

## Observability & Operations

### Execution tracing

Traces persist under `agent_execution_log.metadata` (e.g. `execution_trace`) for support and power users.

### Schedule circuit breaker

On too many consecutive failures, the schedule is paused; Celery calls internal `POST /api/agent-factory/internal/notify-schedule-paused` (see `notify_schedule_paused` in `agent_factory_api.py`). WebSocket payload shape:

```json
{
  "type": "agent_notification",
  "subtype": "schedule_paused",
  "agent_name": "...",
  "title": "...",
  "preview": "...",
  "timestamp": "..."
}
```

Clients key off `type` + `subtype` (not a single dotted name).

### Logging agent execution

Orchestrator/backend log runs via gRPC (`LogAgentExecution`) so costs/tokens/status align with `agent_execution_log`.

---

## Design Decisions (Summary)

- **Dual LangGraph:** Outer runner for profile lifecycle and approvals; inner graph for playbook steps; **shared checkpointer implementation, distinct thread IDs**
- **Unified tool return:** Dict with typed fields + mandatory `formatted`
- **Credential isolation:** Injected at call site; not exposed in LLM tool schemas
- **Safe conditions:** No `eval()` on user-authored playbook strings
- **Auto-route cache:** 60s per user, single active user’s custom routes in registry at a time

---

## Key Source Files (Accurate Paths)

```
llm-orchestrator/orchestrator/
  grpc_service.py                 # Stream dispatch, custom agent entry
  agents/custom_agent_runner.py   # Outer LangGraph
  engines/unified_dispatch.py     # Routing to custom agent
  engines/playbook_graph_builder.py
  engines/pipeline_executor.py
  utils/action_io_registry.py
  routes/definitions/__init__.py  # load_auto_routable_agents, TTL cache
  plugins/base_plugin.py
  plugins/plugin_loader.py
  plugins/integrations/*.py

backend/
  api/agent_factory_api.py
  services/agent_factory_service.py
  services/grpc_tool_service.py   # Tools + ExecuteConnector
  services/celery_tasks/scheduled_agent_tasks.py

connections-service/              # Outbound HTTP for connector execution

frontend/src/components/
  AgentFactoryPage.js
  agent-factory/*.js

docs/
  AGENT_FACTORY.md
  AGENT_FACTORY_TECHNICAL_GUIDE.md
  AGENT_FACTORY_TOOLS.md
  AGENT_FACTORY_UX.md
  AGENT_FACTORY_EXAMPLES.md
```

---

## If You Are Porting This Architecture

Suggested order:

1. Action I/O registry + `formatted` convention  
2. Pipeline executor (variables, coercion, connector/email special cases)  
3. Playbook schema + condition evaluator  
4. Outer `CustomAgentRunner` + Postgres checkpointer  
5. Composer UI + `GET /actions`  
6. HITL (`approval`, `interrupt_before`)  
7. Plugins and connector execution path (isolated credentials)

---

## Summary

Bastion’s Agent Factory combines **LangGraph** execution with a **typed tool registry**, **no-code playbooks**, **connectors** (Tools Service + connections-service), **plugins**, **skills**, and **operational** features (schedules, budgets, approvals, traces). This document reflects the **current** layout under `llm-orchestrator/orchestrator/engines/` for dispatch and execution, and the **actual** WebSocket and gRPC shapes described above.
