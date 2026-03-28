# User Facts Architecture

System overview, data model, injection paths, and configuration for the user fact store (remembered facts) used across chat, automation skills, and Agent Factory.

## Data Model

**Table:** `user_facts`

| Column       | Type      | Description |
|-------------|-----------|-------------|
| id          | SERIAL PK | Row id |
| user_id     | VARCHAR   | Owner |
| fact_key    | VARCHAR   | Unique per user (e.g. `job_title`, `preferred_language`) |
| value       | TEXT      | Fact value |
| category    | VARCHAR   | `general`, `work`, `preferences`, `personal` (for filtering) |
| source      | VARCHAR   | `user_manual`, `agent`, `api` (provenance) |
| confidence  | FLOAT     | 0.0–1.0; manual=1.0, agent=0.8 default |
| expires_at  | TIMESTAMPTZ | Optional TTL; NULL = never expires |
| embedding   | FLOAT[]   | Optional vector for relevance filtering (async-populated) |
| created_at  | TIMESTAMPTZ | Set on insert |
| updated_at  | TIMESTAMPTZ | Set on insert/update |

**Unique:** `(user_id, fact_key)` — upsert semantics.

## Write Paths

1. **Settings UI (manual)**  
   User adds/edits facts in Settings → Profile → Remembered Facts.  
   REST: `POST /api/settings/user/facts` with `fact_key`, `value`, `category`, optional `expires_at`.  
   Stored with `source='user_manual'`, `confidence=1.0`.

2. **REST API**  
   Same endpoint; callers can set `expires_at` (ISO string).  
   Always `source='user_manual'`.

3. **Agent tool**  
   `save_user_fact_tool` (orchestrator) calls backend gRPC `UpsertUserFact` with `source='agent'`.  
   Stored with `confidence=0.8`.  
   Respects user-level **facts_write_enabled**: if disabled, gRPC returns error and no write occurs.

After any write, a Celery task `embed_user_fact_task` runs asynchronously to fill `embedding` for relevance filtering.

## Injection Pathways and Opt-In / Opt-Out

| Path | Who | Opt-in / opt-out | Notes |
|------|-----|-------------------|-------|
| **Chat (static)** | ChatAgent | **User-level opt-out:** `facts_inject_enabled` in user_settings. Default True. | Backend `grpc_context_gatherer` checks before fetching/injecting. |
| **Automation / skills** | AutomationEngine | **Per-skill:** `include_user_facts` on the Skill definition. Default False. | Facts come from gRPC metadata `user_facts` (same string as chat). |
| **Agent Factory** | CustomAgentRunner | **Per-profile:** `include_user_facts` on the agent profile. Optional **category filter:** `include_facts_categories` (list). | Runner fetches via GetUserFacts gRPC; filters by category if non-empty. |
| **Agent Factory playbooks** | Pipeline executor | **Per-step (restrictive only):** optional `user_facts_policy` on each playbook step. | See **Per-playbook-step policy** below. |
| **Explicit read** | Any agent | **Tool call:** `get_user_facts_tool(category=...)` | No automatic injection; agent decides when to query. |

**Agent Factory symmetric behavior (no user facts = no save):** When an agent profile has **include_user_facts** set to False, the pipeline executor does not bind `get_user_facts` or `save_user_fact` for that run. The agent therefore cannot read or save user facts (“vacuum” mode). This applies to both `llm_agent` and `deep_agent` steps. Implemented in `orchestrator/engines/pipeline_executor.py` via `_effective_user_facts_policy()` and `_filter_user_fact_tools_by_policy()`.

### Per-playbook-step policy

Optional field **`user_facts_policy`** on playbook step objects (JSON). Only has effect when the profile has **`include_user_facts` true**; steps cannot expand access beyond the profile.

| Value | Effect |
|-------|--------|
| Omitted or `inherit` | Same as today: `user_facts_str` in the system message for that step (when profile supplies it) and both fact tools may be bound. |
| `no_write` | `save_user_fact` is not bound for that step; injection and `get_user_facts` remain. Deterministic **tool** steps that call `save_user_fact` are blocked with a clear error. |
| `isolated` | `user_facts_str` is omitted from the system message for that step; neither fact tool is bound. Deterministic tool steps for `save_user_fact` / `get_user_facts` are blocked. |

Applies to **`llm_task`**, **`llm_agent`**, **`deep_agent`**, and deterministic **`tool`** steps. For **`deep_agent`**, the parent step’s policy applies to the shared system message for all phases; **`act`** phases inherit the parent policy when calling the nested LLM agent executor.

Orchestrator helpers: `_effective_user_facts_policy()`, `_filter_user_fact_tools_by_policy()`, `_build_system_message(metadata, step)`.

## User-Level Opt-Outs (Static Agents)

Stored in `user_settings` KV:

- **facts_inject_enabled** (boolean, default True)  
  If False, backend does not fetch or inject facts into gRPC metadata (chat and any consumer of that metadata).

- **facts_write_enabled** (boolean, default True)  
  If False, gRPC `UpsertUserFact` returns an error and no fact is written (agents cannot save new facts).

REST: `GET/POST /api/settings/user/facts-preferences` with `{ facts_inject_enabled, facts_write_enabled }`.  
UI: Settings → Profile → Remembered Facts: two toggles.

## Relevance Filtering (Chat Path)

When injecting facts for chat, if the user has **more than 20 facts** and a **query** is present:

1. Embed the query via `EmbeddingServiceWrapper.generate_embeddings([query])`.
2. For facts that have `embedding` set, compute cosine similarity with the query vector.
3. Keep top 12 facts with similarity ≥ 0.35, plus all facts without an embedding yet.
4. Format and inject that subset.

If there are ≤20 facts, or embedding fails, **all** (valid) facts are injected.  
Canonical formatting (filter expired, sort by confidence, then key) is done by `settings_service.format_user_facts_for_prompt()` (backend) and `orchestrator.utils.fact_utils.format_user_facts_for_prompt()` (orchestrator).

## Category Filtering (Agent Factory)

Agent profiles can set **include_facts_categories** (JSONB array), e.g. `["work", "preferences"]`.  
When `include_user_facts` is True and `include_facts_categories` is non-empty, the custom agent runner filters facts to those whose `category` is in the list.  
Empty array = no filter (all categories).

## TTL (expires_at)

- **Set via API:** `POST /api/settings/user/facts` body can include `expires_at` (ISO 8601 string).  
- **Celery Beat:** `purge_expired_facts_task` runs hourly and deletes rows where `expires_at IS NOT NULL AND expires_at < NOW()`.

## save_user_fact Skill

Automation skill `save_user_fact` uses `save_user_fact_tool`; keywords (e.g. “remember”, “save that”) trigger the skill.  
It writes with `source='agent'` and is subject to **facts_write_enabled**.

## get_user_facts_tool

Explicit read tool for agents: `get_user_facts_tool(category=None, user_id=...)`.  
Returns `{ facts, count, formatted }`.  
Optional `category` filters to one category (`general`, `work`, `preferences`, `personal`).  
Use when the agent needs to query what is stored instead of relying on automatic injection.

## Trust Hierarchy (Provenance and Confidence)

- **source** identifies origin: `user_manual`, `agent`, or `api`.
- **confidence** is set at write: user_manual → 1.0, agent → 0.8.
- When formatting for prompts, facts are sorted by **confidence descending**, then by key.  
  Expired facts (where `expires_at < NOW()`) are excluded by the canonical formatter.

## Episodic Memory (user_episodes)

Conversation-derived events are stored separately from semantic facts to support "remember what we worked on?" and workflow learning.

**Table:** `user_episodes`

| Column           | Type      | Description |
|------------------|-----------|-------------|
| id               | SERIAL PK | Row id |
| user_id          | VARCHAR   | Owner |
| conversation_id  | VARCHAR   | Optional link to conversation |
| summary          | TEXT      | One- or two-sentence summary of the turn |
| episode_type     | VARCHAR   | `chat`, `research`, `editing`, `coding`, `automation`, `file_management`, `general` |
| agent_used       | VARCHAR   | Agent that handled the turn |
| tools_used       | JSONB     | List of tool names used |
| key_topics       | JSONB     | List of topic strings |
| outcome          | VARCHAR   | e.g. `completed` |
| embedding        | FLOAT[]   | For relevance filtering |
| created_at       | TIMESTAMPTZ | When the turn occurred |

**Extraction:** After each assistant message is saved (in `grpc_orchestrator_proxy`), a Celery task `extract_episode_task` runs asynchronously. It calls a fast LLM to produce `summary`, `episode_type`, `key_topics`, and `outcome` from the user query and assistant response, then inserts into `user_episodes` and enqueues embedding. Short responses (< 50 chars) are skipped.

**Injection:** When `episodes_inject_enabled` is True (default), the context gatherer loads recent episodes (last 30 days, max 50). If there are more than 10 and a query is present, relevance filtering keeps top 5 by cosine similarity (threshold 0.30). Formatted as `RECENT ACTIVITY:\n- [date] [type]: summary` and set in gRPC metadata `user_episodes`. Chat agent appends this to the system prompt.

**User control:** Settings → Profile → Activity History: toggle "Track activity for AI context", view/delete individual episodes, "Clear all activity". REST: `GET/DELETE /api/settings/user/episodes`, `DELETE /api/settings/user/episodes/{id}`, `GET/POST /api/settings/user/episodes-preferences`. Celery Beat: `purge_old_episodes_task` runs daily and deletes episodes older than 90 days.

## Contradiction Detection and Fact History

When a fact is updated, the system detects conflicts and protects user-authored facts from silent overwrites by agents.

**Table:** `user_fact_history`

| Column          | Type      | Description |
|-----------------|-----------|-------------|
| id              | SERIAL PK | Row id |
| user_id         | VARCHAR   | Owner |
| fact_key        | VARCHAR   | Fact that was changed |
| old_value       | TEXT      | Previous value |
| new_value       | TEXT      | Proposed or applied value |
| old_source      | VARCHAR   | Source of previous value |
| new_source      | VARCHAR   | Source of new value |
| old_confidence  | FLOAT     | Previous confidence |
| new_confidence  | FLOAT     | New confidence |
| resolution      | VARCHAR   | `auto_replaced`, `pending_review`, `user_accepted`, `user_rejected` |
| resolved_at     | TIMESTAMPTZ | When the pending item was resolved |
| created_at      | TIMESTAMPTZ | When the change was recorded |

**Rules in `upsert_user_fact`:**

- **Same value:** No-op; only `updated_at` is touched.
- **Agent overwrites agent or user overwrites anything:** Record row in `user_fact_history` with `resolution='auto_replaced'`, then update the fact.
- **Agent tries to overwrite user-set fact (`old_source='user_manual'`, `new_source='agent'`):** Do **not** update the fact. Insert a `user_fact_history` row with `resolution='pending_review'`. Return `{ success: false, status: 'pending_review', fact_key, current_value, history_id }` so the agent can inform the user that the update was queued for review.

**User review:** Settings → Profile shows "Pending Fact Updates" when any exist. Each item shows fact key, current (user) value, and proposed (agent) value; user can Accept (apply agent value, set resolution to `user_accepted`) or Reject (keep current value, set resolution to `user_rejected`). REST: `GET /api/settings/user/facts/pending`, `POST /api/settings/user/facts/pending/{id}/resolve` with `{ "action": "accept" | "reject" }`.

**Fact history:** All changes are recorded. REST: `GET /api/settings/user/facts/history?fact_key=...` (optional filter). Settings → Profile shows a short "Fact Change History" list when present.

**Agent feedback:** When the gRPC handler returns `success=False` and a message like "Fact 'city' is currently set to 'NYC' by the user. Your proposed update has been queued for user review.", the orchestrator tool `save_user_fact_tool` surfaces this in its `formatted` response so the agent can tell the user naturally.

## Expansion Roadmap

- **Per-conversation scope:** Optional fact visibility scoped to a conversation or project.
- **Qdrant migration:** Move fact embeddings to a dedicated Qdrant collection for scale and richer retrieval (e.g. hybrid or filters).

## References

- Migrations: `backend/sql/migrations/066_enhance_user_facts.sql`, `backend/sql/migrations/067_add_episodic_memory_and_fact_history.sql`
- Backend: `backend/services/settings_service.py`, `backend/services/grpc_context_gatherer.py`, `backend/services/grpc_tool_service.py`, `backend/services/episode_service.py`
- Orchestrator: `llm-orchestrator/orchestrator/utils/fact_utils.py`, `llm-orchestrator/orchestrator/agents/custom_agent_runner.py`, `llm-orchestrator/orchestrator/agents/chat_agent.py`, `llm-orchestrator/orchestrator/tools/user_profile_tools.py`
- Celery: `backend/services/celery_tasks/fact_tasks.py`, `backend/services/celery_tasks/episode_tasks.py`
- API: `backend/api/settings_api.py` (facts, episodes, pending, history)
- Frontend: `frontend/src/components/SettingsPage.js` (Remembered Facts, Activity History, Pending Fact Updates, Fact Change History)
