# AI chat branching (UI, API, LangGraph)

This note describes how **edit-and-resend** creates alternate timelines, how the UI navigates **siblings**, and how **LangGraph checkpoint `thread_id`** stays aligned with the active branch. It is intentionally light on code; see the referenced modules for implementation.

---

## Concepts

| Concept | Meaning |
|--------|---------|
| **Message tree** | Messages are linked by `parent_message_id`. Siblings share the same parent (alternate user turns after an edit-and-resend). |
| **Active path** | The linear transcript from root to the **current leaf**: `conversations.current_node_message_id`, walking parents upward (DB) or the same logic in the UI. |
| **Branch (DB)** | A row in `conversation_branches` created when the user forks from an existing **user** message. It owns a unique `thread_id_suffix` used for LangGraph isolation. |
| **Branch resend** | After the branch API inserts the new user message, the next orchestrator stream runs with `is_branch_resend` + `branch_message_id` so the backend does not double-insert the user row and can attach the assistant reply to the correct `branch_id`. |

---

## Data model (PostgreSQL)

Defined in migration `backend/sql/migrations/112_message_branching.sql` and used by `ConversationService`.

- **`conversations.current_node_message_id`** — Leaf of the timeline the user is viewing. New messages append under this parent (unless a caller supplies an explicit parent). Switching branches updates this column.
- **`conversation_messages.parent_message_id`** — Tree edge to the previous message in that timeline.
- **`conversation_messages.branch_id`** — Optional FK to `conversation_branches`; messages on a forked timeline reference the branch created at the fork.
- **`conversation_branches`** — One row per fork: `forked_from_message_id`, `parent_branch_id` (lineage of forks), `thread_id_suffix` (UUID string, unique), `first_message_id` (first user message on that branch).

**Sibling rule:** Two user messages are “branches” in the UI sense when they have the same `parent_message_id` (and thus the same position in the tree).

---

## API surface

| Endpoint | Role |
|----------|------|
| `POST /api/conversations/{id}/messages/{message_id}/branch` | Body: `{ "new_content": "..." }`. Validates that `message_id` is a **user** message. Creates `conversation_branches`, inserts a **new** user message as sibling of the original (same `parent_message_id` as the original), sets metadata flags (`forked_from_message_id`, `branch_edit_resend`). Returns the new message, `branch_id`, `thread_id_suffix`, and `active_path_messages`. |
| `POST /api/conversations/{id}/switch-branch` | Body: `{ "target_message_id": "..." }`. Moves the UI “cursor” by setting `current_node_message_id` to the **deepest descendant** of the target (at each level, child with highest `sequence_number`). Returns `current_node_message_id` and `active_path`. |
| `GET /api/conversations/{id}/messages/{message_id}/siblings` | Lists messages with the same `parent_message_id` (ordered), plus `current_index` / `total`. |
| `GET /api/conversations/{id}/messages?include_tree=true` | Returns **all** messages in the conversation (all branches), ordered by `sequence_number`, plus `current_node_message_id`. Without `include_tree`, and when `current_node_message_id` is set, the API returns only the **active path** (recursive ancestors of the leaf). |

Orchestrator streaming (`POST /api/async/orchestrator/stream`) accepts:

- `is_branch_resend` — Skip persisting the user message again (already created by the branch endpoint).
- `branch_message_id` — That persisted user message’s id (used to resolve `branch_id` for the assistant message and to drive branch-aware context).

Optional: `base_checkpoint_id` is present on the async request model and gRPC `ChatRequest` proto for **future** “start from this checkpoint” behavior; it is not required for the current edit-and-resend flow.

---

## Request path: branch edit-and-resend → orchestrator

1. **Frontend** calls the branch endpoint, then opens the stream with `is_branch_resend: true` and `branch_message_id` set to the new user message id (`ChatSidebarContext` / `ConversationService`).
2. **`stream_from_grpc_orchestrator`** (`backend/api/grpc_orchestrator_proxy.py`): when `is_branch_resend` is set, it loads the active path, sets `active_path_messages` to **all messages strictly before** the new user turn (`path[:-1]`), and resolves `branch_thread_suffix` via `ConversationService.get_branch_thread_id`.
3. **`GRPCContextGatherer`** (`backend/services/grpc_context_gatherer.py`): if `active_path_messages` is present, it uses that list instead of loading “most recent window” from the DB for history attachment. It also copies `branch_thread_suffix` (and optional `base_checkpoint_id`) onto the gRPC request metadata / fields (`_add_checkpoint_info`).
4. **`grpc_service`** (`llm-orchestrator/orchestrator/grpc_service.py`): merges `request.metadata` into the dict passed into agents (includes `branch_thread_suffix` when set).
5. **`BaseAgent._get_checkpoint_config`** (`llm-orchestrator/orchestrator/agents/base_agent.py`): builds `configurable.thread_id` as `{user_id}:{conversation_id}` and, when `branch_thread_suffix` is present, appends `:branch_{suffix}`. This isolates LangGraph checkpoints per fork while keeping a deterministic namespace.

**Assistant persistence:** After the stream completes, the proxy passes `message_branch_id` from the branch user row into `add_message` for the assistant so the reply stays on the same branch.

---

## UI behavior

- **Full tree** — `ChatSidebarContext` loads messages with `include_tree=true` into `allMessages` and builds `messageTree` via `frontend/src/utils/messageTreeUtils.js`.
- **Visible transcript** — `messages` is the active path from `current_node_message_id` (from the API), with extra handling for streaming tails and conversation switches.
- **Edit and resend** — User edits a past user message; the client calls `editAndBranch`, refetches, then streams with branch flags.
- **Branch navigation** — For any message on the active path that has multiple siblings, `ChatMessagesArea` computes `siblingInfo` and `ChatMessage` renders prev/next controls. `switchBranch` calls the switch API with the adjacent sibling’s id, then refreshes path state from the response / refetch.

---

## LangGraph / checkpoints

- **Default thread:** Checkpoints for a linear chat use `thread_id = "{user_id}:{conversation_id}"` (see `backend/services/orchestrator_utils.py` `normalize_thread_id` for the canonical rules).
- **Forked thread:** Same base plus `:branch_{thread_id_suffix}` so each edit-and-resend timeline has its own checkpoint chain. The suffix comes from `conversation_branches.thread_id_suffix` for the **branch row referenced by the deepest `branch_id` on the active path** (`get_branch_thread_id`).
- **Fresh fork:** A new UUID suffix means a **new** checkpoint namespace; the model still receives prior turns via **serialized conversation history** on the request (`active_path_messages`), not by reading the parent branch’s LangGraph state. Parent-branch checkpoints remain in Postgres but are not selected for the new `thread_id`.

---

## Future: revert vs. branch (same mechanisms)

The same primitives can support different product behaviors without inventing a parallel system:

### Branching (current)

- Insert sibling user message + new `conversation_branches` row + new `thread_id_suffix`.
- Pros: History is preserved; safe default. Cons: More rows and more checkpoint namespaces over time.

### Revert / “truncate here” (possible evolution)

- **Conversation DB:** Move `current_node_message_id` to an earlier message; **soft-delete** or **detach** messages after that point (or mark them inactive) so `include_tree=false` queries and “active path” match the truncated timeline. Optionally compact sibling rows if the product should collapse alternates.
- **LangGraph:** Either keep `thread_id` unchanged and **delete or supersede** checkpoints after a chosen checkpoint id, or **rotate** to a new `thread_id_suffix` while copying state once (heavier). The proto field `base_checkpoint_id` is a natural hook to **load state from a specific checkpoint** when starting the next turn, then continue on the current or a new thread id.
- **API:** Could expose `POST .../revert` or `PATCH .../current-node` with semantics “set leaf to message X and optionally prune descendants” vs. today’s switch-branch which **follows** the deepest child from the selected sibling.

### Unified design angle

- **Branching** = new sibling + new suffix + history override on first request.
- **Reverting** = move `current_node_message_id` backward + align checkpoint storage (truncate, time-travel, or copy-from-checkpoint) so the next message’s `thread_id` and checkpoint history match the user’s expectation of “we continued from here.”

Documenting these as explicit product decisions will keep UI (which path is shown), API (which messages are visible), and LangGraph (`thread_id` + checkpoint lifecycle) in sync.

---

## Primary references

| Layer | Location |
|-------|----------|
| Schema | `backend/sql/migrations/112_message_branching.sql` |
| Branch / switch / path / thread id | `backend/services/conversation_service.py` |
| REST routes | `backend/api/conversation_api.py` |
| Stream + branch context injection | `backend/api/grpc_orchestrator_proxy.py` |
| History + gRPC metadata | `backend/services/grpc_context_gatherer.py` |
| Thread id helper | `backend/services/orchestrator_utils.py` |
| Metadata → checkpoint config | `llm-orchestrator/orchestrator/agents/base_agent.py` |
| UI tree + navigation | `frontend/src/utils/messageTreeUtils.js`, `frontend/src/contexts/ChatSidebarContext.js`, `frontend/src/components/chat/ChatMessage.js` |
| API client | `frontend/src/services/conversation/ConversationService.js` |
