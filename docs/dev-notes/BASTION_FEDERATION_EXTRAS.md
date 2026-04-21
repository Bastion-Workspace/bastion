# Bastion Federation — Extended Feature Opportunities

Beyond real-time messaging (covered in `BASTION_FEDERATION_PLAN.md`), the Bastion-to-Bastion federation trust layer unlocks several other product areas. This document surveys each opportunity, assesses feasibility against the current architecture, and proposes concrete designs.

All features here share a prerequisite: **Phase 1 of the federation messaging plan must be complete** (keypair generation, `federation_peers` table, admin pairing UI, Ed25519 sign/verify utilities). Once two instances have established mutual trust, the transport and identity mechanisms described below are available at no additional cost.

---

## Priority Summary

| Feature | Value | Effort | Earliest Phase |
|---------|-------|--------|----------------|
| RSS feed syndication | Medium | Low | Alongside messaging Phase 2 |
| Knowledge graph query federation | Medium | Low–Medium | After messaging Phase 3 |
| Document / knowledge sharing (read-only) | High | Medium | After messaging Phase 2 |
| Agent profiles / playbooks / skills sharing | High | Medium | After messaging Phase 2 |
| Data workspace (read-only federated access) | Medium–High | Medium | After messaging Phase 3 |
| Inter-instance agent line orchestration | Very High | Very High | Long-term, own phase |

---

## 1. RSS Feed Syndication

### What It Unlocks

Global RSS feeds already exist in Bastion with `user_id = NULL`. Federated feed syndication lets Instance A curate and publish a named collection of feeds — internal news digests, team updates, curated reading lists — that Instance B administrators and users can subscribe to in one click, with authentication handled by the federation keypair rather than a public URL.

Use cases:
- A parent organization publishes internal news feeds that subsidiaries on separate Bastion instances subscribe to automatically.
- A consulting firm publishes curated industry feeds for client instances without exposing an unauthenticated endpoint.
- Any instance can subscribe to another's public article archive as a live, authenticated feed.

### Design

**New endpoint on each instance:**

```
GET /api/federation/feeds
```

Returns a signed JSON feed catalog listing available syndicated feeds, their titles, descriptions, and subscription URLs. Only peers with `allowed_scopes` including `rss` can call this.

**Schema change:**

```sql
ALTER TABLE rss_feeds ADD COLUMN federation_peer_id UUID REFERENCES federation_peers(peer_id);
ALTER TABLE rss_feeds ADD COLUMN federation_remote_feed_id TEXT;
ALTER TABLE rss_feeds ADD COLUMN federation_sync_cursor TIMESTAMPTZ;
```

A feed with `federation_peer_id` set is a **remote feed** — articles are fetched from the peer instance rather than the origin URL. The existing Celery polling task (`rss_tasks.py`) gains a branch: if `federation_peer_id` is set, fetch from `{peer_url}/api/federation/feeds/{remote_feed_id}/articles` with a signed request instead of polling the RSS URL directly.

**Inbound endpoint:**

```
GET /api/federation/feeds/{feed_id}/articles?since={cursor}
```

Returns articles as signed JSON. Instance B's RSS poller calls this on its regular schedule. Authentication uses the same Ed25519 header pattern as messaging.

### Affected Files

| File | Change |
|------|--------|
| `backend/api/federation_api.py` | Add feed catalog + articles endpoints |
| `backend/api/rss_api.py` | Add "subscribe to federated feed" endpoint |
| `backend/services/rss_service.py` | Handle `federation_peer_id` feeds |
| `backend/services/celery_tasks/rss_tasks.py` | Branch for federated feed polling |
| `backend/postgres_init/migrations/` | `federation_peer_id`, `federation_remote_feed_id` on `rss_feeds` |
| `frontend/src/components/` | RSS settings: "Add from federated instance" option |

---

## 2. Knowledge Graph Query Federation

### What It Unlocks

The Neo4j knowledge graph built during document processing is currently instance-local. Federated KG queries let a research agent on Instance B query entities and relationships discovered from documents on Instance A — enriching research results without copying underlying documents.

Use cases:
- A research agent fans out entity lookups across all trusted peer instances, returning a richer graph than any single instance holds alone.
- An analyst on Instance B can ask "what do we know about Company X across all connected organizations?" and get results from both local and peer KGs.
- Legal or compliance teams share entity extraction results across instances for due diligence without sharing the source documents.

### Design

**New endpoint:**

```
POST /api/federation/kg/query
Body: { "cypher": "MATCH (e:Entity {name: $name})-[:MENTIONED_IN]->(d:Document) RETURN e, d LIMIT 20", "params": {"name": "Acme Corp"} }
```

The endpoint accepts a **restricted Cypher query** (read-only, allowlisted patterns — no `MERGE`, `CREATE`, `DELETE`, `SET`). It runs against the local Neo4j instance and returns node/relationship data as JSON, signed in the response.

The research agent in the llm-orchestrator gains a `query_federated_knowledge_graph` tool that:
1. Iterates over active peers with the `knowledge_graph` scope.
2. Sends a signed query to each peer's `/api/federation/kg/query`.
3. Merges results with local KG results, deduplicating by entity name + type.
4. Returns a unified entity + relationship set to the LLM context.

**Cypher allowlist** (enforced server-side before execution):

```python
ALLOWED_KG_QUERY_PATTERNS = [
    r"^MATCH\s",         # Read-only match
    r"RETURN\s",         # Must have a return clause
]
BLOCKED_KG_KEYWORDS = ["MERGE", "CREATE", "DELETE", "SET", "DETACH", "CALL {"]
```

**Scope:** Add `knowledge_graph` to `federation_peers.allowed_scopes[]`. Admins can grant or revoke per peer.

### Affected Files

| File | Change |
|------|--------|
| `backend/api/federation_api.py` | Add KG query endpoint |
| `backend/services/knowledge_graph_service.py` | Add `execute_federated_query()` with allowlist |
| `llm-orchestrator/orchestrator/tools/` | New `federated_kg_query_tool` |
| `backend/services/grpc_tool_service.py` | Expose federated KG query as a gRPC tool |

---

## 3. Document and Knowledge Sharing

### What It Unlocks

Cross-instance document access without requiring file transfers or shared storage. A user on Instance B receives a reference to a document on Instance A; they can read it, search within it, and attach it to agent context — all via authenticated proxy calls back to the source instance.

Use cases:
- A contractor's Bastion (Instance B) references client documents owned on Instance A without the client granting direct DB access.
- A parent organization shares a policy library with subsidiary instances; updates on Instance A are immediately visible to all subscribers.
- Research agents on any federated instance can include documents from peer instances in their context window.

### Two Sharing Modes

#### Mode A: Read-Only Reference (Lightweight)

Instance A issues a signed **document reference** to Instance B. The reference contains metadata only (`title`, `document_id`, `peer_url`, `file_type`, `created_at`). Content is fetched on-demand from Instance A via a signed proxy request.

- No content stored on Instance B.
- Always reflects Instance A's current version.
- Instance B's vector search **cannot** find this document (not indexed locally).

#### Mode B: Full Mirror (Heavyweight)

Instance A pushes document content, chunks, and embeddings to Instance B. Instance B indexes the document locally with a `source_peer_id` marker. Instance A sends update events when the document changes.

- Instance B's search finds the document.
- Adds storage and indexing load to Instance B.
- Suitable for curated "published" documents, not arbitrary sharing.

### Schema Changes

```sql
-- On the receiving instance (Instance B)
ALTER TABLE document_metadata ADD COLUMN federated_peer_id UUID REFERENCES federation_peers(peer_id);
ALTER TABLE document_metadata ADD COLUMN federated_remote_doc_id TEXT;
ALTER TABLE document_metadata ADD COLUMN federation_mode TEXT CHECK (federation_mode IN ('reference', 'mirror'));
ALTER TABLE document_metadata ADD COLUMN federation_synced_at TIMESTAMPTZ;
```

A document with `federated_peer_id` set is a remote document. `document_service_v2.get_document_content()` detects this and proxies the content fetch to the peer instance via a signed request.

**New endpoints:**

```
GET  /api/federation/documents/{doc_id}/metadata    -- metadata for a shared doc
GET  /api/federation/documents/{doc_id}/content     -- content (signed, respects doc ACL)
GET  /api/federation/documents/{doc_id}/chunks      -- for mirror mode: push chunks to receiver
POST /api/federation/documents/share                -- Instance A grants access to Instance B
POST /api/federation/documents/push-update          -- Instance A pushes content update to mirrors
```

**Access control:** The sharing user on Instance A must have at least read access to the document. The `document_shares` table gains a `federated_peer_id` column alongside the existing `user_id` / `team_id` columns.

### Affected Files

| File | Change |
|------|--------|
| `backend/api/federation_api.py` | Document metadata, content, chunk endpoints |
| `backend/services/document_service_v2.py` | Detect federated docs, proxy content fetch |
| `backend/repositories/document_repository.py` | Federated doc queries |
| `backend/services/federation_service.py` | Share grant, mirror push, update events |
| `backend/postgres_init/migrations/` | `federated_peer_id`, `federation_mode` on `document_metadata` and `document_shares` |
| `frontend/src/components/DocumentsPage.js` | "Share with federated instance" action |
| `frontend/src/components/DocumentViewer.js` | Remote origin badge on federated docs |

---

## 4. Agent Profiles, Playbooks, and Skills Sharing

### What It Unlocks

The existing `agent_artifact_shares` system supports user-to-user sharing of agent profiles, playbooks, and skills within one instance. Federation extends this to cross-instance publishing: an organization can publish a curated agent to peer instances, which receive a local copy they can use (and optionally customize).

Use cases:
- A specialist firm publishes a domain-specific agent (e.g., a legal research agent) that client organizations import into their own Bastion instance with one click.
- A DevOps team publishes their deployment playbook to all subsidiary instances that use the same infrastructure.
- A shared skills library across an enterprise federation — any instance can pull the latest version of a skill from the canonical source.

### Sharing Model

Federated artifact sharing uses a **publish/subscribe** pattern:

1. **Publisher** (Instance A): marks an agent profile, playbook, or skill as `federation_published = true`. Instance A exposes it at a signed endpoint.
2. **Subscriber** (Instance B): admin imports the artifact. A local copy is created with `source_peer_id` and `source_artifact_id` tracking the origin.
3. **Updates**: Instance A can push a new version; Instance B receives a notification (via federation messaging or webhook) and the admin can choose to accept the update.

### Dependency Resolution

Agents reference data source connectors, external connections, and tool packs that may not exist on Instance B. The import flow produces a **dependency manifest** listing unresolved references, and the admin maps them to local equivalents or marks them as optional.

### Schema Changes

```sql
ALTER TABLE agent_profiles ADD COLUMN federation_published BOOLEAN DEFAULT FALSE;
ALTER TABLE agent_profiles ADD COLUMN source_peer_id UUID REFERENCES federation_peers(peer_id);
ALTER TABLE agent_profiles ADD COLUMN source_artifact_id TEXT;
ALTER TABLE agent_profiles ADD COLUMN federation_version INTEGER DEFAULT 1;
ALTER TABLE agent_profiles ADD COLUMN federation_locked BOOLEAN DEFAULT FALSE; -- read-only from source

-- Same columns added to: custom_playbooks, agent_skills
```

**New endpoints:**

```
GET  /api/federation/artifacts                           -- catalog of published artifacts
GET  /api/federation/artifacts/{type}/{artifact_id}     -- artifact definition (signed)
POST /api/federation/artifacts/{type}/{artifact_id}/versions  -- push version update (instance A → B)
POST /api/federation/artifacts/import                   -- Instance B imports an artifact
```

### Affected Files

| File | Change |
|------|--------|
| `backend/api/federation_api.py` | Artifact catalog + definition + version push endpoints |
| `backend/api/agent_factory_api.py` | "Publish to federation" action; import endpoint |
| `backend/services/agent_factory_service.py` | Publish/import logic, dependency manifest |
| `backend/postgres_init/migrations/` | `federation_published`, `source_peer_id`, etc. on artifact tables |
| `frontend/src/components/agent-factory/AgentEditor.js` | "Publish" toggle, "Imported from" badge |
| `frontend/src/components/agent-factory/AgentListSidebar.js` | Import from peer, version update notification |

---

## 5. Data Workspace Federated Access

### What It Unlocks

`data_workspace_shares` already supports users and teams. Extending it to federated peers provides read-only (or read-write) SQL access to a workspace across instance boundaries — useful for cross-org analytics and agent-driven data pipelines.

Use cases:
- A supplier shares a live inventory workspace with a customer on a separate Bastion instance; the customer's agents can query it for stock levels.
- Cross-org analytics: an executive dashboard on Instance B queries data from workspaces on both Instance A and Instance B in a single view.
- A data team publishes a curated read-only workspace as a "data product" available to all federated partners.

### Two Access Modes

| Mode | Description | Complexity |
|------|-------------|------------|
| **Federated query** | Instance B sends a signed SQL query to Instance A's data-service; results returned as JSON. No local copy. | Low |
| **Snapshot sync** | Instance A periodically pushes a snapshot of specified tables to Instance B. Instance B gets a local read replica. | High |

Federated query is the right v1 target.

### Design

**New data-service gRPC method:**

```protobuf
rpc ExecuteFederatedQuery (FederatedQueryRequest) returns (FederatedQueryResponse);

message FederatedQueryRequest {
  string peer_id = 1;
  string workspace_id = 2;
  string sql = 3;             // read-only; allowlisted
  string signed_by = 4;       // instance URL
  bytes signature = 5;
}
```

**Backend federation endpoint:**

```
POST /api/federation/data/query
Body: { "workspace_id": "...", "sql": "SELECT ...", "peer_signature": "..." }
```

Query allowlist is enforced at the data-service level (`SELECT` only, no CTEs with write operations, row limits applied).

**Schema change (data-service):**

```sql
ALTER TABLE data_workspace_shares ADD COLUMN shared_with_peer_id TEXT; -- peer_id from federation_peers
ALTER TABLE data_workspace_shares ADD COLUMN federated_access_mode TEXT DEFAULT 'read'; -- read | read_write
```

### Affected Files

| File | Change |
|------|--------|
| `backend/api/federation_api.py` | Federated data query endpoint |
| `backend/services/data_workspace_grpc_client.py` | Call `ExecuteFederatedQuery` |
| `data-service/grpc_service.py` | Implement `ExecuteFederatedQuery` with allowlist |
| `data-service/services/query_service.py` | Read-only query execution for federation |
| `data-service/grpc/data_service.proto` | Add `ExecuteFederatedQuery` RPC |
| `data-service/sql/` | Migration: `shared_with_peer_id` on `data_workspace_shares` |
| `frontend/src/components/data_workspace/DataWorkspaceManager.js` | "Share with peer instance" option |

---

## 6. Inter-Instance Agent Line Orchestration

### What It Unlocks

This is the most architecturally ambitious federation feature and potentially the most distinctive. An **inter-instance agent line** lets organizations compose multi-agent workflows where some agents are hosted on Instance A and some on Instance B. The orchestrator on Instance A can delegate a subtask to a specialist agent on Instance B over federation, receive the result, and continue the workflow.

Use cases:
- A parent organization (Instance A) has a senior orchestrator agent; a subsidiary (Instance B) hosts domain-specific agents. The orchestrator transparently delegates to the subsidiary without direct credential sharing.
- A consulting firm runs an engagement-specific agent on Instance A that invokes a client's internal research agent on Instance B for proprietary data lookups.
- An automated pipeline on Instance A hands off a content generation step to a specialist writing agent on Instance B, then receives the result and continues post-processing locally.

### Design

#### Federated Agent Membership

`agent_line_memberships` today references only local `agent_profiles`. A federated membership adds:

```sql
ALTER TABLE agent_line_memberships ADD COLUMN federated_peer_id UUID REFERENCES federation_peers(peer_id);
ALTER TABLE agent_line_memberships ADD COLUMN federated_agent_id TEXT; -- agent_profile_id on remote instance
-- When federated_peer_id is set, agent_profile_id may be NULL
```

#### Federated Task Invocation

When the llm-orchestrator decides to invoke a remote agent (via line membership lookup), instead of calling the local gRPC `StreamChat`, it:

1. Constructs a signed **federated task request**:
```json
{
  "bfp_version": "1",
  "task_id": "<uuid>",
  "from_instance": "https://bastion.org-a.com",
  "target_agent_id": "<agent_profile_id_on_B>",
  "query": "Summarize the Q3 financials",
  "context": { "shared_memory": {}, "metadata": {} },
  "reply_to": "https://bastion.org-a.com/api/federation/task-result"
}
```
2. POSTs to `https://bastion.org-b.com/api/federation/task`.
3. Instance B verifies the signature, looks up the target agent, invokes it via its local orchestrator.
4. Instance B POSTs the result back to Instance A's `reply_to` endpoint (or streams chunks if the agent supports streaming).

**Execution is synchronous (with timeout) for v1** — the calling workflow waits for the result. Async with callback can come later.

#### New Endpoints

```
POST /api/federation/task              -- receive and execute a federated task
POST /api/federation/task-result       -- receive result from a federated task (async mode)
GET  /api/federation/agents            -- catalog of agents available for federated invocation
```

#### Security Constraints

- The target agent on Instance B must be explicitly marked `federation_invocable = true` by an admin.
- The calling peer must be in `allowed_scopes` including `agent_invocation`.
- The task context (`shared_memory`, `metadata`) is stripped of any local credentials or connection tokens before being sent.
- Result payloads are signed by Instance B so Instance A can verify the response is genuine.

### Orchestrator Changes

The llm-orchestrator's agent routing layer (`orchestrator/agents/`, `orchestrator/engines/`) needs a **federated agent resolver**: when an agent line member has a `federated_peer_id`, the resolver calls the federation transport layer instead of the local gRPC stub.

```python
# Pseudocode in the orchestrator routing layer
async def invoke_agent(member, query, context):
    if member.get("federated_peer_id"):
        return await invoke_federated_agent(
            peer_url=member["peer_url"],
            remote_agent_id=member["federated_agent_id"],
            query=query,
            context=sanitize_context_for_federation(context)
        )
    else:
        return await invoke_local_agent(member["agent_profile_id"], query, context)
```

### Affected Files

| File | Change |
|------|--------|
| `backend/api/federation_api.py` | Task invocation + result endpoints + agent catalog |
| `backend/services/agent_line_service.py` | Federated member resolution |
| `backend/postgres_init/migrations/` | `federated_peer_id`, `federated_agent_id` on `agent_line_memberships`; `federation_invocable` on `agent_profiles` |
| `llm-orchestrator/orchestrator/agents/` | Federated agent resolver |
| `llm-orchestrator/orchestrator/engines/` | Federation transport layer for task dispatch |
| `connections-service/` | May relay federated task streaming if WebSocket-style delivery is needed |
| `frontend/src/components/agent-factory/OrgChartView.js` | Remote agent members shown with peer badge |
| `frontend/src/components/agent-factory/TeamEditor.js` | Add remote agent member from peer instance |

---

## Features Not Recommended for Federation

| Feature | Reason |
|---------|--------|
| **Voice / TTS / STT** | Entirely per-user provider keys (ElevenLabs, OpenAI). No sharing concept exists locally. Federating voice credentials creates credential exposure risk with no clear benefit. |
| **Home dashboards** | User-private by design. Widgets reference local resources. The right approach is sharing the underlying resources (documents, feeds) rather than the dashboard layout itself. |
| **User voice provider configs** | BYOK credentials — must never be transmitted to another instance under any circumstances. |
| **Personas** | Personal identity configuration. Would federate as part of agent profile sharing (a persona can be referenced by a shared agent), not independently. |
| **Presence / user status** | Can be a low-priority addition to messaging Phase 4 (federated contacts' online status) but is not worth its own feature track. |

---

## Shared Implementation Concerns

### Scope Enforcement

Each feature above maps to a named scope. Admins grant scopes per peer:

| Scope | Feature |
|-------|---------|
| `messaging` | Federated rooms and messages (Phase 1 plan) |
| `rss` | Feed syndication |
| `knowledge_graph` | KG query federation |
| `documents` | Document read-only reference and mirror |
| `artifacts` | Agent / playbook / skill sharing |
| `data_workspaces` | Federated data queries |
| `agent_invocation` | Inter-instance agent task delegation |

The `federation_peers.allowed_scopes TEXT[]` column already accounts for this.

### Versioning

The `bfp_version` field in the wire format allows Instance A running federation v2 to negotiate with Instance B still on v1. Each endpoint should respond with a `406 Not Acceptable` if the requested version is unsupported, and the calling instance falls back gracefully.

### Audit Trail

Every inbound federated request should produce an entry in the existing audit log (`audit_log` table or equivalent) with `{peer_id, scope, action, timestamp, payload_hash}`. This gives admins visibility into what peer instances are doing.

---

## References

- Federation core plan (messaging, trust, keypairs): `docs/dev-notes/BASTION_FEDERATION_PLAN.md`
- Messaging system architecture: `docs/dev-notes/MESSAGING_IMPROVEMENTS.md`
- Channels roadmap (Matrix stretch): `docs/dev-notes/CHANNELS_ROADMAP_AND_DEPLOYMENT.md`
- Agent Factory architecture: `docs/dev-notes/AGENT_FACTORY_ARCHITECTURE.md`
- Agent Lines architecture: `docs/dev-notes/AGENT_LINES_ARCHITECTURE.md`
- OpenFang comparison (federation gap): `docs/dev-notes/OPENFANG_COMPARISON.md`
