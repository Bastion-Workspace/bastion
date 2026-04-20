# gRPC microservices architecture (developer guide)

This document describes how Bastion uses **gRPC** between the FastAPI **backend**, the **llm-orchestrator**, the **Tool Service** (in-process with the backend on port 50052), and optional satellite services. It is kept aligned with the **Agent Factory** execution model (`CustomAgentRunner`, playbooks, `BackendToolClient`).

For product-level agent behavior, see [AGENT_FACTORY_ARCHITECTURE.md](./AGENT_FACTORY_ARCHITECTURE.md).

---

## Overview

| Layer | Role | Typical transport |
|-------|------|---------------------|
| **Frontend** | React UI, WebSockets | HTTP(S) to backend |
| **Backend** | REST APIs, auth, documents, Agent Factory CRUD, WebSocket hub | HTTP; gRPC **client** to orchestrator and optional services |
| **llm-orchestrator** | `OrchestratorService`: streaming chat, context assembly, LangGraph run | gRPC **server** (:50051); gRPC **client** to backend Tool Service |
| **Tool Service** | `ToolService`: documents, web, email, org inbox, media, etc. | gRPC server on backend (:50052), called from orchestrator |
| **Satellites** | vector-service, crawl4ai-service, tools-service, … | gRPC or HTTP per service |

**Request path (chat):**

1. Browser calls backend REST/WebSocket APIs.
2. Backend opens a gRPC stream to **llm-orchestrator** (`StreamChat` / related RPCs in `protos/orchestrator.proto`).
3. Orchestrator **`grpc_service.py`** builds `ChatRequest` (metadata, history, shared_memory, etc.) and runs **`CustomAgentRunner`** (LangGraph + PostgreSQL checkpointer).
4. During the run, tools invoke **`backend_tool_client.py`** → **`ToolService`** on the backend (`protos/tool_service.proto`).
5. Chunks stream back over gRPC to the backend and onward to the client.

There is **no** intent-router roster of many Python agent classes in the orchestrator; behavior is driven by **profiles, playbooks, and registered tools**.

---

## Key files

| Concern | Path |
|---------|------|
| Orchestrator gRPC server | `llm-orchestrator/orchestrator/grpc_service.py` |
| Agent execution | `llm-orchestrator/orchestrator/agents/custom_agent_runner.py` |
| Dispatch / playbook wiring | `llm-orchestrator/orchestrator/engines/unified_dispatch.py`, `playbook_graph_builder.py`, `pipeline_executor.py` |
| Orchestrator → backend tools | `llm-orchestrator/orchestrator/backend_tool_client.py` |
| Backend tool gRPC server | `backend/services/grpc_tool_service.py` |
| Tool RPC handlers | `backend/services/grpc_handlers/*.py` |
| Shared contracts | `protos/orchestrator.proto`, `protos/tool_service.proto`, `protos/vector_service.proto`, … |

---

## Why gRPC here

- **Binary, typed contracts** — protobufs catch many integration mistakes early; generated stubs stay in sync.
- **Streaming** — LLM output maps naturally to server-side streaming (`StreamChat`).
- **Process boundaries** — orchestrator can be scaled, restarted, or upgraded separately from the monolith backend (within your deployment choices).
- **One proto tree** — see below; both services build from the same `protos/` directory.

---

## Shared Protocol Buffers (single source of truth)

All service definitions live under the **repository root**:

```
/opt/bastion/protos/
├── orchestrator.proto    # OrchestratorService (e.g. StreamChat)
├── tool_service.proto    # ToolService (document search, web, …)
├── vector_service.proto  # Vector microservice API (when used)
└── README.md
```

Docker builds for `backend` and `llm-orchestrator` use **repository root** as build context so both images can `COPY protos` and run `grpc_tools.protoc` the same way. Do not maintain duplicate copies of `.proto` files under each service tree.

**Changing an RPC**

1. Edit the `.proto` under `protos/`.
2. Rebuild images (or run `grpc_tools.protoc` locally with the same roots as CI/Dockerfile).
3. Implement or adjust the servicer in the backend or orchestrator.
4. Update the matching client (`backend_tool_client` or backend’s orchestrator client).

Field rules (proto3): reserve deleted field numbers; prefer stable field IDs for frequently used fields (1–15 encode smaller).

---

## Communication patterns

| Pattern | Use case | Example in Bastion |
|---------|----------|---------------------|
| Unary request/response | Single result | Many `ToolService` methods |
| Server streaming | Progressive LLM output | `OrchestratorService.StreamChat` → `ChatChunk` stream |
| Client streaming | Rare; batch upload style | As needed by specific RPCs |
| Bidirectional | Interactive sessions | Only if explicitly defined in protos |

Orchestrator clients should **reuse one async channel** (see `BackendToolClient._ensure_connected`) instead of opening a new channel per tool call.

---

## Connection and limits

Typical async Python client options (orchestrator → backend tools):

- Raise **max send/receive message size** for large documents (e.g. tens of MB) if your protos carry big payloads.
- Use **keepalive** options for long-lived streams so idle connections are not dropped blindly by middleboxes.
- Set **deadlines** on unary calls that might run long; for LLM work prefer **streaming** with heartbeats/status chunks rather than one huge unary response.

On errors, map failures to appropriate **`grpc.StatusCode`** (`INVALID_ARGUMENT`, `PERMISSION_DENIED`, `NOT_FOUND`, `INTERNAL`, `DEADLINE_EXCEEDED`, `UNAVAILABLE`, …) so callers can retry or surface UI messages consistently.

---

## Docker and service discovery

Inside Compose, services reach each other by **service name** (e.g. `backend:50052`, `llm-orchestrator:50051`). Configure hosts/ports via environment variables consumed by `backend_tool_client` and the backend’s orchestrator client—avoid hardcoding in application code.

Regenerate protobuf Python modules during **image build**, not at arbitrary runtime on developer laptops only, so production and CI agree on generated code.

---

## Testing and operations

- **Unit tests**: point an async `grpc.aio.insecure_channel` at `localhost:50052` (Tool Service) or the orchestrator port when those services are up in dev.
- **Integration**: exercise `StreamChat` with a test user and conversation id; assert chunk types (`status`, `content`, `complete`, `error`) match your client contract.
- **Health**: use Docker `HEALTHCHECK` or orchestrator/backend health RPCs; health payloads should describe **connectivity** (e.g. backend reachable from orchestrator), not legacy concepts like “agent count”.

---

## Operational checklist (DO / DON’T)

**Do**

- Edit protos only under `protos/` and rebuild all consumers.
- Reuse channels; set timeouts and message size limits explicitly for large payloads.
- Log structured fields (method, latency, status, `user_id` where appropriate) without secrets.

**Don’t**

- Fork duplicate `.proto` trees per service.
- Reuse reserved protobuf field numbers.
- Open a new gRPC channel per request in hot paths.
- Expose gRPC directly to browsers (keep REST/WebSocket on the backend).

---

## Troubleshooting

| Symptom | Likely cause | What to check |
|---------|----------------|----------------|
| Connection refused | Service down or wrong port | `docker compose ps`, logs for `backend` / `llm-orchestrator` |
| `UNAVAILABLE` | Network, crash, or restart | Same; verify Compose network and depends_on |
| `DEADLINE_EXCEEDED` | Slow LLM or tool | Increase deadline for that call; optimize tool; use streaming |
| Generated code out of sync | Proto changed, image not rebuilt | `docker compose build --no-cache` for affected services |

---

## References

- gRPC Python: https://grpc.io/docs/languages/python/
- Protocol Buffers: https://protobuf.dev/
- Repo: `protos/README.md`, [AGENT_FACTORY_ARCHITECTURE.md](./AGENT_FACTORY_ARCHITECTURE.md)

**Last updated:** April 2026 (Agent Factory + `CustomAgentRunner`; legacy multi-agent / feature-flag routing removed from this doc.)
