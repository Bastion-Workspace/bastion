# Bastion vs Ruflo: Agent Factory & Platform Comparison

This note compares Bastion’s Agent Factory, Agent Lines, knowledge base (vector DB + Neo4j), and orchestration with [Ruflo](https://github.com/ruvnet/ruflo) (formerly Claude Flow) — a CLI/MCP-native multi-agent orchestration stack for Claude Code. Ruflo emphasizes swarm coordination, self-learning routing, and IDE-centric tooling; Bastion is a full-stack knowledge platform with LangGraph playbooks and Celery-driven autonomy.

## Where Bastion Is Tops

| Capability | What Bastion Has | Ruflo Equivalent |
| --- | --- | --- |
| **Full knowledge management platform** | Document CRUD, folders, sharing, pins, frontmatter references, file trees | Nothing — it is a CLI tool, not a content platform |
| **Visual agent builder** | Web UI for creating agent profiles, playbooks, skills, data sources | YAML config files and CLI commands |
| **Typed playbook DAGs** | `tool`, `llm_task`, `llm_agent`, `branch`, `loop`, `parallel`, `approval`, `browser_authenticate` steps with `{upstream.field}` wiring | Flat swarm task assignment — no structured pipeline semantics |
| **Entity knowledge graph (Neo4j)** | NER-extracted entities, typed relationships, multi-hop traversal, hybrid RAG reranking | In-memory PageRank over flat memory entries — not a dedicated graph DB |
| **Data Workspace** | User-managed SQL tables with schema injection into agent context | Nothing comparable |
| **External messaging** | Telegram, Discord, Slack, email via connections-service | GitHub-focused integrations |
| **Document-aware agents** | Frontmatter parsing, manuscript editing, reference file resolution, editor operations | Code file editing focus |
| **HITL in pipelines** | `approval` step type at any point in a playbook; LangGraph `interrupt_before` | Claims system for task ownership — coarser grained |
| **Multi-user / multi-org** | User management, document sharing, org-level permissions | Single-user CLI workflow |
| **Plugin integrations** | Trello, Notion, CalDAV, GitHub — typed I/O contracts with Action Registry | npm plugin packages, often dev-tooling oriented |
| **Vector store (Qdrant)** | Persistent, scalable vector DB with document-level collections | HNSW in-process + SQLite — fast but not the same deployment model |

## Where Ruflo Is Great

| Capability | What Ruflo Has | Bastion Equivalent |
| --- | --- | --- |
| **Self-learning routing (SONA)** | Tracks agent performance per task type, adapts routing over time, stores successful patterns | Static routing — `auto_routable` profiles matched by handle; no feedback loop |
| **Cost-tier task routing** | Three tiers: WASM for simple transforms ($0), cheaper models for medium tasks, expensive for complex | Same model often used across steps unless manually overridden |
| **Swarm topologies** | Hierarchical, mesh, ring, star, adaptive | Agent Lines are hierarchical (CEO directs) |
| **Consensus protocols** | Byzantine, Raft, Gossip, CRDT, Quorum for multi-agent decisions | No voting/consensus — CEO decides |
| **Graph analytics (PageRank, communities)** | PageRank + community detection over memory entries | Neo4j holds entity data; GDS-style PageRank/community detection not wired in app code |
| **Hierarchical memory lifecycle** | Working → Episodic → Semantic with forgetting curves and consolidation | Facts/themes exist; no full tiered lifecycle with decay/promotion |
| **Anti-drift mechanisms** | Goal checkpointing, coordinator validation, compliance scoring | No dedicated drift detection for autonomous Line runs |
| **Context-triggered workers** | Many auto-triggered workers (audit, optimize, document, test gaps, etc.) | Celery is schedule- or request-driven; limited event-triggered automation |
| **Multi-provider LLM failover** | Multiple providers with automatic fallback and cost-based selection | Multi-provider via OpenRouter; not the same as built-in failover/cost routing |
| **Execution pattern learning** | ReasoningBank-style trajectories for future retrieval | `agent_execution_log` exists but is not mined for routing patterns |
| **Local embeddings (ONNX)** | Very fast local embeddings without API calls | Embeddings typically via external API |
| **Prompt injection defense** | AIDefence, jailbreak detection, PII scanning (per their docs) | Standard auth and permissions |

## Summary

Bastion leads on **platform breadth**: knowledge management, visual configuration, typed pipelines, Neo4j + Qdrant, external messaging, and multi-user operation. Ruflo leads on **adaptive agent operations**: learning-based routing, cost tiers, swarm shapes, consensus, and operational hooks around long-running agent work. High-value follow-ups for Bastion include cost-aware per-step models, outcome-based routing hints, Neo4j GDS (PageRank / communities) on the existing entity graph, and optional consensus or drift checks for Agent Lines.

## Related

- Neo4j usage and gaps: `backend/services/knowledge_graph_service.py`, `backend/help_docs/knowledge-graph/`, `backend/docs/NEO4J_KNOWLEDGE_GRAPH_IMPROVEMENTS.md` (if present).
- Agent Factory architecture: `docs/AGENT_FACTORY_TECHNICAL_GUIDE.md`.
