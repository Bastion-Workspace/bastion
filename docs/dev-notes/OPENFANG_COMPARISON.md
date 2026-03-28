# OpenFang Comparison — Feature Parity Analysis

Comparison of Bastion’s capabilities against [OpenFang](https://github.com/RightNow-AI/openfang) (RightNow-AI’s “Agent Operating System,” v0.1.0, February 2026). Identifies existing parity and gaps to prioritize.

**Reference:** [OpenFang GitHub](https://github.com/RightNow-AI/openfang), [OpenFang website](https://www.openfang.sh/).

---

## 1. What We Already Have (Parity or Better)

| Capability | Bastion | OpenFang | Verdict |
|------------|---------|----------|---------|
| Research agent | Full multi-round research, query expansion, web + local | Researcher Hand | Parity |
| Memory systems | PostgreSQL + Qdrant + Neo4j | SQLite + vector | Bastion stronger |
| Knowledge graph | Neo4j, entity extraction, analytics | Not mentioned | Bastion stronger |
| Skills system | 4 categories, LLM-based selection, eligibility | 60 bundled SKILL.md | Parity (different approach) |
| Plugin architecture | 6 plugins (Trello, Notion, CalDAV, Slack, GitHub, Victoria) | MCP + A2A | Parity |
| MCP server | Full implementation, 14+ tools | MCP support | Parity |
| A2A communication | Agent invocation tools, handoff patterns | A2A support | Parity |
| Scheduled tasks | Celery Beat, 12+ periodic tasks, Agent Factory cron | Hands on schedules | Parity |
| Agent Factory | Custom agent builder, dynamic execution | “Build your own Hand” | Parity |
| Data workspace | Per-user PostgreSQL workspaces, SQL tools, import/export | Not mentioned | Bastion stronger |
| Document management | CRUD, folders, org-mode, fiction editing | Not core focus | Bastion stronger |
| Audit logging | PostgreSQL triggers, JSONB old/new, email audit | Merkle hash-chain | Different (see gaps) |
| Desktop app | Electron (Windows) | Tauri 2.0 | Parity |
| Architecture | 7 gRPC microservices | Single binary | Bastion more modular |

---

## 2. Gaps — What We Lack

### 2.1 Channel Adapters: 7 vs 40

**Bastion today:** Telegram, Discord, Slack, SMS, Microsoft Graph (email), IMAP/SMTP, CalDAV (`connections-service/providers/`).

**OpenFang claims:** 40 adapters (Core, Enterprise, Social, Community, Privacy, Workplace).

**Missing (by category):**

- **Social:** LINE, Viber, Facebook Messenger, Mastodon, Bluesky, Reddit, LinkedIn, Twitch
- **Enterprise:** Microsoft Teams (chat), Mattermost, Google Chat, Webex, Feishu/Lark, Zulip
- **Community:** IRC, XMPP, Guilded, Revolt, Keybase, Discourse, Gitter
- **Privacy:** Threema, Signal, Nostr, Nextcloud Talk, Rocket.Chat, Ntfy, Gotify
- **Workplace:** Pumble, Flock, Twist, DingTalk, Zalo
- **Core:** WhatsApp (OpenFang has dedicated WhatsApp gateway)

**Practical takeaway:** High-value additions: WhatsApp, Microsoft Teams chat, Signal, Mastodon/Bluesky, Webhooks. Many others are long-tail.

---

### 2.2 LLM Providers: 4 vs 27

**Bastion today:** OpenAI, OpenRouter, Ollama, vLLM (`backend/services/user_llm_provider_service.py`, embedding providers in vector-service).

**OpenFang claims:** 27 providers, 123+ models.

**Missing as direct drivers:** Anthropic, Gemini, Groq, DeepSeek, Together, Mistral, Fireworks, Cohere, Perplexity, xAI, AI21, Cerebras, SambaNova, HuggingFace, Replicate, Qwen, MiniMax, Zhipu, Moonshot, Qianfan, Bedrock.

**Practical takeaway:** OpenRouter already proxies most of these. Highest-value additions: **direct Anthropic and Gemini drivers** to avoid OpenRouter middleman.

---

### 2.3 Autonomous “Hands” — Pre-Built Packages

OpenFang ships 7 pre-built autonomous agents that run independently. Bastion has scheduling (Celery Beat, Agent Factory) but not equivalent pre-packaged workflows.

| OpenFang Hand | Bastion equivalent | Gap |
|---------------|---------------------|-----|
| Researcher | FullResearchAgent + scheduling | Near parity |
| Clip (YouTube → short video) | None | Full gap — video pipeline |
| Lead (lead generation) | None | Full gap — lead scoring/enrichment |
| Collector (OSINT monitoring) | RSS + web scraping | Partial — no change detection, sentiment |
| Predictor (superforecasting) | None | Full gap — calibration, Brier scores |
| Twitter (social manager) | None | Full gap — scheduling, engagement |
| Browser (web automation) | Crawl4AI scraping | Partial — no Playwright form/workflow automation |

---

### 2.4 Security Systems: ~6 vs 16

**Bastion today:** JWT auth, RLS, DB audit logging, Fernet encryption, input validation, CORS (`backend/utils/auth_middleware.py`, `backend/sql/01_init.sql`).

**OpenFang claims:** 16 discrete security layers.

**Missing (conceptually):**

| # | System | Description |
|---|--------|-------------|
| 1 | WASM sandbox | Tool code in WebAssembly with fuel metering |
| 2 | Merkle hash-chain audit | Tamper-evident log chain |
| 3 | Information flow taint tracking | Secret labeling source→sink |
| 4 | Ed25519 signed agent manifests | Cryptographic agent identity |
| 5 | SSRF protection | Block private IPs, cloud metadata, DNS rebinding |
| 6 | Secret zeroization | Wipe API keys from memory when done |
| 7 | Mutual auth (OFP) | HMAC-SHA256 P2P verification |
| 8 | Capability gates | RBAC on declared tools |
| 9 | Security headers | CSP, X-Frame-Options, HSTS, etc. |
| 10 | Health endpoint redaction | Minimal public health; full diag behind auth |
| 11 | Subprocess sandbox | env_clear + selective vars, process isolation |
| 12 | Prompt injection scanner | Override/exfil/shell patterns in skills |
| 13 | Loop guard | SHA256 tool-call loop detection, circuit breaker |
| 14 | Session repair | Message history validation and recovery |
| 15 | Path traversal prevention | Canonicalization, symlink escape |
| 16 | GCRA rate limiter | Cost-aware token bucket, per-IP |

**Practical takeaway:** Highest impact: **SSRF protection**, **prompt injection scanner**, **loop guard**. WASM sandbox is high effort; others are defense-in-depth.

---

### 2.5 Single Binary vs Multi-Container

OpenFang: single ~32MB Rust binary; one install, one command. Bastion: Docker Compose, 10+ containers. Trade-off: their simplicity vs our modularity and independent scaling.

---

### 2.6 CLI / TUI

OpenFang: CLI (`openfang hand activate`, `openfang chat`, `openfang agent spawn`) and TUI dashboard. Bastion: no CLI; all interaction via web UI.

---

### 2.7 OpenAI-Compatible Chat Completions API

OpenFang exposes `/v1/chat/completions` as a drop-in for external tools. Bastion does not expose this; API is Bastion-specific.

---

### 2.8 Migration from Other Frameworks

OpenFang: import from OpenClaw, LangChain, AutoGPT. Bastion: migration for JSON/CSV and ChatGPT export (`backend/services/migration_service.py`), not other agent frameworks.

---

### 2.9 P2P Networking

OpenFang: OFP protocol for P2P agent communication across instances. Bastion: no multi-instance federation.

---

### 2.10 Cross-Platform Desktop

Bastion: Electron app, Windows only (`electron/`). OpenFang: Tauri 2.0, macOS/Linux/Windows, system tray, notifications, global shortcuts.

---

## 3. Priority Ranking for Closing Gaps

| Priority | Gap | Impact | Effort |
|----------|-----|--------|--------|
| P1 | Direct Anthropic + Gemini LLM drivers | High — avoid OpenRouter middleman | Medium |
| P1 | SSRF protection + prompt injection scanner | High — security | Medium |
| P2 | WhatsApp + Microsoft Teams adapters | High — common request | Medium |
| P2 | OpenAI-compatible `/v1/chat/completions` API | High — ecosystem integration | Low |
| P2 | Browser automation Hand (Playwright-style) | High — web workflow automation | High |
| P3 | Pre-built autonomous Hands (e.g. Twitter, Lead Gen) | Medium — differentiation | High |
| P3 | Loop guard / circuit breaker for tool calls | Medium — runaway prevention | Low |
| P3 | CLI for agent management | Medium — power users | Medium |
| P4 | WASM sandbox for tool isolation | Medium — hardening | Very high |
| P4 | Merkle hash-chain audit trail | Low — DB audit sufficient today | High |
| P4 | Cross-platform desktop (e.g. Tauri) | Low — web primary | Very high |
| P5 | P2P federation | Low — niche | Very high |
| P5 | 30+ additional channel adapters | Low — long-tail | High total |

---

## 4. Summary

- **Strengths:** Bastion already matches or exceeds OpenFang on research, memory (PostgreSQL + Qdrant + Neo4j), knowledge graph, skills, plugins, MCP/A2A, scheduling, Agent Factory, data workspace, and document management. Architecture is more modular (gRPC services vs single binary).
- **Gaps that matter most:** Direct LLM drivers (Anthropic, Gemini), security hardening (SSRF, prompt injection, loop guard), high-value channels (WhatsApp, Teams), OpenAI-compatible API, and optionally pre-built autonomous workflow packages (“Hands”) on top of existing Agent Factory.
- **Lower priority:** Many extra channel adapters, P2P, Merkle audit, cross-platform desktop, and single-binary distribution are either niche or large architectural changes.

Focusing on the P1–P2 items above will improve security, cost, and ecosystem fit without chasing raw feature count.
