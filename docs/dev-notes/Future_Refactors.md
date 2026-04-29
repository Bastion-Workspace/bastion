# Future refactors

Notes for modules that exceed the repo’s file-size guidance (~700 lines hard cap; prefer splits when editing files already ~560+ lines). Line counts below are approximate; re-check with `wc -l` before starting work.

## Motivation

Large files increase merge conflicts, slow code review, and hide boundaries between concerns. The tables below focus on the **largest** files per area (typically 900+ lines); many smaller files still sit above 700 lines and should be split opportunistically when touched.

---

## Backend (`backend/`)

| Path | Approx. lines | Notes |
|------|----------------|-------|
| `api/agent_factory_api.py` | ~5300 | Agent Factory REST surface; split by resource (profiles, playbooks, skills, triggers) or route groups |
| `services/chat_service.py` | ~3900 | Core chat orchestration; extract handlers (streaming, tools, persistence) into focused modules |
| `repositories/document_repository.py` | ~2800 | Data access; split by concern (CRUD vs search vs metadata) or by table family |
| `api/document_api.py` | ~2300 | Document HTTP API; group endpoints into routers/modules |
| `api/image_metadata_api.py` | ~2300 | Parallel size to document API; candidate for shared patterns or merge of shared helpers only |
| `services/agent_factory_service.py` | ~2200 | Service layer vs `agent_factory_api`; align splits with API boundaries |
| `services/grpc_context_gatherer.py` | ~2100 | Context building; extract per-domain gatherers |
| `main.py` | ~2000 | App wiring; move route registration / lifespan into smaller include modules |
| `api/conversation_api.py` | ~1700 | Conversation routes; split listing vs mutations vs attachments |
| `services/folder_service.py` | ~1700 | Folder logic vs repository |
| `utils/document_processor.py` | ~1700 | Processing pipeline helpers; isolate format-specific paths |
| `services/conversation_service.py` | ~1600 | Conversation business logic |
| `services/langgraph_tools/document_editing_tools.py` | ~1500 | Tool implementations; align with orchestrator tool placement rules |
| `services/langgraph_tools/image_search_tools.py` | ~1600 | Same as above |
| `services/settings_service.py` | ~1500 | Settings reads/writes; split admin vs user vs org |
| `services/builtin_skill_definitions.py` | ~1600 | Data-heavy definitions; consider generated or split JSON modules |
| `api/teams_api.py` | ~1500 | Teams REST |
| `api/settings_api.py` | ~1400 | Settings REST |
| `services/auth_service.py` | ~1400 | Auth flows |
| `services/grpc_handlers/agent_messaging_handlers.py` | ~1400 | gRPC handlers |
| `services/federation_message_service.py` | ~1300 | Federation messaging |
| `clients/connections_service_client.py` | ~1300 | gRPC client surface |
| `services/messaging/messaging_service.py` | ~1300 | Messaging |
| `services/celery_tasks/agent_tasks.py` | ~1200 | Celery tasks |
| `services/rss_service.py` | ~1200 | RSS |
| `api/org_search_api.py` | ~1200 | Org search API |
| `services/federation_service.py` | ~1200 | Federation |
| `models/agent_response_models.py` | ~1100 | Pydantic models; split by agent/response type |

### Backend — suggested priorities

- **`agent_factory_api.py` + `agent_factory_service.py`**: highest leverage; natural seams are REST resource and service method groups.
- **`chat_service.py`**: extract streaming, tool execution, and DB touchpoints incrementally to avoid a risky big bang.

---

## Document service (`document-service/`)

| Path | Approx. lines | Notes |
|------|----------------|-------|
| `ds_db/document_repository.py` | ~3200 | DB access; split extensions already exist (`document_repository_extensions.py`); continue extracting query families |
| `ds_services/file_watcher_service.py` | ~2500 | Filesystem watcher; isolate platform-specific and debounce logic |
| `ds_services/document_service_v2.py` | ~2400 | Core document operations; split read vs write vs search entrypoints |
| `ds_processing/document_processor.py` | ~1800 | Pipeline stages; one module per stage or per file type |
| `ds_services/folder_service.py` | ~1700 | Folder operations |
| `ds_langgraph_tools/document_editing_tools.py` | ~1500 | Editing tools; mirror backend/orchestrator boundaries |
| `ds_processing/parallel_document_processor.py` | ~1200 | Parallel ingest |
| `ds_handlers/folder_crud_handlers_mixin.py` (and sibling phase-2 mixins) | ~450 each | gRPC phase-2 handlers split from former `phase2_handlers.py` |
| `ds_services/knowledge_graph_service.py` | ~1100 | KG operations |
| `ds_services/file_manager_service.py` | ~1100 | File manager |
| `ds_clients/vector_service_client.py` | ~990 | Vector client |
| `ds_services/parallel_document_service.py` | ~960 | Parallel document API |
| `ds_models/api_models.py` | ~940 | API models package split |
| `ds_services/vector_store_service.py` | ~910 | Vector store |
| `ds_services/direct_search_service.py` | ~760 | Search |
| `ds_services/embedding_service_wrapper.py` | ~800 | Embeddings |
| `ds_handlers/document_handlers_mixin.py` | ~690 | Handler mixins (near cap) |

### Document service — suggested priorities

- **`document_repository.py`**: largest single file in the service; align splits with existing `document_repository_extensions` and query categories.
- **`file_watcher_service.py` + `document_service_v2.py`**: separate reactive filesystem logic from RPC-facing document CRUD.

---

## Frontend (`frontend/src/`)

| Path | Approx. lines | Notes |
|------|----------------|-------|
| `components/FileTreeSidebar.js` | ~5600 | Tree + batch loading; extract hooks, row components, and context menus |
| `components/DocumentViewer.js` | ~4800 | Viewer shell; split tabs, toolbar, save pipeline, and extensions |
| `components/SettingsPage.js` | ~4200 | Tab router; one component per settings tab is already partial — finish extraction |
| `services/exportService.js` | ~2700 | Export formats; split by output type or by pipeline stage |
| `components/agent-factory/StepConfigDrawer.js` | ~2000 | Step editor; split step types (tool, llm, approval, …) |
| `contexts/ChatSidebarContext.js` | ~1900 | Sidebar state; reduce context value surface or split sub-contexts |
| `components/ExternalConnectionsSettings.js` | ~1900 | Provider cards per subdirectory |
| `components/agent-factory/AgentListSidebar.js` | ~1800 | List + filters |
| `components/OrgTodosView.js` | ~1700 | Org todos table and actions |
| `components/MarkdownCMEditor.js` | ~1500 | Editor wrapper; plugins in separate files |
| `components/editor/extensions/liveEditDiffExtension.js` | ~1500 | Extension-only file; consider sub-extensions per concern |
| `components/images/ImageMetadataModal.js` | ~1500 | Modal sections |
| `components/OrgEditorPlugins.js` | ~1400 | Plugin registry |
| `components/chat/ChatMessagesArea.js` | ~1400 | Message list virtualization and actions |
| `components/MediaPage.js` | ~1300 | Media UI |
| `components/data_workspace/DataTableView.js` | ~1300 | Table + schema UI |
| `components/chat/ChatMessage.js` | ~1200 | Single message render paths |
| `components/agent-factory/PlaybookEditor.js` | ~1200 | Playbook graph UI |
| `components/settings/ControlPanesSettingsTab.js` | ~1200 | Control panes |
| `components/OrgCMEditor.js` | ~1000 | Org editor entry |

### Frontend — suggested priorities

- **`FileTreeSidebar.js` and `DocumentViewer.js`**: dominate UI complexity; extract hooks (`useFileTree`, `useDocumentSave`) and dumb presentational components first.
- **`SettingsPage.js`**: ensure each tab is lazy-loaded and lives in its own module to cap growth.

---

## LLM orchestrator (`llm-orchestrator/orchestrator/`)

### Inventory

| Path (under `llm-orchestrator/orchestrator/`) | Approx. lines | Notes |
|-----------------------------------------------|-----------------|-------|
| `backend_tool_client.py` | ~8300 | Single class wrapping tools-service + document-service gRPC; heavy `json.loads` on proto `*_json` fields |
| `engines/pipeline_executor.py` | ~2600 | Step execution, coercion, connector/MCP paths, LLM tool rounds |
| `tools/agent_factory_tools.py` | ~2200 | Playbook/profile/skill CRUD and related tool surfaces |
| `agents/custom_agent_runner.py` | ~1800 | LangGraph wiring for Agent Factory custom agents |
| `grpc_service.py` | ~1300 | gRPC servicer surface |
| `engines/playbook_graph_builder.py` | ~1000 | Dynamic graph from playbook definitions |
| `tools/data_connection_tools.py` | ~1000 | Data connector / workspace oriented tools |
| `engines/tool_resolution.py` | ~830 | Capability manifests, scoped tools, skill injection |
| `agents/base_agent.py` | ~870 | Shared agent / LLM / checkpoint behavior |
| `engines/deep_agent_executor.py` | ~770 | Deep-agent execution path |
| `tools/__init__.py` | ~590 | Side-effect imports to register all tools |

### Suggested split directions

#### `backend_tool_client.py`

- Split by **transport target**: e.g. document-service methods vs tools-service methods (separate modules or a thin facade + delegates).
- Optionally group by **domain** (documents, search, images, org, connectors, …) once the two-channel split exists.
- **Return-shape cleanup:** many methods return a **string** on `grpc.RpcError` / generic `Exception` while success paths return **dict** (or the reverse pattern). There are on the order of **~45+** `return f"Error` string exits (inventory via ripgrep). Normalize to a small set of error dicts or a shared helper so callers and `pipeline_executor` do not branch on `str` vs `dict`. `search_images` exception paths were aligned to dict in a prior hygiene pass; extend that pattern deliberately.

#### `pipeline_executor.py`

- Extract **connector steps** (`connector:…`), **MCP steps** (`mcp:…`), and **connection-scoped** tool dispatch into dedicated modules with stable interfaces.
- Keep the core “resolve inputs → invoke tool → store playbook_state” loop in a shorter orchestrator file.

#### `agent_factory_tools.py`

- Split **read/list/get** tooling from **mutations** (create/update/delete), or split by resource (playbooks vs profiles vs skills).

#### `custom_agent_runner.py`

- Move helpers (`_format_user_context`, step parsing, trace aggregation) into a `custom_agent/` package; keep the `StateGraph` definition and `process()` entry in a smaller main module.

#### `grpc_service.py`

- Group RPC handlers by feature area (streaming vs unary, health-adjacent helpers) into mixins or submodules imported by a thin servicer class.

#### `tools/__init__.py`

- Long-term: replace “import everything for registration” with **explicit registry modules** or entry-point style registration to reduce import-time coupling (larger design change).

---

## ToolService gRPC handler extraction (Phase 0)

**Context:** The tools-service **container** runs [`backend/services/grpc_tool_service.py`](../../backend/services/grpc_tool_service.py). Handler bodies live in [`backend/services/grpc_handlers/`](../../backend/services/grpc_handlers/). The [`tools-service/`](../../tools-service/) package (`tools_service`) already holds some implementations (RSS, weather, navigation, data workspace, file analysis); more logic can move there for clearer boundaries.

**Circular-import rule:** `tools_service` modules must **not** import from `backend.services.grpc_handlers` (which loads `grpc_handlers/__init__.py` and risks cycles). **RLS context dicts** live in [`backend/utils/grpc_rls.py`](../../backend/utils/grpc_rls.py) (`from utils.grpc_rls import grpc_user_rls, grpc_admin_rls`). [`grpc_handlers/rls.py`](../../backend/services/grpc_handlers/rls.py) re-exports for compatibility only. Pass `rls_context` from the mixin into `tools_service` when an op touches `database_helpers` with RLS.

**Golden path (Phase 2+):** Keep each mixin method as **decode gRPC request → call `tools_service` (or leaf `services.*`) → map dicts/dataclasses to protobuf → `context.abort` on failure**. `tools_service` returns **JSON-serializable dicts** (or TypedDicts) unless protobuf construction is explicitly required in that layer.

**Tech debt — [`search_utility_handlers.py`](../../backend/services/grpc_handlers/search_utility_handlers.py):** `SearchEntities` and `GetEntity` are **placeholders** (empty or stub responses). Do not move stubs into `tools_service` as-is; implement real behavior, trim from the public proto surface, or delegate to a concrete service.

### Inventory (mixin modules)

Line counts are approximate (`wc -l` on `backend/services/grpc_handlers/*.py`).

| File | Lines | Primary RPCs / role | Key dependencies | Risk | Suggested phase |
|------|------:|---------------------|------------------|------|-----------------|
| `agent_messaging_handlers.py` | ~1410 | Messaging, teams, goals, tasks, user facts, scratchpad, device proxy | `database_helpers`, `httpx` to backend, services | High | Late; split by domain (chat vs teams vs device) |
| `org_todo_handlers.py` | ~940 | Org inbox, journal, todos, refile | DB helpers, org services | Medium | After neutral RLS helper if extracting DB-heavy paths |
| `web_handlers.py` | ~930 | Web search/crawl, browser session | Crawl/browser services, storage | High | Mid; extract crawl vs browser |
| `email_m365_handlers.py` | ~870 | M365 email/calendar/contacts, graph invoke | M365 / graph clients | Medium | Mid |
| `agent_runtime_handlers.py` | ~710 | Execution log, approvals, agent memory, run history | DB helpers | Medium | Mid |
| `agent_factory_crud_handlers.py` | ~690 | Agent profiles, playbooks, schedules, data source bind | DB / agent factory services | Medium | Mid |
| `data_connector_builder_handlers.py` | ~660 | Connector probe/test/CRUD, bulk scrape, control panes | DB, scraper | Medium | Mid |
| `media_handlers.py` | ~600 | Images, faces, objects, audio | Image/vision services | Medium | Mid |
| `rss_handlers.py` | ~580 | RSS search, feeds, articles, unread | `tools_service.services.rss_service` | Low | Already delegates; thin mixin further |
| `control_pane_handlers.py` | ~580 | Control panes, playbook/profile listings | DB | Medium | Mid |
| `analysis_handlers.py` | ~550 | Weather, charts, file analysis, system design | `tools_service` weather/file analysis | Low–Med | Weather/file already partial |
| `agent_profile_handlers.py` | ~510 | Profiles, team posts, playbook fetch, invocation | DB, services | Medium | Mid |
| `data_workspace_handlers.py` | ~400 | Workspace list/schema/query/mutations | `tools_service.services.data_workspace_service` | Low | Delegate more into `tools_service` |
| `connector_mcp_handlers.py` | ~330 | Connector execute, GitHub, MCP | connections / external | High | Coordinate with connections-service boundaries |
| `navigation_handlers.py` | ~310 | Locations, routes, security scan | `tools_service` navigation/security | Low | Mostly delegated |
| `agent_skills_handlers.py` | ~290 | Skills CRUD/search | DB / vector | Medium | Mid |
| `search_utility_handlers.py` | ~220 | Entities (stubs); other RPCs delegate to `tools_service` | `tools_service` (`help_search`, `search_utility_ops`); stubs only in mixin | Low | **Phase 1 done** for non-stub RPCs |
| `agent_execution_trace_handlers.py` | ~120 | `GetExecutionTrace` | Thin mixin; body in `tools_service.services.agent_execution_trace_ops` | Low | **Phase 2 slice done** |
| `rls.py` | ~5 | Re-export of `utils.grpc_rls` | None | N/A | Canonical: `utils.grpc_rls` |

### Phase 1 status (Search utility — complete)

Non-placeholder SearchUtility gRPC paths consolidate into `tools_service`; the mixin only maps requests/responses and keeps **SearchEntities** / **GetEntity** stubs.

- **`SearchHelpDocs`:** [`tools-service/services/help_search.py`](../../tools-service/services/help_search.py).
- **`FindCoOccurringEntities`, `ExpandQuery`, `SearchConversationCache`:** [`tools-service/services/search_utility_ops.py`](../../tools-service/services/search_utility_ops.py).

### Phase 2+ log (ToolService extraction)

- **RLS:** [`backend/utils/grpc_rls.py`](../../backend/utils/grpc_rls.py); mixins and Celery call sites import `utils.grpc_rls` (not `grpc_handlers`).
- **`GetExecutionTrace`:** [`tools-service/services/agent_execution_trace_ops.py`](../../tools-service/services/agent_execution_trace_ops.py) + thin [`agent_execution_trace_handlers.py`](../../backend/services/grpc_handlers/agent_execution_trace_handlers.py).
- **Data workspace gRPC:** request parsing / response shaping helpers in [`data_workspace_service.py`](../../tools-service/services/data_workspace_service.py); mixin maps dicts to protobuf.
- **Analysis (non-weather):** [`tools-service/services/analysis_ops.py`](../../tools-service/services/analysis_ops.py) — charts, text analysis metrics, system modeling calls.
- **Media (partial):** [`tools-service/services/media_ops.py`](../../tools-service/services/media_ops.py) — image search, identify faces, image generation orchestration.
- **Agent skills:** [`tools-service/services/agent_skills_ops.py`](../../tools-service/services/agent_skills_ops.py) — CRUD/search orchestration; mixin maps JSON payloads to protobuf.

## References

- `.cursor/rules/project-structure.mdc`, `.cursor/rules/separation-of-concerns.mdc` — file size and responsibility rules
- `docs/dev-notes/AGENT_FACTORY_ARCHITECTURE.md` — orchestration context

### How line counts were produced

```bash
find backend -name '*.py' -type f | xargs wc -l | sort -n | tail -30
find document-service -name '*.py' -type f | xargs wc -l | sort -n | tail -25
find frontend/src \( -name '*.tsx' -o -name '*.ts' -o -name '*.jsx' -o -name '*.js' \) -type f | xargs wc -l | sort -n | tail -35
```
