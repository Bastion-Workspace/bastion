# User-Designable Dashboard (Concept)

**Status:** Phases 1–3 (Home layout, multi-dashboard, grid + org/folder/pins widgets + document pins API) are implemented; broader vision (Data Workspace widgets, maps, music, etc.) remains future work.

## Vision

Users can create one or more **dashboard pages**: composed layouts built from **primitives** (widgets). Each primitive binds to existing data or UI surfaces (Data Workspace, folders, RSS, music, agents, etc.). Layout and configuration are persisted per user (and optionally per team later).

Goals:

- **At-a-glance** personal or role-specific home surfaces (finance workspace + charts, news, queues).
- **Composable** experience without shipping a separate “app” for every combination.
- **Reuse** of existing APIs and components where possible; new work mostly orchestration, layout persistence, and thin adapter widgets.

## Existing “Dashboard” Usage (Not This Feature)

The codebase already uses “dashboard” for **fixed** operational views. A user-designable dashboard would be **additive** and should not overload these names without clear UX separation.

| Surface | Purpose | Primary locations |
|--------|---------|-------------------|
| **Operations dashboard** | Agent Factory fleet status, cost summary, activity feed | `frontend/src/components/AgentDashboardPage.js`, route `/agent-dashboard`; `GET /api/agent-factory/dashboard/*` in `backend/api/agent_factory_api.py` |
| **Agent Line dashboard** | Per-line monitoring: budget, timeline, goals, tasks, analytics | `frontend/src/components/agent-factory/LineDashboardPanel.js`, `LineEditor.js`; help: `backend/help_docs/agent-lines/02-dashboard-and-controls.md` and related |
| **Data Workspace Overview tab** | In-workspace tab labeled with a dashboard icon | `frontend/src/components/data_workspace/DataWorkspaceManager.js` (Overview tab) |

No current document defines a **layout builder**, **widget catalog**, or **per-user dashboard page** model.

## Primitive Ideas (Examples)

### Data and analytics

- **Data Workspace table** — Paginated grid for a selected table; column visibility and sort persisted in widget config.
- **Saved query / KPI** — Scalar or small result set from a saved SQL (e.g., spend MTD, open items count); stepping stone to heavier charts.
- **Charts** — Pie/bar/line over workspace data (e.g., spend by category) once chart primitives and aggregation APIs exist or wrap existing query endpoints.

### Navigation and knowledge

- **Folder shortcuts** — Quick access tiles for pinned folders or recent locations (complements full file tree).
- **Pinned documents / reading queue** — User-curated or rule-based list of document IDs for one-click open.
- **Knowledge graph snapshot** — Compact entity/relationship summary or counts when APIs support bounded queries suitable for widgets.

### News and RSS

- **RSS feed strip** — Single feed, combined inbox, or filter by category/tag if metadata supports it.
- **Starred or unread digest** — Surfaces aligned with RSS article list/search tooling (e.g., starred articles across feeds).

### Geo and events (later phase)

- **News on a map** — Requires structured place data (coordinates or geocoded locations) on articles or enriched sidecars; treat as phase 2 unless such fields exist.

### Media

- **Music / audiobook player** — Richer than status-only: album art, queue, transport; backed by existing music/Audiobookshelf integration paths.

### Org mode and capture

- **Org agenda / today** — Condensed agenda or todo list for the current day.
- **Quick capture launcher** — Shortcuts to inbox/journal capture flows (similar intent to Quick Capture UI).

### Collaboration and agents

- **Messaging / team pulse** — Unread counts, recent threads, or @mention highlights (teams + messaging).
- **Personal Agent Factory strip** — Pending approvals, next scheduled run, last execution for **my** agents (subset of operations dashboard data scoped to the user).

### System and connections

- **External connections status** — Which integrations are connected, token health warnings.
- **Status mirror** — Optional duplication of key status bar indicators for a full-width home layout.

## Cross-Cutting Concerns (Implementation Sketch)

- **Persistence:** Dashboard rows in main app PostgreSQL (`user_home_dashboards`: `id`, `user_id`, `name`, `is_default`, `layout_json` JSONB). Legacy copies in `user_settings` are migrated on first dashboard API access.
- **Layout:** Grid stack (e.g., react-grid-layout) or responsive sections; mobile degradation strategy.
- **Security:** Each primitive resolves data through existing auth and workspace boundaries; no new data paths that bypass RLS or user scoping.
- **Extensibility:** Register primitives by type string + JSON schema for config validation; aligns with typed tool/config patterns elsewhere.

## Related Notes and Docs

- Data Workspace stack: `docs/SERVICES.md` (postgres-data, data-service), Data Workspace UI under `frontend/src/components/data_workspace/`.
- Agent operations UI: `docs/dev-notes/AGENT_FACTORY_ARCHITECTURE.md`, `docs/AGENT_FACTORY_UX.md`.
- Broader visualization / intelligence context: `docs/dev-notes/DATA_INTELLIGENCE_ARCHITECTURE.md`.
- Optional future observability overlap: `docs/dev-notes/FUTURE_TOOLING.md` (external metrics dashboards — different product layer).

## Phase 1 (implemented)

**Home** area (nav label **Home**; distinct from **Operations** → `/agent-dashboard`). Per-widget layout schema below.

**Layout schema** (`HomeDashboardLayout`, `schema_version` = `1`): optional `layout_mode`: `stack` (default) or `grid`. `widgets` is an ordered array of `{ id, type, config, grid? }` with discriminated `type`. When `layout_mode` is `grid`, each widget may include `grid: { x, y, w, h }` (12-column layout for react-grid-layout).

| `type` | `config` |
|--------|----------|
| `nav_links` | `{ items: [{ label, path? \| href? }] }` — exactly one of `path` (in-app, leading `/`) or `href` (`http`/`https`). |
| `markdown_card` | `{ title?, body }` — body max 50k chars; rendered as Markdown in the UI. |
| `rss_headlines` | `{ feed_id?, limit }` — optional feed; if omitted, client aggregates recent articles from up to four feeds. |
| `org_agenda` | `{ days_ahead, include_scheduled, include_deadlines, include_appointments }` — uses `GET /api/org/agenda`. |
| `folder_shortcuts` | `{ items: [{ folder_id, label? }] }` — opens `/documents?folder=…`. |
| `pinned_documents` | `{ limit, show_preview }` — lists user pins from `GET /api/document-pins`. |

**Legacy layout-only API (still supported):** reads/writes the **default** dashboard’s layout only.

- `GET /api/home-dashboard` — layout of the default dashboard.
- `PUT /api/home-dashboard` — replace layout on the default dashboard (same validation as Phase 1).

## Phase 2 (implemented)

**Multiple named dashboards** per user, stored as rows in **`user_home_dashboards`** (UUID `id`, `user_id` FK to `users`, `name`, `is_default`, `layout_json` matching `HomeDashboardLayout`). At most one `is_default` per user (partial unique index). Max 20 dashboards per user.

**Migration:** If a user has **no rows** but still has `user_settings` keys `home_dashboards_v2` (envelope `schema_version` `2`) or legacy `home_dashboard_v1`, the server inserts rows from that data and **deletes** those keys. New users get a seeded default dashboard row without using `user_settings`.

**Envelope shape** (historical / API models): `{ schema_version: 2, dashboards: [{ id, name, is_default, layout }] }` — the relational table is the source of truth; each row is one dashboard.

**API (authenticated):**

- `GET /api/home-dashboards` — list `{ id, name, is_default }` for each dashboard.
- `POST /api/home-dashboards` — body `{ name?, duplicate_from_id? }`; returns updated list.
- `PATCH /api/home-dashboards/{id}` — body `{ name?, is_default? }`; returns updated list.
- `DELETE /api/home-dashboards/{id}` — not allowed if only one dashboard remains; if the default was deleted, another becomes default; returns updated list.
- `GET` / `PUT /api/home-dashboards/{id}/layout` — get/save widget layout for that dashboard (same RSS validation as Phase 1).

**Frontend:** Routes `/home` (redirects to default dashboard id) and `/home/:dashboardId`. Dashboard picker, **⋯** menu (new / rename / set default / delete), and layout editing per Phase 1. Implementation: [`frontend/src/components/HomeDashboardPage.js`](../../frontend/src/components/HomeDashboardPage.js), [`frontend/src/components/homeDashboard/HomeDashboardChrome.js`](../../frontend/src/components/homeDashboard/HomeDashboardChrome.js).

## Phase 3 (implemented)

- **Grid layout:** `layout_mode` `stack` \| `grid`; per-widget optional `grid` (`x`, `y`, `w`, `h`); frontend uses `react-grid-layout` in edit mode; toggle **Stack** / **Grid** in the edit toolbar.
- **Widgets:** `org_agenda`, `folder_shortcuts`, `pinned_documents` (see table above).
- **Document pins:** Table `user_document_pins` (migration `115_user_document_pins.sql`); API `GET/POST /api/document-pins`, `DELETE /api/document-pins/{pin_id}`, `PUT /api/document-pins/reorder` (`{ pin_ids }`). Pins are per-user; documents must be owned by the user.
- **Documents deep links:** `/documents?folder={folder_id}` and `/documents?document={document_id}&doc_title=…` for opening from Home widgets.

**Not yet implemented:** Data Workspace table/chart widgets, map/music primitives, richer pin UX from the document viewer (see vision above).

## Document history

- Introduced as concept note; Phase 1 MVP; Phase 2 multi-dashboard storage; Phase 3 grid layout + org/folder/pins widgets + document pins API.
