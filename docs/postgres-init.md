# PostgreSQL init bundle (`backend/postgres_init/`)

PostgreSQL schema initialization and incremental migrations for Bastion. This directory is **image-only**: it is copied into the Postgres image and must not hold Markdown or unrelated files.

## Docker `docker-entrypoint-initdb.d`

Files are copied by [`postgres/Dockerfile`](../postgres/Dockerfile). The entrypoint runs each top-level `.sql` file in a **new** `psql` session, defaulting to **`POSTGRES_DB`** (often `postgres`). [`01_init.sql`](../backend/postgres_init/01_init.sql) connects to **`bastion_knowledge_base`** with `\c`; the wrappers [`02_document_sharing.sql`](../backend/postgres_init/02_document_sharing.sql) through [`09_greenfield_extensions.sql`](../backend/postgres_init/09_greenfield_extensions.sql) also `\c bastion_knowledge_base` before `\ir` so migrations run against the app database.

**If init failed once** (errors in logs, then restarts show *Skipping initialization*): remove the Postgres named data volume (e.g. `docker compose down` then `docker volume rm <project>_bastion_postgres_data`), deploy an image that includes the fixed SQL, and start again so init runs cleanly.

## Current structure

```
backend/postgres_init/
├── 01_init.sql              # Primary greenfield schema (RLS, grants, merged DDL from key migrations)
├── 02_document_sharing.sql  # \c + \ir migrations/118_…
├── 03_document_chunk_index_state.sql
├── 04_document_processing_resilience.sql
├── 05_federation_phases.sql # \c + \ir migrations/143–147 (+146)
├── 06_learning_and_message_branching.sql  # \c + \ir migrations/031, 112
├── 07_messaging_improvements.sql          # \c + \ir migrations/130
├── 08_user_llm_providers.sql             # \c + \ir migrations/055, 080 (Groq)
├── 09_greenfield_extensions.sql          # \c + \ir 039, 048, 054, 071, 083, 156
└── migrations/            # Incremental SQL (run_migration.py, \ir from wrappers)
```

**Coverage matrix (regenerate after migration changes):** run `python backend/scripts/generate_greenfield_coverage.py` → [`GREENFIELD_COVERAGE.md`](../backend/postgres_init/GREENFIELD_COVERAGE.md) (heuristic A/B/C/D/R classification vs `01` and wrappers).

Early numbered files under `migrations/` that duplicated `01_init.sql` (roughly `004`–`041`, except `010_add_hierarchical_exemption.sql` and `031_add_learning_progress.sql`) were **removed from the repo**; new databases rely on `01_init.sql` plus the `02`–`09` wrappers. For legacy databases that still lack those objects, restore the old files from git history or rebuild from a fresh volume.

## Greenfield closure (default `docker compose`)

After a successful first boot (empty data volume), **no extra migration run** is required for these areas; they are covered by `01_init.sql` and/or the numbered wrappers:

| Area | Where applied |
|------|----------------|
| Core app schema, messaging, teams, music, agent lines, docs, RSS, checkpoints, etc. | `01_init.sql` |
| Document sharing locks / 118 deltas | `02` → `118` |
| Chunk index freshness columns on `document_metadata` | `03` → `148` |
| Document processing resilience | `04` → `151` |
| Federation peers/outbox/users/messaging parity | `05` → `143`–`147`, `146` |
| Learning progress (`learning_progress`); message branching (`conversation_branches`, branch columns) | `06` → `031`, `112` |
| Messaging: bot users (`users.is_bot`, `agent_profiles.bot_user_id`); reply-to / edit on `chat_messages` | `07` → `130` |
| User LLM providers and enabled models (`user_llm_providers`, `user_enabled_models`; Groq in CHECK) | `08` → `055`, `080_add_groq_provider_type` |
| AI convo attachments; event watches; edit proposals; browser sessions; chunk pages + metadata FTS; line watches + workspace + grants/RLS | `09` → `039`, `048`, `054`, `071`, `083`, **`156`** |
| `mcp_servers` (brownfield idempotent DDL) | `155_add_mcp_servers.sql` (renamed from duplicate `080_*`; **`mcp_servers` in `01`**) |
| Agent skills + connection types + execution metrics + promotion recs (068, **126**, 131–136, 133) | `01_init.sql` |
| `agent_lines.reference_config`, `agent_lines.data_workspace_config` | `01_init.sql` |
| `document_chunks.qdrant_point_id`; `kg_write_backlog`; `vector_embed_backlog` | `01_init.sql` |
| Per-document encryption metadata on `document_metadata` (141) | `01_init.sql` |
| `user_home_dashboards`; `oregon_trail_saves` | `01_init.sql` |

Other files under `migrations/` remain for **brownfield** upgrades, data repairs, seeds, or legacy-only alters. If you add a feature whose DDL is only in a migration file and not in `01` or `02`–`09`, default greenfield will **not** pick it up until you merge it or add another wrapper.

## Single unified init script (greenfield)

**Most** schema for a new install lives in `01_init.sql`. Wrappers `02`–`09` pull focused migration files that are kept separate for review and reuse on existing databases.

## How it works

### Automatic initialization

On first container start, the Postgres image runs numbered `.sql` files in `/docker-entrypoint-initdb.d/` (each in a new `psql` session). In order: `01_init.sql`, then `02`–`09` as above.

### Fresh database setup

```bash
docker compose down
docker volume rm "${COMPOSE_PROJECT_NAME:-bastion}_bastion_postgres_data"
docker compose up --build
```

Use the exact volume name from `docker volume ls` if it differs.

## Migration bundling policy

Optional **bundled** migration files (single file concatenating multiple numbered migrations) are allowed only when:

- The bundle remains **idempotent** and **ordered** (respect foreign keys and data dependencies).
- It can still be applied the same way as individual files: `backend/scripts/run_migration.py` or manual `psql`.
- Prefer bundling **tightly coupled phases** (for example federation `143`–`147` including `146` in correct order) over arbitrary numeric ranges.

Do **not** bundle migration **049** (`049_agent_factory_ux_cleanup.sql`) with anything that touches `agent_skills` or related risky areas; keep dangerous migrations **isolated** and documented.

Bundling large swaths in one change is discouraged; ship layout or guard fixes separately from experimental bundles when possible.

## What's in 01_init.sql (overview)

High-level areas include document management (including encryption metadata and chunk/Qdrant mapping), PDF processing, authentication, document folders, conversations and messages, background jobs, templates, RSS, LangGraph checkpoints, messaging, teams, music and entertainment sync, email agent, agent factory (including **agent_skills** and related tables), home dashboards, Oregon Trail saves, KG/vector backlogs, RLS and grants. See the file for authoritative DDL.

## For developers

### Adding new tables (greenfield)

1. Edit `backend/postgres_init/01_init.sql` and/or add a new ordered wrapper (e.g. `10_*.sql`) if the DDL is a large bundle.
2. Add the table in the appropriate section with indexes and constraints.
3. Test with a fresh DB volume (`docker compose down`, remove volume, `docker compose up --build`).
4. Update the **Greenfield closure** table in this doc.

### Schema changes on existing databases

- Keep migration scripts under `backend/postgres_init/migrations/`.
- For databases provisioned before a change, run `python backend/scripts/run_migration.py` (or apply the SQL as superuser).

### Agent Factory built-in playbooks

Shared rows `00000000-0001-4000-8000-000000000001` (default ReAct) and `...000003` (deep-research template) are seeded in `01_init.sql` after `custom_playbooks` is created.

## Historical notes

- **November 2025**: FreeForm Notes removed from schema and codebase (see git history).
- **October 2025**: Consolidation toward unified init; duplicate top-level SQL folded into `01_init.sql`.

Duplicate SubSonic-shaped music DDL (`12_music_tables.sql`) was removed; music schema lives in `01_init.sql` and related migrations.
