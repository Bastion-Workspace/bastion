# Changelog

All notable changes to Bastion AI Workspace will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Feature: **`bastion-postgres`** and **`bastion-postgres-data`** images bundle `backend/sql` and `data-service/sql` so deployments work without bind-mounting the repo; CI publishes them alongside other services. Compose uses `postgres/Dockerfile` and `postgres-data/Dockerfile` by default.
- Update: `build-and-push` manual runs support **build scope** `postgres_only` (DB images only) vs `all` (default); tag pushes always build all images.
- Update: `build-and-push` uses separate GHCR package families — **`bastion-<service>`** for production tags (e.g. `v0.70.0` from `main`) and **`bastion-dev-<service>`** for `-dev` versions (e.g. `v0.70.5-dev` from `dev`); workflow validates tag commits against `main`/`dev`. See `.github/workflows/README.md` for visibility (public vs private) on GitHub Container Registry.
- Update: In-app help — **Data Workspaces overview** and getting-started cross-links aligned with the current UI (databases, import formats, Run SQL, grid features, sharing, agent binding); help-docs sync manifest hashes refreshed.
- Fix: `build-local-proxy` Linux CI installs `libpipewire-0.3-dev` so `libspa-sys` (PipeWire) links during `cargo build`.
- Fix: `build-local-proxy` Linux CI installs `libgbm-dev` and `libxdo-dev` so the release link step finds `-lgbm` and `-lxdo` (egui/glutin and tray stack).
- Fix: `local-proxy` — export `CreateDirectoryCapability`, match `git2::BranchType` in `git_info`, and box recursive async helpers (`list_dir_recursive`, `build_tree`, `walk_and_search`) for stable Linux CI builds.
- Feature: `build-and-push` workflow can be run manually (`workflow_dispatch`); optional version input or `VERSION` file when not tagging.
- Fix: `build-and-push` workflow frees runner disk before Docker builds and uses GHA BuildKit cache `mode=min` to reduce disk use during multi-image tag pipelines.
- Update: BBS RSS article list remembers All vs Unread choice for the rest of the telnet session.
- Update: BBS RSS article viewer uses html2text with links and images omitted for cleaner terminal text (article `Link:` line unchanged).
- Feature: RSS article list All/Unread (BBS `[V]` toggle; web RSS viewer segment buttons); BBS wallpaper pane hides the footer bar and exits with Esc (plus q; arrow keys discarded as non-quit).
- Feature: BBS menu prompts use single-key `[letter]` shortcuts without Enter (multi-digit row numbers still read digits then Enter).
- Fix: Frontend Docker build uses Node 22 and `NODE_OPTIONS=--max-old-space-size=4096` so `vite build` meets dependency engine ranges and avoids heap OOM on CI.
- Update: `.env.example` lists every variable referenced in root `docker-compose.yml` (with compose-aligned defaults) plus application-only env vars for migration toward externalized / Kubernetes config.
- Update: `docker-compose.yml` documents `${VAR:-default}` pattern; DB URLs, JWT, Qdrant/Neo4j, LangGraph DB creds, data-workspace Postgres, prod frontend `VITE_*` runtime, and pgbouncer healthcheck honor `.env` overrides with same defaults as before when unset.
- Update: Docker Compose uses single `LOG_LEVEL` for all first-party services (removed per-service `*_LOG_LEVEL` env indirection).
- Fix: `build-and-push` workflow sets `IMAGE_PREFIX` to lowercase `github.repository_owner` so GHCR tags validate (e.g. `ghcr.io/bastion-workspace/bastion-backend:0.70.0`).
- Update: README — drop standalone **Deployment** section; fold operator link into **Technical Architecture** (see `docs/DEPLOYMENT.md`).
- Update: README — dedicated **Agent Factory** section (building blocks, hybrid skill discovery, concrete examples); **Agent Factory** snapshot + skill rows; remove premature **code-workspace** claims; reframe long-form editing as user-composed agents.
- Update: README product overview — platform snapshot, supported formats table, and expanded capability sections for operators and evaluators.
- Update: GitHub Actions `build-and-push` publishes production `frontend` only; `frontend-dev` (Vite HMR) is not built in CI—use `docker compose --profile dev --build` locally.
- Update: Added `docs/DEPLOYMENT.md` (compose scenarios, env grouping, GHCR, operations) and linked it from `README.md`.

## [0.70.0] - 2026-04-20

Release 0.70.0 aggregates a large set of improvements across the web app, BBS, document pipeline, Agent Factory / agent lines, Docker stacks, tooling, and operational hardening since 0.55.0. Prior per-change entries were retired in favor of this milestone cut; see git history for granular commits.

- Fix: GitHub Actions image builds use annotated tag `v0.70.0` (pattern `v*`), replacing non-matching tag `0.70.0`.
