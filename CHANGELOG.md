# Changelog

All notable changes to Bastion AI Workspace will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
