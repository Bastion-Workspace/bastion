-- Runs after 04_document_processing_resilience.sql on fresh PostgreSQL (docker-entrypoint-initdb.d).
-- Bundles federation DDL (migrations 143–147 + 146 policy). Idempotent per-file; order matters for FKs.
--
-- Each init file runs in a new psql session (default DB = POSTGRES_DB, usually "postgres").
\c bastion_knowledge_base

\ir migrations/143_federation_phase1.sql
\ir migrations/144_federation_phase2.sql
\ir migrations/145_federation_phase3.sql
\ir migrations/146_federation_peers_participant_read.sql
\ir migrations/147_federation_phase4.sql
