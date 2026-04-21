-- Runs after 05_federation_phases.sql on fresh PostgreSQL (docker-entrypoint-initdb.d).
-- Pulls learning progress (031) and message branching (112). Idempotent per-file; order: 031 then 112.
--
-- Each init file runs in a new psql session (default DB = POSTGRES_DB, usually "postgres").
\c bastion_knowledge_base

\ir migrations/031_add_learning_progress.sql
\ir migrations/112_message_branching.sql

GRANT SELECT, INSERT, UPDATE, DELETE ON learning_progress TO bastion_user;
GRANT USAGE, SELECT ON SEQUENCE learning_progress_id_seq TO bastion_user;
