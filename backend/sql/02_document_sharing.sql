-- Runs after 01_init.sql on fresh PostgreSQL containers (docker-entrypoint-initdb.d).
-- Keep in sync with backend/sql/migrations/118_document_sharing.sql
--
-- Each init file runs in a new psql session (default DB = POSTGRES_DB, usually "postgres").
\c bastion_knowledge_base

\ir migrations/118_document_sharing.sql
