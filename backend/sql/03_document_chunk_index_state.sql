-- Runs after 02_document_sharing.sql on fresh PostgreSQL (docker-entrypoint-initdb.d).
-- Keep in sync with backend/sql/migrations/148_document_chunk_index_state.sql
-- Existing volumes: run migration 148 manually if these columns are missing (see migrations/148 header).
--
-- Each init file runs in a new psql session (default DB = POSTGRES_DB, usually "postgres").
\c bastion_knowledge_base

\ir migrations/148_document_chunk_index_state.sql
