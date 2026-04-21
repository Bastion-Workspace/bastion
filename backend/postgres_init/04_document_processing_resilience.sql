-- Runs after 03_document_chunk_index_state.sql on fresh PostgreSQL (docker-entrypoint-initdb.d).
-- Keep in sync with backend/postgres_init/migrations/151_document_processing_resilience.sql
--
-- Each init file runs in a new psql session (default DB = POSTGRES_DB, usually "postgres").
\c bastion_knowledge_base

\ir migrations/151_document_processing_resilience.sql
