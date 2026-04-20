-- Runs after 03_document_chunk_index_state.sql on fresh PostgreSQL (docker-entrypoint-initdb.d).
-- Keep in sync with backend/sql/migrations/151_document_processing_resilience.sql

\ir migrations/151_document_processing_resilience.sql
