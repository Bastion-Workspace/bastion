-- Runs after 01_init.sql on fresh PostgreSQL containers (docker-entrypoint-initdb.d).
-- Keep in sync with backend/sql/migrations/118_document_sharing.sql

\ir migrations/118_document_sharing.sql
