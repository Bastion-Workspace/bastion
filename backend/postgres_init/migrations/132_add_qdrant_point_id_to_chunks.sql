-- Greenfield: column is in backend/postgres_init/01_init.sql; keep for existing DBs.
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS qdrant_point_id VARCHAR(36);
CREATE INDEX IF NOT EXISTS idx_document_chunks_qdrant_point_id ON document_chunks(qdrant_point_id);
