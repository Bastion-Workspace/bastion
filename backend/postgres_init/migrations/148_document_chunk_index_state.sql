-- Primary chunk index freshness (source of truth for idempotent reprocess / backfill).
-- Fresh Compose DBs: included via backend/postgres_init/03_document_chunk_index_state.sql (init order after 02_*).
-- Existing DB missing columns: docker exec -i bastion-postgres psql -U postgres -d bastion_knowledge_base < backend/postgres_init/migrations/148_document_chunk_index_state.sql
ALTER TABLE document_metadata
    ADD COLUMN IF NOT EXISTS chunk_indexed_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS chunk_indexed_file_hash VARCHAR(64),
    ADD COLUMN IF NOT EXISTS chunk_index_schema_version INTEGER NOT NULL DEFAULT 0;

COMMENT ON COLUMN document_metadata.chunk_indexed_at IS 'Last successful primary chunk+vector index completion';
COMMENT ON COLUMN document_metadata.chunk_indexed_file_hash IS 'document_metadata.file_hash at last successful index';
COMMENT ON COLUMN document_metadata.chunk_index_schema_version IS 'Schema version of index pipeline; bump in bastion_indexing.policy when contract changes';

CREATE INDEX IF NOT EXISTS idx_document_metadata_chunk_index_backfill
    ON document_metadata (processing_status, doc_type)
    WHERE processing_status = 'completed'
      AND (exempt_from_vectorization IS FALSE OR exempt_from_vectorization IS NULL);
