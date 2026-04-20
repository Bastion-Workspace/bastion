-- Document processing: leases, retries, progress, attempt history.
-- Fresh Compose DBs: included via backend/sql/04_document_processing_resilience.sql (init order after 03_*).
-- Existing DB: docker exec -i bastion-postgres psql -U postgres -d bastion_knowledge_base < backend/sql/migrations/151_document_processing_resilience.sql

ALTER TABLE document_metadata
    ADD COLUMN IF NOT EXISTS processing_started_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS processing_completed_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS processing_stage VARCHAR(50),
    ADD COLUMN IF NOT EXISTS processing_progress_done INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS processing_progress_total INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS attempt_count INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS last_error TEXT,
    ADD COLUMN IF NOT EXISTS last_error_kind VARCHAR(32),
    ADD COLUMN IF NOT EXISTS next_attempt_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS locked_by TEXT,
    ADD COLUMN IF NOT EXISTS locked_until TIMESTAMPTZ;

COMMENT ON COLUMN document_metadata.processing_stage IS 'Coarse pipeline stage: queued, parsing, chunking, embedding, kg, done';
COMMENT ON COLUMN document_metadata.processing_progress_done IS 'Chunks or units completed within current stage (paired with processing_progress_total)';
COMMENT ON COLUMN document_metadata.processing_progress_total IS 'Total units for current stage progress';
COMMENT ON COLUMN document_metadata.attempt_count IS 'Number of processing attempts started for this document';
COMMENT ON COLUMN document_metadata.last_error_kind IS 'transient, terminal, timeout, dependency';
COMMENT ON COLUMN document_metadata.next_attempt_at IS 'When a retry_scheduled document should be requeued';
COMMENT ON COLUMN document_metadata.locked_by IS 'Worker id holding processing lease';
COMMENT ON COLUMN document_metadata.locked_until IS 'Lease expiry time';

CREATE TABLE IF NOT EXISTS document_processing_attempts (
    attempt_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id VARCHAR(255) NOT NULL REFERENCES document_metadata(document_id) ON DELETE CASCADE,
    attempt_number INTEGER NOT NULL,
    stage VARCHAR(50),
    started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMPTZ,
    status VARCHAR(32) NOT NULL,
    error_kind VARCHAR(32),
    error_message TEXT,
    error_traceback TEXT,
    worker_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_doc_processing_attempts_document
    ON document_processing_attempts (document_id, started_at DESC);

CREATE INDEX IF NOT EXISTS idx_document_metadata_lease_reaper
    ON document_metadata (locked_until)
    WHERE locked_by IS NOT NULL AND locked_until IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_document_metadata_retry_queue
    ON document_metadata (next_attempt_at)
    WHERE processing_status = 'retry_scheduled' AND next_attempt_at IS NOT NULL;
