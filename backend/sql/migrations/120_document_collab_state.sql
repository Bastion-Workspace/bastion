-- Persist Y.Doc binary state for collaborative editing (fast room restore; periodic snapshots).

CREATE TABLE IF NOT EXISTS document_collab_state (
    document_id TEXT PRIMARY KEY REFERENCES document_metadata(document_id) ON DELETE CASCADE,
    ydoc_state BYTEA NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_document_collab_state_updated
    ON document_collab_state (updated_at DESC);
