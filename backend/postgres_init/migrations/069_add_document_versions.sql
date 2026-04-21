-- Document version history (Phase 1: snapshots, rollback, diff)

CREATE TABLE IF NOT EXISTS document_versions (
    version_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     VARCHAR(255) NOT NULL
                    REFERENCES document_metadata(document_id) ON DELETE CASCADE,
    version_number  INTEGER NOT NULL,
    content_hash    VARCHAR(64) NOT NULL,
    file_size       BIGINT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    created_by      VARCHAR(255),
    change_source   VARCHAR(50) NOT NULL,
    change_summary  TEXT,
    parent_version  UUID REFERENCES document_versions(version_id),
    operations_json JSONB,
    storage_path    TEXT NOT NULL,
    is_current      BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_document_versions_document
    ON document_versions(document_id, version_number DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_document_versions_unique
    ON document_versions(document_id, version_number);
CREATE INDEX IF NOT EXISTS idx_document_versions_current
    ON document_versions(document_id) WHERE is_current = TRUE;
