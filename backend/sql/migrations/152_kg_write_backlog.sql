-- Persist Neo4j write operations when the graph is unavailable for later replay.

CREATE TABLE IF NOT EXISTS kg_write_backlog (
    id BIGSERIAL PRIMARY KEY,
    op VARCHAR(32) NOT NULL,
    document_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255),
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    attempts INT NOT NULL DEFAULT 0,
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS kg_write_backlog_doc_idx ON kg_write_backlog(document_id);
CREATE INDEX IF NOT EXISTS kg_write_backlog_attempts_idx ON kg_write_backlog(attempts, id);
