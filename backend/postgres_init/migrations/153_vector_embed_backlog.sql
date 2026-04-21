-- Persist vector/embedding sync when vector-service or Qdrant is unavailable; drain when healthy.
-- Greenfield: table is in backend/postgres_init/01_init.sql; keep for existing DBs.

CREATE TABLE IF NOT EXISTS vector_embed_backlog (
    id BIGSERIAL PRIMARY KEY,
    op VARCHAR(64) NOT NULL,
    document_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255),
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    attempts INT NOT NULL DEFAULT 0,
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS vector_embed_backlog_doc_idx ON vector_embed_backlog(document_id);
CREATE INDEX IF NOT EXISTS vector_embed_backlog_attempts_idx ON vector_embed_backlog(attempts, id);
