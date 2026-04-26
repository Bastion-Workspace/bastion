-- Indexed code chunks for local code workspaces (FTS + optional Qdrant point id)

CREATE TABLE IF NOT EXISTS code_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    workspace_id UUID NOT NULL REFERENCES code_workspaces(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    chunk_index INT NOT NULL,
    start_line INT NOT NULL DEFAULT 1,
    end_line INT NOT NULL DEFAULT 1,
    content TEXT NOT NULL,
    language TEXT,
    git_sha TEXT,
    content_tsv tsvector,
    qdrant_point_id TEXT,
    embedding_pending BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_code_chunks_workspace_file_chunk UNIQUE (workspace_id, file_path, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_code_chunks_user_workspace ON code_chunks (user_id, workspace_id);
CREATE INDEX IF NOT EXISTS idx_code_chunks_workspace ON code_chunks (workspace_id);
CREATE INDEX IF NOT EXISTS idx_code_chunks_tsv ON code_chunks USING GIN (content_tsv);
CREATE INDEX IF NOT EXISTS idx_code_chunks_pending ON code_chunks (workspace_id) WHERE embedding_pending IS TRUE;

GRANT SELECT, INSERT, UPDATE, DELETE ON code_chunks TO bastion_user;
