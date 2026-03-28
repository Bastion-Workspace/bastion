-- Full-text search: document_chunks table for PostgreSQL GIN search with RLS
-- Run after teams tables exist. Idempotent (IF NOT EXISTS / DROP POLICY IF EXISTS).

CREATE TABLE IF NOT EXISTS document_chunks (
    id              SERIAL PRIMARY KEY,
    chunk_id        VARCHAR(255) UNIQUE NOT NULL,
    document_id     VARCHAR(255) NOT NULL REFERENCES document_metadata(document_id) ON DELETE CASCADE,
    content         TEXT NOT NULL,
    chunk_index     INTEGER NOT NULL DEFAULT 0,
    user_id         VARCHAR(255),
    collection_type VARCHAR(50) DEFAULT 'user',
    team_id         UUID,
    is_image_sidecar BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    content_tsv     TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

CREATE INDEX IF NOT EXISTS idx_document_chunks_content_tsv ON document_chunks USING GIN(content_tsv);
CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_user_id ON document_chunks(user_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_collection_type ON document_chunks(collection_type);
CREATE INDEX IF NOT EXISTS idx_document_chunks_team_id ON document_chunks(team_id) WHERE team_id IS NOT NULL;

GRANT SELECT, INSERT, UPDATE, DELETE ON document_chunks TO bastion_user;

ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS document_chunks_select_policy ON document_chunks;
DROP POLICY IF EXISTS document_chunks_update_policy ON document_chunks;
DROP POLICY IF EXISTS document_chunks_delete_policy ON document_chunks;
DROP POLICY IF EXISTS document_chunks_insert_policy ON document_chunks;

CREATE POLICY document_chunks_select_policy ON document_chunks
    FOR SELECT USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR collection_type = 'global'
        OR (team_id IS NOT NULL AND team_id IN (
            SELECT team_id FROM team_members
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
        ))
        OR (user_id IS NOT NULL AND team_id IS NULL AND current_setting('app.current_user_role', true) = 'admin')
    );

CREATE POLICY document_chunks_update_policy ON document_chunks
    FOR UPDATE
    USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR (team_id IS NOT NULL AND team_id IN (
            SELECT team_id FROM team_members
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
            AND role = 'admin'
        ))
        OR (team_id IS NULL AND current_setting('app.current_user_role', true) = 'admin')
    )
    WITH CHECK (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR (team_id IS NOT NULL AND team_id IN (
            SELECT team_id FROM team_members
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
            AND role = 'admin'
        ))
        OR (team_id IS NULL AND current_setting('app.current_user_role', true) = 'admin')
    );

CREATE POLICY document_chunks_delete_policy ON document_chunks
    FOR DELETE USING (
        current_setting('app.current_user_role', true) = 'admin'
        OR user_id = current_setting('app.current_user_id', true)::varchar
        OR (team_id IS NOT NULL AND team_id IN (
            SELECT team_id FROM team_members
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
            AND role = 'admin'
        ))
        OR (user_id IS NULL AND current_setting('app.current_user_role', true) = 'admin')
    );

CREATE POLICY document_chunks_insert_policy ON document_chunks
    FOR INSERT WITH CHECK (
        current_setting('app.current_user_role', true) = 'admin'
        OR user_id = current_setting('app.current_user_id', true)::varchar
        OR collection_type = 'global'
        OR (team_id IS NOT NULL AND team_id IN (
            SELECT team_id FROM team_members
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
        ))
    );
