-- ========================================
-- DOCUMENT LINKS (file relation graph)
-- ========================================
-- Adds document_links table and RLS for Obsidian-style file relation cloud.
-- Idempotent: safe to run multiple times.
--
-- Run from host (postgres container must be up):
--   docker exec -i bastion-postgres psql -U postgres -d bastion_knowledge_base < backend/sql/migrations/041_add_document_links.sql
-- ========================================

CREATE TABLE IF NOT EXISTS document_links (
    id SERIAL PRIMARY KEY,
    source_document_id VARCHAR(255) NOT NULL REFERENCES document_metadata(document_id) ON DELETE CASCADE,
    target_document_id VARCHAR(255) REFERENCES document_metadata(document_id) ON DELETE SET NULL,
    target_raw_path TEXT NOT NULL,
    link_type VARCHAR(50) NOT NULL,
    description TEXT,
    line_number INTEGER,
    user_id VARCHAR(255),
    collection_type VARCHAR(20),
    team_id UUID REFERENCES teams(team_id) ON DELETE SET NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_document_id, target_raw_path, line_number)
);

CREATE INDEX IF NOT EXISTS idx_document_links_source ON document_links(source_document_id);
CREATE INDEX IF NOT EXISTS idx_document_links_target ON document_links(target_document_id);
CREATE INDEX IF NOT EXISTS idx_document_links_user_id ON document_links(user_id) WHERE user_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_document_links_team_id ON document_links(team_id) WHERE team_id IS NOT NULL;

ALTER TABLE document_links ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS document_links_select_policy ON document_links;
DROP POLICY IF EXISTS document_links_update_policy ON document_links;
DROP POLICY IF EXISTS document_links_delete_policy ON document_links;
DROP POLICY IF EXISTS document_links_insert_policy ON document_links;

CREATE POLICY document_links_select_policy ON document_links
    FOR SELECT USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR collection_type = 'global'
        OR (team_id IS NOT NULL AND team_id IN (
            SELECT team_id FROM team_members
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
        ))
        OR (user_id IS NOT NULL AND team_id IS NULL AND current_setting('app.current_user_role', true) = 'admin')
    );

CREATE POLICY document_links_update_policy ON document_links
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

CREATE POLICY document_links_delete_policy ON document_links
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

CREATE POLICY document_links_insert_policy ON document_links
    FOR INSERT WITH CHECK (
        current_setting('app.current_user_role', true) = 'admin'
        OR user_id = current_setting('app.current_user_id', true)::varchar
        OR collection_type = 'global'
        OR (team_id IS NOT NULL AND team_id IN (
            SELECT team_id FROM team_members
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
        ))
    );
