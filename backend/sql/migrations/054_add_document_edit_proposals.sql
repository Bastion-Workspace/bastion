-- ========================================
-- DOCUMENT EDIT PROPOSALS (persistent)
-- ========================================
-- Moves in-editor document edit proposals from in-memory to PostgreSQL.
-- Idempotent: safe to run multiple times.
-- ========================================

CREATE TABLE IF NOT EXISTS document_edit_proposals (
    id SERIAL PRIMARY KEY,
    proposal_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    document_id VARCHAR(255) NOT NULL REFERENCES document_metadata(document_id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,

    edit_type VARCHAR(20) NOT NULL CHECK (edit_type IN ('operations', 'content')),
    operations JSONB DEFAULT '[]',
    content_edit JSONB,

    agent_name VARCHAR(100) NOT NULL DEFAULT 'unknown',
    summary TEXT,
    requires_preview BOOLEAN DEFAULT TRUE,

    content_hash VARCHAR(64),
    expires_at TIMESTAMP WITH TIME ZONE,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_document_edit_proposals_document_id ON document_edit_proposals(document_id);
CREATE INDEX IF NOT EXISTS idx_document_edit_proposals_user_id ON document_edit_proposals(user_id);
CREATE INDEX IF NOT EXISTS idx_document_edit_proposals_expires_at ON document_edit_proposals(expires_at) WHERE expires_at IS NOT NULL;

ALTER TABLE document_edit_proposals ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS document_edit_proposals_select_policy ON document_edit_proposals;
DROP POLICY IF EXISTS document_edit_proposals_insert_policy ON document_edit_proposals;
DROP POLICY IF EXISTS document_edit_proposals_update_policy ON document_edit_proposals;
DROP POLICY IF EXISTS document_edit_proposals_delete_policy ON document_edit_proposals;

CREATE POLICY document_edit_proposals_select_policy ON document_edit_proposals
    FOR SELECT USING (
        user_id = current_setting('app.current_user_id', true)::varchar
    );

CREATE POLICY document_edit_proposals_insert_policy ON document_edit_proposals
    FOR INSERT WITH CHECK (
        user_id = current_setting('app.current_user_id', true)::varchar
    );

CREATE POLICY document_edit_proposals_update_policy ON document_edit_proposals
    FOR UPDATE USING (
        user_id = current_setting('app.current_user_id', true)::varchar
    );

CREATE POLICY document_edit_proposals_delete_policy ON document_edit_proposals
    FOR DELETE USING (
        user_id = current_setting('app.current_user_id', true)::varchar
    );

GRANT SELECT, INSERT, UPDATE, DELETE ON document_edit_proposals TO bastion_user;
GRANT USAGE, SELECT ON document_edit_proposals_id_seq TO bastion_user;

-- Function for Celery cleanup task: delete expired proposals (bypasses RLS via SECURITY DEFINER)
CREATE OR REPLACE FUNCTION cleanup_expired_document_edit_proposals()
RETURNS integer
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  deleted integer;
BEGIN
  DELETE FROM document_edit_proposals WHERE expires_at < NOW();
  GET DIAGNOSTICS deleted = ROW_COUNT;
  RETURN deleted;
END;
$$;
GRANT EXECUTE ON FUNCTION cleanup_expired_document_edit_proposals() TO bastion_user;
