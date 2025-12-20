-- ========================================
-- FIX TEAM PRIVACY RLS POLICIES
-- Remove admin bypass for team content to enforce team privacy
-- ========================================
-- This migration fixes a security issue where system admins could see
-- all team folders and documents even if they weren't team members.
--
-- After this migration:
-- - System admins can see user documents and global documents
-- - System admins CANNOT see team documents unless they're team members
-- - Team privacy is enforced for all users, including system admins
--
-- Usage:
-- docker exec -i <postgres-container> psql -U postgres -d bastion_knowledge_base < backend/sql/migrations/010_fix_team_privacy_rls_policies.sql
-- ========================================

-- Drop existing policies
DROP POLICY IF EXISTS document_metadata_select_policy ON document_metadata;
DROP POLICY IF EXISTS document_metadata_update_policy ON document_metadata;
DROP POLICY IF EXISTS document_folders_select_policy ON document_folders;
DROP POLICY IF EXISTS document_folders_update_policy ON document_folders;

-- ========================================
-- DOCUMENT METADATA POLICIES (FIXED)
-- ========================================

-- RLS Policy: Users can see their own docs, global docs, or team docs they're members of
-- System admins can see user docs and global docs, but NOT team docs unless they're members
CREATE POLICY document_metadata_select_policy ON document_metadata
    FOR SELECT USING (
        -- User's own documents
        user_id = current_setting('app.current_user_id', true)::varchar
        -- Global documents (everyone can see)
        OR collection_type = 'global'
        -- Team documents (only if user is a team member - NO admin bypass for privacy)
        OR (team_id IS NOT NULL AND team_id IN (
            SELECT team_id FROM team_members 
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
        ))
        -- System admins can see user documents (but NOT team documents unless they're members)
        OR (user_id IS NOT NULL AND team_id IS NULL AND current_setting('app.current_user_role', true) = 'admin')
    );

-- RLS Policy: Users can update their own docs, team admins can update team docs
-- System admins can update user docs and global docs, but NOT team docs unless they're team admins
CREATE POLICY document_metadata_update_policy ON document_metadata
    FOR UPDATE 
    USING (
        -- User's own documents
        user_id = current_setting('app.current_user_id', true)::varchar
        -- Team documents (only team admins can update - NO system admin bypass for privacy)
        OR (team_id IS NOT NULL AND team_id IN (
            SELECT team_id FROM team_members 
            WHERE user_id = current_setting('app.current_user_id', true)::varchar 
            AND role = 'admin'
        ))
        -- System admins can update user documents and global documents (but NOT team documents unless they're team admins)
        OR (team_id IS NULL AND current_setting('app.current_user_role', true) = 'admin')
    )
    WITH CHECK (
        -- User's own documents
        user_id = current_setting('app.current_user_id', true)::varchar
        -- Team documents (only team admins can update - NO system admin bypass for privacy)
        OR (team_id IS NOT NULL AND team_id IN (
            SELECT team_id FROM team_members 
            WHERE user_id = current_setting('app.current_user_id', true)::varchar 
            AND role = 'admin'
        ))
        -- System admins can update user documents and global documents (but NOT team documents unless they're team admins)
        OR (team_id IS NULL AND current_setting('app.current_user_role', true) = 'admin')
    );

-- ========================================
-- DOCUMENT FOLDERS POLICIES (FIXED)
-- ========================================

-- RLS Policy: Users can see their own folders, global folders, or team folders they're members of
-- System admins can see user folders and global folders, but NOT team folders unless they're members
CREATE POLICY document_folders_select_policy ON document_folders
    FOR SELECT USING (
        -- Users can see their own folders
        user_id = current_setting('app.current_user_id', true)::varchar
        -- Everyone can see global folders
        OR collection_type = 'global'
        -- Team members can see team folders (NO admin bypass for privacy)
        OR (team_id IS NOT NULL AND team_id IN (
            SELECT team_id FROM team_members 
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
        ))
        -- System admins can see user folders (but NOT team folders unless they're members)
        OR (user_id IS NOT NULL AND team_id IS NULL AND current_setting('app.current_user_role', true) = 'admin')
    );

-- RLS Policy: Users can update their own folders, team admins can update team folders
-- System admins can update user folders and global folders, but NOT team folders unless they're team admins
CREATE POLICY document_folders_update_policy ON document_folders
    FOR UPDATE 
    USING (
        -- Users can update their own folders
        user_id = current_setting('app.current_user_id', true)::varchar
        -- Team admins can update team folders (NO system admin bypass for privacy)
        OR (team_id IS NOT NULL AND team_id IN (
            SELECT team_id FROM team_members 
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
            AND role = 'admin'
        ))
        -- System admins can update user folders (but NOT team folders unless they're team admins)
        OR (user_id IS NOT NULL AND team_id IS NULL AND current_setting('app.current_user_role', true) = 'admin')
    )
    WITH CHECK (
        -- Users can update their own folders (same as USING)
        user_id = current_setting('app.current_user_id', true)::varchar
        -- Team admins can update team folders (NO system admin bypass for privacy)
        OR (team_id IS NOT NULL AND team_id IN (
            SELECT team_id FROM team_members 
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
            AND role = 'admin'
        ))
        -- System admins can update user folders (but NOT team folders unless they're team admins)
        OR (user_id IS NOT NULL AND team_id IS NULL AND current_setting('app.current_user_role', true) = 'admin')
    );

-- Add comments for documentation
COMMENT ON POLICY document_metadata_select_policy ON document_metadata IS 'Users see own docs, global docs, or team docs they are members of. System admins cannot see team docs unless they are members.';
COMMENT ON POLICY document_metadata_update_policy ON document_metadata IS 'Users update own docs, team admins update team docs. System admins cannot update team docs unless they are team admins.';
COMMENT ON POLICY document_folders_select_policy ON document_folders IS 'Users see own folders, global folders, or team folders they are members of. System admins cannot see team folders unless they are members.';
COMMENT ON POLICY document_folders_update_policy ON document_folders IS 'Users update own folders, team admins update team folders. System admins cannot update team folders unless they are team admins.';



