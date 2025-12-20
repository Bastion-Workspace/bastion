-- ========================================
-- ADD FOLDER OWNERSHIP TRACKING
-- Add created_by column to document_folders to track folder ownership
-- This enables regular team members to delete folders they created
-- while preventing them from deleting other members' folders
-- ========================================
-- Usage:
-- docker exec -i <postgres-container> psql -U bastion_user -d bastion_knowledge_base < backend/sql/migrations/016_add_folder_ownership.sql
-- Or from within container:
-- psql -U bastion_user -d bastion_knowledge_base -f /docker-entrypoint-initdb.d/migrations/016_add_folder_ownership.sql
-- ========================================

-- Add created_by column to document_folders
ALTER TABLE document_folders
ADD COLUMN IF NOT EXISTS created_by VARCHAR(255) REFERENCES users(user_id) ON DELETE SET NULL;

-- Create index for created_by lookups
CREATE INDEX IF NOT EXISTS idx_document_folders_created_by ON document_folders(created_by);

-- Update existing team folders: Set created_by to team creator
-- For team root folders, use the team's created_by
UPDATE document_folders df
SET created_by = t.created_by
FROM teams t
WHERE df.team_id = t.team_id
  AND df.collection_type = 'team'
  AND df.parent_folder_id IS NULL
  AND df.created_by IS NULL;

-- For team subfolders, we can't determine the original creator from existing data
-- Set them to NULL (they'll be protected by team admin policy)
-- New folders will have created_by set correctly going forward

-- Drop existing DELETE policy
DROP POLICY IF EXISTS document_folders_delete_policy ON document_folders;

-- Recreate DELETE policy with ownership support
CREATE POLICY document_folders_delete_policy ON document_folders
    FOR DELETE USING (
        -- Users can delete their own user folders
        user_id = current_setting('app.current_user_id', true)::varchar
        -- Team admins can delete any team folder
        OR (team_id IS NOT NULL AND team_id IN (
            SELECT team_id FROM team_members 
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
            AND role = 'admin'
        ))
        -- Regular team members can delete team folders they created
        OR (team_id IS NOT NULL AND created_by = current_setting('app.current_user_id', true)::varchar)
        -- System admins can delete all folders
        OR current_setting('app.current_user_role', true) = 'admin'
    );



