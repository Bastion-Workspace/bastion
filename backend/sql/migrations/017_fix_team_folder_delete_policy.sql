-- Migration 017: Fix team folder delete policy to allow team admins full control
-- 
-- Problem: Team admins can only delete folders they personally created, not folders
-- created by other team members.
--
-- Solution: Allow team admins to delete ANY folder in their team, while regular members
-- can only delete their own folders.

-- Drop and recreate the document_folders delete policy
DROP POLICY IF EXISTS document_folders_delete_policy ON document_folders;

CREATE POLICY document_folders_delete_policy ON document_folders
FOR DELETE USING (
    -- User folders: user can delete their own
    (user_id = current_setting('app.current_user_id', true)::varchar 
     AND collection_type = 'user')
    
    -- Global folders: only admins can delete
    OR (collection_type = 'global' 
        AND current_setting('app.current_user_role', true) = 'admin')
    
    -- Team folders: creator can delete, OR team admin can delete any team folder
    OR (team_id IS NOT NULL 
        AND (
            -- Creator can delete their own folder
            created_by = current_setting('app.current_user_id', true)::varchar
            
            -- Team admin can delete ANY folder in their team
            OR EXISTS (
                SELECT 1 FROM team_members tm 
                WHERE tm.team_id = document_folders.team_id 
                AND tm.user_id = current_setting('app.current_user_id', true)::varchar 
                AND tm.role = 'admin'
            )
        )
    )
);


