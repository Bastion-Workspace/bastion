-- 159: Allow system admins to UPDATE global document_folders rows (user_id IS NULL).
--
-- The previous UPDATE policy's admin branch was (user_id IS NOT NULL AND team_id IS NULL AND role = 'admin),
-- which never matched global root rows. INSERT ... ON CONFLICT DO UPDATE for global root upserts
-- therefore failed RLS (INSERT could succeed; DO UPDATE could not). Greenfield and folder ensure
-- paths use admin RLS for global; this aligns the row policy with DELETE (which already allows
-- admin for collection_type = 'global').

DROP POLICY IF EXISTS document_folders_update_policy ON document_folders;

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
        -- System admins can update user-owned top-level folders (user_id set, not team)
        OR (user_id IS NOT NULL AND team_id IS NULL AND current_setting('app.current_user_role', true) = 'admin')
        -- Global root rows (user_id NULL): admins only (matches delete policy spirit)
        OR (collection_type = 'global' AND team_id IS NULL AND current_setting('app.current_user_role', true) = 'admin')
    )
    WITH CHECK (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR (team_id IS NOT NULL AND team_id IN (
            SELECT team_id FROM team_members
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
            AND role = 'admin'
        ))
        OR (user_id IS NOT NULL AND team_id IS NULL AND current_setting('app.current_user_role', true) = 'admin')
        OR (collection_type = 'global' AND team_id IS NULL AND current_setting('app.current_user_role', true) = 'admin')
    );

COMMENT ON POLICY document_folders_update_policy ON document_folders IS
  'Users update own folders; team admins update team folders; admins update user top-level and global (user_id NULL) folders.';
