-- Per-user document and folder shares, edit locks, and RLS extensions.

CREATE TABLE IF NOT EXISTS document_shares (
    id SERIAL PRIMARY KEY,
    share_id VARCHAR(255) UNIQUE NOT NULL,
    document_id VARCHAR(255) REFERENCES document_metadata(document_id) ON DELETE CASCADE,
    folder_id VARCHAR(255) REFERENCES document_folders(folder_id) ON DELETE CASCADE,
    shared_by_user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    shared_with_user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    share_type VARCHAR(10) NOT NULL CHECK (share_type IN ('read', 'write')),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMPTZ,
    CONSTRAINT document_shares_target_check CHECK (
        (document_id IS NOT NULL AND folder_id IS NULL)
        OR (document_id IS NULL AND folder_id IS NOT NULL)
    ),
    CONSTRAINT document_shares_no_self_share CHECK (shared_by_user_id <> shared_with_user_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_document_shares_unique_document
    ON document_shares (shared_with_user_id, document_id)
    WHERE document_id IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_document_shares_unique_folder
    ON document_shares (shared_with_user_id, folder_id)
    WHERE folder_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_document_shares_shared_with ON document_shares (shared_with_user_id);
CREATE INDEX IF NOT EXISTS idx_document_shares_shared_by ON document_shares (shared_by_user_id);
CREATE INDEX IF NOT EXISTS idx_document_shares_document ON document_shares (document_id) WHERE document_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_document_shares_folder ON document_shares (folder_id) WHERE folder_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS document_locks (
    document_id VARCHAR(255) PRIMARY KEY REFERENCES document_metadata(document_id) ON DELETE CASCADE,
    locked_by_user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    acquired_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMPTZ NOT NULL,
    heartbeat_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_document_locks_locked_by ON document_locks (locked_by_user_id);
CREATE INDEX IF NOT EXISTS idx_document_locks_expires ON document_locks (expires_at);

-- Share access for documents/chunks: direct document share or document under a shared folder subtree.
CREATE OR REPLACE FUNCTION document_user_has_share_access(
    p_document_id VARCHAR,
    p_folder_id VARCHAR,
    p_user_id VARCHAR,
    p_require_write BOOLEAN
)
RETURNS BOOLEAN AS $$
BEGIN
    IF p_user_id IS NULL OR p_user_id = '' THEN
        RETURN FALSE;
    END IF;

    IF EXISTS (
        SELECT 1 FROM document_shares ds
        WHERE ds.shared_with_user_id = p_user_id
          AND ds.document_id IS NOT NULL
          AND ds.document_id = p_document_id
          AND (ds.expires_at IS NULL OR ds.expires_at > NOW())
          AND (NOT p_require_write OR ds.share_type = 'write')
          AND (p_require_write OR ds.share_type IN ('read', 'write'))
    ) THEN
        RETURN TRUE;
    END IF;

    IF p_folder_id IS NOT NULL AND EXISTS (
        SELECT 1 FROM document_shares ds
        INNER JOIN LATERAL (
            WITH RECURSIVE descendants AS (
                SELECT df.folder_id
                FROM document_folders df
                WHERE df.folder_id = ds.folder_id
                UNION ALL
                SELECT c.folder_id
                FROM document_folders c
                INNER JOIN descendants d ON c.parent_folder_id = d.folder_id
            )
            SELECT d.folder_id FROM descendants d
        ) sub ON sub.folder_id = p_folder_id
        WHERE ds.shared_with_user_id = p_user_id
          AND ds.folder_id IS NOT NULL
          AND (ds.expires_at IS NULL OR ds.expires_at > NOW())
          AND (NOT p_require_write OR ds.share_type = 'write')
          AND (p_require_write OR ds.share_type IN ('read', 'write'))
    ) THEN
        RETURN TRUE;
    END IF;

    RETURN FALSE;
END;
$$ LANGUAGE plpgsql STABLE SECURITY DEFINER;

-- Folder visible if it is under a subtree rooted at a folder shared with the user.
CREATE OR REPLACE FUNCTION document_folder_shared_with_user(
    p_folder_id VARCHAR,
    p_user_id VARCHAR
)
RETURNS BOOLEAN AS $$
BEGIN
    IF p_folder_id IS NULL OR p_user_id IS NULL OR p_user_id = '' THEN
        RETURN FALSE;
    END IF;

    RETURN EXISTS (
        SELECT 1 FROM document_shares ds
        INNER JOIN LATERAL (
            WITH RECURSIVE descendants AS (
                SELECT df.folder_id
                FROM document_folders df
                WHERE df.folder_id = ds.folder_id
                UNION ALL
                SELECT c.folder_id
                FROM document_folders c
                INNER JOIN descendants d ON c.parent_folder_id = d.folder_id
            )
            SELECT d.folder_id FROM descendants d
        ) sub ON sub.folder_id = p_folder_id
        WHERE ds.folder_id IS NOT NULL
          AND ds.shared_with_user_id = p_user_id
          AND (ds.expires_at IS NULL OR ds.expires_at > NOW())
    );
END;
$$ LANGUAGE plpgsql STABLE SECURITY DEFINER;

-- Amend document_metadata RLS
DROP POLICY IF EXISTS document_metadata_select_policy ON document_metadata;
DROP POLICY IF EXISTS document_metadata_update_policy ON document_metadata;

CREATE POLICY document_metadata_select_policy ON document_metadata
    FOR SELECT USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR collection_type = 'global'
        OR (team_id IS NOT NULL AND team_id IN (
            SELECT team_id FROM team_members
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
        ))
        OR (user_id IS NOT NULL AND team_id IS NULL AND current_setting('app.current_user_role', true) = 'admin')
        OR document_user_has_share_access(
            document_id,
            folder_id,
            current_setting('app.current_user_id', true)::varchar,
            FALSE
        )
    );

CREATE POLICY document_metadata_update_policy ON document_metadata
    FOR UPDATE
    USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR (team_id IS NOT NULL AND team_id IN (
            SELECT team_id FROM team_members
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
            AND role = 'admin'
        ))
        OR (team_id IS NULL AND current_setting('app.current_user_role', true) = 'admin')
        OR document_user_has_share_access(
            document_id,
            folder_id,
            current_setting('app.current_user_id', true)::varchar,
            TRUE
        )
    )
    WITH CHECK (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR (team_id IS NOT NULL AND team_id IN (
            SELECT team_id FROM team_members
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
            AND role = 'admin'
        ))
        OR (team_id IS NULL AND current_setting('app.current_user_role', true) = 'admin')
        OR document_user_has_share_access(
            document_id,
            folder_id,
            current_setting('app.current_user_id', true)::varchar,
            TRUE
        )
    );

-- Amend document_chunks RLS
DROP POLICY IF EXISTS document_chunks_select_policy ON document_chunks;
DROP POLICY IF EXISTS document_chunks_update_policy ON document_chunks;

CREATE POLICY document_chunks_select_policy ON document_chunks
    FOR SELECT USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR collection_type = 'global'
        OR (team_id IS NOT NULL AND team_id IN (
            SELECT team_id FROM team_members
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
        ))
        OR (user_id IS NOT NULL AND team_id IS NULL AND current_setting('app.current_user_role', true) = 'admin')
        OR document_user_has_share_access(
            document_id,
            (SELECT dm.folder_id FROM document_metadata dm WHERE dm.document_id = document_chunks.document_id LIMIT 1),
            current_setting('app.current_user_id', true)::varchar,
            FALSE
        )
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
        OR document_user_has_share_access(
            document_id,
            (SELECT dm.folder_id FROM document_metadata dm WHERE dm.document_id = document_chunks.document_id LIMIT 1),
            current_setting('app.current_user_id', true)::varchar,
            TRUE
        )
    )
    WITH CHECK (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR (team_id IS NOT NULL AND team_id IN (
            SELECT team_id FROM team_members
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
            AND role = 'admin'
        ))
        OR (team_id IS NULL AND current_setting('app.current_user_role', true) = 'admin')
        OR document_user_has_share_access(
            document_id,
            (SELECT dm.folder_id FROM document_metadata dm WHERE dm.document_id = document_chunks.document_id LIMIT 1),
            current_setting('app.current_user_id', true)::varchar,
            TRUE
        )
    );

-- Shared folders: recipients can see folder rows in shared subtrees (not write/delete via share alone).
DROP POLICY IF EXISTS document_folders_select_policy ON document_folders;

CREATE POLICY document_folders_select_policy ON document_folders
    FOR SELECT USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR collection_type = 'global'
        OR (team_id IS NOT NULL AND team_id IN (
            SELECT team_id FROM team_members
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
        ))
        OR (user_id IS NOT NULL AND team_id IS NULL AND current_setting('app.current_user_role', true) = 'admin')
        OR document_folder_shared_with_user(
            folder_id,
            current_setting('app.current_user_id', true)::varchar
        )
    );

ALTER TABLE document_shares ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_locks ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS document_shares_select_policy ON document_shares;
DROP POLICY IF EXISTS document_shares_insert_policy ON document_shares;
DROP POLICY IF EXISTS document_shares_update_policy ON document_shares;
DROP POLICY IF EXISTS document_shares_delete_policy ON document_shares;
DROP POLICY IF EXISTS document_locks_select_policy ON document_locks;
DROP POLICY IF EXISTS document_locks_insert_policy ON document_locks;
DROP POLICY IF EXISTS document_locks_update_policy ON document_locks;
DROP POLICY IF EXISTS document_locks_delete_policy ON document_locks;

CREATE POLICY document_shares_select_policy ON document_shares
    FOR SELECT USING (
        shared_by_user_id = current_setting('app.current_user_id', true)::varchar
        OR shared_with_user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE POLICY document_shares_insert_policy ON document_shares
    FOR INSERT WITH CHECK (
        shared_by_user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE POLICY document_shares_update_policy ON document_shares
    FOR UPDATE USING (
        shared_by_user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE POLICY document_shares_delete_policy ON document_shares
    FOR DELETE USING (
        shared_by_user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE POLICY document_locks_select_policy ON document_locks
    FOR SELECT USING (
        locked_by_user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
        OR EXISTS (
            SELECT 1 FROM document_metadata dm
            WHERE dm.document_id = document_locks.document_id
              AND dm.user_id = current_setting('app.current_user_id', true)::varchar
        )
        OR document_user_has_share_access(
            document_locks.document_id,
            (SELECT dm.folder_id FROM document_metadata dm WHERE dm.document_id = document_locks.document_id LIMIT 1),
            current_setting('app.current_user_id', true)::varchar,
            FALSE
        )
    );

CREATE POLICY document_locks_insert_policy ON document_locks
    FOR INSERT WITH CHECK (
        locked_by_user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE POLICY document_locks_update_policy ON document_locks
    FOR UPDATE USING (
        locked_by_user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    )
    WITH CHECK (
        locked_by_user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE POLICY document_locks_delete_policy ON document_locks
    FOR DELETE USING (
        locked_by_user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

GRANT SELECT, INSERT, UPDATE, DELETE ON document_shares TO bastion_user;
GRANT SELECT, UPDATE, USAGE ON SEQUENCE document_shares_id_seq TO bastion_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON document_locks TO bastion_user;
