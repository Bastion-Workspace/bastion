-- Migration: Enable Row-Level Security (RLS) for data workspace tables
-- This migration adds comprehensive RLS policies to enforce data isolation
-- at the database level, providing defense-in-depth security

-- Helper function to get accessible workspace IDs for current user
-- This function checks ownership, direct shares, team shares, and public shares
CREATE OR REPLACE FUNCTION get_accessible_workspace_ids()
RETURNS TABLE(workspace_id VARCHAR) AS $$
BEGIN
    RETURN QUERY
    -- Owned workspaces
    SELECT dw.workspace_id::VARCHAR
    FROM data_workspaces dw
    WHERE dw.user_id = current_setting('app.current_user_id', true)::varchar
    
    UNION
    
    -- Direct user shares
    SELECT dws.workspace_id::VARCHAR
    FROM data_workspace_shares dws
    WHERE dws.shared_with_user_id = current_setting('app.current_user_id', true)::varchar
    AND (dws.expires_at IS NULL OR dws.expires_at > NOW())
    
    UNION
    
    -- Team shares (check if user's team IDs match)
    SELECT dws.workspace_id::VARCHAR
    FROM data_workspace_shares dws
    WHERE dws.shared_with_team_id IS NOT NULL
    AND dws.shared_with_team_id = ANY(
        string_to_array(
            NULLIF(current_setting('app.current_user_team_ids', true)::varchar, ''),
            ','
        )
    )
    AND (dws.expires_at IS NULL OR dws.expires_at > NOW())
    
    UNION
    
    -- Public shares
    SELECT dws.workspace_id::VARCHAR
    FROM data_workspace_shares dws
    WHERE dws.is_public = TRUE
    AND (dws.expires_at IS NULL OR dws.expires_at > NOW());
END;
$$ LANGUAGE plpgsql STABLE;

-- Enable RLS on data_workspaces
ALTER TABLE data_workspaces ENABLE ROW LEVEL SECURITY;

-- Policy for data_workspaces: users see owned + shared workspaces
CREATE POLICY workspaces_user_policy ON data_workspaces
    FOR ALL USING (
        workspace_id IN (SELECT workspace_id FROM get_accessible_workspace_ids())
        OR user_id = current_setting('app.current_user_id', true)::varchar
    );

-- Enable RLS on custom_databases
ALTER TABLE custom_databases ENABLE ROW LEVEL SECURITY;

-- Policy for custom_databases: accessible via workspace permissions
CREATE POLICY databases_user_policy ON custom_databases
    FOR ALL USING (
        workspace_id IN (SELECT workspace_id FROM get_accessible_workspace_ids())
    );

-- Enable RLS on custom_tables
ALTER TABLE custom_tables ENABLE ROW LEVEL SECURITY;

-- Policy for custom_tables: accessible via parent database → workspace chain
CREATE POLICY tables_user_policy ON custom_tables
    FOR ALL USING (
        database_id IN (
            SELECT database_id FROM custom_databases
            WHERE workspace_id IN (SELECT workspace_id FROM get_accessible_workspace_ids())
        )
    );

-- Enable RLS on custom_data_rows
ALTER TABLE custom_data_rows ENABLE ROW LEVEL SECURITY;

-- Policy for custom_data_rows: accessible via parent table → database → workspace chain
CREATE POLICY data_rows_user_policy ON custom_data_rows
    FOR ALL USING (
        table_id IN (
            SELECT table_id FROM custom_tables
            WHERE database_id IN (
                SELECT database_id FROM custom_databases
                WHERE workspace_id IN (SELECT workspace_id FROM get_accessible_workspace_ids())
            )
        )
    );

-- Enable RLS on data_visualizations
ALTER TABLE data_visualizations ENABLE ROW LEVEL SECURITY;

-- Policy for data_visualizations: workspace-level access
CREATE POLICY visualizations_user_policy ON data_visualizations
    FOR ALL USING (
        workspace_id IN (SELECT workspace_id FROM get_accessible_workspace_ids())
    );

-- Enable RLS on data_queries
ALTER TABLE data_queries ENABLE ROW LEVEL SECURITY;

-- Policy for data_queries: users see their own queries + workspace access
CREATE POLICY queries_user_policy ON data_queries
    FOR ALL USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR workspace_id IN (SELECT workspace_id FROM get_accessible_workspace_ids())
    );

-- Enable RLS on data_import_jobs
ALTER TABLE data_import_jobs ENABLE ROW LEVEL SECURITY;

-- Policy for data_import_jobs: workspace-level access
CREATE POLICY import_jobs_user_policy ON data_import_jobs
    FOR ALL USING (
        workspace_id IN (SELECT workspace_id FROM get_accessible_workspace_ids())
    );

-- Enable RLS on external_db_connections
ALTER TABLE external_db_connections ENABLE ROW LEVEL SECURITY;

-- Policy for external_db_connections: strict workspace access (sensitive credentials)
CREATE POLICY external_connections_user_policy ON external_db_connections
    FOR ALL USING (
        workspace_id IN (SELECT workspace_id FROM get_accessible_workspace_ids())
    );

-- Enable RLS on data_transformations
ALTER TABLE data_transformations ENABLE ROW LEVEL SECURITY;

-- Policy for data_transformations: accessible via parent table → database → workspace chain
CREATE POLICY transformations_user_policy ON data_transformations
    FOR ALL USING (
        table_id IN (
            SELECT table_id FROM custom_tables
            WHERE database_id IN (
                SELECT database_id FROM custom_databases
                WHERE workspace_id IN (SELECT workspace_id FROM get_accessible_workspace_ids())
            )
        )
    );

-- Enable RLS on styling_rules
ALTER TABLE styling_rules ENABLE ROW LEVEL SECURITY;

-- Policy for styling_rules: accessible via parent table → database → workspace chain
CREATE POLICY styling_rules_user_policy ON styling_rules
    FOR ALL USING (
        table_id IN (
            SELECT table_id FROM custom_tables
            WHERE database_id IN (
                SELECT database_id FROM custom_databases
                WHERE workspace_id IN (SELECT workspace_id FROM get_accessible_workspace_ids())
            )
        )
    );

-- Note: data_workspace_shares table does NOT have RLS enabled
-- This table is used by the RLS policies themselves, so it must be accessible
-- Application-level checks ensure only workspace owners can view/manage shares

-- Create indexes to optimize RLS policy queries
-- Note: Partial indexes with NOW() cannot be created (NOW() is not IMMUTABLE)
-- These indexes will help with general queries, but expiration checks happen at query time
CREATE INDEX IF NOT EXISTS idx_workspace_shares_user_active 
    ON data_workspace_shares(shared_with_user_id) 
    WHERE shared_with_user_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_workspace_shares_team_active 
    ON data_workspace_shares(shared_with_team_id) 
    WHERE shared_with_team_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_workspace_shares_public_active 
    ON data_workspace_shares(workspace_id) 
    WHERE is_public = TRUE;

-- Additional index on expires_at for efficient expiration filtering
CREATE INDEX IF NOT EXISTS idx_workspace_shares_expires_at 
    ON data_workspace_shares(expires_at) 
    WHERE expires_at IS NOT NULL;

