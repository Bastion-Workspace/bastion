-- Data Workspace Platform - Database Schema
-- Isolated database for user data workspaces

-- Workspaces (top-level container for databases)
CREATE TABLE data_workspaces (
    workspace_id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    icon VARCHAR(50),
    color VARCHAR(20),
    is_pinned BOOLEAN DEFAULT FALSE,
    metadata_json JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_by VARCHAR(255)
);

CREATE INDEX idx_workspaces_user ON data_workspaces(user_id);
CREATE INDEX idx_workspaces_created_at ON data_workspaces(created_at DESC);
CREATE INDEX idx_workspaces_updated_by ON data_workspaces(updated_by);

-- Custom Databases within workspaces
CREATE TABLE custom_databases (
    database_id VARCHAR(255) PRIMARY KEY,
    workspace_id VARCHAR(255) REFERENCES data_workspaces(workspace_id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    source_type VARCHAR(50) NOT NULL,
    connection_config JSONB,
    table_count INTEGER DEFAULT 0,
    total_rows BIGINT DEFAULT 0,
    metadata_json JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(255),
    updated_by VARCHAR(255)
);

CREATE INDEX idx_databases_workspace ON custom_databases(workspace_id);
CREATE INDEX idx_databases_created_by ON custom_databases(created_by);
CREATE INDEX idx_databases_updated_by ON custom_databases(updated_by);

-- Tables with styling support
CREATE TABLE custom_tables (
    table_id VARCHAR(255) PRIMARY KEY,
    database_id VARCHAR(255) REFERENCES custom_databases(database_id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    row_count INTEGER DEFAULT 0,
    schema_json JSONB NOT NULL,
    styling_rules_json JSONB,
    indexes_json JSONB,
    constraints_json JSONB,
    metadata_json JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(255),
    updated_by VARCHAR(255)
);

CREATE INDEX idx_tables_database ON custom_tables(database_id);
CREATE INDEX idx_tables_created_by ON custom_tables(created_by);
CREATE INDEX idx_tables_updated_by ON custom_tables(updated_by);

-- Data rows (flexible JSONB storage with formula support)
CREATE TABLE custom_data_rows (
    row_id VARCHAR(255) PRIMARY KEY,
    table_id VARCHAR(255) REFERENCES custom_tables(table_id) ON DELETE CASCADE,
    row_data JSONB NOT NULL,
    row_index INTEGER NOT NULL,
    row_color VARCHAR(20),
    formula_data JSONB, -- Stores formulas per column: {"column_name": "=A1+B1", ...}
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(255),
    updated_by VARCHAR(255)
);

CREATE INDEX idx_rows_table ON custom_data_rows(table_id);
CREATE INDEX idx_rows_table_index ON custom_data_rows(table_id, row_index);
CREATE INDEX idx_rows_data_gin ON custom_data_rows USING gin(row_data);
CREATE INDEX idx_rows_formula_gin ON custom_data_rows USING gin(formula_data);
CREATE INDEX idx_rows_created_by ON custom_data_rows(created_by);
CREATE INDEX idx_rows_updated_by ON custom_data_rows(updated_by);

COMMENT ON COLUMN custom_data_rows.formula_data IS 'JSONB object storing formulas per column: {"column_name": "=A1+B1", ...}';

-- External database connections
CREATE TABLE external_db_connections (
    connection_id VARCHAR(255) PRIMARY KEY,
    workspace_id VARCHAR(255) REFERENCES data_workspaces(workspace_id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    db_type VARCHAR(50) NOT NULL,
    host VARCHAR(255),
    port INTEGER,
    database_name VARCHAR(255),
    username VARCHAR(255),
    password_encrypted TEXT,
    connection_options JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    last_tested TIMESTAMP WITH TIME ZONE,
    last_sync TIMESTAMP WITH TIME ZONE,
    metadata_json JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(255),
    updated_by VARCHAR(255)
);

CREATE INDEX idx_connections_workspace ON external_db_connections(workspace_id);
CREATE INDEX idx_connections_created_by ON external_db_connections(created_by);

-- Data transformations
CREATE TABLE data_transformations (
    transformation_id VARCHAR(255) PRIMARY KEY,
    table_id VARCHAR(255) REFERENCES custom_tables(table_id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    operation_type VARCHAR(50) NOT NULL,
    config_json JSONB NOT NULL,
    result_preview_json JSONB,
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_transformations_table ON data_transformations(table_id);

-- Visualizations with color schemes
CREATE TABLE data_visualizations (
    visualization_id VARCHAR(255) PRIMARY KEY,
    workspace_id VARCHAR(255) REFERENCES data_workspaces(workspace_id) ON DELETE CASCADE,
    table_id VARCHAR(255) REFERENCES custom_tables(table_id),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    viz_type VARCHAR(50) NOT NULL,
    config_json JSONB NOT NULL,
    color_scheme VARCHAR(50),
    thumbnail_url VARCHAR(500),
    is_pinned BOOLEAN DEFAULT FALSE,
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_by VARCHAR(255)
);

CREATE INDEX idx_visualizations_workspace ON data_visualizations(workspace_id);
CREATE INDEX idx_visualizations_table ON data_visualizations(table_id);
CREATE INDEX idx_visualizations_updated_by ON data_visualizations(updated_by);

-- Import jobs
CREATE TABLE data_import_jobs (
    job_id VARCHAR(255) PRIMARY KEY,
    workspace_id VARCHAR(255) REFERENCES data_workspaces(workspace_id) ON DELETE CASCADE,
    database_id VARCHAR(255) REFERENCES custom_databases(database_id) ON DELETE SET NULL,
    table_id VARCHAR(255) REFERENCES custom_tables(table_id) ON DELETE SET NULL,
    status VARCHAR(50) NOT NULL,
    source_file VARCHAR(500),
    file_size BIGINT,
    rows_processed INTEGER DEFAULT 0,
    rows_total INTEGER,
    field_mapping_json JSONB,
    error_log TEXT,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(255)
);

CREATE INDEX idx_import_jobs_workspace ON data_import_jobs(workspace_id);
CREATE INDEX idx_import_jobs_status ON data_import_jobs(status);
CREATE INDEX idx_import_jobs_created_by ON data_import_jobs(created_by);

-- Query history
CREATE TABLE data_queries (
    query_id VARCHAR(255) PRIMARY KEY,
    workspace_id VARCHAR(255) REFERENCES data_workspaces(workspace_id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL,
    natural_language_query TEXT NOT NULL,
    query_intent VARCHAR(100),
    generated_sql TEXT,
    included_documents BOOLEAN DEFAULT FALSE,
    results_json JSONB,
    result_count INTEGER,
    execution_time_ms INTEGER,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_queries_workspace ON data_queries(workspace_id);
CREATE INDEX idx_queries_user ON data_queries(user_id);
CREATE INDEX idx_queries_created_at ON data_queries(created_at DESC);

-- Color styling rules (detailed conditional formatting)
CREATE TABLE styling_rules (
    rule_id VARCHAR(255) PRIMARY KEY,
    table_id VARCHAR(255) REFERENCES custom_tables(table_id) ON DELETE CASCADE,
    rule_name VARCHAR(255) NOT NULL,
    rule_type VARCHAR(50) NOT NULL,
    target_column VARCHAR(255),
    condition_json JSONB,
    color VARCHAR(20) NOT NULL,
    background_color VARCHAR(20),
    priority INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_styling_rules_table ON styling_rules(table_id);
CREATE INDEX idx_styling_rules_active ON styling_rules(table_id, is_active, priority);

-- Workspace sharing table
CREATE TABLE data_workspace_shares (
    id SERIAL PRIMARY KEY,
    share_id VARCHAR(255) UNIQUE NOT NULL,
    workspace_id VARCHAR(255) NOT NULL REFERENCES data_workspaces(workspace_id) ON DELETE CASCADE,
    shared_by_user_id VARCHAR(255) NOT NULL,
    shared_with_user_id VARCHAR(255), -- NULL for team/public shares
    shared_with_team_id VARCHAR(255), -- UUID as string, NULL for user/public shares
    permission_level VARCHAR(20) NOT NULL CHECK (permission_level IN ('read', 'write', 'admin')),
    is_public BOOLEAN DEFAULT FALSE,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    access_count INTEGER DEFAULT 0
);

CREATE INDEX idx_workspace_shares_workspace ON data_workspace_shares(workspace_id);
CREATE INDEX idx_workspace_shares_user ON data_workspace_shares(shared_with_user_id);
CREATE INDEX idx_workspace_shares_team ON data_workspace_shares(shared_with_team_id);
CREATE INDEX idx_workspace_shares_public ON data_workspace_shares(is_public) WHERE is_public = TRUE;

-- ============================================================================
-- Row-Level Security (RLS) Configuration
-- ============================================================================
-- Comprehensive RLS policies to enforce data isolation at the database level,
-- providing defense-in-depth security

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





