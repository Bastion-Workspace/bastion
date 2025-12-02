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

-- Data rows (flexible JSONB storage)
CREATE TABLE custom_data_rows (
    row_id VARCHAR(255) PRIMARY KEY,
    table_id VARCHAR(255) REFERENCES custom_tables(table_id) ON DELETE CASCADE,
    row_data JSONB NOT NULL,
    row_index INTEGER NOT NULL,
    row_color VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(255),
    updated_by VARCHAR(255)
);

CREATE INDEX idx_rows_table ON custom_data_rows(table_id);
CREATE INDEX idx_rows_table_index ON custom_data_rows(table_id, row_index);
CREATE INDEX idx_rows_data_gin ON custom_data_rows USING gin(row_data);
CREATE INDEX idx_rows_created_by ON custom_data_rows(created_by);
CREATE INDEX idx_rows_updated_by ON custom_data_rows(updated_by);

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









