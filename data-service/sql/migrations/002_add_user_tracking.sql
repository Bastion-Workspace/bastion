-- Migration: Add user tracking (created_by, updated_by) to data workspace tables
-- This migration adds user tracking columns to track who created and modified records

-- Add updated_by to data_workspaces (created_by is already tracked as user_id)
ALTER TABLE data_workspaces 
ADD COLUMN IF NOT EXISTS updated_by VARCHAR(255);

-- Set updated_by to user_id for existing records (backward compatibility)
UPDATE data_workspaces 
SET updated_by = user_id 
WHERE updated_by IS NULL;

-- Add created_by and updated_by to custom_databases
ALTER TABLE custom_databases 
ADD COLUMN IF NOT EXISTS created_by VARCHAR(255),
ADD COLUMN IF NOT EXISTS updated_by VARCHAR(255);

-- Add created_by and updated_by to custom_tables
ALTER TABLE custom_tables 
ADD COLUMN IF NOT EXISTS created_by VARCHAR(255),
ADD COLUMN IF NOT EXISTS updated_by VARCHAR(255);

-- Add created_by and updated_by to custom_data_rows
ALTER TABLE custom_data_rows 
ADD COLUMN IF NOT EXISTS created_by VARCHAR(255),
ADD COLUMN IF NOT EXISTS updated_by VARCHAR(255);

-- Add created_by and updated_by to external_db_connections
ALTER TABLE external_db_connections 
ADD COLUMN IF NOT EXISTS created_by VARCHAR(255),
ADD COLUMN IF NOT EXISTS updated_by VARCHAR(255);

-- Add created_by to data_import_jobs (already has created_at)
ALTER TABLE data_import_jobs 
ADD COLUMN IF NOT EXISTS created_by VARCHAR(255);

-- Add updated_by to data_visualizations (already has created_by)
ALTER TABLE data_visualizations 
ADD COLUMN IF NOT EXISTS updated_by VARCHAR(255);

-- Create indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_workspaces_updated_by ON data_workspaces(updated_by);
CREATE INDEX IF NOT EXISTS idx_databases_created_by ON custom_databases(created_by);
CREATE INDEX IF NOT EXISTS idx_databases_updated_by ON custom_databases(updated_by);
CREATE INDEX IF NOT EXISTS idx_tables_created_by ON custom_tables(created_by);
CREATE INDEX IF NOT EXISTS idx_tables_updated_by ON custom_tables(updated_by);
CREATE INDEX IF NOT EXISTS idx_rows_created_by ON custom_data_rows(created_by);
CREATE INDEX IF NOT EXISTS idx_rows_updated_by ON custom_data_rows(updated_by);
CREATE INDEX IF NOT EXISTS idx_connections_created_by ON external_db_connections(created_by);
CREATE INDEX IF NOT EXISTS idx_import_jobs_created_by ON data_import_jobs(created_by);
CREATE INDEX IF NOT EXISTS idx_visualizations_updated_by ON data_visualizations(updated_by);

