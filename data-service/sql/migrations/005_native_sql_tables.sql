-- Migration: Add storage_type to custom_tables for native SQL tables
-- native = real PostgreSQL table in workspace schema; jsonb = legacy custom_data_rows storage

ALTER TABLE custom_tables
ADD COLUMN IF NOT EXISTS storage_type VARCHAR(20) NOT NULL DEFAULT 'jsonb';

COMMENT ON COLUMN custom_tables.storage_type IS 'jsonb = rows in custom_data_rows; native = real table in ws_<workspace_id> schema';

CREATE INDEX IF NOT EXISTS idx_tables_storage_type ON custom_tables(storage_type);
