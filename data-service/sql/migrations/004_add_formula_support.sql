-- Migration: Add formula support to data workspace tables
-- This migration adds formula_data JSONB column to store cell formulas separately from values

-- Add formula_data column to custom_data_rows table
ALTER TABLE custom_data_rows 
ADD COLUMN IF NOT EXISTS formula_data JSONB;

-- Create GIN index for efficient formula queries
CREATE INDEX idx_rows_formula_gin ON custom_data_rows USING gin(formula_data);

-- Add comment explaining the column structure
COMMENT ON COLUMN custom_data_rows.formula_data IS 'JSONB object storing formulas per column: {"column_name": "=A1+B1", ...}';

