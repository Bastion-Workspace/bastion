-- Migration: Add vectorization exemption support
-- Adds exemption flags to document_metadata and document_folders tables

-- Add exemption flag to document_metadata table
ALTER TABLE document_metadata 
ADD COLUMN IF NOT EXISTS exempt_from_vectorization BOOLEAN DEFAULT FALSE;

-- Add exemption flag to document_folders table
ALTER TABLE document_folders 
ADD COLUMN IF NOT EXISTS exempt_from_vectorization BOOLEAN DEFAULT FALSE;

-- Create indexes for exemption queries
CREATE INDEX IF NOT EXISTS idx_document_metadata_exempt ON document_metadata(exempt_from_vectorization);
CREATE INDEX IF NOT EXISTS idx_document_folders_exempt ON document_folders(exempt_from_vectorization);

-- Add comments
COMMENT ON COLUMN document_metadata.exempt_from_vectorization IS 'If true, document is exempt from vectorization and knowledge graph processing';
COMMENT ON COLUMN document_folders.exempt_from_vectorization IS 'If true, folder and all descendants are exempt from vectorization and knowledge graph processing';

