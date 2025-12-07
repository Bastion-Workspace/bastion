-- Migration: Add hierarchical vectorization exemption support
-- Enables three-state exemption: TRUE=exempt, FALSE=override (not exempt), NULL=inherit from parent

-- Allow NULL in exempt_from_vectorization columns
-- First drop the DEFAULT constraint, then allow NULL
ALTER TABLE document_metadata 
ALTER COLUMN exempt_from_vectorization DROP DEFAULT;

ALTER TABLE document_metadata 
ALTER COLUMN exempt_from_vectorization DROP NOT NULL;

ALTER TABLE document_folders 
ALTER COLUMN exempt_from_vectorization DROP DEFAULT;

ALTER TABLE document_folders 
ALTER COLUMN exempt_from_vectorization DROP NOT NULL;

-- Update comments to explain three-state system
COMMENT ON COLUMN document_metadata.exempt_from_vectorization IS 
'Three-state exemption: TRUE=exempt, FALSE=not exempt (override), NULL=inherit from folder';

COMMENT ON COLUMN document_folders.exempt_from_vectorization IS 
'Three-state exemption: TRUE=exempt, FALSE=not exempt (override parent), NULL=inherit from parent folder';

