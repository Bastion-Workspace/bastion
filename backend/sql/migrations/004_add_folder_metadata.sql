-- Migration: Add category and tags to document folders
-- **ROOSEVELT'S FOLDER TAGGING DOCTRINE**: Folders organize, tags follow!

-- Add category and tags columns to document_folders
ALTER TABLE document_folders 
ADD COLUMN IF NOT EXISTS category VARCHAR(100),
ADD COLUMN IF NOT EXISTS tags TEXT[] DEFAULT '{}',
ADD COLUMN IF NOT EXISTS inherit_tags BOOLEAN DEFAULT TRUE;

-- Add indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_document_folders_category ON document_folders(category);
CREATE INDEX IF NOT EXISTS idx_document_folders_tags ON document_folders USING GIN(tags);

-- Add comments
COMMENT ON COLUMN document_folders.category IS 'Folder category (inherited by documents uploaded to this folder)';
COMMENT ON COLUMN document_folders.tags IS 'Folder tags (automatically applied to documents uploaded here)';
COMMENT ON COLUMN document_folders.inherit_tags IS 'Whether documents uploaded to this folder should inherit its tags';

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON document_folders TO plato_user;

-- Log migration
DO $$
BEGIN
    RAISE NOTICE '✅ BULLY! Folder metadata migration complete - folders can now be tagged!';
END $$;


