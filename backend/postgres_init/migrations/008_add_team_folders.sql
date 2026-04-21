-- Migration 008: Add team folders support
-- Description: Add team_id column to document_folders table to support team-specific folders

-- Add team_id column to document_folders (UUID to match teams table)
ALTER TABLE document_folders
ADD COLUMN IF NOT EXISTS team_id UUID REFERENCES teams(team_id) ON DELETE CASCADE;

-- Create index for team_id lookups
CREATE INDEX IF NOT EXISTS idx_document_folders_team_id ON document_folders(team_id);

-- Create unique partial indexes for team folders (required for ON CONFLICT)
-- For TEAM root folders (parent_folder_id IS NULL AND team_id IS NOT NULL)
CREATE UNIQUE INDEX IF NOT EXISTS idx_document_folders_unique_team_root 
ON document_folders(team_id, name, collection_type)
WHERE parent_folder_id IS NULL AND team_id IS NOT NULL;

-- For TEAM non-root folders (parent_folder_id IS NOT NULL AND team_id IS NOT NULL)
CREATE UNIQUE INDEX IF NOT EXISTS idx_document_folders_unique_team_with_parent
ON document_folders(team_id, name, parent_folder_id, collection_type)
WHERE parent_folder_id IS NOT NULL AND team_id IS NOT NULL;

-- Update collection_type constraint to include 'team' type
ALTER TABLE document_folders
DROP CONSTRAINT IF EXISTS document_folders_collection_type_check;

ALTER TABLE document_folders
ADD CONSTRAINT document_folders_collection_type_check 
CHECK (collection_type IN ('user', 'global', 'team'));

-- Update the user_id constraint to handle team folders
ALTER TABLE document_folders
DROP CONSTRAINT IF EXISTS check_user_id_for_collection_type;

ALTER TABLE document_folders
ADD CONSTRAINT check_user_id_for_collection_type CHECK (
    (collection_type = 'user' AND user_id IS NOT NULL AND team_id IS NULL) OR
    (collection_type = 'global' AND user_id IS NULL AND team_id IS NULL) OR
    (collection_type = 'team' AND team_id IS NOT NULL AND user_id IS NULL)
);

-- Create root folders for existing teams
DO $$
DECLARE
    team_record RECORD;
    new_folder_id VARCHAR(255);
BEGIN
    FOR team_record IN SELECT team_id, team_name FROM teams LOOP
        -- Generate UUID for folder
        new_folder_id := gen_random_uuid()::text;
        
        -- Create team root folder if it doesn't exist
        INSERT INTO document_folders (
            folder_id, 
            team_id, 
            name, 
            collection_type, 
            parent_folder_id,
            description
        )
        VALUES (
            new_folder_id,
            team_record.team_id,
            team_record.team_name || ' Documents',
            'team',
            NULL,
            'Team documents for ' || team_record.team_name
        )
        ON CONFLICT DO NOTHING;
        
        RAISE NOTICE 'Created folder % for team %', new_folder_id, team_record.team_name;
    END LOOP;
END $$;

