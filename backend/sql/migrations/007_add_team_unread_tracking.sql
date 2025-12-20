-- ========================================
-- TEAM UNREAD POST TRACKING MIGRATION
-- Adds unread post tracking and muting support
-- ========================================
-- This migration can be run on existing databases without data loss
-- Idempotent design - safe to run multiple times

-- Add last_read_at and muted fields to team_members
DO $$ 
BEGIN
    -- Add last_read_at column if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'team_members' AND column_name = 'last_read_at'
    ) THEN
        ALTER TABLE team_members 
        ADD COLUMN last_read_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP;
    END IF;

    -- Add muted column if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'team_members' AND column_name = 'muted'
    ) THEN
        ALTER TABLE team_members 
        ADD COLUMN muted BOOLEAN DEFAULT FALSE;
    END IF;
END $$;

-- Create index for unread queries
CREATE INDEX IF NOT EXISTS idx_team_members_last_read ON team_members(team_id, user_id, last_read_at);

-- Update existing members to have last_read_at set to joined_at
UPDATE team_members 
SET last_read_at = joined_at 
WHERE last_read_at IS NULL;

COMMENT ON COLUMN team_members.last_read_at IS 'Timestamp when user last read team posts';
COMMENT ON COLUMN team_members.muted IS 'Whether user has muted notifications for this team';







