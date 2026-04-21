-- Migration 014: Add service_type to music_cache_metadata
-- Makes cache metadata per-service instead of per-user

-- Add service_type column to music_cache_metadata if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'music_cache_metadata' 
        AND column_name = 'service_type'
    ) THEN
        ALTER TABLE music_cache_metadata 
        ADD COLUMN service_type VARCHAR(50) DEFAULT 'subsonic';
        
        -- Update existing rows to have 'subsonic' as service_type
        UPDATE music_cache_metadata 
        SET service_type = 'subsonic' 
        WHERE service_type IS NULL;
        
        COMMENT ON COLUMN music_cache_metadata.service_type IS 'Music service provider type: subsonic, audiobookshelf, etc.';
    END IF;
END $$;

-- Drop old unique constraint if it exists
DO $$ 
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'music_cache_metadata_user_id_key'
    ) THEN
        ALTER TABLE music_cache_metadata 
        DROP CONSTRAINT music_cache_metadata_user_id_key;
    END IF;
END $$;

-- Add new unique constraint for (user_id, service_type)
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'music_cache_metadata_user_service_unique'
    ) THEN
        ALTER TABLE music_cache_metadata 
        ADD CONSTRAINT music_cache_metadata_user_service_unique 
        UNIQUE (user_id, service_type);
    END IF;
END $$;

-- Create index for service_type in music_cache_metadata
CREATE INDEX IF NOT EXISTS idx_music_cache_metadata_service_type ON music_cache_metadata(service_type);
CREATE INDEX IF NOT EXISTS idx_music_cache_metadata_user_service ON music_cache_metadata(user_id, service_type);

