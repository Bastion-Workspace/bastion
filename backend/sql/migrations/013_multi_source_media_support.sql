-- Migration 013: Multi-Source Media Support
-- Enables multiple media sources per user (SubSonic + Audiobookshelf, etc.)

-- Add service_name column for user-friendly display names
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'music_service_configs' 
        AND column_name = 'service_name'
    ) THEN
        ALTER TABLE music_service_configs 
        ADD COLUMN service_name VARCHAR(255);
        
        COMMENT ON COLUMN music_service_configs.service_name IS 'User-friendly display name for the media source';
    END IF;
END $$;

-- Add is_active column to allow disabling without deleting
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'music_service_configs' 
        AND column_name = 'is_active'
    ) THEN
        ALTER TABLE music_service_configs 
        ADD COLUMN is_active BOOLEAN DEFAULT TRUE;
        
        COMMENT ON COLUMN music_service_configs.is_active IS 'Whether this media source is active (can be disabled without deleting)';
    END IF;
END $$;

-- Ensure service_type exists and has default value
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'music_service_configs' 
        AND column_name = 'service_type'
    ) THEN
        ALTER TABLE music_service_configs 
        ADD COLUMN service_type VARCHAR(50) DEFAULT 'subsonic';
        
        UPDATE music_service_configs 
        SET service_type = 'subsonic' 
        WHERE service_type IS NULL;
    END IF;
END $$;

-- Drop old unique constraint if it exists
DO $$ 
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'music_service_configs_user_id_key'
    ) THEN
        ALTER TABLE music_service_configs 
        DROP CONSTRAINT music_service_configs_user_id_key;
    END IF;
END $$;

-- Add new unique constraint for (user_id, service_type)
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'music_service_configs_user_service_unique'
    ) THEN
        ALTER TABLE music_service_configs 
        ADD CONSTRAINT music_service_configs_user_service_unique 
        UNIQUE (user_id, service_type);
    END IF;
END $$;

-- Add service_type column to music_cache if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'music_cache' 
        AND column_name = 'service_type'
    ) THEN
        ALTER TABLE music_cache 
        ADD COLUMN service_type VARCHAR(50) DEFAULT 'subsonic';
        
        -- Update existing rows to have 'subsonic' as service_type
        UPDATE music_cache 
        SET service_type = 'subsonic' 
        WHERE service_type IS NULL;
        
        COMMENT ON COLUMN music_cache.service_type IS 'Music service provider type: subsonic, audiobookshelf, etc.';
    END IF;
END $$;

-- Drop old unique constraint on music_cache if it exists
DO $$ 
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'music_cache_user_id_cache_type_item_id_key'
    ) THEN
        ALTER TABLE music_cache 
        DROP CONSTRAINT music_cache_user_id_cache_type_item_id_key;
    END IF;
END $$;

-- Add new unique constraint for (user_id, service_type, cache_type, item_id)
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'music_cache_user_service_type_item_unique'
    ) THEN
        ALTER TABLE music_cache 
        ADD CONSTRAINT music_cache_user_service_type_item_unique 
        UNIQUE (user_id, service_type, cache_type, item_id);
    END IF;
END $$;

-- Create index for service_type in music_cache for better query performance
CREATE INDEX IF NOT EXISTS idx_music_cache_service_type ON music_cache(service_type);
CREATE INDEX IF NOT EXISTS idx_music_cache_user_service_type ON music_cache(user_id, service_type);

