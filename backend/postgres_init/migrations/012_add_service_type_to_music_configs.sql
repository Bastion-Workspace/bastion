-- Migration: Add service_type to music_service_configs
-- This allows support for multiple music service providers (SubSonic, Plex, Emby, etc.)

-- Add service_type column if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'music_service_configs' 
        AND column_name = 'service_type'
    ) THEN
        ALTER TABLE music_service_configs 
        ADD COLUMN service_type VARCHAR(50) DEFAULT 'subsonic';
        
        -- Update existing rows to have 'subsonic' as service_type
        UPDATE music_service_configs 
        SET service_type = 'subsonic' 
        WHERE service_type IS NULL;
        
        -- Add comment
        COMMENT ON COLUMN music_service_configs.service_type IS 'Music service provider type: subsonic, plex, emby, jellyfin, etc.';
    END IF;
END $$;

