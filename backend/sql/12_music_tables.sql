-- Music Service Tables
-- Stores SubSonic-compatible music service configurations and cached metadata

-- Create music_service_configs table for storing encrypted music service credentials
CREATE TABLE IF NOT EXISTS music_service_configs (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    server_url VARCHAR(500) NOT NULL,
    username VARCHAR(255) NOT NULL,
    encrypted_password TEXT NOT NULL,
    salt VARCHAR(255) NOT NULL,
    auth_type VARCHAR(50) DEFAULT 'password', -- 'password' or 'token'
    service_type VARCHAR(50) DEFAULT 'subsonic', -- 'subsonic', 'plex', 'emby', 'jellyfin', etc.
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id)
);

-- Create indexes for music_service_configs
CREATE INDEX IF NOT EXISTS idx_music_configs_user_id ON music_service_configs(user_id);

-- Create music_cache table for storing cached library metadata
CREATE TABLE IF NOT EXISTS music_cache (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    cache_type VARCHAR(50) NOT NULL, -- 'album', 'artist', 'playlist', 'track'
    item_id VARCHAR(255) NOT NULL, -- SubSonic item ID
    parent_id VARCHAR(255), -- For tracks: album/playlist ID
    title VARCHAR(500) NOT NULL,
    artist VARCHAR(500),
    album VARCHAR(500),
    duration INTEGER, -- Duration in seconds
    track_number INTEGER,
    cover_art_id VARCHAR(255),
    metadata_json JSONB, -- Additional metadata from SubSonic
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, cache_type, item_id)
);

-- Create indexes for music_cache
CREATE INDEX IF NOT EXISTS idx_music_cache_user_id ON music_cache(user_id);
CREATE INDEX IF NOT EXISTS idx_music_cache_type ON music_cache(cache_type);
CREATE INDEX IF NOT EXISTS idx_music_cache_item_id ON music_cache(item_id);
CREATE INDEX IF NOT EXISTS idx_music_cache_parent_id ON music_cache(parent_id);
CREATE INDEX IF NOT EXISTS idx_music_cache_user_type ON music_cache(user_id, cache_type);
CREATE INDEX IF NOT EXISTS idx_music_cache_metadata_json ON music_cache USING GIN(metadata_json);

-- Create music_cache_metadata table for tracking sync status
CREATE TABLE IF NOT EXISTS music_cache_metadata (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL UNIQUE,
    last_sync_at TIMESTAMP WITH TIME ZONE,
    sync_status VARCHAR(50) DEFAULT 'pending', -- 'pending', 'syncing', 'completed', 'failed'
    sync_error TEXT,
    total_albums INTEGER DEFAULT 0,
    total_artists INTEGER DEFAULT 0,
    total_playlists INTEGER DEFAULT 0,
    total_tracks INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create index for music_cache_metadata
CREATE INDEX IF NOT EXISTS idx_music_cache_metadata_user_id ON music_cache_metadata(user_id);

-- Add comments
COMMENT ON TABLE music_service_configs IS 'Encrypted music service configurations per user (SubSonic, Plex, Emby, etc.)';
COMMENT ON COLUMN music_service_configs.service_type IS 'Music service provider type: subsonic, plex, emby, jellyfin, etc.';
COMMENT ON TABLE music_cache IS 'Cached music library metadata from music servers';
COMMENT ON TABLE music_cache_metadata IS 'Metadata about music cache sync status and timestamps';

