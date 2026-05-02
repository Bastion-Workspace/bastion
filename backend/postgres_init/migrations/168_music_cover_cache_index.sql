-- Per-user music cover art disk cache index (RLS matches music_cache).

CREATE TABLE IF NOT EXISTS music_cover_cache_index (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    service_type VARCHAR(50) NOT NULL,
    cover_art_id TEXT NOT NULL,
    size INTEGER NOT NULL,
    bytes INTEGER NOT NULL,
    etag VARCHAR(64) NOT NULL,
    content_type VARCHAR(100) NOT NULL,
    path TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    accessed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (user_id, service_type, cover_art_id, size)
);

CREATE INDEX IF NOT EXISTS idx_music_cover_cache_user_service ON music_cover_cache_index(user_id, service_type);
CREATE INDEX IF NOT EXISTS idx_music_cover_cache_user_lru ON music_cover_cache_index(user_id, accessed_at);

COMMENT ON TABLE music_cover_cache_index IS 'Index for per-user proxied music cover art blobs on disk';

ALTER TABLE music_cover_cache_index ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS music_cover_cache_index_select_policy ON music_cover_cache_index;
CREATE POLICY music_cover_cache_index_select_policy ON music_cover_cache_index
FOR SELECT
USING (user_id = current_setting('app.current_user_id', true));

DROP POLICY IF EXISTS music_cover_cache_index_insert_policy ON music_cover_cache_index;
CREATE POLICY music_cover_cache_index_insert_policy ON music_cover_cache_index
FOR INSERT
WITH CHECK (user_id = current_setting('app.current_user_id', true));

DROP POLICY IF EXISTS music_cover_cache_index_update_policy ON music_cover_cache_index;
CREATE POLICY music_cover_cache_index_update_policy ON music_cover_cache_index
FOR UPDATE
USING (user_id = current_setting('app.current_user_id', true))
WITH CHECK (user_id = current_setting('app.current_user_id', true));

DROP POLICY IF EXISTS music_cover_cache_index_delete_policy ON music_cover_cache_index;
CREATE POLICY music_cover_cache_index_delete_policy ON music_cover_cache_index
FOR DELETE
USING (user_id = current_setting('app.current_user_id', true));

GRANT ALL PRIVILEGES ON music_cover_cache_index TO bastion_user;
GRANT ALL PRIVILEGES ON SEQUENCE music_cover_cache_index_id_seq TO bastion_user;
