-- ========================================
-- ADD USER LOCATIONS TABLE
-- Create table for storing user location points (private and global)
-- ========================================
-- This migration creates the user_locations table for the map feature.
-- Users can create private locations (Home, Work, etc.) and admins can
-- create global locations visible to all users with map access.
--
-- Usage:
-- docker exec -i <postgres-container> psql -U postgres -d bastion_knowledge_base < backend/sql/migrations/030_add_user_locations.sql
-- Or from within container:
-- psql -U postgres -d bastion_knowledge_base -f /docker-entrypoint-initdb.d/migrations/030_add_user_locations.sql
-- ========================================

CREATE TABLE IF NOT EXISTS user_locations (
    location_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    address TEXT,
    latitude DECIMAL(10, 8) NOT NULL,
    longitude DECIMAL(11, 8) NOT NULL,
    notes TEXT,
    is_global BOOLEAN DEFAULT FALSE,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_locations_user_id ON user_locations(user_id);
CREATE INDEX IF NOT EXISTS idx_user_locations_is_global ON user_locations(is_global);
CREATE INDEX IF NOT EXISTS idx_user_locations_name ON user_locations(name);

GRANT ALL PRIVILEGES ON user_locations TO bastion_user;

-- Enable RLS
ALTER TABLE user_locations ENABLE ROW LEVEL SECURITY;

-- Users see their own private locations + all global locations
CREATE POLICY locations_select_policy ON user_locations
    FOR SELECT USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR is_global = true
        OR current_setting('app.current_user_role', true) = 'admin'
    );

-- Users can only insert their own private locations, admins can create global
CREATE POLICY locations_insert_policy ON user_locations
    FOR INSERT WITH CHECK (
        (user_id = current_setting('app.current_user_id', true)::varchar AND is_global = false)
        OR current_setting('app.current_user_role', true) = 'admin'
    );

-- Users can only update/delete their own locations, admins can modify global
CREATE POLICY locations_update_policy ON user_locations
    FOR UPDATE USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE POLICY locations_delete_policy ON user_locations
    FOR DELETE USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

-- Trigger for updated_at
CREATE TRIGGER update_user_locations_updated_at
    BEFORE UPDATE ON user_locations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

COMMENT ON TABLE user_locations IS 'Stores user location points for map visualization. Supports both private (per-user) and global (shared) locations.';
COMMENT ON COLUMN user_locations.is_global IS 'If true, location is visible to all users with map access. Only admins can create global locations.';
