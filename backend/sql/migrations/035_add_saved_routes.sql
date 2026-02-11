-- ========================================
-- ADD SAVED ROUTES TABLE
-- Stores user-saved road routes (OSRM geometry, steps, waypoints)
-- ========================================

CREATE TABLE IF NOT EXISTS saved_routes (
    route_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    waypoints JSONB NOT NULL DEFAULT '[]',
    geometry JSONB NOT NULL,
    steps JSONB NOT NULL DEFAULT '[]',
    distance_meters NUMERIC(12, 2) NOT NULL,
    duration_seconds NUMERIC(12, 2) NOT NULL,
    profile VARCHAR(64) DEFAULT 'driving',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_saved_routes_user_id ON saved_routes(user_id);
CREATE INDEX IF NOT EXISTS idx_saved_routes_created_at ON saved_routes(created_at DESC);

GRANT ALL PRIVILEGES ON saved_routes TO bastion_user;

ALTER TABLE saved_routes ENABLE ROW LEVEL SECURITY;

CREATE POLICY saved_routes_select_policy ON saved_routes
    FOR SELECT USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE POLICY saved_routes_insert_policy ON saved_routes
    FOR INSERT WITH CHECK (
        user_id = current_setting('app.current_user_id', true)::varchar
    );

CREATE POLICY saved_routes_update_policy ON saved_routes
    FOR UPDATE USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE POLICY saved_routes_delete_policy ON saved_routes
    FOR DELETE USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE TRIGGER update_saved_routes_updated_at
    BEFORE UPDATE ON saved_routes
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

COMMENT ON TABLE saved_routes IS 'Stores user-saved road routes from OSRM (geometry, turn-by-turn steps, waypoints).';
