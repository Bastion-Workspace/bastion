-- ========================================
-- EXTERNAL CONNECTIONS (OAuth / Email / Chat bots)
-- ========================================
-- Adds external_connections, connection_sync_state, connection_data_cache,
-- and system_settings if missing (e.g. DB created before these were in 01_init).
-- Idempotent: safe to run multiple times.
--
-- Run from host (postgres container must be up):
--   docker exec -i bastion-postgres psql -U postgres -d bastion_knowledge_base < backend/sql/migrations/040_add_external_connections.sql
-- Or from backend container with access to sql dir:
--   psql -U postgres -h postgres -d bastion_knowledge_base -f /path/to/040_add_external_connections.sql
-- ========================================

-- external_connections: OAuth tokens and connection metadata (Microsoft, Telegram, etc.)
CREATE TABLE IF NOT EXISTS external_connections (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    provider VARCHAR(50) NOT NULL,
    connection_type VARCHAR(50) NOT NULL,
    account_identifier VARCHAR(255) NOT NULL,
    display_name VARCHAR(255),
    encrypted_access_token TEXT NOT NULL,
    encrypted_refresh_token TEXT NOT NULL,
    token_expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    scopes TEXT[] NOT NULL,
    provider_metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_sync_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    connection_status VARCHAR(50) DEFAULT 'active',
    UNIQUE(user_id, provider, connection_type, account_identifier)
);

CREATE INDEX IF NOT EXISTS idx_external_connections_user ON external_connections(user_id);
CREATE INDEX IF NOT EXISTS idx_external_connections_active ON external_connections(user_id, is_active);
CREATE INDEX IF NOT EXISTS idx_external_connections_type ON external_connections(provider, connection_type);

-- connection_sync_state: per-connection sync cursors (e.g. email delta)
CREATE TABLE IF NOT EXISTS connection_sync_state (
    id BIGSERIAL PRIMARY KEY,
    connection_id BIGINT NOT NULL REFERENCES external_connections(id) ON DELETE CASCADE,
    resource_type VARCHAR(100) NOT NULL,
    resource_id VARCHAR(255) NOT NULL,
    resource_name VARCHAR(255),
    last_sync_timestamp TIMESTAMP WITH TIME ZONE,
    delta_token TEXT,
    sync_status VARCHAR(50) DEFAULT 'pending',
    last_error TEXT,
    metadata JSONB,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(connection_id, resource_type, resource_id)
);

CREATE INDEX IF NOT EXISTS idx_sync_state_connection ON connection_sync_state(connection_id);

-- connection_data_cache: cached API data (e.g. email folders)
CREATE TABLE IF NOT EXISTS connection_data_cache (
    id BIGSERIAL PRIMARY KEY,
    connection_id BIGINT NOT NULL REFERENCES external_connections(id) ON DELETE CASCADE,
    data_type VARCHAR(100) NOT NULL,
    external_id VARCHAR(500) NOT NULL,
    data JSONB NOT NULL,
    metadata JSONB,
    cached_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    UNIQUE(connection_id, data_type, external_id)
);

CREATE INDEX IF NOT EXISTS idx_data_cache_connection ON connection_data_cache(connection_id);
CREATE INDEX IF NOT EXISTS idx_data_cache_type ON connection_data_cache(connection_id, data_type);
CREATE INDEX IF NOT EXISTS idx_data_cache_metadata ON connection_data_cache USING gin(metadata);

-- system_settings: admin key-value (e.g. system_email_connection_id)
CREATE TABLE IF NOT EXISTS system_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL DEFAULT ''
);

GRANT ALL PRIVILEGES ON external_connections TO bastion_user;
GRANT ALL PRIVILEGES ON connection_sync_state TO bastion_user;
GRANT ALL PRIVILEGES ON connection_data_cache TO bastion_user;
GRANT ALL PRIVILEGES ON system_settings TO bastion_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO bastion_user;

ALTER TABLE external_connections ENABLE ROW LEVEL SECURITY;
ALTER TABLE connection_sync_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE connection_data_cache ENABLE ROW LEVEL SECURITY;
ALTER TABLE system_settings ENABLE ROW LEVEL SECURITY;

-- Policies: drop then create so migration is idempotent
DROP POLICY IF EXISTS external_connections_select_policy ON external_connections;
DROP POLICY IF EXISTS external_connections_insert_policy ON external_connections;
DROP POLICY IF EXISTS external_connections_update_policy ON external_connections;
DROP POLICY IF EXISTS external_connections_delete_policy ON external_connections;

CREATE POLICY external_connections_select_policy ON external_connections
    FOR SELECT USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE POLICY external_connections_insert_policy ON external_connections
    FOR INSERT WITH CHECK (user_id = current_setting('app.current_user_id', true)::varchar);

CREATE POLICY external_connections_update_policy ON external_connections
    FOR UPDATE USING (user_id = current_setting('app.current_user_id', true)::varchar OR current_setting('app.current_user_role', true) = 'admin');

CREATE POLICY external_connections_delete_policy ON external_connections
    FOR DELETE USING (user_id = current_setting('app.current_user_id', true)::varchar OR current_setting('app.current_user_role', true) = 'admin');

DROP POLICY IF EXISTS connection_sync_state_select_policy ON connection_sync_state;
DROP POLICY IF EXISTS connection_sync_state_insert_policy ON connection_sync_state;
DROP POLICY IF EXISTS connection_sync_state_update_policy ON connection_sync_state;
DROP POLICY IF EXISTS connection_sync_state_delete_policy ON connection_sync_state;

CREATE POLICY connection_sync_state_select_policy ON connection_sync_state
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM external_connections ec
            WHERE ec.id = connection_sync_state.connection_id
            AND (ec.user_id = current_setting('app.current_user_id', true)::varchar OR current_setting('app.current_user_role', true) = 'admin')
        )
    );

CREATE POLICY connection_sync_state_insert_policy ON connection_sync_state
    FOR INSERT WITH CHECK (
        EXISTS (
            SELECT 1 FROM external_connections ec
            WHERE ec.id = connection_sync_state.connection_id
            AND ec.user_id = current_setting('app.current_user_id', true)::varchar
        )
    );

CREATE POLICY connection_sync_state_update_policy ON connection_sync_state
    FOR UPDATE USING (
        EXISTS (
            SELECT 1 FROM external_connections ec
            WHERE ec.id = connection_sync_state.connection_id
            AND (ec.user_id = current_setting('app.current_user_id', true)::varchar OR current_setting('app.current_user_role', true) = 'admin')
        )
    );

CREATE POLICY connection_sync_state_delete_policy ON connection_sync_state
    FOR DELETE USING (
        EXISTS (
            SELECT 1 FROM external_connections ec
            WHERE ec.id = connection_sync_state.connection_id
            AND (ec.user_id = current_setting('app.current_user_id', true)::varchar OR current_setting('app.current_user_role', true) = 'admin')
        )
    );

DROP POLICY IF EXISTS connection_data_cache_select_policy ON connection_data_cache;
DROP POLICY IF EXISTS connection_data_cache_insert_policy ON connection_data_cache;
DROP POLICY IF EXISTS connection_data_cache_update_policy ON connection_data_cache;
DROP POLICY IF EXISTS connection_data_cache_delete_policy ON connection_data_cache;

CREATE POLICY connection_data_cache_select_policy ON connection_data_cache
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM external_connections ec
            WHERE ec.id = connection_data_cache.connection_id
            AND (ec.user_id = current_setting('app.current_user_id', true)::varchar OR current_setting('app.current_user_role', true) = 'admin')
        )
    );

CREATE POLICY connection_data_cache_insert_policy ON connection_data_cache
    FOR INSERT WITH CHECK (
        EXISTS (
            SELECT 1 FROM external_connections ec
            WHERE ec.id = connection_data_cache.connection_id
            AND ec.user_id = current_setting('app.current_user_id', true)::varchar
        )
    );

CREATE POLICY connection_data_cache_update_policy ON connection_data_cache
    FOR UPDATE USING (
        EXISTS (
            SELECT 1 FROM external_connections ec
            WHERE ec.id = connection_data_cache.connection_id
            AND (ec.user_id = current_setting('app.current_user_id', true)::varchar OR current_setting('app.current_user_role', true) = 'admin')
        )
    );

CREATE POLICY connection_data_cache_delete_policy ON connection_data_cache
    FOR DELETE USING (
        EXISTS (
            SELECT 1 FROM external_connections ec
            WHERE ec.id = connection_data_cache.connection_id
            AND (ec.user_id = current_setting('app.current_user_id', true)::varchar OR current_setting('app.current_user_role', true) = 'admin')
        )
    );

DROP POLICY IF EXISTS system_settings_select_policy ON system_settings;
DROP POLICY IF EXISTS system_settings_insert_policy ON system_settings;
DROP POLICY IF EXISTS system_settings_update_policy ON system_settings;
DROP POLICY IF EXISTS system_settings_delete_policy ON system_settings;

CREATE POLICY system_settings_select_policy ON system_settings
    FOR SELECT USING (current_setting('app.current_user_role', true) = 'admin');

CREATE POLICY system_settings_insert_policy ON system_settings
    FOR INSERT WITH CHECK (current_setting('app.current_user_role', true) = 'admin');

CREATE POLICY system_settings_update_policy ON system_settings
    FOR UPDATE USING (current_setting('app.current_user_role', true) = 'admin');

CREATE POLICY system_settings_delete_policy ON system_settings
    FOR DELETE USING (current_setting('app.current_user_role', true) = 'admin');
