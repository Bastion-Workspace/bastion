-- Zettelkasten user settings (JSONB per user, mirrors org_settings pattern)
CREATE TABLE IF NOT EXISTS zettelkasten_settings (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL UNIQUE,
    settings_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_zettelkasten_settings_user_id ON zettelkasten_settings(user_id);
CREATE INDEX IF NOT EXISTS idx_zettelkasten_settings_json ON zettelkasten_settings USING GIN(settings_json);

COMMENT ON TABLE zettelkasten_settings IS 'Zettelkasten / PKM preferences per user';

ALTER TABLE zettelkasten_settings ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS zettelkasten_settings_select_policy ON zettelkasten_settings;
DROP POLICY IF EXISTS zettelkasten_settings_insert_policy ON zettelkasten_settings;
DROP POLICY IF EXISTS zettelkasten_settings_update_policy ON zettelkasten_settings;
DROP POLICY IF EXISTS zettelkasten_settings_delete_policy ON zettelkasten_settings;

CREATE POLICY zettelkasten_settings_select_policy ON zettelkasten_settings
    FOR SELECT USING (user_id = current_setting('app.current_user_id', true)::varchar);

CREATE POLICY zettelkasten_settings_insert_policy ON zettelkasten_settings
    FOR INSERT WITH CHECK (user_id = current_setting('app.current_user_id', true)::varchar);

CREATE POLICY zettelkasten_settings_update_policy ON zettelkasten_settings
    FOR UPDATE USING (user_id = current_setting('app.current_user_id', true)::varchar)
    WITH CHECK (user_id = current_setting('app.current_user_id', true)::varchar);

CREATE POLICY zettelkasten_settings_delete_policy ON zettelkasten_settings
    FOR DELETE USING (user_id = current_setting('app.current_user_id', true)::varchar);
