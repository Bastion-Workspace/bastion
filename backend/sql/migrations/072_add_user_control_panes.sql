-- User control panes (status bar custom controls wired to data connectors)

CREATE TABLE IF NOT EXISTS user_control_panes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    icon VARCHAR(100) NOT NULL DEFAULT 'Tune',
    connector_id UUID NOT NULL REFERENCES data_source_connectors(id) ON DELETE CASCADE,
    credentials_encrypted JSONB DEFAULT '{}',
    connection_id BIGINT,
    controls JSONB NOT NULL DEFAULT '[]',
    is_visible BOOLEAN DEFAULT true,
    sort_order INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_control_panes_user ON user_control_panes(user_id);
CREATE INDEX IF NOT EXISTS idx_control_panes_visible ON user_control_panes(user_id, is_visible);
