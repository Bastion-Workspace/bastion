-- User-defined sidebar categories (folders) per section for Agent Factory.
-- section: agents | playbooks | skills | connectors
CREATE TABLE IF NOT EXISTS agent_factory_sidebar_categories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    section VARCHAR(50) NOT NULL,
    name VARCHAR(255) NOT NULL,
    sort_order INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (user_id, section, name)
);
CREATE INDEX IF NOT EXISTS idx_sidebar_categories_user_section ON agent_factory_sidebar_categories(user_id, section);
COMMENT ON TABLE agent_factory_sidebar_categories IS 'User-created category names and order for Agent Factory sidebar sections.';
