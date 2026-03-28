-- ========================================
-- Agent Factory: Plugin credentials per profile
-- ========================================
-- Idempotent: uses CREATE TABLE IF NOT EXISTS.
-- Run: docker exec -i bastion-postgres psql -U postgres -d bastion_knowledge_base < backend/sql/migrations/044_add_agent_plugin_configs.sql
-- ========================================

CREATE TABLE IF NOT EXISTS agent_plugin_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_profile_id UUID NOT NULL REFERENCES agent_profiles(id) ON DELETE CASCADE,
    plugin_name VARCHAR(100) NOT NULL,
    credentials_encrypted JSONB,
    is_enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (agent_profile_id, plugin_name)
);

CREATE INDEX IF NOT EXISTS idx_agent_plugin_configs_profile ON agent_plugin_configs(agent_profile_id);
CREATE INDEX IF NOT EXISTS idx_agent_plugin_configs_plugin ON agent_plugin_configs(plugin_name);
