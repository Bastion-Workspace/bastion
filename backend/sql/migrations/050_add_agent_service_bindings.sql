-- ========================================
-- AGENT SERVICE BINDINGS
-- Links agent profiles to external connections (email, messaging, etc.)
-- so playbook steps can use account-scoped tools (email:<connection_id>:send_email).
-- ========================================
-- Idempotent: uses CREATE TABLE IF NOT EXISTS and CREATE INDEX IF NOT EXISTS.
-- Run on existing DB: docker exec -i bastion-postgres psql -U postgres -d bastion_knowledge_base < backend/sql/migrations/050_add_agent_service_bindings.sql
-- ========================================

CREATE TABLE IF NOT EXISTS agent_service_bindings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_profile_id UUID NOT NULL REFERENCES agent_profiles(id) ON DELETE CASCADE,
    connection_id BIGINT NOT NULL REFERENCES external_connections(id) ON DELETE CASCADE,
    service_type VARCHAR(50) NOT NULL,
    is_enabled BOOLEAN DEFAULT true,
    config JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(agent_profile_id, connection_id)
);

CREATE INDEX IF NOT EXISTS idx_agent_service_bindings_profile ON agent_service_bindings(agent_profile_id);
CREATE INDEX IF NOT EXISTS idx_agent_service_bindings_connection ON agent_service_bindings(connection_id);
CREATE INDEX IF NOT EXISTS idx_agent_service_bindings_service_type ON agent_service_bindings(agent_profile_id, service_type);
