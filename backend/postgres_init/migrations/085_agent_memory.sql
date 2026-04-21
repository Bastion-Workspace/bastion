-- Persistent agent memory (Phase 1 autonomous operations)

ALTER TABLE agent_profiles ADD COLUMN IF NOT EXISTS include_agent_memory BOOLEAN NOT NULL DEFAULT false;

CREATE TABLE IF NOT EXISTS agent_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_profile_id UUID NOT NULL REFERENCES agent_profiles(id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    memory_key VARCHAR(500) NOT NULL,
    memory_value JSONB NOT NULL,
    memory_type VARCHAR(50) DEFAULT 'kv',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    UNIQUE(agent_profile_id, memory_key)
);

CREATE INDEX IF NOT EXISTS idx_agent_memory_profile ON agent_memory(agent_profile_id);
CREATE INDEX IF NOT EXISTS idx_agent_memory_user ON agent_memory(user_id);
CREATE INDEX IF NOT EXISTS idx_agent_memory_key_prefix ON agent_memory(agent_profile_id, memory_key);
CREATE INDEX IF NOT EXISTS idx_agent_memory_expires ON agent_memory(expires_at) WHERE expires_at IS NOT NULL;
