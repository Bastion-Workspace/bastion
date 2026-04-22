-- MCP servers for user-configured external tools (Agent Factory).
-- Safe to run on existing databases; mirrors 01_init.sql definition.
-- Renamed from 080_add_mcp_servers.sql (080 is reserved for Groq provider_type in run_migration / init 08).

CREATE TABLE IF NOT EXISTS mcp_servers (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    transport VARCHAR(32) NOT NULL,
    url TEXT,
    command TEXT,
    args JSONB DEFAULT '[]'::jsonb,
    env JSONB DEFAULT '{}'::jsonb,
    headers JSONB DEFAULT '{}'::jsonb,
    is_active BOOLEAN DEFAULT true,
    discovered_tools JSONB DEFAULT '[]'::jsonb,
    last_discovery_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (user_id, name)
);
CREATE INDEX IF NOT EXISTS idx_mcp_servers_user_id ON mcp_servers(user_id);
CREATE INDEX IF NOT EXISTS idx_mcp_servers_active ON mcp_servers(user_id, is_active) WHERE is_active = true;
