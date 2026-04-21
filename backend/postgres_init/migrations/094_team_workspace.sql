-- Team workspace: shared key-value scratchpad for agent teams (Blackboard pattern).
-- Agents can write_to_workspace(key, value) and read_workspace(key) to share artifacts.

CREATE TABLE IF NOT EXISTS team_workspace (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id UUID NOT NULL REFERENCES agent_teams(id) ON DELETE CASCADE,
    key VARCHAR(255) NOT NULL,
    value TEXT NOT NULL,
    updated_by_agent_id UUID REFERENCES agent_profiles(id) ON DELETE SET NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(team_id, key)
);

CREATE INDEX IF NOT EXISTS idx_team_workspace_team_id ON team_workspace(team_id);
