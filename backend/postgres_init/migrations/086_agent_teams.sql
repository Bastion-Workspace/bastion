-- Autonomous Agent Teams: team container and org chart memberships

CREATE TABLE IF NOT EXISTS agent_teams (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    mission_statement TEXT,
    status VARCHAR(50) NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'paused', 'archived')),
    heartbeat_config JSONB DEFAULT '{}',
    governance_policy JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_agent_teams_user ON agent_teams(user_id);
CREATE INDEX IF NOT EXISTS idx_agent_teams_status ON agent_teams(status);

CREATE TABLE IF NOT EXISTS agent_team_memberships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id UUID NOT NULL REFERENCES agent_teams(id) ON DELETE CASCADE,
    agent_profile_id UUID NOT NULL REFERENCES agent_profiles(id) ON DELETE CASCADE,
    role VARCHAR(100) NOT NULL DEFAULT 'worker',
    reports_to UUID REFERENCES agent_team_memberships(id) ON DELETE SET NULL,
    hire_approved BOOLEAN DEFAULT false,
    hire_approved_at TIMESTAMPTZ,
    joined_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(team_id, agent_profile_id)
);
CREATE INDEX IF NOT EXISTS idx_agent_team_memberships_team ON agent_team_memberships(team_id);
CREATE INDEX IF NOT EXISTS idx_agent_team_memberships_profile ON agent_team_memberships(agent_profile_id);
CREATE INDEX IF NOT EXISTS idx_agent_team_memberships_reports_to ON agent_team_memberships(reports_to);
