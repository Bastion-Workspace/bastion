-- Goal hierarchy for agent teams (Phase 3)

CREATE TABLE IF NOT EXISTS agent_team_goals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id UUID NOT NULL REFERENCES agent_teams(id) ON DELETE CASCADE,
    parent_goal_id UUID REFERENCES agent_team_goals(id) ON DELETE CASCADE,
    title VARCHAR(500) NOT NULL,
    description TEXT,
    status VARCHAR(50) NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'completed', 'blocked', 'cancelled')),
    assigned_agent_id UUID REFERENCES agent_profiles(id) ON DELETE SET NULL,
    priority INTEGER DEFAULT 0,
    progress_pct INTEGER DEFAULT 0 CHECK (progress_pct >= 0 AND progress_pct <= 100),
    due_date DATE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_team_goals_team ON agent_team_goals(team_id);
CREATE INDEX IF NOT EXISTS idx_agent_team_goals_parent ON agent_team_goals(team_id, parent_goal_id);
CREATE INDEX IF NOT EXISTS idx_agent_team_goals_assigned ON agent_team_goals(assigned_agent_id);
