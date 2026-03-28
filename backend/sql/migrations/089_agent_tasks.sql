-- Agent task/ticket system for teams (Phase 4)

CREATE TABLE IF NOT EXISTS agent_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id UUID NOT NULL REFERENCES agent_teams(id) ON DELETE CASCADE,
    title VARCHAR(500) NOT NULL,
    description TEXT,
    status VARCHAR(50) NOT NULL DEFAULT 'backlog'
        CHECK (status IN ('backlog', 'assigned', 'in_progress', 'review', 'done', 'cancelled')),
    assigned_agent_id UUID REFERENCES agent_profiles(id) ON DELETE SET NULL,
    created_by_agent_id UUID REFERENCES agent_profiles(id) ON DELETE SET NULL,
    goal_id UUID REFERENCES agent_team_goals(id) ON DELETE SET NULL,
    priority INTEGER DEFAULT 0,
    thread_id UUID REFERENCES agent_messages(id) ON DELETE SET NULL,
    execution_id UUID REFERENCES agent_execution_log(id) ON DELETE SET NULL,
    metadata JSONB DEFAULT '{}',
    due_date DATE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_tasks_team_status ON agent_tasks(team_id, status);
CREATE INDEX IF NOT EXISTS idx_agent_tasks_assigned_status ON agent_tasks(assigned_agent_id, status);
