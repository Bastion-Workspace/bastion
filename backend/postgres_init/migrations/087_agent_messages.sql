-- Inter-agent messages and timeline (Phase 2 autonomous teams)

CREATE TABLE IF NOT EXISTS agent_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id UUID NOT NULL REFERENCES agent_teams(id) ON DELETE CASCADE,
    from_agent_id UUID REFERENCES agent_profiles(id) ON DELETE SET NULL,
    to_agent_id UUID REFERENCES agent_profiles(id) ON DELETE SET NULL,
    message_type VARCHAR(50) NOT NULL DEFAULT 'report',
    content TEXT NOT NULL DEFAULT '',
    metadata JSONB DEFAULT '{}',
    parent_message_id UUID REFERENCES agent_messages(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_messages_team_created ON agent_messages(team_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_messages_from_agent ON agent_messages(from_agent_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_messages_to_agent ON agent_messages(to_agent_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_messages_parent ON agent_messages(parent_message_id) WHERE parent_message_id IS NOT NULL;
