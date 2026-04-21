-- ========================================
-- Agent team watches: which agents watch which teams
-- ========================================
-- Run: docker exec -i bastion-postgres psql -U postgres -d bastion_knowledge_base < backend/postgres_init/migrations/047_add_agent_team_watches.sql
-- ========================================

CREATE TABLE IF NOT EXISTS agent_team_watches (
    agent_profile_id UUID NOT NULL REFERENCES agent_profiles(id) ON DELETE CASCADE,
    team_id UUID NOT NULL REFERENCES teams(team_id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    trigger_on_new_post BOOLEAN DEFAULT true,
    respond_as VARCHAR(20) DEFAULT 'comment',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (agent_profile_id, team_id)
);

CREATE INDEX IF NOT EXISTS idx_agent_team_watches_team ON agent_team_watches(team_id) WHERE trigger_on_new_post = true;
