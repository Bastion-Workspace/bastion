-- Agent Artifact Sharing: user-to-user "use" shares for agent profiles, playbooks, and skills
-- with transitive cascading (sharing a playbook auto-shares its skills).

CREATE TABLE IF NOT EXISTS agent_artifact_shares (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    artifact_type VARCHAR(50) NOT NULL
        CHECK (artifact_type IN ('agent_profile', 'playbook', 'skill')),
    artifact_id UUID NOT NULL,
    owner_user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    shared_with_user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    is_transitive BOOLEAN DEFAULT false,
    parent_share_id UUID REFERENCES agent_artifact_shares(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (artifact_type, artifact_id, shared_with_user_id)
);

CREATE INDEX IF NOT EXISTS idx_artifact_shares_recipient ON agent_artifact_shares(shared_with_user_id);
CREATE INDEX IF NOT EXISTS idx_artifact_shares_owner ON agent_artifact_shares(owner_user_id);
CREATE INDEX IF NOT EXISTS idx_artifact_shares_artifact ON agent_artifact_shares(artifact_type, artifact_id);
CREATE INDEX IF NOT EXISTS idx_artifact_shares_parent ON agent_artifact_shares(parent_share_id);
