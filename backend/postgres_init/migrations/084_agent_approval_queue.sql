-- Background approval queue for scheduled/background agent runs (Phase 1 autonomous operations)

CREATE TABLE IF NOT EXISTS agent_approval_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    agent_profile_id UUID REFERENCES agent_profiles(id) ON DELETE SET NULL,
    execution_id UUID REFERENCES agent_execution_log(id) ON DELETE CASCADE,
    step_name VARCHAR(255) NOT NULL,
    prompt TEXT NOT NULL,
    preview_data JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'pending',
    thread_id VARCHAR(500),
    checkpoint_ns VARCHAR(255),
    playbook_config JSONB,
    timeout_at TIMESTAMPTZ,
    responded_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_approval_queue_user ON agent_approval_queue(user_id);
CREATE INDEX IF NOT EXISTS idx_agent_approval_queue_status ON agent_approval_queue(status) WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS idx_agent_approval_queue_created ON agent_approval_queue(created_at DESC);
