-- Agent Factory: step-level execution traces and playbook version history

-- Per-step execution trace (referenced by agent_execution_log)
CREATE TABLE IF NOT EXISTS agent_execution_steps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    execution_id UUID NOT NULL REFERENCES agent_execution_log(id) ON DELETE CASCADE,
    step_index INTEGER NOT NULL,
    step_name VARCHAR(255) NOT NULL,
    step_type VARCHAR(50) NOT NULL,
    action_name VARCHAR(255),
    status VARCHAR(50) NOT NULL,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    duration_ms INTEGER,
    inputs_json JSONB DEFAULT '{}',
    outputs_json JSONB DEFAULT '{}',
    error_details TEXT
);
CREATE INDEX IF NOT EXISTS idx_execution_steps_execution_id ON agent_execution_steps(execution_id);

-- Playbook version history (snapshots on update)
CREATE TABLE IF NOT EXISTS playbook_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    playbook_id UUID NOT NULL REFERENCES custom_playbooks(id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL,
    label VARCHAR(255),
    definition JSONB NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by VARCHAR(255)
);
CREATE INDEX IF NOT EXISTS idx_playbook_versions_playbook_id ON playbook_versions(playbook_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_playbook_versions_unique ON playbook_versions(playbook_id, version_number);
