-- Code workspaces (local proxy-backed)

CREATE TABLE IF NOT EXISTS code_workspaces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    device_name TEXT,
    workspace_path TEXT NOT NULL,
    last_file_tree JSONB,
    last_git_branch TEXT,
    settings JSONB DEFAULT '{}'::jsonb,
    conversation_id UUID,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_code_workspaces_user_id ON code_workspaces(user_id);
