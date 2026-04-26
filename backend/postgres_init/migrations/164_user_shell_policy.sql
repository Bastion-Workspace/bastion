-- Per-user shell command policy (allow / deny / require_approval), optional per code workspace scope.

CREATE TABLE IF NOT EXISTS user_shell_policy (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    pattern TEXT NOT NULL,
    match_mode VARCHAR(20) NOT NULL DEFAULT 'prefix',
    action VARCHAR(20) NOT NULL,
    scope_workspace_id UUID REFERENCES code_workspaces(id) ON DELETE CASCADE,
    label TEXT,
    priority INT NOT NULL DEFAULT 50,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_user_shell_policy_match_mode CHECK (match_mode IN ('prefix', 'contains', 'glob')),
    CONSTRAINT chk_user_shell_policy_action CHECK (action IN ('allow', 'deny', 'require_approval'))
);

CREATE INDEX IF NOT EXISTS idx_user_shell_policy_user ON user_shell_policy(user_id);
CREATE INDEX IF NOT EXISTS idx_user_shell_policy_user_priority ON user_shell_policy(user_id, priority);

ALTER TABLE user_shell_policy ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS user_shell_policy_all ON user_shell_policy;
CREATE POLICY user_shell_policy_all ON user_shell_policy FOR ALL
    USING (user_id = current_setting('app.current_user_id', true)::varchar);

GRANT SELECT, INSERT, UPDATE, DELETE ON user_shell_policy TO bastion_user;
