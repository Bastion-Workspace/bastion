-- Append-only brief snapshots for agent lines (diff / history).
CREATE TABLE IF NOT EXISTS agent_line_brief_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    line_id UUID NOT NULL REFERENCES agent_lines(id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    source VARCHAR(200) NOT NULL DEFAULT 'heartbeat_report',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_line_brief_snapshots_line_created
    ON agent_line_brief_snapshots(line_id, created_at DESC);

ALTER TABLE agent_line_brief_snapshots ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS agent_line_brief_snapshots_all ON agent_line_brief_snapshots;
CREATE POLICY agent_line_brief_snapshots_all ON agent_line_brief_snapshots FOR ALL
  USING (
    current_setting('app.current_user_role', true) = 'admin'
    OR user_id = current_setting('app.current_user_id', true)::varchar
  )
  WITH CHECK (
    current_setting('app.current_user_role', true) = 'admin'
    OR user_id = current_setting('app.current_user_id', true)::varchar
  );
