-- Row-level security for code workspaces and indexed chunks (parity with agent_memory / agent_approval_queue).

ALTER TABLE code_workspaces ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS code_workspaces_all ON code_workspaces;
CREATE POLICY code_workspaces_all ON code_workspaces FOR ALL
  USING (
    current_setting('app.current_user_role', true) = 'admin'
    OR user_id = current_setting('app.current_user_id', true)::varchar
  )
  WITH CHECK (
    current_setting('app.current_user_role', true) = 'admin'
    OR user_id = current_setting('app.current_user_id', true)::varchar
  );

ALTER TABLE code_chunks ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS code_chunks_all ON code_chunks;
CREATE POLICY code_chunks_all ON code_chunks FOR ALL
  USING (
    current_setting('app.current_user_role', true) = 'admin'
    OR user_id = current_setting('app.current_user_id', true)::varchar
  )
  WITH CHECK (
    current_setting('app.current_user_role', true) = 'admin'
    OR user_id = current_setting('app.current_user_id', true)::varchar
  );
