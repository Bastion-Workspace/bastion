-- Greenfield-only: final agent line watch + workspace DDL (not 047/094/101 brownfield chain).
-- Run from 09_greenfield_extensions.sql after agent_lines exists in 01_init.sql.

-- Line watches (which agents watch which lines for team room triggers)
CREATE TABLE IF NOT EXISTS agent_line_watches (
    agent_profile_id UUID NOT NULL REFERENCES agent_profiles(id) ON DELETE CASCADE,
    line_id UUID NOT NULL REFERENCES agent_lines(id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    trigger_on_new_post BOOLEAN DEFAULT true,
    respond_as VARCHAR(20) DEFAULT 'comment',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (agent_profile_id, line_id)
);

CREATE INDEX IF NOT EXISTS idx_agent_line_watches_line
    ON agent_line_watches(line_id) WHERE trigger_on_new_post = true;

-- Shared key-value workspace per line (Blackboard pattern)
CREATE TABLE IF NOT EXISTS agent_line_workspace (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    line_id UUID NOT NULL REFERENCES agent_lines(id) ON DELETE CASCADE,
    key VARCHAR(255) NOT NULL,
    value TEXT NOT NULL,
    updated_by_agent_id UUID REFERENCES agent_profiles(id) ON DELETE SET NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(line_id, key)
);

CREATE INDEX IF NOT EXISTS idx_agent_line_workspace_line_id ON agent_line_workspace(line_id);

-- Grants (048/039/071 migrations omit these; 01 RLS DO blocks ran before these tables existed)
GRANT SELECT, INSERT, UPDATE, DELETE ON conversation_message_attachments TO bastion_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON agent_email_watches TO bastion_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON agent_folder_watches TO bastion_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON agent_conversation_watches TO bastion_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON browser_session_states TO bastion_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON agent_line_watches TO bastion_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON agent_line_workspace TO bastion_user;

-- conversation_message_attachments: access via owning conversation user
ALTER TABLE conversation_message_attachments ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS conversation_message_attachments_all ON conversation_message_attachments;
CREATE POLICY conversation_message_attachments_all ON conversation_message_attachments FOR ALL
  USING (
    current_setting('app.current_user_role', true) = 'admin'
    OR EXISTS (
      SELECT 1
      FROM conversation_messages m
      JOIN conversations c ON c.conversation_id = m.conversation_id
      WHERE m.message_id = conversation_message_attachments.message_id
        AND c.user_id = current_setting('app.current_user_id', true)::varchar
    )
  )
  WITH CHECK (
    current_setting('app.current_user_role', true) = 'admin'
    OR EXISTS (
      SELECT 1
      FROM conversation_messages m
      JOIN conversations c ON c.conversation_id = m.conversation_id
      WHERE m.message_id = conversation_message_attachments.message_id
        AND c.user_id = current_setting('app.current_user_id', true)::varchar
    )
  );

ALTER TABLE browser_session_states ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS browser_session_states_user ON browser_session_states;
CREATE POLICY browser_session_states_user ON browser_session_states FOR ALL
  USING (
    current_setting('app.current_user_role', true) = 'admin'
    OR user_id = current_setting('app.current_user_id', true)::varchar
  )
  WITH CHECK (
    current_setting('app.current_user_role', true) = 'admin'
    OR user_id = current_setting('app.current_user_id', true)::varchar
  );

-- RLS for 048 watch tables (policies mirror 01_init.sql DO block)
ALTER TABLE agent_email_watches ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS agent_email_watches_all ON agent_email_watches;
CREATE POLICY agent_email_watches_all ON agent_email_watches FOR ALL
  USING (
    current_setting('app.current_user_role', true) = 'admin'
    OR user_id = current_setting('app.current_user_id', true)::varchar
  )
  WITH CHECK (
    current_setting('app.current_user_role', true) = 'admin'
    OR user_id = current_setting('app.current_user_id', true)::varchar
  );

ALTER TABLE agent_folder_watches ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS agent_folder_watches_all ON agent_folder_watches;
CREATE POLICY agent_folder_watches_all ON agent_folder_watches FOR ALL
  USING (
    current_setting('app.current_user_role', true) = 'admin'
    OR user_id = current_setting('app.current_user_id', true)::varchar
  )
  WITH CHECK (
    current_setting('app.current_user_role', true) = 'admin'
    OR user_id = current_setting('app.current_user_id', true)::varchar
  );

ALTER TABLE agent_conversation_watches ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS agent_conversation_watches_all ON agent_conversation_watches;
CREATE POLICY agent_conversation_watches_all ON agent_conversation_watches FOR ALL
  USING (
    current_setting('app.current_user_role', true) = 'admin'
    OR user_id = current_setting('app.current_user_id', true)::varchar
  )
  WITH CHECK (
    current_setting('app.current_user_role', true) = 'admin'
    OR user_id = current_setting('app.current_user_id', true)::varchar
  );

ALTER TABLE agent_line_watches ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS agent_line_watches_all ON agent_line_watches;
CREATE POLICY agent_line_watches_all ON agent_line_watches FOR ALL
  USING (
    current_setting('app.current_user_role', true) = 'admin'
    OR user_id = current_setting('app.current_user_id', true)::varchar
  )
  WITH CHECK (
    current_setting('app.current_user_role', true) = 'admin'
    OR user_id = current_setting('app.current_user_id', true)::varchar
  );

ALTER TABLE agent_line_workspace ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS agent_line_workspace_all ON agent_line_workspace;
CREATE POLICY agent_line_workspace_all ON agent_line_workspace FOR ALL
  USING (
    current_setting('app.current_user_role', true) = 'admin'
    OR EXISTS (
      SELECT 1 FROM agent_lines t
      WHERE t.id = agent_line_workspace.line_id
        AND t.user_id = current_setting('app.current_user_id', true)::varchar
    )
  )
  WITH CHECK (
    current_setting('app.current_user_role', true) = 'admin'
    OR EXISTS (
      SELECT 1 FROM agent_lines t
      WHERE t.id = agent_line_workspace.line_id
        AND t.user_id = current_setting('app.current_user_id', true)::varchar
    )
  );
