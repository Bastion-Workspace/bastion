-- Message tree branching: edit-and-resend with fork navigation and branch-aware checkpoints
-- Greenfield: also applied via backend/postgres_init/06_learning_and_message_branching.sql; keep for existing DBs.

ALTER TABLE conversations ADD COLUMN IF NOT EXISTS current_node_message_id VARCHAR(255);

CREATE TABLE IF NOT EXISTS conversation_branches (
    branch_id VARCHAR(255) PRIMARY KEY DEFAULT gen_random_uuid()::text,
    conversation_id VARCHAR(255) NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    parent_branch_id VARCHAR(255) REFERENCES conversation_branches(branch_id) ON DELETE SET NULL,
    forked_from_message_id VARCHAR(255) NOT NULL,
    first_message_id VARCHAR(255),
    thread_id_suffix VARCHAR(255) NOT NULL UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_branches_conversation ON conversation_branches(conversation_id);
CREATE INDEX IF NOT EXISTS idx_branches_forked_from ON conversation_branches(forked_from_message_id);

ALTER TABLE conversation_messages ADD COLUMN IF NOT EXISTS branch_id VARCHAR(255);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'fk_conversation_messages_branch_id'
    ) THEN
        ALTER TABLE conversation_messages
            ADD CONSTRAINT fk_conversation_messages_branch_id
            FOREIGN KEY (branch_id) REFERENCES conversation_branches(branch_id) ON DELETE SET NULL;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_messages_branch ON conversation_messages(branch_id);

-- Backfill linear parent chain where missing
WITH ordered AS (
    SELECT message_id, conversation_id, sequence_number,
           LAG(message_id) OVER (PARTITION BY conversation_id ORDER BY sequence_number) AS prev_message_id
    FROM conversation_messages
)
UPDATE conversation_messages cm
SET parent_message_id = o.prev_message_id
FROM ordered o
WHERE cm.message_id = o.message_id
  AND (cm.parent_message_id IS NULL OR cm.parent_message_id = '')
  AND o.prev_message_id IS NOT NULL;

-- Backfill current node to latest message per conversation
UPDATE conversations c
SET current_node_message_id = sub.message_id
FROM (
    SELECT DISTINCT ON (conversation_id) conversation_id, message_id
    FROM conversation_messages
    ORDER BY conversation_id, sequence_number DESC
) sub
WHERE c.conversation_id = sub.conversation_id
  AND (c.current_node_message_id IS NULL OR c.current_node_message_id = '');

ALTER TABLE conversation_branches ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS conversation_branches_select_policy ON conversation_branches;
CREATE POLICY conversation_branches_select_policy ON conversation_branches
    FOR SELECT USING (
        conversation_id IN (
            SELECT conversation_id FROM conversations
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
            OR current_setting('app.current_user_role', true) = 'admin'
        )
    );

DROP POLICY IF EXISTS conversation_branches_insert_policy ON conversation_branches;
CREATE POLICY conversation_branches_insert_policy ON conversation_branches
    FOR INSERT WITH CHECK (true);

DROP POLICY IF EXISTS conversation_branches_update_policy ON conversation_branches;
CREATE POLICY conversation_branches_update_policy ON conversation_branches
    FOR UPDATE USING (
        conversation_id IN (
            SELECT conversation_id FROM conversations
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
            OR current_setting('app.current_user_role', true) = 'admin'
        )
    );

DROP POLICY IF EXISTS conversation_branches_delete_policy ON conversation_branches;
CREATE POLICY conversation_branches_delete_policy ON conversation_branches
    FOR DELETE USING (
        conversation_id IN (
            SELECT conversation_id FROM conversations
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
            OR current_setting('app.current_user_role', true) = 'admin'
        )
    );

GRANT ALL PRIVILEGES ON conversation_branches TO bastion_user;
