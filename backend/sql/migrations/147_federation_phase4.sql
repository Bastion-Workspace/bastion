-- ========================================
-- FEDERATION PHASE 4 — messaging parity, abuse controls, privacy
-- Idempotent: safe to run multiple times.
-- ========================================

ALTER TABLE chat_messages
    ADD COLUMN IF NOT EXISTS federation_delivery_status TEXT;

ALTER TABLE federated_users
    ADD COLUMN IF NOT EXISTS is_blocked BOOLEAN NOT NULL DEFAULT FALSE;

ALTER TABLE federated_users
    ADD COLUMN IF NOT EXISTS presence_status TEXT;

ALTER TABLE federated_users
    ADD COLUMN IF NOT EXISTS presence_updated_at TIMESTAMPTZ;

ALTER TABLE users
    ADD COLUMN IF NOT EXISTS federation_share_read_receipts BOOLEAN NOT NULL DEFAULT TRUE;

ALTER TABLE users
    ADD COLUMN IF NOT EXISTS federation_share_presence BOOLEAN NOT NULL DEFAULT TRUE;

CREATE INDEX IF NOT EXISTS idx_federated_users_blocked
    ON federated_users (peer_id) WHERE is_blocked = TRUE;

-- Reactions from federated users (no local users.user_id row)
ALTER TABLE message_reactions
    ADD COLUMN IF NOT EXISTS federated_user_id UUID REFERENCES federated_users(federated_user_id) ON DELETE CASCADE;

ALTER TABLE message_reactions ALTER COLUMN user_id DROP NOT NULL;

ALTER TABLE message_reactions DROP CONSTRAINT IF EXISTS message_reactions_message_id_user_id_emoji_key;

ALTER TABLE message_reactions
    ADD CONSTRAINT message_reactions_actor_chk CHECK (
        (user_id IS NOT NULL AND federated_user_id IS NULL)
        OR (user_id IS NULL AND federated_user_id IS NOT NULL)
    );

CREATE UNIQUE INDEX IF NOT EXISTS idx_message_reactions_local_user_emoji
    ON message_reactions (message_id, user_id, emoji)
    WHERE user_id IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_message_reactions_federated_user_emoji
    ON message_reactions (message_id, federated_user_id, emoji)
    WHERE federated_user_id IS NOT NULL;

DROP POLICY IF EXISTS message_reactions_insert_policy ON message_reactions;
CREATE POLICY message_reactions_insert_policy ON message_reactions
    FOR INSERT WITH CHECK (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

DROP POLICY IF EXISTS message_reactions_delete_policy ON message_reactions;
CREATE POLICY message_reactions_delete_policy ON message_reactions
    FOR DELETE USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );
