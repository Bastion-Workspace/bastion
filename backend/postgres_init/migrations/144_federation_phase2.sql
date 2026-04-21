-- ========================================
-- FEDERATION PHASE 2 — federated users, federated rooms, nullable sender
-- Idempotent: safe to run multiple times.
-- ========================================

-- Extend room_type_enum (pattern from 006_add_teams_system.sql)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_enum
        WHERE enumlabel = 'federated'
          AND enumtypid = (SELECT oid FROM pg_type WHERE typname = 'room_type_enum')
    ) THEN
        ALTER TYPE room_type_enum ADD VALUE 'federated';
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS federated_users (
    federated_user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    peer_id           UUID NOT NULL REFERENCES federation_peers(peer_id) ON DELETE CASCADE,
    remote_user_id    TEXT NOT NULL,
    federated_address TEXT NOT NULL,
    display_name      TEXT,
    avatar_url        TEXT,
    last_seen_at      TIMESTAMPTZ,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (peer_id, remote_user_id),
    UNIQUE (federated_address)
);

CREATE INDEX IF NOT EXISTS idx_federated_users_peer ON federated_users (peer_id);

ALTER TABLE chat_rooms ADD COLUMN IF NOT EXISTS federation_metadata JSONB;

ALTER TABLE chat_messages
    ADD COLUMN IF NOT EXISTS federated_sender_id UUID REFERENCES federated_users(federated_user_id);

ALTER TABLE chat_messages
    ALTER COLUMN sender_id DROP NOT NULL;

CREATE INDEX IF NOT EXISTS idx_chat_messages_federated_sender
    ON chat_messages (federated_sender_id)
    WHERE federated_sender_id IS NOT NULL;

GRANT ALL PRIVILEGES ON federated_users TO bastion_user;

ALTER TABLE federated_users ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS federated_users_select_policy ON federated_users;
DROP POLICY IF EXISTS federated_users_insert_policy ON federated_users;
DROP POLICY IF EXISTS federated_users_update_policy ON federated_users;
DROP POLICY IF EXISTS federated_users_delete_policy ON federated_users;

CREATE POLICY federated_users_select_policy ON federated_users
    FOR SELECT USING (current_setting('app.current_user_role', true) = 'admin');

CREATE POLICY federated_users_insert_policy ON federated_users
    FOR INSERT WITH CHECK (current_setting('app.current_user_role', true) = 'admin');

CREATE POLICY federated_users_update_policy ON federated_users
    FOR UPDATE USING (current_setting('app.current_user_role', true) = 'admin');

CREATE POLICY federated_users_delete_policy ON federated_users
    FOR DELETE USING (current_setting('app.current_user_role', true) = 'admin');
