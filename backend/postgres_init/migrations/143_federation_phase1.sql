-- ========================================
-- FEDERATION PHASE 1 — peers + outbox
-- Idempotent: safe to run multiple times.
-- ========================================

CREATE TABLE IF NOT EXISTS federation_peers (
    peer_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    peer_url           TEXT NOT NULL,
    peer_public_key    TEXT NOT NULL,
    display_name       TEXT,
    status             TEXT NOT NULL DEFAULT 'pending',
    connectivity_mode  TEXT NOT NULL DEFAULT 'bidirectional',
    allowed_scopes     TEXT[] DEFAULT ARRAY['messaging']::TEXT[],
    initiated_by       VARCHAR(255) REFERENCES users(user_id) ON DELETE SET NULL,
    is_inbound         BOOLEAN NOT NULL DEFAULT FALSE,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    activated_at       TIMESTAMPTZ,
    metadata           JSONB NOT NULL DEFAULT '{}'::jsonb,
    CONSTRAINT federation_peers_status_check
        CHECK (status IN ('pending', 'active', 'suspended', 'revoked')),
    CONSTRAINT federation_peers_connectivity_check
        CHECK (connectivity_mode IN ('bidirectional', 'asymmetric_caller', 'asymmetric_listener')),
    CONSTRAINT federation_peers_peer_url_unique UNIQUE (peer_url)
);

CREATE INDEX IF NOT EXISTS idx_federation_peers_status
    ON federation_peers (status);
CREATE INDEX IF NOT EXISTS idx_federation_peers_inbound_pending
    ON federation_peers (is_inbound, status)
    WHERE status = 'pending' AND is_inbound = TRUE;

CREATE TABLE IF NOT EXISTS federation_outbox (
    outbox_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    peer_id         UUID NOT NULL REFERENCES federation_peers(peer_id) ON DELETE CASCADE,
    event_type      TEXT NOT NULL,
    payload         JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    picked_up_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_federation_outbox_pending
    ON federation_outbox (peer_id, created_at)
    WHERE picked_up_at IS NULL;

GRANT ALL PRIVILEGES ON federation_peers TO bastion_user;
GRANT ALL PRIVILEGES ON federation_outbox TO bastion_user;

ALTER TABLE federation_peers ENABLE ROW LEVEL SECURITY;
ALTER TABLE federation_outbox ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS federation_peers_select_policy ON federation_peers;
DROP POLICY IF EXISTS federation_peers_insert_policy ON federation_peers;
DROP POLICY IF EXISTS federation_peers_update_policy ON federation_peers;
DROP POLICY IF EXISTS federation_peers_delete_policy ON federation_peers;

CREATE POLICY federation_peers_select_policy ON federation_peers
    FOR SELECT USING (current_setting('app.current_user_role', true) = 'admin');

CREATE POLICY federation_peers_insert_policy ON federation_peers
    FOR INSERT WITH CHECK (current_setting('app.current_user_role', true) = 'admin');

CREATE POLICY federation_peers_update_policy ON federation_peers
    FOR UPDATE USING (current_setting('app.current_user_role', true) = 'admin');

CREATE POLICY federation_peers_delete_policy ON federation_peers
    FOR DELETE USING (current_setting('app.current_user_role', true) = 'admin');

DROP POLICY IF EXISTS federation_outbox_select_policy ON federation_outbox;
DROP POLICY IF EXISTS federation_outbox_insert_policy ON federation_outbox;
DROP POLICY IF EXISTS federation_outbox_update_policy ON federation_outbox;
DROP POLICY IF EXISTS federation_outbox_delete_policy ON federation_outbox;

CREATE POLICY federation_outbox_select_policy ON federation_outbox
    FOR SELECT USING (current_setting('app.current_user_role', true) = 'admin');

CREATE POLICY federation_outbox_insert_policy ON federation_outbox
    FOR INSERT WITH CHECK (current_setting('app.current_user_role', true) = 'admin');

CREATE POLICY federation_outbox_update_policy ON federation_outbox
    FOR UPDATE USING (current_setting('app.current_user_role', true) = 'admin');

CREATE POLICY federation_outbox_delete_policy ON federation_outbox
    FOR DELETE USING (current_setting('app.current_user_role', true) = 'admin');
