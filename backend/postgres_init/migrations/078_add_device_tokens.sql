-- Device tokens for Bastion Local Proxy daemon authentication.
-- Tokens are created in the UI; the daemon uses the raw token to connect via WebSocket.

CREATE TABLE IF NOT EXISTS device_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL,
    device_name TEXT NOT NULL,
    last_connected_at TIMESTAMPTZ,
    last_ip TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    revoked_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_device_tokens_user_id ON device_tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_device_tokens_token_hash ON device_tokens(token_hash);
CREATE INDEX IF NOT EXISTS idx_device_tokens_revoked ON device_tokens(revoked_at) WHERE revoked_at IS NULL;
