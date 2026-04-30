-- Mobile push tokens (Expo / FCM relay) and notification delivery audit log.
-- Greenfield: also merged into 01_init.sql.

CREATE TABLE IF NOT EXISTS mobile_push_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    token TEXT NOT NULL,
    platform VARCHAR(20) NOT NULL,
    device_id VARCHAR(255) NOT NULL,
    app_version TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_used_at TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ,
    UNIQUE (user_id, device_id)
);

CREATE INDEX IF NOT EXISTS idx_mobile_push_tokens_user_id ON mobile_push_tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_mobile_push_tokens_revoked ON mobile_push_tokens(revoked_at) WHERE revoked_at IS NULL;

CREATE TABLE IF NOT EXISTS notification_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    notification_id UUID NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    conversation_id VARCHAR(255),
    channel VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_notification_log_user_created ON notification_log(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_notification_log_notification_id ON notification_log(notification_id);
