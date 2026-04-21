-- Per-user pinned library documents (Home dashboard widget + future UI)

CREATE TABLE IF NOT EXISTS user_document_pins (
    pin_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    document_id VARCHAR(255) NOT NULL REFERENCES document_metadata(document_id) ON DELETE CASCADE,
    label TEXT,
    sort_order INT NOT NULL DEFAULT 0,
    pinned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, document_id)
);

CREATE INDEX IF NOT EXISTS idx_user_document_pins_user_sort
    ON user_document_pins(user_id, sort_order);

GRANT SELECT, INSERT, UPDATE, DELETE ON user_document_pins TO bastion_user;
