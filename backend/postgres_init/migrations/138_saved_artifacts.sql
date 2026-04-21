-- Saved chat artifacts: persistent storage independent of conversations (dashboard widgets, share, export).

CREATE TABLE IF NOT EXISTS saved_artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    artifact_type VARCHAR(20) NOT NULL
        CHECK (artifact_type IN ('html', 'mermaid', 'chart', 'svg', 'react')),
    code TEXT NOT NULL,
    language VARCHAR(20),
    share_token VARCHAR(64) UNIQUE,
    is_public BOOLEAN DEFAULT FALSE,
    source_conversation_id VARCHAR(255),
    source_message_id VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_saved_artifacts_user
    ON saved_artifacts (user_id);

CREATE INDEX IF NOT EXISTS idx_saved_artifacts_share_token
    ON saved_artifacts (share_token) WHERE share_token IS NOT NULL;
