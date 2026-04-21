-- Session-level memory: flag conversations for post-idle combined analysis; tiered episode retention.
-- Greenfield: columns + indexes are also in backend/postgres_init/01_init.sql; this file remains for existing DBs.
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS needs_session_summary BOOLEAN NOT NULL DEFAULT FALSE;

ALTER TABLE user_episodes ADD COLUMN IF NOT EXISTS is_aged BOOLEAN NOT NULL DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_conversations_needs_summary_updated
    ON conversations (needs_session_summary, updated_at)
    WHERE needs_session_summary = TRUE;

CREATE INDEX IF NOT EXISTS idx_user_episodes_created_aged ON user_episodes (created_at, is_aged);
