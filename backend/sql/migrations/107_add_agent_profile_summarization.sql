-- Agent Factory: persist summarization settings for custom agent history compression
ALTER TABLE agent_profiles ADD COLUMN IF NOT EXISTS summary_threshold_tokens INTEGER NOT NULL DEFAULT 5000;
ALTER TABLE agent_profiles ADD COLUMN IF NOT EXISTS summary_keep_messages INTEGER NOT NULL DEFAULT 10;
