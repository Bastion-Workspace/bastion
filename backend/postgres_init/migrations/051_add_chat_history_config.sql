-- Add per-profile chat history configuration for custom agents
ALTER TABLE agent_profiles ADD COLUMN IF NOT EXISTS chat_history_enabled BOOLEAN DEFAULT false;
ALTER TABLE agent_profiles ADD COLUMN IF NOT EXISTS chat_history_lookback INTEGER DEFAULT 10;
