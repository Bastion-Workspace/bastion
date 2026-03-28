-- Add include_user_context flag for Agent Factory system prompt layer
ALTER TABLE agent_profiles ADD COLUMN IF NOT EXISTS include_user_context BOOLEAN NOT NULL DEFAULT false;
