-- Add include_user_facts flag for Agent Factory (inject user fact store into system prompt for every LLM step)
ALTER TABLE agent_profiles ADD COLUMN IF NOT EXISTS include_user_facts BOOLEAN NOT NULL DEFAULT false;
