-- Add per-profile persona toggle for custom agents
ALTER TABLE agent_profiles ADD COLUMN IF NOT EXISTS persona_enabled BOOLEAN DEFAULT false;
