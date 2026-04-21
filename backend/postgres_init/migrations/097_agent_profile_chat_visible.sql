-- Whether this agent appears in the chat @mention dropdown. Does not affect inter-agent handle resolution.
ALTER TABLE agent_profiles ADD COLUMN IF NOT EXISTS chat_visible BOOLEAN NOT NULL DEFAULT true;
COMMENT ON COLUMN agent_profiles.chat_visible IS 'Whether this agent appears in the chat @mention dropdown. Does not affect inter-agent handle resolution.';
