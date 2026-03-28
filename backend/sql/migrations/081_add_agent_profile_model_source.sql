-- Add optional model source metadata to agent_profiles for Agent Factory soft-retarget and notices.

ALTER TABLE agent_profiles
  ADD COLUMN IF NOT EXISTS model_source VARCHAR(50),
  ADD COLUMN IF NOT EXISTS model_provider_type VARCHAR(50);

COMMENT ON COLUMN agent_profiles.model_source IS 'Model source when model_preference was set: admin or user';
COMMENT ON COLUMN agent_profiles.model_provider_type IS 'Provider type when model_preference was set: openai, openrouter, groq, ollama, vllm';
