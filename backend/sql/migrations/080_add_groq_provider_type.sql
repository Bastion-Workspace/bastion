-- Add groq to allowed user_llm_providers provider_type.

ALTER TABLE user_llm_providers DROP CONSTRAINT IF EXISTS chk_provider_type;
ALTER TABLE user_llm_providers ADD CONSTRAINT chk_provider_type
    CHECK (provider_type IN ('openai', 'openrouter', 'ollama', 'vllm', 'groq'));
