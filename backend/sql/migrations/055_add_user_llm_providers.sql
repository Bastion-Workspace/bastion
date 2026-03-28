-- User-level LLM providers: per-user API keys and base URLs for OpenAI, OpenRouter, Ollama, vLLM.
-- use_admin_models toggle lives in user_settings (key: use_admin_models, default: true).

CREATE TABLE user_llm_providers (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    provider_type VARCHAR(50) NOT NULL,
    display_name VARCHAR(255),
    encrypted_api_key TEXT,
    base_url VARCHAR(500),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_provider_type CHECK (provider_type IN ('openai', 'openrouter', 'ollama', 'vllm'))
);

CREATE UNIQUE INDEX idx_user_llm_providers_user_type_base
ON user_llm_providers (user_id, provider_type, COALESCE(base_url, ''));

CREATE INDEX idx_user_llm_providers_user_id ON user_llm_providers(user_id);

CREATE TABLE user_enabled_models (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    provider_id BIGINT NOT NULL REFERENCES user_llm_providers(id) ON DELETE CASCADE,
    model_id VARCHAR(255) NOT NULL,
    display_name VARCHAR(255),
    is_enabled BOOLEAN DEFAULT true,
    UNIQUE(user_id, provider_id, model_id)
);

CREATE INDEX idx_user_enabled_models_user_id ON user_enabled_models(user_id);
CREATE INDEX idx_user_enabled_models_provider_id ON user_enabled_models(provider_id);
