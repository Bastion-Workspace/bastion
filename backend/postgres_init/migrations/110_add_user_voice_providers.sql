-- Per-user TTS/STT API keys (BYOK). Preferences in user_settings:
-- use_admin_tts, use_admin_stt (default true), user_tts_provider_id, user_tts_voice_id, user_stt_provider_id

CREATE TABLE user_voice_providers (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    provider_type VARCHAR(50) NOT NULL,
    provider_role VARCHAR(10) NOT NULL,
    display_name VARCHAR(255),
    encrypted_api_key TEXT,
    base_url VARCHAR(512),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_voice_provider_type CHECK (
        provider_type IN ('elevenlabs', 'openai', 'deepgram', 'whisper_api')
    ),
    CONSTRAINT chk_voice_provider_role CHECK (provider_role IN ('tts', 'stt'))
);

CREATE UNIQUE INDEX idx_user_voice_providers_user_role_type_base
ON user_voice_providers (user_id, provider_role, provider_type, COALESCE(base_url, ''));

CREATE INDEX idx_user_voice_providers_user_id ON user_voice_providers(user_id);
