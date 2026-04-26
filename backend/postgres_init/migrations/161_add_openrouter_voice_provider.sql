-- OpenRouter TTS as BYOK / voice-service provider (OpenAI-compatible /audio/speech).

ALTER TABLE user_voice_providers DROP CONSTRAINT IF EXISTS chk_voice_provider_type;

ALTER TABLE user_voice_providers ADD CONSTRAINT chk_voice_provider_type CHECK (
    provider_type IN (
        'elevenlabs',
        'openai',
        'deepgram',
        'whisper_api',
        'hedra',
        'openrouter'
    )
);
