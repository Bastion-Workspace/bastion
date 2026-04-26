-- Empty user_admin_tts_provider previously meant "voice-service / deployer default".
-- Application default is now browser TTS for greenfield users (missing key).
-- Persist explicit 'server' for accounts that already stored blank server-default intent.
UPDATE user_settings
SET value = 'server',
    updated_at = CURRENT_TIMESTAMP
WHERE key = 'user_admin_tts_provider'
  AND (value IS NULL OR btrim(value) = '');
