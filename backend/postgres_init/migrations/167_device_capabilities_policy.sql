-- Add capabilities_policy JSONB column to device_tokens for per-proxy capability + path/command policy.
-- Policy is authoritative on the server: pushed to the daemon via WebSocket on register and on save.

ALTER TABLE device_tokens
    ADD COLUMN IF NOT EXISTS capabilities_policy JSONB NOT NULL DEFAULT '{}'::jsonb;
