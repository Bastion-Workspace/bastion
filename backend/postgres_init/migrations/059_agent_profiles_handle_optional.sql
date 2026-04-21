-- Allow agent_profiles.handle to be NULL for schedule/Run-only agents (not @mentionable in chat).
-- UNIQUE(user_id, handle) still applies; multiple NULL handles per user are allowed in PostgreSQL.
ALTER TABLE agent_profiles ALTER COLUMN handle DROP NOT NULL;
