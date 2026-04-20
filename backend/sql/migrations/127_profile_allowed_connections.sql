-- Profile-level allowlist for external connections (email, calendar, code_platform, etc.).
-- Empty array = no restriction (all user connections allowed; backward compatible).

ALTER TABLE agent_profiles
  ADD COLUMN IF NOT EXISTS allowed_connections JSONB DEFAULT '[]'::jsonb;

COMMENT ON COLUMN agent_profiles.allowed_connections IS
  'Allowlist of external connections this agent may use. Empty array = all user connections allowed.';
