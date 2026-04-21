-- Skills: optional external connection types required for skill to apply (email, calendar, code_platform).
-- Greenfield: column is in backend/postgres_init/01_init.sql CREATE agent_skills; keep for existing DBs.
ALTER TABLE agent_skills
  ADD COLUMN IF NOT EXISTS required_connection_types TEXT[] DEFAULT '{}';
