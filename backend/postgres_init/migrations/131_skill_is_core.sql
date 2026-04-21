-- Skills: is_core flag controls catalog visibility (core skills always shown in condensed catalog).
-- Greenfield: column is in backend/postgres_init/01_init.sql; keep for existing DBs.
ALTER TABLE agent_skills
  ADD COLUMN IF NOT EXISTS is_core BOOLEAN DEFAULT false;
