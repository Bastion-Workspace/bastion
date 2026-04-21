-- Skill candidate versioning: A/B testing for skill versions.
-- Greenfield: columns are in backend/postgres_init/01_init.sql; keep for existing DBs.
ALTER TABLE agent_skills
    ADD COLUMN IF NOT EXISTS is_candidate BOOLEAN DEFAULT false,
    ADD COLUMN IF NOT EXISTS candidate_weight INTEGER DEFAULT 0;
