-- Skill composition: depends_on allows skills to declare dependencies on other skills by slug.
-- Greenfield: column is in backend/postgres_init/01_init.sql; keep for existing DBs.
ALTER TABLE agent_skills
    ADD COLUMN IF NOT EXISTS depends_on TEXT[] DEFAULT '{}';
