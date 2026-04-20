-- Skill composition: depends_on allows skills to declare dependencies on other skills by slug.
ALTER TABLE agent_skills
    ADD COLUMN IF NOT EXISTS depends_on TEXT[] DEFAULT '{}';
