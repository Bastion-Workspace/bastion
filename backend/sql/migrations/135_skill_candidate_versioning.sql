-- Skill candidate versioning: A/B testing for skill versions.
ALTER TABLE agent_skills
    ADD COLUMN IF NOT EXISTS is_candidate BOOLEAN DEFAULT false,
    ADD COLUMN IF NOT EXISTS candidate_weight INTEGER DEFAULT 0;
