-- Remove profile-level skill_ids; skills are now step-level only.
ALTER TABLE agent_profiles DROP COLUMN IF EXISTS skill_ids;
