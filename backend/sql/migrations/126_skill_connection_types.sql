-- Skills: optional external connection types required for skill to apply (email, calendar, code_platform).
ALTER TABLE agent_skills
  ADD COLUMN IF NOT EXISTS required_connection_types TEXT[] DEFAULT '{}';
