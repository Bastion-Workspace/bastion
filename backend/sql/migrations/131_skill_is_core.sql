-- Skills: is_core flag controls catalog visibility (core skills always shown in condensed catalog).
ALTER TABLE agent_skills
  ADD COLUMN IF NOT EXISTS is_core BOOLEAN DEFAULT false;
